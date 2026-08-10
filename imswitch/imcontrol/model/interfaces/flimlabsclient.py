"""
flimlabsclient.py - Backend (Python) client for the FLIM LABS flim-imager 2.x
standalone server.

The server is a Rust/warp process (normally the `flim-imager-server` Docker
container, port 5249) that owns the ZestSC3 acquisition card. It exposes REST
for control and a binary WebSocket (``/data``) for the acquisition stream.

This module is the backend counterpart of the browser-side client in
``frontend/src/backendapi/apiFlimLabs.js`` +
``frontend/src/utils/flimBinaryParser.js``. Having it in Python lets ImSwitch
treat the FLIM card as a normal 2D detector (see FLIMLabsDetectorManager), so
acquisitions can be driven from ExperimentController while the stage moves,
instead of only from an open browser tab.

Image geometry note: the card derives frame/line/pixel boundaries purely from
the external trigger bits produced by the galvo scanner (frame=bit31,
line=bit30, pixel=bit29 of each photon dword). The scanner must therefore be
running for frames to arrive - this client only arms the card and assembles
what the card sends.

Only one consumer may drain ``/data`` at a time (upstream design): if the
ImSwitch backend is streaming, the FLIM LABS web UI must not be connected to
the same server, and vice versa.
"""

import json
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

from imswitch.imcommon.model import initLogger

try:
    import websocket  # websocket-client
    IS_WEBSOCKET_AVAILABLE = True
except ImportError:
    IS_WEBSOCKET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Binary WebSocket protocol (flim-imager 2.x)
# ---------------------------------------------------------------------------
# The server concatenates up to 100 messages into one binary frame with no
# framing headers, so every tag must be decodable: an unknown tag has unknown
# length and forces dropping the rest of the frame. All tags are therefore
# handled here even though the detector only consumes a few of them.

class FlimMsg:
    LINE = 0
    CURVE = 1
    CALIBRATION = 2
    PHASOR = 3
    IMAGING_END = 4
    END_EXPERIMENT = 6
    LASER_PERIOD = 8
    CHANNELS_DETECTION = 10
    SKIP_DATA = 11
    CPS = 12
    SBR = 13
    INTENSITY_LIMITS = 14
    INTENSITY_HISTOGRAM = 15
    LINE_INSTA_FLIM = 16
    INSTA_FLIM_LIMITS = 17
    INSTA_FLIM_HISTOGRAM = 18
    INTENSITY_TRACE = 19
    PEAK_KCPS_MATRIX = 20
    COMPLETED_FRAME_LINES = 21
    COMPLETED_FRAME_INSTA_FLIM_LINES = 22
    IMAGE_DIAGNOSTIC = 23
    IMAGE_DIAGNOSTIC_END = 24
    ROI_CURVE = 25
    PIXEL_DWELL_TIME = 26
    LASER_FREQUENCY = 27


class _Reader:
    """Little-endian cursor over one concatenated WebSocket chunk."""

    __slots__ = ('buf', 'off')

    def __init__(self, buf: bytes):
        self.buf = buf
        self.off = 0

    def remaining(self) -> int:
        return len(self.buf) - self.off

    def u8(self) -> int:
        v = self.buf[self.off]
        self.off += 1
        return v

    def u32(self) -> int:
        v = struct.unpack_from('<I', self.buf, self.off)[0]
        self.off += 4
        return v

    def u64(self) -> int:
        v = struct.unpack_from('<Q', self.buf, self.off)[0]
        self.off += 8
        return v

    def f32(self) -> float:
        v = struct.unpack_from('<f', self.buf, self.off)[0]
        self.off += 4
        return v

    def f64(self) -> float:
        v = struct.unpack_from('<d', self.buf, self.off)[0]
        self.off += 8
        return v

    def arr(self, n: int, dtype: str, itemsize: int) -> np.ndarray:
        a = np.frombuffer(self.buf, dtype=dtype, count=n, offset=self.off)
        self.off += n * itemsize
        return a

    def u32_array(self, n: int) -> np.ndarray:
        return self.arr(n, '<u4', 4)

    def f32_array(self, n: int) -> np.ndarray:
        return self.arr(n, '<f4', 4)

    def f64_array(self, n: int) -> np.ndarray:
        return self.arr(n, '<f8', 8)

    def skip(self, n: int) -> None:
        self.off += n


def parse_flim_chunk(buf: bytes) -> List[Dict[str, Any]]:
    """Parse one binary WebSocket chunk into a list of message dicts.

    Truncated or unknown trailing data stops parsing gracefully; whatever was
    decoded before that point is returned.
    """
    r = _Reader(buf)
    out: List[Dict[str, Any]] = []
    try:
        while r.remaining() >= 1:
            t = r.u8()
            if t == FlimMsg.LINE:
                step = r.u8(); frame = r.u32(); line = r.u32(); channel = r.u32()
                pixels = r.u32_array(r.u32())
                out.append({'type': t, 'step': step, 'frame': frame, 'line': line,
                            'channel': channel, 'pixels': pixels})
            elif t in (FlimMsg.CURVE, FlimMsg.ROI_CURVE):
                step = r.u8(); frame = r.u32(); channel = r.u32()
                data = r.u32_array(r.u32())
                out.append({'type': t, 'step': step, 'frame': frame,
                            'channel': channel, 'data': data})
            elif t == FlimMsg.CALIBRATION:
                out.append({'type': t, 'frame': r.u32(), 'channel': r.u32(),
                            'harmonic': r.u32(), 'phase': r.f64(),
                            'modulation': r.f64()})
            elif t == FlimMsg.PHASOR:
                step = r.u8(); frame = r.u32(); channel = r.u32(); harmonic = r.u32()
                g_rows = r.u32(); g_cols = r.u32()
                g_data = r.f64_array(g_rows * g_cols)
                s_rows = r.u32(); s_cols = r.u32()
                s_data = r.f64_array(s_rows * s_cols)
                out.append({'type': t, 'step': step, 'frame': frame, 'channel': channel,
                            'harmonic': harmonic, 'g': g_data, 's': s_data,
                            'shape': (g_rows, g_cols)})
            elif t == FlimMsg.IMAGING_END:
                n = r.u32()
                data_file = None
                if n > 0:
                    data_file = r.buf[r.off:r.off + n].decode('utf-8', 'replace')
                    r.skip(n)
                out.append({'type': t, 'dataFile': data_file,
                            'lastCompleteFrame': r.u32()})
            elif t in (FlimMsg.END_EXPERIMENT, FlimMsg.SKIP_DATA):
                out.append({'type': t})
            elif t == FlimMsg.LASER_PERIOD:
                out.append({'type': t, 'laserPeriod': r.f64(), 'frequency': r.f64()})
            elif t == FlimMsg.CHANNELS_DETECTION:
                sma = r.u32_array(r.u32())
                usb = r.u32_array(r.u32())
                flags = r.buf[r.off:r.off + 9]
                r.skip(9)
                out.append({'type': t, 'sma': sma, 'usb': usb, 'flags': flags})
            elif t == FlimMsg.CPS:
                out.append({'type': t, 'step': r.u8(), 'frame': r.u32(),
                            'channel': r.u32(), 'cps': r.u64()})
            elif t == FlimMsg.SBR:
                out.append({'type': t, 'step': r.u8(), 'frame': r.u32(),
                            'channel': r.u32(), 'sbr': r.f64()})
            elif t == FlimMsg.INTENSITY_LIMITS:
                out.append({'type': t, 'step': r.u8(), 'frame': r.u32(),
                            'channel': r.u32(), 'max': r.u64(), 'min': r.u64()})
            elif t == FlimMsg.INTENSITY_HISTOGRAM:
                step = r.u8(); frame = r.u32(); channel = r.u32()
                labels = r.u32_array(r.u32())
                counts = r.u32_array(r.u32())
                out.append({'type': t, 'step': step, 'frame': frame, 'channel': channel,
                            'labels': labels, 'counts': counts})
            elif t == FlimMsg.LINE_INSTA_FLIM:
                step = r.u8(); frame = r.u32(); line = r.u32(); channel = r.u32()
                n = r.u32()
                r.skip(n * 8)  # n interleaved (u32 intensity, f32 lifetime) pairs
                out.append({'type': t, 'step': step, 'frame': frame, 'line': line,
                            'channel': channel, 'n': n})
            elif t == FlimMsg.INSTA_FLIM_LIMITS:
                out.append({'type': t, 'step': r.u8(), 'frame': r.u32(),
                            'channel': r.u32(), 'max': r.f32(), 'min': r.f32()})
            elif t == FlimMsg.INSTA_FLIM_HISTOGRAM:
                step = r.u8(); frame = r.u32(); channel = r.u32()
                labels = r.f32_array(r.u32())
                counts = r.u32_array(r.u32())
                out.append({'type': t, 'step': step, 'frame': frame, 'channel': channel,
                            'labels': labels, 'counts': counts})
            elif t == FlimMsg.INTENSITY_TRACE:
                step = r.u8(); frame = r.u32(); channel = r.u32()
                bin_width = r.u32()
                bins = r.f32_array(r.u32())
                out.append({'type': t, 'step': step, 'frame': frame, 'channel': channel,
                            'binWidthMicros': bin_width, 'bins': bins})
            elif t == FlimMsg.PEAK_KCPS_MATRIX:
                step = r.u8(); frame = r.u32(); channel = r.u32()
                data = r.f32_array(r.u32())
                out.append({'type': t, 'step': step, 'frame': frame,
                            'channel': channel, 'data': data})
            elif t == FlimMsg.COMPLETED_FRAME_LINES:
                step = r.u8(); frame = r.u32(); channel = r.u32()
                n_lines = r.u32()
                lines = [r.u32_array(r.u32()) for _ in range(n_lines)]
                out.append({'type': t, 'step': step, 'frame': frame,
                            'channel': channel, 'lines': lines})
            elif t == FlimMsg.COMPLETED_FRAME_INSTA_FLIM_LINES:
                step = r.u8(); frame = r.u32(); channel = r.u32()
                n_lines = r.u32()
                for _ in range(n_lines):
                    r.skip(r.u32() * 8)
                out.append({'type': t, 'step': step, 'frame': frame,
                            'channel': channel, 'nLines': n_lines})
            elif t == FlimMsg.IMAGE_DIAGNOSTIC:
                frame = r.u32()
                n_cps = r.u32()
                cps = [{'channel': r.u32(), 'cps': r.u64()} for _ in range(n_cps)]
                pixel_dwell = r.f64() if r.u8() else None
                pixels_per_line = r.u32() if r.u8() else None
                line_time = r.f64() if r.u8() else None
                line_count = r.u32() if r.u8() else None
                frame_time = r.f64()
                det_w = r.u32() if r.u8() else None
                det_h = r.u32() if r.u8() else None
                out.append({'type': t, 'frame': frame, 'cpsPerChannel': cps,
                            'pixelDwellTime': pixel_dwell,
                            'pixelCountPerLine': pixels_per_line,
                            'lineTime': line_time, 'lineCount': line_count,
                            'frameTime': frame_time, 'detectedWidth': det_w,
                            'detectedHeight': det_h,
                            'pixelDetected': r.u8() != 0,
                            'lineDetected': r.u8() != 0,
                            'frameDetected': r.u8() != 0})
            elif t == FlimMsg.IMAGE_DIAGNOSTIC_END:
                out.append({'type': t, 'lastFrame': r.u32()})
            elif t == FlimMsg.PIXEL_DWELL_TIME:
                out.append({'type': t, 'step': r.u8(), 'dwellTime': r.f64()})
            elif t == FlimMsg.LASER_FREQUENCY:
                out.append({'type': t, 'frequency': r.f64(), 'laserPeriod': r.f64()})
            else:
                # Unknown tag: length unknown, the rest of the chunk is unusable
                break
    except (struct.error, IndexError, ValueError):
        # Truncated message at chunk end - keep what we parsed
        pass
    return out


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

# The step formerly called "imaging" is "tcspc" in v2; "scouting" is unchanged.
_V2_STEP = {'scouting': 'scouting', 'imaging': 'tcspc', 'tcspc': 'tcspc',
            'calibration': 'calibration', 'phasors': 'phasors'}

# Phasor density-histogram geometry. The universal semicircle lives at
# g in [0,1], s in [0,0.5]; the margins keep points on the rim visible.
PHASOR_W, PHASOR_H = 420, 280
PHASOR_G_MIN, PHASOR_G_MAX = -0.05, 1.05
PHASOR_S_MIN, PHASOR_S_MAX = -0.02, 0.65


def flim_calibration_reference_path(timestamp_sec: int,
                                    home: str = '/home/appuser') -> str:
    """Server-side path of the calibration JSON written by a calibration run.

    With export disabled the server writes
    ``<home>/.flim-labs/data/<timestamp>_imaging_calibration.json`` (see
    flim-lib ``export_calibration_data``). In the Docker container the server
    runs as ``appuser``.
    """
    return f'{home}/.flim-labs/data/{timestamp_sec}_imaging_calibration.json'


class FLIMLabsClient:
    """REST + WebSocket client that assembles FLIM intensity frames.

    The WebSocket runs on a background thread. LINE messages are accumulated
    into a working buffer; a frame is considered complete when the card starts
    reporting a new frame index (or sends COMPLETED_FRAME_LINES), at which
    point it is published as the "latest frame" and the internal frame counter
    is incremented.
    """

    def __init__(self, host: str = 'localhost', port: int = 5249,
                 image_width: int = 256, image_height: int = 256,
                 timeout: float = 5.0, name: str = 'FLIMLabs'):
        self.__logger = initLogger(self, instanceName=name)
        self.host = host
        self.port = int(port)
        self.timeout = timeout

        self._width = int(image_width)
        self._height = int(image_height)

        self._lock = threading.Lock()
        self._frame_event = threading.Condition(self._lock)
        # Working buffer for the frame currently being received
        self._acc = np.zeros((self._height, self._width), dtype=np.uint32)
        self._acc_frame_idx: Optional[int] = None
        # Last complete frame + monotonic counter (the "frame number")
        self._latest = np.zeros((self._height, self._width), dtype=np.uint32)
        self._frame_number = 0

        self._ws: Optional["websocket.WebSocketApp"] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._cps = 0
        self._last_data_file: Optional[str] = None
        self._experiment_ended = threading.Event()
        # Optional external observers (calibration / phasor consumers)
        self._message_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # Calibration results, keyed by (channel, harmonic)
        self._calibration: Dict[Tuple[int, int], Dict[str, float]] = {}
        # Accumulated phasor density histogram (owned here so the UI can be a
        # thin client - it just fetches the sparse non-zero cells)
        self._phasor_hist = np.zeros((PHASOR_H, PHASOR_W), dtype=np.uint32)
        self._step = 'scouting'

    # -- URLs ------------------------------------------------------------
    @property
    def base_url(self) -> str:
        h = str(self.host or 'localhost').strip()
        if not h.startswith(('http://', 'https://')):
            h = f'http://{h}'
        h = h.rstrip('/')
        # keep an explicit port if the host already carries one
        hostpart = h.split('://', 1)[1]
        return h if ':' in hostpart else f'{h}:{self.port}'

    @property
    def ws_url(self) -> str:
        return self.base_url.replace('http', 'ws', 1) + '/data'

    # -- REST ------------------------------------------------------------
    def _get(self, path: str) -> Any:
        r = requests.get(f'{self.base_url}{path}', timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: Optional[dict] = None) -> Any:
        r = requests.post(f'{self.base_url}{path}', json=payload or {},
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            self._get('/health')
            return True
        except Exception:
            return False

    def check_card(self) -> Optional[str]:
        """Return the card serial number, or None when no card is reachable."""
        try:
            res = self._get('/api/card/check')
            return res.get('data')
        except Exception as e:
            self.__logger.debug(f'card check failed: {e}')
            return None

    def resolve_firmware(self, sync: str = 'in', frequency_mhz: float = 40,
                         channels: Optional[List[int]] = None,
                         reconstruction: str = 'PLF',
                         is_pico_mode: bool = False) -> str:
        res = self._post('/api/firmware/resolve', {
            'sync': sync,
            'frequency_mhz': frequency_mhz,
            'channels': channels or [1],
            'channel': 'sma',
            'sync_connection': 'sma',
            'reconstruction': reconstruction,
            'is_pico_mode': is_pico_mode,
        })
        return res.get('data')

    def detect_laser_frequency(self) -> Optional[float]:
        try:
            res = self._post('/api/laser/detect-frequency', {})
            return float(res.get('data'))
        except Exception as e:
            self.__logger.warning(f'laser frequency detection failed: {e}')
            return None

    # -- Payload ---------------------------------------------------------
    def build_imaging_payload(self, firmware: str, step: str = 'scouting',
                              frequency_mhz: float = 40,
                              reconstruction: str = 'PLF',
                              image_width: int = 256, image_height: int = 256,
                              offsets: Tuple[int, int, int, int] = (0, 0, 0, 0),
                              channels: Optional[List[bool]] = None,
                              dwell_time_us: float = 5,
                              max_frames: Optional[int] = None,
                              is_pico_mode: bool = False,
                              tau_ns: Optional[float] = None,
                              harmonics: int = 1,
                              reference_file: Optional[str] = None,
                              export: Optional[dict] = None,
                              acquisition_timestamp: Optional[int] = None) -> dict:
        """Build a /start payload (flim-imager 2.x schema).

        offsets is ``(top, right, bottom, left)``. ``acquisition_timestamp`` is
        in SECONDS - the server derives export/calibration filenames from it.
        """
        channels = channels or [False, True] + [False] * 6
        enabled_idx = [i for i, on in enumerate(channels) if on]
        top, right, bottom, left = offsets
        geometry = {
            'scan_width': image_width + left + right,
            'scan_height': image_height + top + bottom,
            'image_width': image_width,
            'image_height': image_height,
            'offset_top': top,
            'offset_right': right,
            'offset_bottom': bottom,
            'offset_left': left,
        }
        exp = export or {}
        ts = int(acquisition_timestamp if acquisition_timestamp is not None
                 else time.time())
        params: Dict[str, Any] = {
            'acquisition_setup': 'Default',
            'reconstruction': reconstruction,
            'stop_acquisition_mode': 'Frames Count',
            'step': _V2_STEP.get(step, step),
            'is_preview': False,
            'frequency_mhz': frequency_mhz,
            'is_pico_mode': is_pico_mode,
            'calibration': step == 'calibration',
            'tau_ns': tau_ns,
            'harmonics': harmonics or 1,
            **geometry,
            'skip_frames': 0,
            'calibration_offsets': [[] for _ in range(8)],
            'channels': channels,
            'bg_active_channels': enabled_idx,
            'channels_to_show': enabled_idx,
            'show_cps': True,
            'show_kcps': False,
            'show_sbr': True,
            'show_intensity_traces': False,
            'show_realtime_phasors': step == 'phasors',
            'bin_width': 3000,
            'export_params': {
                'export_data': bool(exp.get('enabled')),
                'export_filename': exp.get('filename', '') if exp.get('enabled') else '',
                'export_path': exp.get('path', '') if exp.get('enabled') else '',
                'export_frames': bool(exp.get('enabled') and exp.get('frames', True)),
                'export_global_image': bool(exp.get('enabled') and exp.get('global_image')),
                'export_notes': exp.get('notes', ''),
                'export_tags': exp.get('tags', []),
                'channels_metadata': [{'id': i, 'alias': f'Channel {i + 1}'}
                                      for i in enabled_idx],
            },
            'acquisition_timestamp': ts,
            'reference_file': reference_file if step == 'phasors' else None,
            'interleaving_type': None,
            'decay_roi': dict(geometry),
        }
        # The card slices lines by dwell time only in LF/F reconstruction;
        # with per-pixel markers (PLF) the geometry comes from the pixel clock.
        if reconstruction != 'PLF':
            params['dwell_time'] = dwell_time_us
        if max_frames and max_frames > 0:
            params['max_frames'] = max_frames
        return {'firmware': firmware, 'frequency': frequency_mhz,
                'experiment': {'type': 'imaging', 'params': params}}

    # -- Frame assembly (WebSocket thread) -------------------------------
    def set_image_size(self, width: int, height: int) -> None:
        with self._lock:
            if width == self._width and height == self._height:
                return
            self._width, self._height = int(width), int(height)
            self._acc = np.zeros((self._height, self._width), dtype=np.uint32)
            self._latest = np.zeros((self._height, self._width), dtype=np.uint32)
            self._acc_frame_idx = None

    def set_message_callback(self, cb: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Register an observer for every decoded message (calibration/phasor)."""
        self._message_callback = cb

    def _accumulate_phasor(self, g: np.ndarray, s: np.ndarray) -> None:
        """Bin one frame's G/S pixels into the density histogram."""
        n = min(len(g), len(s))
        if n == 0:
            return
        g = np.asarray(g[:n], dtype=np.float64)
        s = np.asarray(s[:n], dtype=np.float64)
        valid = np.isfinite(g) & np.isfinite(s) & ~((g == 0) & (s == 0))
        if not valid.any():
            return
        g, s = g[valid], s[valid]
        x = ((g - PHASOR_G_MIN) / (PHASOR_G_MAX - PHASOR_G_MIN) * PHASOR_W).astype(np.int64)
        y = (PHASOR_H - (s - PHASOR_S_MIN) / (PHASOR_S_MAX - PHASOR_S_MIN) * PHASOR_H).astype(np.int64)
        inside = (x >= 0) & (x < PHASOR_W) & (y >= 0) & (y < PHASOR_H)
        if not inside.any():
            return
        with self._lock:
            np.add.at(self._phasor_hist, (y[inside], x[inside]), 1)

    def get_phasor_sparse(self, max_points: int = 20000) -> Dict[str, Any]:
        """Return the phasor density as sparse ``[x, y, count]`` triplets.

        Phasor clouds are sparse, so this is far smaller than the full
        420x280 grid and lets the UI keep its own rendering (semicircle,
        colormap) without receiving raw per-pixel G/S matrices.
        """
        with self._lock:
            hist = self._phasor_hist
            ys, xs = np.nonzero(hist)
            counts = hist[ys, xs]
        if len(xs) > max_points:
            # Keep the densest cells - they dominate the visual anyway
            keep = np.argpartition(counts, -max_points)[-max_points:]
            xs, ys, counts = xs[keep], ys[keep], counts[keep]
        return {
            'width': PHASOR_W, 'height': PHASOR_H,
            'gMin': PHASOR_G_MIN, 'gMax': PHASOR_G_MAX,
            'sMin': PHASOR_S_MIN, 'sMax': PHASOR_S_MAX,
            'max': int(counts.max()) if len(counts) else 0,
            'points': np.stack([xs, ys, counts], axis=1).astype(int).tolist(),
        }

    def get_calibration_results(self) -> List[Dict[str, float]]:
        with self._lock:
            return sorted(self._calibration.values(),
                          key=lambda r: (r['channel'], r['harmonic']))

    def clear_analysis(self) -> None:
        """Drop calibration results and the phasor histogram."""
        with self._lock:
            self._calibration.clear()
            self._phasor_hist.fill(0)

    def _publish_frame(self) -> None:
        """Promote the working buffer to 'latest frame'. Caller holds the lock."""
        self._latest = self._acc.copy()
        self._frame_number += 1
        self._acc.fill(0)
        self._frame_event.notify_all()

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        t = msg['type']
        if t == FlimMsg.LINE:
            line = msg['line']
            frame = msg['frame']
            with self._lock:
                if self._acc_frame_idx is None:
                    self._acc_frame_idx = frame
                elif frame != self._acc_frame_idx:
                    # The card moved on to a new frame -> the previous one is done
                    self._publish_frame()
                    self._acc_frame_idx = frame
                if 0 <= line < self._height:
                    px = msg['pixels']
                    n = min(len(px), self._width)
                    # Sum over enabled channels into one intensity image
                    self._acc[line, :n] += px[:n].astype(np.uint32, copy=False)
        elif t == FlimMsg.COMPLETED_FRAME_LINES:
            lines = msg['lines']
            with self._lock:
                for i, px in enumerate(lines):
                    if i >= self._height:
                        break
                    n = min(len(px), self._width)
                    self._acc[i, :n] += px[:n].astype(np.uint32, copy=False)
                self._acc_frame_idx = msg['frame']
                self._publish_frame()
        elif t == FlimMsg.CPS:
            self._cps = msg['cps']
        elif t == FlimMsg.CALIBRATION:
            with self._lock:
                self._calibration[(msg['channel'], msg['harmonic'])] = {
                    'channel': msg['channel'],
                    'harmonic': msg['harmonic'],
                    'phase': msg['phase'],
                    'modulation': msg['modulation'],
                }
        elif t == FlimMsg.PHASOR:
            self._accumulate_phasor(msg['g'], msg['s'])
        elif t in (FlimMsg.IMAGING_END, FlimMsg.END_EXPERIMENT):
            if t == FlimMsg.IMAGING_END and msg.get('dataFile'):
                self._last_data_file = msg['dataFile']
            with self._lock:
                # Flush a partially-received final frame so a max_frames run
                # always yields its last frame
                if self._acc_frame_idx is not None and self._acc.any():
                    self._publish_frame()
                self._acc_frame_idx = None
                self._running = False
                self._frame_event.notify_all()
            self._experiment_ended.set()

        if self._message_callback is not None:
            try:
                self._message_callback(msg)
            except Exception:
                self.__logger.debug('FLIM message callback raised', exc_info=True)

    def _on_ws_message(self, _ws, data) -> None:
        if isinstance(data, str):
            return  # control/JSON frames are not used by the data socket
        for msg in parse_flim_chunk(data):
            self._handle_message(msg)

    def _open_ws(self) -> None:
        if not IS_WEBSOCKET_AVAILABLE:
            raise RuntimeError('websocket-client is not installed '
                               '(pip install websocket-client)')
        self.close_ws()
        self._experiment_ended.clear()
        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_ws_message,
            on_error=lambda _w, e: self.__logger.warning(f'FLIM websocket error: {e}'),
        )
        self._ws = ws
        self._ws_thread = threading.Thread(
            target=ws.run_forever, kwargs={'ping_interval': 0},
            daemon=True, name='FLIMLabsWS')
        self._ws_thread.start()
        # Give the socket a moment to connect before the card starts sending
        time.sleep(0.3)

    def close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        thread, self._ws_thread = self._ws_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    # -- Acquisition -----------------------------------------------------
    def start(self, payload: dict, image_width: int, image_height: int) -> None:
        """Open the data socket, then arm the card with ``payload``."""
        self.set_image_size(image_width, image_height)
        with self._lock:
            self._acc.fill(0)
            self._acc_frame_idx = None
            self._frame_number = 0
            self._last_data_file = None
        self._open_ws()
        try:
            self._post('/start', payload)
        except Exception:
            self.close_ws()
            raise
        self._running = True

    def stop(self) -> None:
        try:
            self._post('/stop')
        except Exception as e:
            self.__logger.debug(f'FLIM stop failed: {e}')
        finally:
            self._running = False
            self.close_ws()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def cps(self) -> int:
        return self._cps

    @property
    def last_data_file(self) -> Optional[str]:
        return self._last_data_file

    def get_latest_frame(self, return_frame_number: bool = False):
        with self._lock:
            frame = self._latest.copy()
            n = self._frame_number
        return (frame, n) if return_frame_number else frame

    def get_frame_number(self) -> int:
        with self._lock:
            return self._frame_number

    def get_display_frame(self) -> np.ndarray:
        """Frame for live display: the partially-filled current frame while it
        is being received, otherwise the last complete one. FLIM frames take
        seconds, so showing the progressive fill matters for usability."""
        with self._lock:
            if self._acc_frame_idx is not None and self._acc.any():
                return self._acc.copy()
            return self._latest.copy()

    def wait_for_next_frame(self, timeout: float = 10.0,
                            min_new_frames: int = 1) -> Optional[np.ndarray]:
        """Block until ``min_new_frames`` complete frames have arrived.

        Returns the newest complete frame, or None on timeout. Used by
        ``snapSync`` so a grab after a stage move never returns an image that
        was integrated while the stage was still travelling.
        """
        deadline = time.time() + timeout
        with self._frame_event:
            target = self._frame_number + max(1, min_new_frames)
            while self._frame_number < target:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._frame_event.wait(timeout=min(remaining, 0.5))
                if not self._running and self._frame_number < target:
                    # Acquisition ended without reaching the target
                    break
            return self._latest.copy() if self._frame_number >= target else None

    def flush(self) -> None:
        """Discard the partially-integrated frame (post-move flush)."""
        with self._lock:
            self._acc.fill(0)
            self._acc_frame_idx = None


# Copyright (C) 2020-2025 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
