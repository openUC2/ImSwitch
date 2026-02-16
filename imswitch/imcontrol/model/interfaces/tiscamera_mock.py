# pip install opencv-python numpy tifffile
import threading
import collections
from enum import Enum
from pathlib import Path
import time
import datetime

import cv2
import numpy as np
import tifffile as tif

try:
    import NanoImagingPack as nip
    HAS_NIP = True
except ImportError:
    HAS_NIP = False


class TriggerSource(str, Enum):
    CONTINUOUS = "Continuous"
    SOFTWARE = "Internal trigger"
    EXTERNAL = "External trigger"


class MockCameraTIS:
    """Drop-in replacement for CameraHIK when no hardware is present.

    CPU-friendly default:
      - "normal" produces a black frame with current time via cv2.putText
      - frame production throttled to max_fps (and also respects exposure_ms if slower)
      - continuous loop sleeps via Event.wait (no busy polling)
      - optional pull behavior: getLast can synthesize at most max_fps if buffer is empty
    """

    def __init__(
        self,
        mocktype: str = "normal",
        mockstackpath: str | None = None,
        isRGB: bool = False,
        width: int = 512,
        height: int = 512,
        nbuffer: int = 256,
        max_fps: float = 5.0,
        time_format: str = "%H:%M:%S",
    ):
        # public metadata
        self.model = "mock"
        self.SensorWidth = width
        self.SensorHeight = height
        self.pixelSize = 1.0

        # acquisition parameters
        self.exposure_ms = 50.0
        self.gain = 1.0
        self.trigger_source = TriggerSource.CONTINUOUS

        # internal state
        self._frame_counter = 0
        self._is_running = False
        self.isRGB = isRGB
        self._mocktype = mocktype
        self._stackpath = Path(mockstackpath) if mockstackpath else None
        self._stack_reader = None

        # throttling
        self._max_fps = float(max_fps) if max_fps and max_fps > 0 else 5.0
        self._last_emit_t = 0.0

        # timestamp rendering
        self._time_format = time_format
        self._text_origin = (32, 64)  # (x, y) baseline
        self._font_face = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 2
        self._font_thickness = 2
        self._line_type = cv2.LINE_AA

        # buffers
        self.frame_buffer = collections.deque(maxlen=nbuffer)
        self.frameid_buffer = collections.deque(maxlen=nbuffer)

        # background thread
        self._thread = None
        self._stop_evt = threading.Event()

        # reusable scratch buffers (avoid per-frame allocations)
        self._alloc_scratch_buffers()

        # prepare data source depending on mocktype
        self._prepare_source()

    # ---------------------------
    # Public API
    # ---------------------------

    def start_live(self):
        if self._is_running:
            return
        self._stop_evt.clear()
        self._is_running = True
        if self.trigger_source == TriggerSource.CONTINUOUS and self._thread is None:
            self._thread = threading.Thread(target=self._continuous_loop, daemon=True)
            self._thread.start()

    def stop_live(self):
        if not self._is_running:
            return
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._thread = None
        self._is_running = False

    suspend_live = stop_live

    def setROI(self, x: int, y: int, w: int, h: int):
        self.SensorWidth = int(w)
        self.SensorHeight = int(h)
        self.flushBuffer()
        self._alloc_scratch_buffers()
        self._prepare_source()

    def getTriggerSource(self):
        return self.trigger_source.value

    def getTriggerTypes(self):
        return [
            TriggerSource.CONTINUOUS.value,
            TriggerSource.SOFTWARE.value,
            TriggerSource.EXTERNAL.value,
        ]

    def flushBuffer(self):
        self.frame_buffer.clear()
        self.frameid_buffer.clear()

    def setTriggerSource(self, source: str):
        self.trigger_source = TriggerSource(source)

        if not self._is_running:
            return

        if self.trigger_source == TriggerSource.CONTINUOUS:
            if self._thread is None:
                self._stop_evt.clear()
                self._thread = threading.Thread(target=self._continuous_loop, daemon=True)
                self._thread.start()
        else:
            self._stop_evt.set()
            if self._thread and self._thread.is_alive():
                self._thread.join()
            self._thread = None
            self._stop_evt.clear()

    def send_trigger(self):
        if self.trigger_source != TriggerSource.SOFTWARE:
            return False
        self._emit_frame_throttled()
        return True

    def external_pulse(self):
        if self.trigger_source != TriggerSource.EXTERNAL:
            return False
        self._emit_frame_throttled()
        return True

    def getLast(self, returnFrameNumber: bool = False):
        # pull behavior: if nothing buffered, synthesize one (throttled)
        if not self.frame_buffer:
            self._emit_frame_throttled(force=True)

        if not self.frame_buffer:
            return (None, -1) if returnFrameNumber else None

        if returnFrameNumber:
            return self.frame_buffer[-1], self.frameid_buffer[-1]
        return self.frame_buffer[-1]

    def getLastChunk(self):
        ids = np.asarray(self.frameid_buffer)
        stack = np.asarray(self.frame_buffer)
        self.flushBuffer()
        return stack, ids

    def setPropertyValue(self, name, value):
        if name == "exposure":
            self.exposure_ms = float(value)
        elif name == "gain":
            self.gain = float(value)
        elif name == "trigger_source":
            self.setTriggerSource(value)
        return value

    def getPropertyValue(self, name):
        if name == "image_width":
            return self.SensorWidth
        if name == "image_height":
            return self.SensorHeight
        if name == "frame_number":
            return self._frame_counter
        return None

    def openPropertiesGUI(self):
        pass

    def close(self):
        self.stop_live()

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _alloc_scratch_buffers(self):
        h, w = self.SensorHeight, self.SensorWidth
        self._scratch_u8 = np.zeros((h, w), dtype=np.uint8)
        self._scratch_u16 = np.zeros((h, w), dtype=np.uint16)
        self._scratch_rgb_u16 = np.zeros((h, w, 3), dtype=np.uint16)

    def _prepare_source(self):
        self._stack_reader = None
        if self._mocktype == "STORM" and self._stackpath and self._stackpath.exists():
            self._stack_reader = tif.TiffFile(self._stackpath)
        elif self._mocktype == "OffAxisHolo":
            self._holo_intensity_image = self._generate_hologram()

    def _min_interval_s(self) -> float:
        return max(self.exposure_ms / 1000.0, 1.0 / self._max_fps)

    def _continuous_loop(self):
        while not self._stop_evt.is_set():
            interval = self._min_interval_s()
            now = time.monotonic()
            due_in = max(0.0, (self._last_emit_t + interval) - now)
            if self._stop_evt.wait(timeout=due_in):
                break
            self._emit_frame_throttled()

    def _emit_frame_throttled(self, force: bool = False):
        now = time.monotonic()
        interval = self._min_interval_s()
        if (not force) and (now - self._last_emit_t) < interval:
            return False
        self._last_emit_t = now
        self._emit_frame()
        return True

    def _emit_frame(self):
        img = self._simulate_frame()
        fid = self._frame_counter
        self._frame_counter += 1
        self.frame_buffer.append(img)
        self.frameid_buffer.append(fid)

    def _simulate_frame(self):
        if self._mocktype in ("normal", "clock", "timestamp"):
            img8 = self._scratch_u8
            img8.fill(0)

            t = datetime.datetime.now().strftime(self._time_format)
            cv2.putText(
                img8,
                "The camera is not connected, "+t,
                self._text_origin,
                self._font_face,
                self._font_scale,
                255,
                self._font_thickness,
                self._line_type,
            )

            img16 = self._scratch_u16
            # 0..255 -> 0..65535 without allocating (multiply into preallocated buffer)
            np.multiply(img8, 257, out=img16, dtype=np.uint16)

            if self.isRGB:
                rgb16 = self._scratch_rgb_u16
                rgb16[..., 0] = img16
                rgb16[..., 1] = img16
                rgb16[..., 2] = img16
                return rgb16.copy()
            return img16.copy()

        if self._mocktype == "focus_lock":
            img = np.zeros((self.SensorHeight, self.SensorWidth), dtype=np.uint16)
            cy = int(np.random.randn() * 1 + self.SensorHeight / 2)
            cx = int(np.random.randn() * 30 + self.SensorWidth / 2)
            img[max(cy - 10, 0): cy + 10, max(cx - 10, 0): cx + 10] = 4095
            return np.stack([img, img, img], axis=-1) if self.isRGB else img

        if self._mocktype == "random_peak":
            img = self._random_peak_frame()
            return np.stack([img, img, img], axis=-1) if self.isRGB else img

        if self._mocktype == "STORM" and self._stack_reader:
            idx = self._frame_counter % len(self._stack_reader.pages)
            img = self._stack_reader.pages[idx].asarray()
            if img.dtype != np.uint16:
                img = img.astype(np.uint16, copy=False)
            return np.stack([img, img, img], axis=-1) if self.isRGB else img

        if self._mocktype == "OffAxisHolo":
            img = self._holo_intensity_image
            if img.dtype != np.uint16:
                # normalize to uint16 once per call, still cheap at 5 fps
                f = img.astype(np.float32, copy=False)
                mn, mx = float(f.min()), float(f.max())
                if mx > mn:
                    f = (f - mn) * (65535.0 / (mx - mn))
                img_u16 = f.astype(np.uint16)
            else:
                img_u16 = img
            return np.stack([img_u16, img_u16, img_u16], axis=-1) if self.isRGB else img_u16

        img = np.zeros((self.SensorHeight, self.SensorWidth), dtype=np.uint16)
        return np.stack([img, img, img], axis=-1) if self.isRGB else img

    def _random_peak_frame(self):
        from scipy.stats import multivariate_normal  # imported only if used

        imgsize = (self.SensorHeight, self.SensorWidth)
        img = np.zeros(imgsize, dtype=np.float32)

        if np.random.rand() > 0.8:
            x, y = np.meshgrid(np.arange(imgsize[1]), np.arange(imgsize[0]))
            xc = (np.random.rand() * 2 - 1) * imgsize[0] / 2 + imgsize[0] / 2
            yc = (np.random.rand() * 2 - 1) * imgsize[1] / 2 + imgsize[1] / 2
            rv = multivariate_normal([yc, xc], [[60, 0], [0, 60]])
            img += 2000 * rv.pdf(np.dstack((y, x)))

        img += np.random.poisson(lam=15, size=imgsize)
        return img.astype(np.uint16)

    def _generate_hologram(self):
        width, height = self.SensorWidth, self.SensorHeight
        wavelength = 0.6328e-6
        k = 2 * np.pi / wavelength
        angle = np.pi / 10

        x = np.linspace(-np.pi, np.pi, width)
        y = np.linspace(-np.pi, np.pi, height)
        X, Y = np.meshgrid(x, y)
        pupil = (X**2 + Y**2) < np.pi*2
        if HAS_NIP:
            mPhase = nip.extract(nip.readim(), ROIsize=(width,height))
            mPhase = mPhase/np.max(mPhase)*2*np.pi
            phase_sample = np.exp(1j * mPhase)
        else:
            phase_sample = np.exp(1j * ((X**2 + Y**2) < 50))
        tilt_x = k * np.sin(angle)
        tilt_y = k * np.sin(angle)
        Xpix, Ypix = np.meshgrid(np.arange(width), np.arange(height))
        plane_wave = np.exp(1j * (tilt_x * Xpix + tilt_y * Ypix))

        filtered = np.fft.ifft2(np.fft.fftshift(pupil) * np.fft.fft2(phase_sample))
        holo = filtered + plane_wave
        return np.real(holo * np.conjugate(holo)).astype(np.float32)
