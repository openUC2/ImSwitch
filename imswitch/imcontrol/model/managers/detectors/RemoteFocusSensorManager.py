import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

import numpy as np

from imswitch.imcommon.model import initLogger
from .DetectorManager import (
    DetectorManager, DetectorNumberParameter, DetectorListParameter
)


class RemoteFocusSensorManager(DetectorManager):
    """ Detector backed by an external openUC2 Pi focus sensor.

    The sensor is a satellite Raspberry Pi that owns its own camera and does
    the two-spot peak estimation itself, reporting a single focus value over a
    USB gadget-ethernet (or plain network) link.  That takes the second USB3
    camera, its bandwidth, and the per-frame maths off the host entirely — the
    host only reads a float.

    What makes this a *detector* rather than a bespoke device is that
    ``FocusLockController`` already knows how to drive detectors: cropping,
    exposure and gain all map onto the sensor's REST API, and
    ``getLatestFrame()`` still returns real pixels so the focus lock widget and
    the alignment view keep working.  The one addition is ``getFocusValue()``:
    when a detector offers it, the controller uses the sensor's estimate
    instead of computing its own.

    Two channels are used:

    * **WebSocket** ``/ws/focus`` — a background thread keeps the newest sample
      cached, so ``getFocusValue()`` is a dictionary read with no network round
      trip in the control loop.
    * **REST** for setup and for the debug frame, and as the fallback when the
      socket cannot be established.

    Manager properties (all optional except ``host``):

    - ``host``           -- sensor address, e.g. ``"192.168.7.2"`` over USB
      gadget ethernet, or ``"focussensor.local"``.
    - ``port``           -- default 8321.
    - ``timeout``        -- REST timeout in seconds, default 2.0.
    - ``useWebsocket``   -- default True; set False to poll REST instead.
    - ``maxSampleAgeMs`` -- samples older than this are reported invalid,
      default 500. Guards against a frozen link silently feeding the PI loop a
      stale value forever.
    - ``cameraParams``   -- applied at startup, e.g.
      ``{"exposure_us": 5000, "gain": 1.0, "roi": {...}, "fps_target": 100}``.
    - ``focusParams``    -- estimator parameters applied at startup, e.g.
      ``{"gaussian_sigma": 3.0, "peak_distance": 40}``.

    Note on units: ``exposure`` is exposed to ImSwitch in **milliseconds**, as
    everywhere else in ImSwitch, and converted to the sensor's microseconds on
    the wire.
    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)

        properties = detectorInfo.managerProperties or {}
        self._host = properties.get('host', '192.168.7.2')
        self._port = int(properties.get('port', 8321))
        self._timeout = float(properties.get('timeout', 2.0))
        self._useWebsocket = bool(properties.get('useWebsocket', True))
        self._maxSampleAge = float(properties.get('maxSampleAgeMs', 500.0)) / 1000.0
        self._base = f'http://{self._host}:{self._port}'

        self._running = False
        self._online = False
        self._simulated = False
        self._latestSample: Optional[Dict[str, Any]] = None
        self._latestSampleAt = 0.0
        self._sampleLock = threading.Lock()
        self._wsThread: Optional[threading.Thread] = None
        self._wsStop = threading.Event()

        # Cache the debug frame briefly: the focus lock widget asks for it at
        # UI rate, and each fetch is a full ROI over the wire.
        self._frameCache: Optional[np.ndarray] = None
        self._frameCacheAt = 0.0
        self._frameCacheTtl = float(properties.get('frameCacheMs', 100.0)) / 1000.0

        status = self._getStatus(quiet=True)
        if status:
            camera = status.get('camera', {})
            fullShape = (int(camera.get('full_shape', {}).get('width', 1456)),
                         int(camera.get('full_shape', {}).get('height', 1088)))
            frameShape = (int(camera.get('frame_shape', {}).get('width', 640)),
                          int(camera.get('frame_shape', {}).get('height', 200)))
            roi = camera.get('roi', {})
            model = f"openuc2-focussensor:{camera.get('model', 'unknown')}"
            exposureMs = float(camera.get('exposure_us', 5000)) / 1000.0
            gain = float(camera.get('gain', 1.0))
            self._simulated = bool(status.get('simulated', False))
            self.__logger.info(
                f"Connected to {status.get('name', 'focus sensor')} at {self._base} "
                f"({model}{', SIMULATED' if self._simulated else ''})")
        else:
            # A satellite Pi can easily boot slower than the host. Come up with
            # sensible placeholders and reconnect lazily rather than failing the
            # whole setup and leaving the user without a focus lock entry.
            self.__logger.warning(
                f"Focus sensor at {self._base} not reachable yet; will retry on access.")
            fullShape, frameShape = (1456, 1088), (640, 200)
            roi, model = {}, 'openuc2-focussensor:offline'
            exposureMs, gain = 5.0, 1.0

        self._frameStart = (int(roi.get('x', 0)), int(roi.get('y', 0)))
        self._shape = frameShape

        parameters = {
            'exposure': DetectorNumberParameter(group='Focus sensor', value=exposureMs,
                                                valueUnits='ms', editable=True),
            'gain': DetectorNumberParameter(group='Focus sensor', value=gain,
                                            valueUnits='arb.u.', editable=True),
            'fps_target': DetectorNumberParameter(group='Focus sensor', value=100.0,
                                                  valueUnits='fps', editable=True),
            'gaussian_sigma': DetectorNumberParameter(group='Estimator', value=3.0,
                                                      valueUnits='px', editable=True),
            'peak_distance': DetectorNumberParameter(group='Estimator', value=40,
                                                     valueUnits='px', editable=True),
            'peak_prominence_mad': DetectorNumberParameter(group='Estimator', value=4.0,
                                                           valueUnits='MAD', editable=True),
            'projection_mode': DetectorListParameter(group='Estimator', value='max',
                                                     options=['max', 'mean'],
                                                     editable=True),
        }

        super().__init__(detectorInfo, name, fullShape=fullShape, supportedBinnings=[1, 2, 4],
                         model=model, parameters=parameters, croppable=True)

        for key, value in (properties.get('cameraParams') or {}).items():
            try:
                self._post('/api/camera/params', {key: value})
            except Exception as exc:
                self.__logger.warning(f"Could not apply camera param {key}: {exc}")
        focusParams = properties.get('focusParams') or {}
        if focusParams:
            try:
                self._post('/api/focus/params', focusParams)
            except Exception as exc:
                self.__logger.warning(f"Could not apply focus params: {exc}")
        self._syncParametersFromSensor()

    # ------------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, payload: Any = None,
                 raw: bool = False, timeout: Optional[float] = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self._base + path, data=data, method=method,
            headers={'Content-Type': 'application/json'} if data else {})
        with urllib.request.urlopen(request, timeout=timeout or self._timeout) as response:
            body = response.read()
        self._online = True
        return body if raw else json.loads(body)

    def _get(self, path: str, raw: bool = False, timeout: Optional[float] = None) -> Any:
        return self._request('GET', path, raw=raw, timeout=timeout)

    def _post(self, path: str, payload: Any = None,
              timeout: Optional[float] = None) -> Any:
        return self._request('POST', path, payload if payload is not None else {},
                             timeout=timeout)

    def _getStatus(self, quiet: bool = False) -> Optional[Dict[str, Any]]:
        try:
            return self._get('/api/status')
        except Exception as exc:
            self._online = False
            if not quiet:
                self.__logger.debug(f"Focus sensor status unavailable: {exc}")
            return None

    def _syncParametersFromSensor(self) -> None:
        """Pull the sensor's actual settings into the parameter list."""
        status = self._getStatus(quiet=True)
        if not status:
            return
        camera, focus = status.get('camera', {}), status.get('focus_params', {})
        self._simulated = bool(status.get('simulated', False))
        updates = {
            'exposure': float(camera.get('exposure_us', 5000)) / 1000.0,
            'gain': float(camera.get('gain', 1.0)),
            'fps_target': float(camera.get('fps_target', 100.0)),
            'gaussian_sigma': float(focus.get('gaussian_sigma', 3.0)),
            'peak_distance': int(focus.get('peak_distance', 40)),
            'peak_prominence_mad': float(focus.get('peak_prominence_mad', 4.0)),
            'projection_mode': str(focus.get('projection_mode', 'max')),
        }
        for key, value in updates.items():
            if key in self.parameters:
                self.parameters[key].value = value
        roi = camera.get('roi')
        frame = camera.get('frame_shape')
        if roi:
            self._frameStart = (int(roi['x']), int(roi['y']))
        if frame:
            self._shape = (int(frame['width']), int(frame['height']))

    # ------------------------------------------------------------ properties
    @property
    def isOnline(self) -> bool:
        return self._online

    @property
    def isSimulated(self) -> bool:
        """ True when the sensor is rendering synthetic pixels.

        ``FocusLockController`` uses this to decide whether to mirror the stage
        position into the sensor, which is what makes a full end-to-end
        simulation of the focus lock possible with no hardware at all.
        """
        return self._simulated

    @property
    def pixelSizeUm(self):
        return [1, 1, 1]

    # --------------------------------------------------------------- sampling
    def getFocusValue(self, maxAgeS: Optional[float] = None) -> Dict[str, Any]:
        """ The sensor's latest focus estimate.

        Returns the sensor's sample dictionary with an added ``age_s``, or a
        dictionary with ``valid=False`` when nothing recent is available. The
        keys match ``focusmetrics.PeakMetric.compute`` (``focus``,
        ``left_peak_x``, ``right_peak_x``, ``x_peak_distance``) so the
        controller can consume either source unchanged.
        """
        maxAge = self._maxSampleAge if maxAgeS is None else maxAgeS

        with self._sampleLock:
            sample = dict(self._latestSample) if self._latestSample else None
            age = time.monotonic() - self._latestSampleAt if sample else None

        if sample is None or age is None or age > maxAge:
            # Socket is cold or stale — fall back to a direct read.
            try:
                sample = self._get('/api/focus')
                age = float(sample.get('age_s', 0.0))
                with self._sampleLock:
                    self._latestSample = sample
                    self._latestSampleAt = time.monotonic()
            except Exception as exc:
                self.__logger.debug(f"Focus read failed: {exc}")
                return {'valid': False, 'focus': None, 'age_s': None,
                        'error': str(exc)}

        sample['age_s'] = age
        if age is not None and age > maxAge:
            sample['valid'] = False
            sample['stale'] = True
        return sample

    def setSimulatedZ(self, z_um: float) -> Optional[Dict[str, Any]]:
        """ Mirror the host's stage position into a simulated sensor.

        Returns the sample that was exposed *after* the move, so the caller can
        use it directly instead of racing a frame that is already in flight.
        No-op (returns None) on a real sensor.
        """
        if not self._simulated:
            return None
        try:
            result = self._post('/api/sim/state', {'z_um': float(z_um), 'wait': True})
        except Exception as exc:
            self.__logger.debug(f"Could not set simulated z: {exc}")
            return None
        sample = result.get('sample')
        if sample:
            with self._sampleLock:
                self._latestSample = sample
                self._latestSampleAt = time.monotonic()
        return sample

    # ------------------------------------------------------------- websocket
    def _startWebsocket(self) -> None:
        if not self._useWebsocket or (self._wsThread and self._wsThread.is_alive()):
            return
        self._wsStop.clear()
        self._wsThread = threading.Thread(target=self._websocketLoop,
                                          name=f'{self.name}-ws', daemon=True)
        self._wsThread.start()

    def _stopWebsocket(self) -> None:
        self._wsStop.set()
        thread = self._wsThread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._wsThread = None

    def _websocketLoop(self) -> None:
        """ Keep ``_latestSample`` fresh at the sensor's own rate.

        Runs its own asyncio loop in this thread; ``websockets`` ships with
        ImSwitch already (uvicorn[standard]). If it is missing or the socket
        cannot be established, this exits quietly and ``getFocusValue()``
        falls back to REST reads.
        """
        try:
            import asyncio
            import websockets
        except ImportError:
            self.__logger.info(
                "websockets not available; focus sensor will be read over REST")
            return

        url = f'ws://{self._host}:{self._port}/ws/focus'

        async def consume() -> None:
            backoff = 0.5
            while not self._wsStop.is_set():
                try:
                    async with websockets.connect(url, max_queue=8,
                                                  open_timeout=self._timeout) as socket:
                        backoff = 0.5
                        self._online = True
                        while not self._wsStop.is_set():
                            message = await asyncio.wait_for(socket.recv(), timeout=5.0)
                            sample = json.loads(message)
                            with self._sampleLock:
                                self._latestSample = sample
                                self._latestSampleAt = time.monotonic()
                except Exception as exc:
                    if self._wsStop.is_set():
                        break
                    self.__logger.debug(f"Focus sensor socket dropped: {exc}")
                    self._online = False
                    await asyncio.sleep(backoff)
                    backoff = min(5.0, backoff * 2)

        try:
            asyncio.run(consume())
        except Exception:
            self.__logger.debug("Focus sensor socket loop ended", exc_info=True)

    # ----------------------------------------------------------------- frames
    def getLatestFrame(self, is_save: bool = False,
                       returnFrameNumber: bool = False):
        """ The sensor's current ROI frame, for the alignment view.

        ``returnFrameNumber`` defaults to False to match every other detector
        manager: callers such as ``FocusLockController.returnLastImage`` and
        ``DetectorManager.updateLatestFrame`` expect a bare array. The frame
        number, when asked for, is the sensor's own frame counter (``seq``),
        which is what advances once per exposure.

        The cached and freshly-fetched paths must agree on the return type --
        returning a tuple on a cache miss and an array on a hit makes the
        shape of the result depend on how recently someone else looked.
        """
        now = time.monotonic()
        if self._frameCache is not None and (now - self._frameCacheAt) < self._frameCacheTtl:
            frame = self._frameCache
        else:
            try:
                payload = self._get('/api/frame.npy', raw=True)
                frame = np.load(io.BytesIO(payload), allow_pickle=False)
                self._frameCache = frame
                self._frameCacheAt = now
                self._shape = (frame.shape[1], frame.shape[0])
            except Exception as exc:
                self.__logger.debug(f"Frame fetch failed: {exc}")
                frame = self._frameCache

        if returnFrameNumber:
            with self._sampleLock:
                number = self._latestSample.get('seq') if self._latestSample else None
            return frame, number
        return frame

    def getChunk(self) -> Optional[np.ndarray]:
        frame = self.getLatestFrame()
        return None if frame is None else frame[np.newaxis, ...]

    def flushBuffers(self) -> None:
        pass

    # -------------------------------------------------------------- lifecycle
    def startAcquisition(self) -> None:
        if self._running:
            return
        self._running = True
        self._startWebsocket()
        self._syncParametersFromSensor()
        self.__logger.debug('Focus sensor acquisition started')

    def stopAcquisition(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stopWebsocket()
        self.__logger.debug('Focus sensor acquisition stopped')

    def stopAcquisitionForROIChange(self) -> None:
        pass

    def finalize(self) -> None:
        self._stopWebsocket()

    def closeEvent(self) -> None:
        self._stopWebsocket()

    # ------------------------------------------------------------- parameters
    def setParameter(self, name: str, value: Any) -> Dict[str, Any]:
        super().setParameter(name, value)

        cameraKeys = {'exposure': 'exposure_us', 'gain': 'gain',
                      'fps_target': 'fps_target'}
        focusKeys = {'gaussian_sigma', 'peak_distance', 'peak_prominence_mad',
                     'peak_height_mad', 'projection_mode', 'min_quality',
                     'baseline_percentile', 'background_threshold'}
        try:
            if name in cameraKeys:
                # ImSwitch speaks milliseconds, the sensor speaks microseconds.
                wire = float(value) * 1000.0 if name == 'exposure' else value
                self._post('/api/camera/params', {cameraKeys[name]: wire})
            elif name in focusKeys:
                self._post('/api/focus/params', {name: value})
        except Exception as exc:
            self.__logger.error(f"Could not set '{name}' on the focus sensor: {exc}")
        return self.parameters

    def getParameter(self, name: str) -> Any:
        if name not in self.parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')
        return self.parameters[name].value

    def refreshParameters(self) -> Dict[str, Any]:
        self._syncParametersFromSensor()
        return self.parameters

    def setBinning(self, binning: int) -> None:
        super().setBinning(binning)
        try:
            self._post('/api/camera/params', {'binning': int(binning)})
        except Exception as exc:
            self.__logger.error(f"Could not set binning on the focus sensor: {exc}")

    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int) -> None:
        """ Move the sensor's readout window — a real ROI change, not a
        numpy crop: a narrow window is what lets the sensor run fast. """
        try:
            params = self._post('/api/camera/params', {
                'roi': {'x': int(hpos), 'y': int(vpos),
                        'width': int(hsize), 'height': int(vsize)}})
            roi, frame = params['roi'], params['frame_shape']
            self._frameStart = (int(roi['x']), int(roi['y']))
            self._shape = (int(frame['width']), int(frame['height']))
            self._frameCache = None
        except Exception as exc:
            self.__logger.error(f"Could not set focus sensor ROI: {exc}")

    def openPropertiesDialog(self) -> None:
        self.__logger.info(f"Focus sensor debug view: {self._base}/")


# Copyright (C) ImSwitch developers 2021
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
