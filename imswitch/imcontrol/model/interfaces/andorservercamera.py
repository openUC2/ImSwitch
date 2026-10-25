"""Client for andor-server, the HTTP sidecar that owns an Andor SDK2 camera (iXon).

The Andor SDK is x86_64-only, so on a Raspberry Pi the camera runs in its own
(emulated) container and ImSwitch talks to it over HTTP. This class offers the
same camera-object interface as CameraToupcam (getLast, start_live,
setPropertyValue, setROI, ...) so AndorServerManager can mirror ToupCamManager.

Robustness:
- ImSwitch may start before the server; the client then comes up disconnected
  and keeps reconnecting in the background.
- A server restart (new boot id) is detected; the settings and cooler
  configuration from the setup file are pushed again.
- Frame numbers are counted here, so they stay monotonic across server restarts.
"""

import collections
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from imswitch.imcommon.model import initLogger

# ImSwitch parameter name -> andor-server setting
SETTING_NAMES = {
    'exposure': 'exposure_ms',
    'gain': 'em_gain',
    'em_gain_mode': 'em_gain_mode',
    'em_advanced': 'em_advanced',
    'output_amplifier': 'output_amplifier',
    'readout_rate': 'hs_speed_mhz',
    'preamp_gain': 'preamp_gain',
    'vs_speed': 'vs_speed_us',
    'vs_amplitude': 'vs_amplitude',
    'baseline_clamp': 'baseline_clamp',
    'blacklevel': 'baseline_offset',
    'frame_transfer': 'frame_transfer',
    'shutter_mode': 'shutter_mode',
    'frame_rate': 'live_max_fps',
}
AMPLIFIER_LABELS = {'em': 'Electron Multiplying', 'conventional': 'Conventional'}
TRIGGER_SOURCES = ['Continous', 'Internal trigger', 'External trigger']


def fmt_mhz(value: Optional[float]) -> str:
    return '' if value is None else f'{value:g} MHz'


def fmt_us(value: Optional[float]) -> str:
    return '' if value is None else f'{value:g} us'


def fmt_gain(value: Optional[float]) -> str:
    return '' if value is None else f'{value:g}x'


def parse_number(text: Any) -> float:
    if isinstance(text, (int, float)):
        return float(text)
    return float(str(text).split()[0].rstrip('x'))


class AndorServerError(RuntimeError):
    pass


class CameraAndorServer:
    def __init__(self, host: str = 'andor-server', port: int = 8000, basePath: str = '',
                 requestTimeout: float = 10.0, connectTimeout: float = 30.0,
                 settings: Optional[Dict[str, Any]] = None, cooler: Optional[Dict[str, Any]] = None,
                 flipImage: Tuple[bool, bool] = (False, False), longExposureMs: float = 2000.0,
                 sensorShape: Tuple[int, int] = (1024, 1024)):
        self.__logger = initLogger(self, tryInheritParent=False)
        self.baseUrl = f'http://{host}:{int(port)}{basePath.rstrip("/")}'
        self.requestTimeout = requestTimeout
        self.flipImage = flipImage
        self.longExposureMs = longExposureMs
        self._setupSettings = dict(settings or {})
        self._setupCooler = dict(cooler or {})
        self._session = requests.Session()

        self.model = 'Andor (andor-server)'
        self.SensorWidth, self.SensorHeight = sensorShape
        self.pixelSizeUm = 13.0
        self.info: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}
        self.settings: Dict[str, Any] = {}
        self.status: Dict[str, Any] = {}
        self.is_connected = False
        self._bootId: Optional[str] = None
        self._listeners: List = []  # callables invoked after (re)connecting

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frameNumber = 0
        self._serverFrame = 0
        self._chunk = collections.deque(maxlen=64)
        self._frameTimes = collections.deque(maxlen=30)
        self._acquiring = threading.Event()
        self._generation = 0
        self._softwareTrigger = False
        self._mode = 'idle'  # idle, live, snap (repeated long exposures), software
        self._stop = threading.Event()

        deadline = time.monotonic() + connectTimeout
        while not self._connect():
            if time.monotonic() > deadline:
                self.__logger.warning(f'andor-server at {self.baseUrl} not reachable; continuing '
                                      f'disconnected and retrying in the background')
                break
            time.sleep(1.0)
        self._thread = threading.Thread(target=self._frameLoop, name='andor-server-frames', daemon=True)
        self._thread.start()
        self._watch = threading.Thread(target=self._watchLoop, name='andor-server-watch', daemon=True)
        self._watch.start()

    # ------------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, timeout: Optional[float] = None, **kwargs) -> requests.Response:
        try:
            r = self._session.request(method, self.baseUrl + path, timeout=timeout or self.requestTimeout, **kwargs)
        except requests.RequestException as e:
            self.is_connected = False
            raise AndorServerError(f'andor-server unreachable: {e}') from e
        if r.status_code >= 400:
            try:
                detail = r.json().get('detail')
            except ValueError:
                detail = r.text
            raise AndorServerError(f'{method} {path} -> {r.status_code}: {detail}')
        return r

    def _json(self, method: str, path: str, **kwargs) -> Any:
        return self._request(method, path, **kwargs).json()

    def _connect(self) -> bool:
        try:
            info = self._json('GET', '/info', timeout=3.0)
            status = self._json('GET', '/status', timeout=3.0)
            if status.get('state') not in ('idle', 'live', 'exposing'):
                self.__logger.debug(f'andor-server is {status.get("state")}: {status.get("error")}')
                return False
            caps = self._json('GET', '/capabilities', timeout=3.0)
        except AndorServerError as e:
            self.__logger.debug(str(e))
            return False
        self.info, self.capabilities, self.status = info, caps, status
        cam = info.get('camera', {})
        sensor = cam.get('sensor') or {}
        self.model = cam.get('head_model') or self.model
        self.SensorWidth = sensor.get('width', self.SensorWidth)
        self.SensorHeight = sensor.get('height', self.SensorHeight)
        self.pixelSizeUm = (sensor.get('pixel_size_um') or [self.pixelSizeUm])[0]
        newBoot = status.get('boot_id')
        restarted = self._bootId is not None and newBoot != self._bootId
        firstConnect = self._bootId is None
        self._bootId = newBoot
        self.is_connected = True
        try:
            if firstConnect or restarted:
                if self._setupSettings:
                    self._json('PATCH', '/settings', json=self._setupSettings)
                if self._setupCooler:
                    self._json('PUT', '/cooler', json=self._setupCooler)
            self.settings = self._json('GET', '/settings')
            if restarted and self._acquiring.is_set():
                self._startMode(self._mode)
        except AndorServerError as e:
            self.__logger.error(f'configuring andor-server failed: {e}')
        self.__logger.info(f'connected to andor-server {self.baseUrl}: {self.model} '
                           f'({"restarted" if restarted else "boot " + str(newBoot)})')
        for listener in list(self._listeners):
            try:
                listener()
            except Exception as e:  # noqa: BLE001
                self.__logger.debug(f'listener failed: {e}')
        return True

    def addConnectListener(self, callback) -> None:
        self._listeners.append(callback)

    def _watchLoop(self) -> None:
        while not self._stop.wait(2.0):
            try:
                self.status = self._json('GET', '/status', timeout=3.0)
                if self.status.get('boot_id') != self._bootId or not self.is_connected:
                    self._connect()
                else:
                    self.is_connected = True
            except AndorServerError:
                if self.is_connected:
                    self.__logger.warning('lost the connection to andor-server; retrying')
                self.is_connected = False

    # ------------------------------------------------------------------ frames
    def _frameLoop(self) -> None:
        while not self._stop.is_set():
            if not self._acquiring.wait(0.5):
                continue
            if not self.is_connected:
                time.sleep(1.0)
                continue
            generation = self._generation
            try:
                r = self._request('GET', '/frames/next', timeout=10.0,
                                  params={'after': self._serverFrame, 'timeout_s': 2.0, 'format': 'raw'})
            except AndorServerError as e:
                self.__logger.debug(f'frame request failed: {e}')
                time.sleep(1.0)
                continue
            if generation != self._generation or not self._acquiring.is_set():
                continue  # answer to a request from before the last start/stop
            current = int(r.headers.get('X-Frame-Number', self._serverFrame))
            if r.status_code == 204:
                if current < self._serverFrame:  # the server restarted and counts from 0 again
                    self._serverFrame = current
                continue
            h, w = (int(v) for v in r.headers['X-Frame-Shape'].split(','))
            frame = np.frombuffer(r.content, dtype='<u2').reshape(h, w)
            self._deliver(frame, current)
            if self._mode == 'snap':
                # Long exposures are taken as single snaps (server-side TIFF copy, progress);
                # keep taking them while the acquisition stays armed. The pause lets a
                # one-off snap (which disarms as soon as it has its frame) end first.
                if not self._stop.wait(0.5) and generation == self._generation and self._acquiring.is_set():
                    self._startSnap()

    def _deliver(self, frame: np.ndarray, serverNumber: int) -> None:
        flipY, flipX = self.flipImage
        if flipY:
            frame = np.flipud(frame)
        if flipX:
            frame = np.fliplr(frame)
        with self._lock:
            self._frame = frame
            self._frameNumber += 1
            self._serverFrame = serverNumber
            self._chunk.append(frame)
            self._frameTimes.append(time.monotonic())

    def getLast(self, returnFrameNumber: bool = False):
        with self._lock:
            frame, number = self._frame, self._frameNumber
        return (frame, number) if returnFrameNumber else frame

    def getLastChunk(self) -> Optional[np.ndarray]:
        with self._lock:
            frames = list(self._chunk)
            self._chunk.clear()
        return np.stack(frames) if frames else None

    def flushBuffer(self) -> None:
        with self._lock:
            self._chunk.clear()

    def getFrameNumber(self) -> int:
        return self._frameNumber

    # ------------------------------------------------------------------ acquisition
    def isLongExposure(self) -> bool:
        return float(self.settings.get('exposure_ms', 0) or 0) >= self.longExposureMs

    def _startMode(self, mode: str) -> None:
        if mode == 'live':
            self._json('POST', '/live/start')
        elif mode == 'snap':
            self._startSnap()

    def _startSnap(self) -> Optional[Dict[str, Any]]:
        try:
            return self._json('POST', '/snap', json={'resume_live': False})
        except AndorServerError as e:
            self.__logger.error(f'starting an exposure failed: {e}')
            return None

    def start_live(self) -> None:
        """Arm continuous acquisition: live mode, or repeated snaps for long exposures."""
        if self._softwareTrigger:
            self._mode = 'software'
        elif self.isLongExposure():
            self._mode = 'snap'
        else:
            self._mode = 'live'
        self._generation += 1
        try:
            # frames up to the current one are old; only later ones count as fresh
            self.status = self._json('GET', '/status', timeout=3.0)
            self._serverFrame = int(self.status.get('frame_number') or 0)
            self._acquiring.set()
            self._startMode(self._mode)
        except AndorServerError as e:
            self._acquiring.set()  # the frame loop picks up once the server is back
            self.__logger.error(f'starting the acquisition failed: {e}')

    def suspend_live(self) -> None:
        self._acquiring.clear()
        self._generation += 1
        mode, self._mode = self._mode, 'idle'
        if mode in ('live', 'snap') and self.is_connected:
            try:
                self._json('POST', '/abort')
            except AndorServerError as e:
                self.__logger.warning(f'stopping the acquisition failed: {e}')

    def stop_live(self) -> None:
        self.suspend_live()

    def snapSoftwareTrigger(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """One exposure, returned when read out (ExperimentController's snapSync)."""
        exposureS = float(self.settings.get('exposure_ms', 0) or 0) / 1000.0
        wait = max(timeout, exposureS * 1.5 + 10.0)
        try:
            snap = self._json('POST', '/snap', timeout=wait + 5.0,
                              json={'wait': True, 'timeout_s': wait, 'resume_live': True})
            if snap.get('state') != 'done':
                self.__logger.warning(f'snap {snap.get("id")} ended {snap.get("state")}: {snap.get("error")}')
                return None
            r = self._request('GET', f'/snaps/{snap["id"]}/frame', params={'format': 'raw'})
        except AndorServerError as e:
            self.__logger.error(f'snap failed: {e}')
            return None
        h, w = (int(v) for v in r.headers['X-Frame-Shape'].split(','))
        frame = np.frombuffer(r.content, dtype='<u2').reshape(h, w)
        number = int(r.headers['X-Frame-Number'])
        if number > self._serverFrame:  # not yet delivered by the frame stream
            self._deliver(frame, number)
            return self.getLast()
        return frame

    def send_trigger(self) -> bool:
        """Start one exposure; its frame arrives through the frame stream."""
        self._acquiring.set()
        return self._startSnap() is not None

    def abort(self) -> None:
        self._json('POST', '/abort')

    # ------------------------------------------------------------------ settings
    def updateSettings(self, changes: Dict[str, Any], abort: bool = False) -> Tuple[Dict[str, Any], List[str]]:
        body = self._json('PATCH', '/settings', params={'abort': abort}, json=changes)
        self.settings = body['settings']
        if self._mode in ('live', 'snap') and self._acquiring.is_set():
            # an exposure change can cross the long-exposure threshold
            wanted = 'snap' if self.isLongExposure() else 'live'
            if wanted != self._mode:
                self.suspend_live()
                self.start_live()
        for warning in body.get('warnings') or []:
            self.__logger.warning(f'andor-server: {warning}')
        return self.settings, body.get('warnings') or []

    def setPropertyValue(self, name: str, value: Any) -> Any:
        if name in ('target_temperature', 'cooler', 'fan_mode'):
            key = {'target_temperature': 'setpoint_c', 'cooler': 'on', 'fan_mode': 'fan_mode'}[name]
            if name == 'target_temperature':
                value = int(round(float(value)))
            elif name == 'cooler':
                value = bool(value)
            cooler = self._json('PUT', '/cooler', json={key: value})
            for warning in cooler.pop('warnings', None) or []:
                self.__logger.warning(f'andor-server: {warning}')
            self.status = {**self.status, 'temperature': cooler}
            return {'setpoint_c': cooler.get('setpoint_c'), 'on': cooler.get('cooler_on'),
                    'fan_mode': cooler.get('fan_mode')}[key]
        if name == 'trigger_source':
            self.setTriggerSource(value)
            return value
        if name not in SETTING_NAMES:
            raise AttributeError(f'andor-server has no property "{name}"')
        key = SETTING_NAMES[name]
        if name == 'output_amplifier':
            value = 'conventional' if str(value).lower().startswith('conv') else 'em'
            if value != self.settings.get('output_amplifier'):
                # readout rates and valid pre-amp gains differ per amplifier: start from its defaults
                self.updateSettings({key: value, 'hs_speed_mhz': None, 'preamp_gain': None})
                return self.getPropertyValue(name)
        elif name in ('readout_rate', 'preamp_gain', 'vs_speed'):
            value = parse_number(value)
        elif name == 'vs_amplitude':
            options = self.capabilities.get('vs_amplitudes') or []
            value = options.index(value) if value in options else int(value)
        elif name in ('gain', 'blacklevel'):
            value = int(round(float(value)))
        elif name in ('exposure', 'frame_rate'):
            value = float(value)
        self.updateSettings({key: value})
        return self.getPropertyValue(name)

    def getPropertyValue(self, name: str) -> Any:
        if name in ('temperature', 'target_temperature', 'cooler', 'fan_mode', 'temperature_status'):
            t = self.status.get('temperature') or {}
            return {'temperature': t.get('celsius'), 'target_temperature': t.get('setpoint_c'),
                    'cooler': t.get('cooler_on'), 'fan_mode': t.get('fan_mode'),
                    'temperature_status': t.get('status')}[name]
        if name == 'frame_number':
            return self._frameNumber
        if name == 'trigger_source':
            return self.getTriggerSource()
        if name in ('image_width', 'image_height'):
            image = self.status.get('image') or {}
            return image.get('width' if name == 'image_width' else 'height')
        if name not in SETTING_NAMES:
            return None
        value = self.settings.get(SETTING_NAMES[name])
        if name == 'output_amplifier':
            return AMPLIFIER_LABELS.get(value, value)
        if name == 'readout_rate':
            return fmt_mhz(value)
        if name == 'preamp_gain':
            return fmt_gain(value)
        if name == 'vs_speed':
            return fmt_us(value)
        if name == 'vs_amplitude':
            options = self.capabilities.get('vs_amplitudes') or []
            return options[value] if isinstance(value, int) and value < len(options) else value
        return value

    def refreshSettings(self) -> Dict[str, Any]:
        self.settings = self._json('GET', '/settings')
        return self.settings

    # ------------------------------------------------------------------ options for list parameters
    def readoutRateOptions(self) -> List[str]:
        amps = self.capabilities.get('amplifiers') or []
        index = 1 if self.settings.get('output_amplifier') == 'conventional' and len(amps) > 1 else 0
        return [fmt_mhz(v) for v in (amps[index]['hs_speeds_mhz'] if amps else [])]

    def preampGainOptions(self) -> List[str]:
        caps = self.capabilities
        gains = caps.get('preamp_gains') or []
        amps = caps.get('amplifiers') or []
        amp = 1 if self.settings.get('output_amplifier') == 'conventional' and len(amps) > 1 else 0
        speeds = amps[amp]['hs_speeds_mhz'] if amps else []
        hs = self.settings.get('hs_speed_mhz')
        hsIndex = speeds.index(hs) if hs in speeds else 0
        valid = (caps.get('preamp_valid') or {}).get(f'{amp}:{hsIndex}', range(len(gains)))
        return [fmt_gain(gains[i]) for i in valid if i < len(gains) and gains[i] is not None]

    def vsSpeedOptions(self) -> List[str]:
        return [fmt_us(v) for v in self.capabilities.get('vs_speeds_us') or []]

    # ------------------------------------------------------------------ geometry, trigger, cooling
    def setROI(self, hpos: int, vpos: int, hsize: int, vsize: int) -> None:
        """hpos/vpos/hsize/vsize in unbinned sensor pixels; the full sensor clears the ROI."""
        if (hpos, vpos, hsize, vsize) == (0, 0, self.SensorWidth, self.SensorHeight):
            self.updateSettings({'roi': None})
        else:
            self.updateSettings({'roi': {'x': int(hpos), 'y': int(vpos), 'width': int(hsize), 'height': int(vsize)}})

    def setBinning(self, binning: int) -> None:
        self.updateSettings({'binning': int(binning)})

    @property
    def binning(self) -> int:
        return int(self.settings.get('binning', 1))

    @property
    def imageShape(self) -> Tuple[int, int]:
        """(width, height) of the frames with the current binning and ROI."""
        b = self.binning
        roi = self.settings.get('roi')
        w, h = (roi['width'], roi['height']) if roi else (self.SensorWidth, self.SensorHeight)
        return w // b, h // b

    def setTriggerSource(self, source: str) -> None:
        """'Internal trigger' (software): one exposure per trigger/snapSync, taken as an internally
        triggered single scan (/snap), since iXon heads lack the SDK's software trigger mode.
        'External trigger': the camera waits for its TTL input. 'Continous': free-running."""
        s = str(source).lower()
        self._softwareTrigger = s in ('internal trigger', 'software', 'software trigger')
        mode = 'external' if s.startswith('external') else 'internal'
        if self.settings.get('trigger_mode') != mode:
            self.updateSettings({'trigger_mode': mode}, abort=True)

    def getTriggerSource(self) -> str:
        if self._softwareTrigger:
            return 'Internal trigger'
        if str(self.settings.get('trigger_mode', 'internal')).startswith('external'):
            return 'External trigger'
        return 'Continous'

    def getTriggerTypes(self) -> List[str]:
        return list(TRIGGER_SOURCES)

    def get_temperature(self) -> Optional[float]:
        return (self.status.get('temperature') or {}).get('celsius')

    # ------------------------------------------------------------------ diagnostics, lifecycle
    def getStreamDiagnostics(self) -> Dict[str, Any]:
        times = list(self._frameTimes)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
        return {'connected': self.is_connected, 'mode': self._mode, 'clientFps': fps,
                'frameNumber': self._frameNumber, 'serverFrame': self._serverFrame,
                'server': {k: self.status.get(k) for k in ('state', 'live', 'exposure', 'error')}}

    def get_camera_parameters(self) -> Dict[str, Any]:
        return {'settings': self.settings, 'status': self.status, 'server': self.baseUrl}

    def openPropertiesGUI(self) -> None:
        self.__logger.info(f'andor-server API documentation: {self.baseUrl}/docs')

    def close(self) -> None:
        self._stop.set()
        if self._mode in ('live', 'snap'):
            try:
                self.suspend_live()
            except Exception:  # noqa: BLE001
                pass
        self._acquiring.clear()
