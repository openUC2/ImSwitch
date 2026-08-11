"""
FLIMLabsDetectorManager.py - Exposes a FLIM LABS acquisition card (via the
flim-imager 2.x standalone server) as a normal ImSwitch 2D detector.

Why a DetectorManager and not a bespoke controller: ExperimentController (and
the autofocus / pixel-calibration / snap paths) already speak the
DetectorManager contract - ``startAcquisition``, ``getLatestFrame``,
``snapSync``, ``flushBuffer``. Implementing that contract means FLIM images can
be acquired position-by-position by the existing experiment machinery with no
special-casing, and the live-stream plumbing (sigImageUpdated) comes for free.

Model
-----
The card has no notion of geometry: frame/line/pixel boundaries come entirely
from the galvo scanner's trigger outputs. So this detector behaves like a
free-running camera whose "exposure" is one galvo frame:

  * ``startAcquisition()`` arms the card in scouting mode with max_frames=0
    (continuous) and opens the binary WebSocket. The galvo scanner must be
    scanning for frames to arrive - either started separately, or by this
    manager when ``autoStartGalvo`` is enabled and a scanner has been bound
    via ``setGalvoScanner()``.
  * ``getLatestFrame()`` returns the most recent COMPLETE frame.
  * ``snapSync()`` waits for the *next* complete frame, so a grab issued right
    after a stage move never returns an image that was integrated while the
    stage was still travelling. ``framesToIntegrate`` > 1 sums consecutive
    frames for a better SNR at slow scan rates.

Setup file example
------------------
{
  "detectors": {
    "FLIM": {
      "managerName": "FLIMLabsDetectorManager",
      "managerProperties": {
        "host": "localhost",
        "port": 5249,
        "imageWidth": 256,
        "imageHeight": 256,
        "frequencyMhz": 40,
        "laserSync": "in",
        "reconstruction": "PLF",
        "channels": [false, true, false, false, false, false, false, false],
        "dwellTimeUs": 25,
        "framesToIntegrate": 1,
        "galvoScanner": "ESP32Galvo",
        "autoStartGalvo": true,
        "pixelSizeUm": 1.0
      },
      "forAcquisition": true
    }
  }
}
"""

import time

import numpy as np

from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.interfaces.flimlabsclient import (
    flim_calibration_reference_path
)
from .DetectorManager import (
    DetectorManager, DetectorNumberParameter, DetectorListParameter
)


class _FLIMCameraShim:
    """The `_camera` surface generic controllers reach for on a detector.

    ObjectiveController, ExperimentController, WorkflowController and friends
    read `detector._camera.SensorWidth/SensorHeight/isRGB/exposure_time`
    directly instead of going through the DetectorManager API. Without this
    attribute those controllers raise on construction, which takes the FLIM
    detector out of the live view entirely.

    Everything is derived from the manager, so a galvo-driven geometry change
    is reflected without re-creating the shim.
    """

    def __init__(self, manager: 'FLIMLabsDetectorManager'):
        self._manager = manager
        self.model = 'FLIMLabs'
        self.isRGB = False

    @property
    def SensorWidth(self) -> int:
        return int(self._manager.shape[0])

    @property
    def SensorHeight(self) -> int:
        return int(self._manager.shape[1])

    @property
    def exposure_time(self) -> float:
        """Frame integration time in microseconds (callers divide by 1e6)."""
        return float(self._manager.parameters['exposure'].value) * 1e3

    def setParameter(self, name, value):
        return self._manager.setParameter(name, value)

    def getStreamDiagnostics(self) -> dict:
        return {
            'model': self.model,
            'width': self.SensorWidth,
            'height': self.SensorHeight,
            'running': self._manager.isRunning,
            'frameNumber': self._manager.client.get_frame_number(),
        }


class FLIMLabsDetectorManager(DetectorManager):
    """DetectorManager for a FLIM LABS card served by flim-imager 2.x."""

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)

        props = detectorInfo.managerProperties or {}
        self._host = props.get('host', 'localhost')
        self._port = int(props.get('port', 5249))
        width = int(props.get('imageWidth', 256))
        height = int(props.get('imageHeight', 256))
        self._pixelSizeUm = float(props.get('pixelSizeUm', 1.0))

        self._frequencyMhz = float(props.get('frequencyMhz', 40))
        self._laserSync = props.get('laserSync', 'in')
        self._reconstruction = props.get('reconstruction', 'PLF')
        self._channels = list(props.get(
            'channels', [False, True, False, False, False, False, False, False]))
        self._isPicoMode = bool(props.get('isPicoMode', False))

        # Galvo coupling (late-bound: detectors are constructed before the
        # galvo scanners manager exists, so the name is resolved on first use
        # via setGalvoScanner()).
        self._galvoScannerName = props.get('galvoScanner', None)
        self._autoStartGalvo = bool(props.get('autoStartGalvo', True))
        # Take the frame size from the galvo scan config (nx/ny) rather than
        # the static setup-file value - the scan pattern defines the frame.
        self._syncWithGalvo = bool(props.get('syncWithGalvo', True))
        self._galvoScanner = None

        self._running = False
        self._firmware = None
        self._step = 'scouting'
        self._acquisitionTimestamp = None
        self._calibrationReference = None

        parameters = {
            'dwell_time': DetectorNumberParameter(
                group='FLIM', value=float(props.get('dwellTimeUs', 25)),
                valueUnits='us', editable=True),
            'frames_to_integrate': DetectorNumberParameter(
                group='FLIM', value=int(props.get('framesToIntegrate', 1)),
                valueUnits='frames', editable=True),
            'frequency_mhz': DetectorNumberParameter(
                group='FLIM', value=self._frequencyMhz, valueUnits='MHz',
                editable=True),
            'reconstruction': DetectorListParameter(
                group='FLIM', value=self._reconstruction,
                options=['PLF', 'LF', 'F'], editable=True),
            # "exposure" is what generic code (ExperimentController timeouts,
            # the UI) looks for; for FLIM one exposure is one integrated frame.
            'exposure': DetectorNumberParameter(
                group='FLIM', value=float(props.get('frameTimeoutMs', 5000)),
                valueUnits='ms', editable=True),
            # Scan-region crop: pixels discarded from each line/frame edge.
            # scan (= galvo nx/ny) = image + offsets. Use offset_left to drop
            # the first column(s) where flyback/settle photons accumulate,
            # exactly like the SCAN AREA panel of the FLIM LABS UI.
            'offset_top': DetectorNumberParameter(
                group='Scan Area', value=int(props.get('offsetTop', 0)),
                valueUnits='px', editable=True),
            'offset_right': DetectorNumberParameter(
                group='Scan Area', value=int(props.get('offsetRight', 0)),
                valueUnits='px', editable=True),
            'offset_bottom': DetectorNumberParameter(
                group='Scan Area', value=int(props.get('offsetBottom', 0)),
                valueUnits='px', editable=True),
            'offset_left': DetectorNumberParameter(
                group='Scan Area', value=int(props.get('offsetLeft', 0)),
                valueUnits='px', editable=True),
        }

        super().__init__(detectorInfo, name, fullShape=(width, height),
                         supportedBinnings=[1], model='FLIMLabs',
                         parameters=parameters, actions={}, croppable=False)

        # The SCAN is what the galvo traces (= trigger pattern); the IMAGE is
        # the scan minus the crop offsets. imageWidth/imageHeight in the setup
        # file describe the scan; galvo sync overwrites it at runtime.
        self._scanShape = (width, height)

        from imswitch.imcontrol.model.interfaces.flimlabsclient import FLIMLabsClient
        self._client = FLIMLabsClient(
            host=self._host, port=self._port,
            image_width=width, image_height=height, name=name)

        # Generic controllers reach into detector._camera; see _FLIMCameraShim.
        self._camera = _FLIMCameraShim(self)

        # Probe with a short timeout: an unreachable server must not stall the
        # whole ImSwitch startup behind the normal request timeout.
        probeTimeout, self._client.timeout = self._client.timeout, 1.5
        try:
            serial = self._client.check_card()
        finally:
            self._client.timeout = probeTimeout
        if serial:
            self.__logger.info(f'FLIM card detected: {serial} @ {self._client.base_url}')
        else:
            self.__logger.warning(
                f'No FLIM card reachable at {self._client.base_url} - the detector '
                'will return blank frames until the server/card is available.')

    # -- Galvo coupling ---------------------------------------------------
    def setGalvoScanner(self, scanner) -> None:
        """Bind the galvo scanner manager that produces the trigger pattern.

        Called after startup (detectors are built before galvoScannersManager),
        e.g. by GalvoScannerController.
        """
        self._galvoScanner = scanner

    @property
    def galvoScannerName(self):
        return self._galvoScannerName

    def _syncGeometryFromGalvo(self) -> None:
        """Adopt the galvo scan's nx/ny as the SCAN size.

        The card has no geometry of its own - the scan is exactly what the
        scanner's triggers delimit. If the user changes nx/ny in the galvo tab
        and the detector kept its setup-file size, every frame would be
        mis-shaped, so re-read it before each acquisition.
        """
        if not (self._syncWithGalvo and self._galvoScanner is not None):
            return
        try:
            cfg = self._galvoScanner.config
            nx, ny = int(cfg.nx), int(cfg.ny)
        except Exception as e:
            self.__logger.debug(f'Could not read galvo geometry: {e}')
            return
        if nx > 0 and ny > 0 and (nx, ny) != self._scanShape:
            self._scanShape = (nx, ny)
            self.__logger.info(f'FLIM scan geometry synced from galvo: {nx}x{ny}')

    def _offsets(self):
        """Crop offsets (top, right, bottom, left), clamped non-negative."""
        get = lambda k: max(0, int(self.parameters[k].value))
        return (get('offset_top'), get('offset_right'),
                get('offset_bottom'), get('offset_left'))

    def _applyGeometry(self):
        """image = scan - offsets; resize the frame assembler accordingly."""
        scan_w, scan_h = self._scanShape
        top, right, bottom, left = self._offsets()
        img_w = max(1, scan_w - left - right)
        img_h = max(1, scan_h - top - bottom)
        self._shape = (img_w, img_h)
        self._DetectorManager__fullShape = (img_w, img_h)
        self._client.set_image_size(img_w, img_h)
        return img_w, img_h, (top, right, bottom, left)

    def _startGalvo(self) -> None:
        if not (self._autoStartGalvo and self._galvoScanner is not None):
            return
        try:
            # Continuous scan: the detector decides how many frames to keep
            self._galvoScanner.start_scan(frame_count=0, enable_trigger=1)
        except Exception as e:
            self.__logger.warning(f'Could not start galvo scan: {e}')

    def _stopGalvo(self) -> None:
        if not (self._autoStartGalvo and self._galvoScanner is not None):
            return
        try:
            self._galvoScanner.stop_scan()
        except Exception as e:
            self.__logger.warning(f'Could not stop galvo scan: {e}')

    # -- Acquisition ------------------------------------------------------
    def startAcquisition(self) -> None:
        """Free-running scouting acquisition (the detector/camera use case)."""
        self.startAcquisitionStep('scouting')

    def startAcquisitionStep(self, step: str = 'scouting', maxFrames: int = 0,
                             tauNs=None, harmonics: int = 1,
                             referenceFile=None, export=None) -> dict:
        """Arm the card for any FLIM step.

        ``scouting`` is the continuous intensity mode behind startAcquisition();
        ``calibration`` and ``phasors`` are the analysis modes the UI drives
        through FLIMLabsController. Returns a small status dict for the API.
        """
        if self._running:
            self.stopAcquisition()
        self._syncGeometryFromGalvo()
        width, height, offsets = self._applyGeometry()
        try:
            enabled = [i + 1 for i, on in enumerate(self._channels) if on] or [1]
            self._firmware = self._client.resolve_firmware(
                sync=self._laserSync,
                frequency_mhz=self.parameters['frequency_mhz'].value,
                channels=enabled,
                reconstruction=self.parameters['reconstruction'].value,
                is_pico_mode=self._isPicoMode)
            timestamp = int(time.time())
            if step in ('calibration', 'phasors'):
                self._client.clear_analysis()
            payload = self._client.build_imaging_payload(
                firmware=self._firmware,
                step=step,
                frequency_mhz=self.parameters['frequency_mhz'].value,
                reconstruction=self.parameters['reconstruction'].value,
                image_width=width, image_height=height,
                offsets=offsets,  # scan = image + offsets = galvo nx/ny
                channels=self._channels,
                dwell_time_us=self.parameters['dwell_time'].value,
                max_frames=maxFrames,
                is_pico_mode=self._isPicoMode,
                tau_ns=tauNs, harmonics=harmonics,
                reference_file=referenceFile, export=export,
                acquisition_timestamp=timestamp)
            self._client.start(payload, width, height)
            self._running = True
            self._step = step
            self._acquisitionTimestamp = timestamp
            if step == 'calibration':
                # Where the server will write its reference JSON
                self._calibrationReference = flim_calibration_reference_path(timestamp)
            self.__logger.info(f'FLIM {step} acquisition started ({width}x{height})')
        except Exception as e:
            self.__logger.error(f'Failed to start FLIM {step} acquisition: {e}')
            return {'error': str(e)}
        self._startGalvo()
        return {'status': 'started', 'step': step, 'firmware': self._firmware,
                'acquisitionTimestamp': timestamp}

    def stopAcquisition(self) -> None:
        if not self._running:
            return
        self._stopGalvo()
        try:
            self._client.stop()
        except Exception as e:
            self.__logger.warning(f'Failed to stop FLIM acquisition: {e}')
        self._running = False

    @staticmethod
    def _toU16(frame):
        """Photon-count frames are uint32 internally; the ImSwitch pipeline
        (JPEGStreamWorker live view, TIFF writers) expects uint8/uint16 -
        cv2.imencode rejects uint32 outright."""
        if frame is None or frame.dtype == np.uint16:
            return frame
        return np.clip(frame, 0, 65535).astype(np.uint16)

    def getLatestFrame(self, is_resize=True, returnFrameNumber=False):
        """Return the most recent COMPLETE frame (no blocking), as uint16."""
        if returnFrameNumber:
            frame, n = self._client.get_latest_frame(return_frame_number=True)
            return self._toU16(frame), n
        return self._toU16(self._client.get_latest_frame())

    def getFrameNumber(self) -> int:
        return self._client.get_frame_number()

    def snapSync(self, timeout: float = None):
        """Return a frame guaranteed to have been integrated after this call.

        Discards the partially-integrated frame first, then waits for
        ``frames_to_integrate`` fresh complete frames and returns their sum.
        This is the fast path ExperimentController uses after a stage move.
        """
        if timeout is None:
            timeout = max(2.0, float(self.parameters['exposure'].value) / 1000.0)
        nFrames = max(1, int(self.parameters['frames_to_integrate'].value))
        self._client.flush()
        if nFrames == 1:
            frame = self._client.wait_for_next_frame(timeout=timeout)
        else:
            # Sum consecutive complete frames for SNR
            acc = None
            for i in range(nFrames):
                f = self._client.wait_for_next_frame(timeout=timeout)
                if f is None:
                    break
                acc = f.astype(np.uint32) if acc is None else acc + f
            frame = acc
        if frame is None:
            self.__logger.warning(
                f'snapSync timed out after {timeout:.1f}s - is the galvo scanning '
                'and are the trigger lines connected?')
            return self._toU16(self._client.get_latest_frame())
        return self._toU16(frame)

    def flushBuffer(self) -> None:
        """Drop the partially-integrated frame so the next grab is post-move."""
        self._client.flush()

    def flushBuffers(self) -> None:
        """:meta private: DetectorManager spells it plural."""
        self.flushBuffer()

    def getChunk(self):
        return np.expand_dims(self.getLatestFrame(), 0)

    # -- Parameters / info ------------------------------------------------
    def setParameter(self, name, value):
        super().setParameter(name, value)
        # Geometry/firmware-affecting changes require an acquisition restart
        if name in ('reconstruction', 'frequency_mhz', 'dwell_time',
                    'offset_top', 'offset_right', 'offset_bottom',
                    'offset_left') and self._running:
            self.stopAcquisition()
            self.startAcquisition()
        return self.parameters

    def getParameter(self, name):
        if name not in self.parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')
        return self.parameters[name].value

    def getCameraStatus(self):
        return {
            'name': self.name,
            'model': self.model,
            'running': self._running,
            'serverUrl': self._client.base_url,
            'cardSerial': self._client.check_card(),
            'firmware': self._firmware,
            'cps': self._client.cps,
            'frameNumber': self._client.get_frame_number(),
            'lastDataFile': self._client.last_data_file,
            'galvoScanner': self._galvoScannerName,
        }

    # -- FLIM-specific accessors used by FLIMLabsController -----------------
    @property
    def client(self):
        return self._client

    @property
    def isRunning(self) -> bool:
        # The server ends max_frames runs on its own; mirror that here so the
        # UI shows "stopped" without the user pressing anything.
        if self._running and not self._client.is_running:
            self._running = False
        return self._running

    @property
    def step(self) -> str:
        return self._step

    @property
    def calibrationReference(self):
        return self._calibrationReference

    def setCalibrationReference(self, path) -> None:
        self._calibrationReference = path

    def getDisplayFrame(self):
        """Progressive frame for live display (partial frame while filling)."""
        return self._client.get_display_frame()

    @property
    def pixelSizeUm(self):
        return [1, self._pixelSizeUm, self._pixelSizeUm]

    def crop(self, hpos, vpos, hsize, vsize):
        """Not croppable: the frame size is defined by the galvo scan pattern."""
        pass

    def setBinning(self, binning):
        super().setBinning(binning)

    def finalize(self) -> None:
        self.stopAcquisition()
        self._client.close_ws()


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
