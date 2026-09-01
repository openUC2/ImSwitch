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

  * ``arm()`` claims the card for ImSwitch. Nothing else does: the flim-imager
    server's /data stream is a single-consumer queue drain, so an ImSwitch
    connection opened just because a browser tab connected would lock out (and
    silently split the stream with) the FLIM LABS web UI. Until armed,
    ``startAcquisition()`` - which ImSwitch calls for every ``forAcquisition``
    detector at once - is a no-op here. ``disarm()`` stops the run and drops
    the socket, handing the card back. ``"autoArm": true`` restores the old
    arm-on-connect behaviour.
  * ``startAcquisition()`` arms the card in scouting mode with max_frames=0
    (continuous) and opens the binary WebSocket. The galvo scanner must be
    scanning for frames to arrive - either started separately, or by this
    manager when ``autoStartGalvo`` is enabled and a scanner has been bound
    via ``setGalvoScanner()``.
  * ``getLatestFrame()`` returns the most recent COMPLETE frame, or None while
    disarmed/stopped so the live view shows nothing rather than a stale frame.
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

import json
import os
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
        # Scanner -> sample calibration. The galvo is commanded in DAC counts
        # (0..DAC_FULL_SCALE-1); how much sample that spans depends on galvo
        # gain, scan lens and objective, so it has to be measured once (grid
        # slide / known bead spacing) and put in the setup file. Give either
        # umPerDacUnit{X,Y} directly, or fovUmFullScale{X,Y} (the µm spanned by
        # the FULL DAC range) and let it be derived. With neither, pixelSizeUm
        # stays the static setup value and nothing below changes behaviour.
        self._umPerDac = [
            self._resolveUmPerDac(props, 'X'), self._resolveUmPerDac(props, 'Y')]
        # Derived per-axis pixel size (x, y); falls back to the static value.
        self._pixelSizeUmXY = (self._pixelSizeUm, self._pixelSizeUm)

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
        # Arming gate. ImSwitch starts every forAcquisition detector together
        # (DetectorsManager.startAcquisition -> execOnAll, and
        # LiveViewController.startLiveView), so merely opening the frontend
        # used to arm the FLIM card and grab the flim-imager server's /data
        # socket - which is single-consumer, so it also locked out the FLIM
        # LABS web UI. The card is therefore only started when the FLIM panel
        # explicitly asks for it; the generic startAcquisition() is a no-op
        # until then. Set "autoArm": true in the setup file for the old
        # behaviour.
        self._armed = bool(props.get('autoArm', False))
        self._firmware = None
        self._step = 'scouting'
        self._acquisitionTimestamp = None
        self._calibrationReference = None
        self._calibrations = []

        parameters = {
            'dwell_time': DetectorNumberParameter(
                group='FLIM', value=float(props.get('dwellTimeUs', 25)),
                valueUnits='us', editable=True),
            'frames_to_integrate': DetectorNumberParameter(
                group='FLIM', value=int(props.get('framesToIntegrate', 1)),
                valueUnits='frames', editable=True),
            # How many consecutive frames the LIVE image sums over. The card
            # emits one frame per galvo sweep, so at low count rates a single
            # frame is mostly zeros; raising this trades latency for SNR.
            # Affects display only - snapSync() integrates via
            # frames_to_integrate and is unaffected.
            'display_frames': DetectorNumberParameter(
                group='FLIM', value=int(props.get('displayFrames', 1)),
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
        self._calibrations = self._loadCalibrations()
        self._client.set_display_frames(self.parameters['display_frames'].value)

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

    # DAC range the galvo scanner is commanded over (x_min/x_max are 0..4095).
    DAC_FULL_SCALE = 4096

    @classmethod
    def _resolveUmPerDac(cls, props, axis: str):
        """µm of sample per DAC count for one axis, or None if uncalibrated."""
        direct = props.get(f'umPerDacUnit{axis}', props.get('umPerDacUnit'))
        if direct:
            return float(direct)
        fov = props.get(f'fovUmFullScale{axis}', props.get('fovUmFullScale'))
        if fov:
            return float(fov) / cls.DAC_FULL_SCALE
        return None

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
        # Read separately: a scanner without a DAC range still has a valid
        # nx/ny, and losing the shape sync over a missing range would be worse
        # than simply not deriving a pixel size.
        try:
            spanX = abs(int(cfg.x_max) - int(cfg.x_min))
            spanY = abs(int(cfg.y_max) - int(cfg.y_min))
        except Exception as e:
            self.__logger.debug(f'Could not read galvo scan range: {e}')
            return
        self._syncPixelSize(nx, ny, spanX, spanY)

    def _syncPixelSize(self, nx: int, ny: int, spanX: int, spanY: int) -> None:
        """Derive pixel size from the scanned DAC range and the sample count.

        pixel = (scanned DAC span) * (µm per DAC count) / (samples across it).
        Uncalibrated axes keep the static setup value.
        """
        umPerDacX, umPerDacY = self._umPerDac
        if umPerDacX is None or umPerDacY is None or nx <= 0 or ny <= 0:
            return
        psX = umPerDacX * spanX / nx
        psY = umPerDacY * spanY / ny
        if psX <= 0 or psY <= 0:
            return
        if (psX, psY) != self._pixelSizeUmXY:
            self._pixelSizeUmXY = (psX, psY)
            # Keep the scalar in step for anything reading it directly
            self._pixelSizeUm = psX
            self.__logger.info(
                f'FLIM pixel size from galvo: {psX:.4f} x {psY:.4f} um/px '
                f'(FOV {umPerDacX * spanX:.1f} x {umPerDacY * spanY:.1f} um)')

    def applyGeometryLive(self) -> dict:
        """Re-read the galvo geometry and push it to a running acquisition.

        The server applies ControlMessage::UpdateDimensions to the running
        experiment, so a scan-parameter change in the galvo tab takes effect
        without stopping and re-arming the card. Only the scouting step
        consumes these upstream; other steps need the restart.
        """
        self._syncGeometryFromGalvo()
        width, height, offsets = self._applyGeometry()
        scan_w, scan_h = self._scanShape
        pushed = False
        if self._running:
            if self._step == 'scouting':
                pushed = self._client.send_dimensions(
                    scan_w, scan_h, width, height, offsets)
                if not pushed:
                    self.__logger.debug(
                        'Live dimension update not delivered; restarting')
            if not pushed:
                # Steps other than scouting ignore the control message, and a
                # closed socket means it never arrived - re-arm instead.
                self.stopAcquisition()
                self.startAcquisitionStep(self._step)
        return {'scanWidth': scan_w, 'scanHeight': scan_h,
                'imageWidth': width, 'imageHeight': height,
                'offsets': offsets, 'live': pushed,
                'pixelSizeUm': list(self._pixelSizeUmXY)}

    def setFieldOfViewUm(self, fovUmX=None, fovUmY=None,
                         nx=None, ny=None) -> dict:
        """Set the scanned field of view in micrometres, and optionally the
        sampling (nx/ny).

        The galvo is commanded in DAC counts, so the requested extent is
        converted with the scanner calibration and centred on the scan's
        current centre. Pixel size follows as FOV/samples; the new geometry is
        pushed to a running acquisition rather than re-arming the card.
        """
        if self._galvoScanner is None:
            return {'error': 'No galvo scanner bound to this detector'}
        umPerDacX, umPerDacY = self._umPerDac
        if (fovUmX and umPerDacX is None) or (fovUmY and umPerDacY is None):
            return {'error': 'Scanner is not calibrated: set umPerDacUnitX/Y '
                             'or fovUmFullScaleX/Y in the detector properties'}
        try:
            cfg = self._galvoScanner.config
        except Exception as e:
            return {'error': f'Could not read galvo config: {e}'}

        def dacRange(lo, hi, fovUm, umPerDac):
            """DAC window of the requested width, centred where the old one was
            and clamped into the device range."""
            span = max(1, int(round(fovUm / umPerDac)))
            span = min(span, self.DAC_FULL_SCALE - 1)
            centre = (int(lo) + int(hi)) // 2
            newLo = centre - span // 2
            newLo = max(0, min(newLo, self.DAC_FULL_SCALE - 1 - span))
            return newLo, newLo + span

        update = {}
        if nx:
            update['nx'] = int(nx)
        if ny:
            update['ny'] = int(ny)
        if fovUmX:
            update['x_min'], update['x_max'] = dacRange(
                cfg.x_min, cfg.x_max, float(fovUmX), umPerDacX)
        if fovUmY:
            update['y_min'], update['y_max'] = dacRange(
                cfg.y_min, cfg.y_max, float(fovUmY), umPerDacY)
        if update:
            try:
                self._galvoScanner.update_config(**update)
            except Exception as e:
                return {'error': f'Could not update galvo config: {e}'}
        result = self.applyGeometryLive()
        result['galvo'] = update
        return result

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
        """Free-running scouting acquisition (the detector/camera use case).

        This is the generic entry point ImSwitch calls for every detector at
        once, so it must not touch the card unless the FLIM panel has armed
        it - see the ``_armed`` note in __init__. arm() / startAcquisitionStep()
        are the deliberate paths.
        """
        if not self._armed:
            self.__logger.debug(
                'startAcquisition ignored: FLIM is disarmed. Arm it from the '
                'FLIM panel (or set "autoArm": true in the setup file).')
            return
        self.startAcquisitionStep('scouting')

    # -- Arming -----------------------------------------------------------
    @property
    def isArmed(self) -> bool:
        return self._armed

    def arm(self, start: bool = True) -> dict:
        """Hand the FLIM card to ImSwitch.

        Until this is called, ImSwitch never opens the flim-imager server's
        /data socket, which leaves the card free for the FLIM LABS web UI.
        With ``start=False`` the detector is only marked armed, so the next
        live view / experiment start picks it up.
        """
        self._armed = True
        if start and not self._running:
            return self.startAcquisitionStep('scouting')
        return {'status': 'armed', 'running': self._running}

    def disarm(self) -> dict:
        """Stop the card and release the server's /data socket.

        Releasing the socket is the point: the flim-imager /data queue is a
        single-consumer drain, so ImSwitch has to let go before the FLIM LABS
        web UI can stream.
        """
        self._armed = False
        self.stopAcquisition(releaseSocket=True)
        # stopAcquisition() returns early when nothing was running, so drop the
        # socket here too - it may have been left open by a previous run.
        try:
            self._client.close_ws()
        except Exception as e:
            self.__logger.warning(f'Could not close the FLIM /data socket: {e}')
        return {'status': 'disarmed'}

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
        # An explicit run request is itself the arming decision, so a later
        # generic startAcquisition() (live view, ExperimentController) keeps
        # working without a separate arm() call.
        self._armed = True
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
                self._recordCalibration(timestamp, tauNs, harmonics)
            self.__logger.info(f'FLIM {step} acquisition started ({width}x{height})')
        except Exception as e:
            self.__logger.error(f'Failed to start FLIM {step} acquisition: {e}')
            return {'error': str(e)}
        self._startGalvo()
        return {'status': 'started', 'step': step, 'firmware': self._firmware,
                'acquisitionTimestamp': timestamp}

    def stopAcquisition(self, releaseSocket: bool = False) -> None:
        """Stop the run. ``releaseSocket`` also drops /data, handing the FLIM
        server back to its own web UI (see disarm())."""
        if not self._running:
            return
        wasCalibration = self._step == 'calibration'
        calibrationTimestamp = self._acquisitionTimestamp
        self._stopGalvo()
        try:
            self._client.stop(close_socket=releaseSocket)
        except Exception as e:
            self.__logger.warning(f'Failed to stop FLIM acquisition: {e}')
        self._running = False
        if wasCalibration and calibrationTimestamp is not None:
            # Only now can the run be judged usable: the card has returned its
            # phase/modulation table and the server has written the reference.
            self._confirmCalibration(calibrationTimestamp)

    @staticmethod
    def _toU16(frame):
        """Photon-count frames are uint32 internally; the ImSwitch pipeline
        (JPEGStreamWorker live view, TIFF writers) expects uint8/uint16 -
        cv2.imencode rejects uint32 outright."""
        if frame is None or frame.dtype == np.uint16:
            return frame
        return np.clip(frame, 0, 65535).astype(np.uint16)

    def getLatestFrame(self, is_resize=True, returnFrameNumber=False):
        """Return the most recent COMPLETE frame (no blocking), as uint16.

        Returns None while the card is not acquiring: the client's buffer keeps
        the last frame (and zeros before the first one), and streaming that
        into the live view would show a stale or black FLIM image next to a
        running camera. LiveViewController skips None frames.
        """
        if not self._running:
            return (None, self._client.get_frame_number()) if returnFrameNumber else None
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
        frame = self.getLatestFrame()
        if frame is None:
            return np.zeros((0, *self.shape), dtype=np.uint16)
        return np.expand_dims(frame, 0)

    # -- Parameters / info ------------------------------------------------
    def setParameter(self, name, value):
        super().setParameter(name, value)
        if name == 'display_frames':
            # Display-only: never touches the acquisition.
            self._client.set_display_frames(self.parameters[name].value)
            return self.parameters
        if name in ('offset_top', 'offset_right', 'offset_bottom',
                    'offset_left'):
            # Crop offsets are part of the dimension update the server accepts
            # live, so these no longer cost a restart while scouting.
            self.applyGeometryLive()
            return self.parameters
        # Firmware-affecting changes still require re-arming the card
        if name in ('reconstruction', 'frequency_mhz', 'dwell_time') and self._running:
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
    def channels(self):
        """Per-channel enable flags (8 entries), as sent to the server."""
        return list(self._channels)

    @property
    def calibrationReference(self):
        return self._calibrationReference

    def setCalibrationReference(self, path) -> None:
        self._calibrationReference = path

    # -- Calibration registry ---------------------------------------------
    # A phasors run needs the reference JSON produced by an earlier
    # calibration, and that file lives on the FLIM SERVER (inside its
    # container), so ImSwitch cannot enumerate it. Keep our own index of the
    # calibrations we started, with the metadata the server validates against,
    # so the UI can offer them as a dropdown and mismatches can be caught
    # before the server answers 400.

    @property
    def calibrationsFile(self) -> str:
        from imswitch.imcommon.model.dirtools import UserFileDirs
        return os.path.join(UserFileDirs.Config, 'flim_calibrations.json')

    def _loadCalibrations(self) -> list:
        try:
            with open(self.calibrationsFile, 'r') as f:
                entries = json.load(f)
            return entries if isinstance(entries, list) else []
        except FileNotFoundError:
            return []
        except Exception as e:
            self.__logger.warning(f'Could not read calibration index: {e}')
            return []

    def _saveCalibrations(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.calibrationsFile), exist_ok=True)
            with open(self.calibrationsFile, 'w') as f:
                json.dump(self._calibrations, f, indent=2)
        except Exception as e:
            self.__logger.warning(f'Could not write calibration index: {e}')

    def _recordCalibration(self, timestamp: int, tauNs, harmonics: int) -> None:
        """Register a calibration run that has just been started."""
        self._calibrations = [c for c in self._calibrations
                              if c.get('timestamp') != timestamp]
        self._calibrations.append({
            'timestamp': timestamp,
            'referenceFile': flim_calibration_reference_path(timestamp),
            'tauNs': tauNs,
            'harmonics': int(harmonics),
            # The server checks the reference against the ENABLED channel count
            # and the harmonic count, so both have to travel with the entry.
            'channels': list(self._channels),
            'channelCount': sum(1 for c in self._channels if c),
            'frequencyMhz': float(self.parameters['frequency_mhz'].value),
            'label': f'tau={tauNs}ns h={harmonics} @{timestamp}',
            'confirmed': False,
        })
        self._saveCalibrations()

    def _confirmCalibration(self, timestamp: int) -> None:
        """Mark a calibration usable once the card actually returned results."""
        results = self._client.get_calibration_results()
        for entry in self._calibrations:
            if entry.get('timestamp') == timestamp:
                entry['confirmed'] = bool(results)
                entry['results'] = results
                self._saveCalibrations()
                return

    def listCalibrations(self, confirmedOnly: bool = False) -> list:
        """Calibrations available for a phasors run, newest first.

        The server is the authority on which reference files exist -- our local
        index records intent, and a calibration whose file never got written
        (an unwritable data volume, a crashed run) would otherwise be offered
        and then fail with REFERENCE_FILE_NOT_FOUND. Server entries are
        enriched with the local metadata (tau, channels) where timestamps
        match, and the local-only ones are marked so the UI can grey them out.
        """
        local = {c.get('timestamp'): c for c in self._calibrations}
        merged = []
        for remote in self._client.list_calibrations():
            entry = dict(local.get(remote.get('timestamp'), {}))
            entry.update({
                'timestamp': remote.get('timestamp'),
                'referenceFile': remote.get('file'),
                'harmonics': remote.get('harmonics', entry.get('harmonics')),
                'channelCount': remote.get('channels', entry.get('channelCount')),
                'laserPeriodNs': remote.get('laser_period_ns'),
                'onServer': True,
                'confirmed': True,
            })
            entry.setdefault('label', f'@{remote.get("timestamp")}')
            merged.append(entry)

        seen = {e['timestamp'] for e in merged}
        for entry in self._calibrations:
            if entry.get('timestamp') in seen:
                continue
            entry = dict(entry)
            entry['onServer'] = False
            # Never usable: the reference file is not on the server.
            entry['confirmed'] = False
            merged.append(entry)

        if confirmedOnly:
            merged = [c for c in merged if c.get('confirmed')]
        return sorted(merged, key=lambda c: c.get('timestamp') or 0, reverse=True)

    def resolveCalibration(self, timestamp=None, referenceFile=None) -> dict:
        """Pick the calibration to use for a phasors run.

        Returns {'referenceFile', 'harmonics', ...} or {'error': ...}. An
        explicit referenceFile wins; then a timestamp; otherwise the most
        recent confirmed calibration.
        """
        if referenceFile:
            return {'referenceFile': referenceFile}
        available = self.listCalibrations()
        if timestamp is not None:
            for entry in available:
                if entry.get('timestamp') == int(timestamp):
                    if not entry.get('onServer'):
                        return {'error': f'Calibration {timestamp} is not on the '
                                         f'server - its reference file was never '
                                         f'written. Re-run the calibration.'}
                    return dict(entry)
            return {'error': f'No calibration with timestamp {timestamp}'}
        usable = [c for c in available if c.get('confirmed')]
        if usable:
            return dict(usable[0])
        if available:
            return {'error': 'Calibration runs were recorded but none of their '
                             'reference files are on the server - check that the '
                             "server's data directory is writable, then re-run a "
                             'calibration.'}
        return {'error': 'No calibration available - run a calibration first'}

    def getDisplayFrame(self):
        """Progressive frame for live display (partial frame while filling)."""
        return self._client.get_display_frame()

    @property
    def pixelSizeUm(self):
        """[z, y, x] in µm. x/y come from the galvo scan range when the
        scanner is calibrated (see umPerDacUnit / fovUmFullScale)."""
        psX, psY = self._pixelSizeUmXY
        return [1, psY, psX]

    @property
    def fovUm(self):
        """Scanned field of view (x, y) in µm, or None if uncalibrated."""
        psX, psY = self._pixelSizeUmXY
        scan_w, scan_h = self._scanShape
        if None in self._umPerDac:
            return None
        return (psX * scan_w, psY * scan_h)

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
