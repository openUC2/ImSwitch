"""
FLIMLabsController.py - REST facade for the FLIM LABS card.

The ImSwitch backend is the SINGLE owner of the flim-imager server's ``/data``
WebSocket (it is a single-consumer queue drain upstream: two clients would
split the stream between them). This controller therefore exposes everything
the UI needs over the normal ImSwitch API, so the frontend never talks to the
FLIM server directly:

  * control        - start/stop scouting, calibration and phasors runs
  * status         - card serial, health, CPS, frame number, calibration table
  * live intensity - base64 PNG snapshots of the (progressively filling) frame
  * phasors        - the accumulated density histogram as sparse triplets

All hardware access goes through the FLIMLabsDetectorManager, which owns the
client, so an ExperimentController acquisition and the FLIM panel share one
connection and one frame assembler.
"""

import base64
import json
from typing import Any, Dict, List, Optional

import numpy as np

from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FLIMLabsController(ImConWidgetController):
    """Controls the FLIM LABS detector and serves its data to the frontend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self, tryInheritParent=True)
        self._detectorName = None
        self._resolveDetector()

    # ------------------------------------------------------------------
    # Detector resolution
    # ------------------------------------------------------------------
    def _resolveDetector(self):
        """Find the FLIM detector among the configured detectors (if any)."""
        try:
            detectors = self._master.detectorsManager
        except Exception:
            return None
        if self._detectorName is not None:
            try:
                return detectors[self._detectorName]
            except Exception:
                self._detectorName = None
        for name in detectors.getAllDeviceNames():
            det = detectors[name]
            if type(det).__name__ == 'FLIMLabsDetectorManager':
                self._detectorName = name
                self.__logger.info(f"FLIM detector resolved: '{name}'")
                return det
        return None

    @property
    def _detector(self):
        return self._resolveDetector()

    # ------------------------------------------------------------------
    # Status / health
    # ------------------------------------------------------------------
    @APIExport()
    def getFlimStatus(self) -> Dict[str, Any]:
        """Full FLIM state for the UI: connection, acquisition and analysis.

        Example:
            GET /api/FLIMLabsController/getFlimStatus
        """
        det = self._detector
        if det is None:
            return {'available': False,
                    'error': 'No FLIMLabsDetectorManager configured in the setup file'}
        client = det.client
        width, height = det.shape
        scanW, scanH = getattr(det, '_scanShape', (width, height))
        debug = client.get_debug_stats()
        running = det.isRunning
        # A published frame carrying far fewer lines than the image height is
        # the signature of a SECOND /data consumer splitting the stream (e.g.
        # the FLIM LABS web UI open in another browser tab) - surface it.
        hint = None
        linesLast = debug.get('linesInLastFrame', 0)
        if running and debug.get('publishes', 0) > 0 and linesLast < 0.8 * height:
            hint = (f'Only {linesLast}/{height} lines in the last frame - is another '
                    f'/data client (e.g. the FLIM LABS web UI) connected to the '
                    f'same server? Close it; the stream supports one consumer.')
        return {
            'available': True,
            'detectorName': det.name,
            'serverUrl': client.base_url,
            'serverHealthy': client.health(),
            # While running, the card is held by the acquisition -- probing it
            # would contend with that, so answer from the cached serial.
            'cardSerial': client.check_card(use_cache=det.isRunning),
            'running': det.isRunning,
            'step': det.step,
            'firmware': getattr(det, '_firmware', None),
            'cps': client.cps,
            'frameNumber': client.get_frame_number(),
            'imageWidth': width,
            'imageHeight': height,
            'scanWidth': scanW,
            'scanHeight': scanH,
            'lastDataFile': client.last_data_file,
            'calibrationResults': client.get_calibration_results(),
            'calibrationReference': det.calibrationReference,
            'galvoScanner': det.galvoScannerName,
            'debug': debug,
            'hint': hint,
            # [z, y, x] um. x/y are derived from the galvo scan range when the
            # scanner is calibrated (umPerDacUnit / fovUmFullScale in the setup
            # file); otherwise they are the static setup value.
            'pixelSizeUm': det.pixelSizeUm,
            'fovUm': det.fovUm,
            'parameters': {
                'dwell_time': det.parameters['dwell_time'].value,
                'frames_to_integrate': det.parameters['frames_to_integrate'].value,
                'display_frames': det.parameters['display_frames'].value,
                'frequency_mhz': det.parameters['frequency_mhz'].value,
                'reconstruction': det.parameters['reconstruction'].value,
                'offset_top': det.parameters['offset_top'].value,
                'offset_right': det.parameters['offset_right'].value,
                'offset_bottom': det.parameters['offset_bottom'].value,
                'offset_left': det.parameters['offset_left'].value,
            },
        }

    @APIExport()
    def detectFlimLaserFrequency(self) -> Dict[str, Any]:
        """Run the card's frequency meter once (laser sync must be connected).

        Example:
            GET /api/FLIMLabsController/detectFlimLaserFrequency
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        if det.isRunning:
            return {'error': 'Stop the running acquisition before detecting the frequency'}
        try:
            freq = det.client.detect_laser_frequency()
        except Exception as e:
            # The flim-server maps all errors to 400; the body carries the
            # reason. 'ProcessAlreadyRunning' = the card is busy - either an
            # acquisition just stopped (retry in a moment) or ANOTHER client
            # (e.g. the FLIM LABS web UI) is using the same server.
            return {'error': f'Frequency detection failed: {e}'}
        if not freq:
            return {'error': 'Frequency detection failed: no signal detected'}
        # Keep the MEASURED value (e.g. 39.99974 MHz), not the nominal grid
        # point. Firmware selection snaps to the nearest grid frequency
        # server-side anyway, while the laser period used for the lifetime
        # maths comes from this number - rounding it to 40 MHz would both
        # misreport the laser and bias the decay axis.
        nominal = min([20, 40, 80, 100], key=lambda f: abs(f - freq))
        det.setParameter('frequency_mhz', freq)
        return {'frequency': freq, 'nominal': nominal}

    @APIExport()
    def getFlimDecayCurve(self, roi: bool = False) -> Dict[str, Any]:
        """Decay histogram per channel, ready to plot.

        256 bins spanning one laser period. ``timeNs`` is the matching x axis in
        nanoseconds, derived from the current laser frequency, so a client can
        plot counts against time without knowing the encoding.

        Example:
            GET /api/FLIMLabsController/getFlimDecayCurve
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        data = det.client.get_decay_curves(roi=bool(roi))
        freqMhz = float(det.parameters['frequency_mhz'].value) or 40.0
        laserPeriodNs = 1000.0 / freqMhz
        bins = data.get('bins', 256)
        data['laserPeriodNs'] = laserPeriodNs
        data['frequencyMhz'] = freqMhz
        data['timeNs'] = [i * laserPeriodNs / bins for i in range(bins)]
        return data

    # ------------------------------------------------------------------
    # Geometry / field of view
    # ------------------------------------------------------------------
    @APIExport(runOnUIThread=True)
    def setFlimFieldOfView(self, fovUmX: Optional[float] = None,
                           fovUmY: Optional[float] = None,
                           nx: Optional[int] = None,
                           ny: Optional[int] = None) -> Dict[str, Any]:
        """Set the scanned field of view in micrometres and/or the sampling.

        The galvo is commanded in DAC counts, so this needs the scanner
        calibration (``umPerDacUnitX/Y`` or ``fovUmFullScaleX/Y`` in the
        detector's setup properties). Pixel size follows as FOV / samples, and
        a running scouting acquisition picks the new geometry up without being
        re-armed.

        Example:
            GET /api/FLIMLabsController/setFlimFieldOfView?fovUmX=200&fovUmY=200&nx=512&ny=512
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        return det.setFieldOfViewUm(fovUmX=fovUmX, fovUmY=fovUmY, nx=nx, ny=ny)

    # ------------------------------------------------------------------
    # Acquisition control
    # ------------------------------------------------------------------
    @APIExport()
    def listFlimCalibrations(self) -> Dict[str, Any]:
        """Calibrations available for a phasors run, newest first.

        The reference JSON lives on the FLIM server (inside its container), so
        this index is kept by ImSwitch as calibrations are run. Feed it to a
        dropdown and pass the chosen ``timestamp`` to startFlimAcquisition.

        Example:
            GET /api/FLIMLabsController/listFlimCalibrations
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        return {'calibrations': det.listCalibrations(),
                'current': det.calibrationReference}

    @APIExport(runOnUIThread=False)
    def startFlimAcquisition(self, step: str = 'scouting',
                             maxFrames: int = 0,
                             tauNs: Optional[float] = None,
                             harmonics: int = 1,
                             exportData: bool = False,
                             exportFilename: str = '',
                             exportPath: str = '',
                             calibrationTimestamp: Optional[int] = None,
                             referenceFile: Optional[str] = None) -> Dict[str, Any]:
        """Start a FLIM run.

        Args:
            step: ``scouting`` (live intensity), ``calibration`` (solid
                calibrator, needs tauNs) or ``phasors`` (needs a prior
                calibration).
            maxFrames: 0 = continuous.
            tauNs: known lifetime of the solid calibrator.
            harmonics: number of harmonics for calibration/phasors.
            exportData: write the acquisition on the FLIM server.
            exportPath: server-side directory for the exports; empty lets the
                server use its own data folder (~/.flim-labs/data).
            calibrationTimestamp: which stored calibration to use for phasors
                (see listFlimCalibrations). Defaults to the newest confirmed
                one; its frequency and harmonics are reused.
            referenceFile: explicit server-side path, overriding the above.

        Example:
            GET /api/FLIMLabsController/startFlimAcquisition?step=calibration&tauNs=4&maxFrames=10
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        if step == 'phasors':
            chosen = det.resolveCalibration(timestamp=calibrationTimestamp,
                                            referenceFile=referenceFile)
            if 'error' in chosen:
                return chosen
            referenceFile = chosen['referenceFile']
            # The server validates the reference against the harmonic count and
            # the number of ENABLED channels and answers 400 on a mismatch;
            # checking here says which of the two is wrong.
            if chosen.get('harmonics') is not None and int(chosen['harmonics']) != int(harmonics):
                return {'error': f'Calibration was made with harmonics='
                                 f'{chosen["harmonics"]}, this run asks for '
                                 f'{harmonics}. Use the same harmonic count.',
                        'calibration': chosen}
            enabled = sum(1 for c in det.channels if c)
            if chosen.get('channelCount') is not None and int(chosen['channelCount']) != enabled:
                return {'error': f'Calibration covers {chosen["channelCount"]} '
                                 f'channel(s) but {enabled} are enabled now. '
                                 f'Enable the same channels or recalibrate.',
                        'calibration': chosen}
            # Reuse the frequency the calibration was taken at: the phase /
            # modulation offsets in the reference are only valid for that laser
            # period, so drifting to a freshly measured value would silently
            # bias every lifetime.
            calFreq = chosen.get('frequencyMhz')
            if calFreq:
                current = float(det.parameters['frequency_mhz'].value)
                if abs(current - float(calFreq)) > 1e-6:
                    det.setParameter('frequency_mhz', float(calFreq))
                    self.__logger.info(
                        f'Restored calibration frequency {calFreq} MHz '
                        f'(was {current} MHz) for the phasors run')
            self.__logger.info(f'Phasors run using calibration {referenceFile}')
        else:
            referenceFile = None
        # Empty path = let the server pick its own data directory. It is the
        # only side that knows where it can write, and an empty path used to
        # mean "write to /", which fails as Permission denied.
        export = {'enabled': bool(exportData),
                  'filename': exportFilename or 'imswitch_flim',
                  'path': exportPath or ''}
        return det.startAcquisitionStep(
            step=step, maxFrames=maxFrames, tauNs=tauNs, harmonics=harmonics,
            referenceFile=referenceFile, export=export)

    @APIExport(runOnUIThread=False)
    def stopFlimAcquisition(self) -> Dict[str, Any]:
        """Stop the running acquisition (and the galvo scan, if auto-driven).

        Example:
            GET /api/FLIMLabsController/stopFlimAcquisition
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        det.stopAcquisition()
        return {'status': 'stopped', 'calibrationReference': det.calibrationReference}

    @APIExport()
    def resetFlimBuffers(self) -> Dict[str, Any]:
        """Clear the accumulated image, phasor histogram and calibration table.

        Example:
            GET /api/FLIMLabsController/resetFlimBuffers
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        det.client.reset_buffers()
        return {'status': 'reset'}

    @APIExport()
    def setFlimParameter(self, name: str, value: float) -> Dict[str, Any]:
        """Set a FLIM detector parameter (dwell_time, frames_to_integrate,
        frequency_mhz, reconstruction).

        Example:
            GET /api/FLIMLabsController/setFlimParameter?name=dwell_time&value=25
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        try:
            det.setParameter(name, value)
            return {'status': 'ok', 'name': name, 'value': det.getParameter(name)}
        except Exception as e:
            return {'error': str(e)}

    # ------------------------------------------------------------------
    # Data for the UI
    # ------------------------------------------------------------------
    @APIExport()
    def getFlimImage(self, maxSize: int = 512) -> Dict[str, Any]:
        """Latest intensity frame as a base64 PNG data URL.

        Returns the progressively-filling current frame while one is being
        received, so the UI updates during the (multi-second) FLIM frame.

        Example:
            GET /api/FLIMLabsController/getFlimImage?maxSize=512
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        frame = det.getDisplayFrame()
        if frame is None or frame.size == 0:
            return {'image': None, 'frameNumber': det.client.get_frame_number()}
        maxVal = int(frame.max())
        img8 = self._toUint8(frame)
        if img8 is None:
            return {'error': 'OpenCV not available for PNG encoding'}
        dataUrl = self._encodePng(img8, maxSize)
        return {
            'image': dataUrl,
            'frameNumber': det.client.get_frame_number(),
            'max': maxVal,
            'width': int(frame.shape[1]),
            'height': int(frame.shape[0]),
            'cps': det.client.cps,
            'running': det.isRunning,
        }

    @APIExport()
    def getFlimPhasor(self, maxPoints: int = 20000) -> Dict[str, Any]:
        """Accumulated phasor density as sparse ``[x, y, count]`` triplets.

        The UI renders these against the universal semicircle; sending sparse
        cells instead of the raw per-pixel G/S matrices keeps the payload small.

        Example:
            GET /api/FLIMLabsController/getFlimPhasor
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        return det.client.get_phasor_sparse(max_points=maxPoints)

    @APIExport()
    def saveFlimData(self, filename: str = '') -> Dict[str, Any]:
        """Save the current FLIM data into ImSwitch's data folder.

        Writes into ``<DataPath>/FLIM/``:
          - ``<name>_intensity.tif``  - cumulative photon counts (uint16 TIFF)
          - ``<name>_decay.json``     - per-channel decay histograms + time axis
          - ``<name>_meta.json``      - geometry, laser frequency, calibration...

        Also returns the TIFF as a base64 data URL so the browser can offer a
        direct download of the raw data (not just the colormapped PNG).

        Example:
            GET /api/FLIMLabsController/saveFlimData?filename=sample1
        """
        det = self._detector
        if det is None:
            return {'error': 'No FLIM detector configured'}
        frame = det.getDisplayFrame()
        if frame is None or frame.size == 0 or not frame.any():
            return {'error': 'No FLIM data acquired yet'}
        if not HAS_CV2:
            return {'error': 'OpenCV not available for TIFF encoding'}

        import os
        import time as _time
        from imswitch.imcommon.model import dirtools
        stamp = _time.strftime('%Y%m%d_%H%M%S')
        name = (filename or 'flim').strip().replace(os.sep, '_') or 'flim'
        base = f'{name}_{stamp}'
        outDir = os.path.join(dirtools.UserFileDirs.Data, 'FLIM')
        os.makedirs(outDir, exist_ok=True)

        # Raw counts as 16-bit TIFF (photon counts per pixel rarely exceed
        # 65535; uint32 TIFFs are poorly supported by downstream viewers)
        tif16 = np.clip(frame, 0, 65535).astype(np.uint16)
        tiffPath = os.path.join(outDir, f'{base}_intensity.tif')
        cv2.imwrite(tiffPath, tif16)

        decay = self.getFlimDecayCurve()
        decayPath = os.path.join(outDir, f'{base}_decay.json')
        with open(decayPath, 'w') as f:
            json.dump(decay, f, indent=2)

        status = self.getFlimStatus()
        meta = {k: status.get(k) for k in (
            'detectorName', 'serverUrl', 'cardSerial', 'step', 'firmware',
            'frameNumber', 'imageWidth', 'imageHeight', 'scanWidth',
            'scanHeight', 'calibrationReference', 'galvoScanner',
            'pixelSizeUm', 'fovUm', 'parameters', 'cps')}
        meta['savedAt'] = stamp
        meta['maxCount'] = int(frame.max())
        metaPath = os.path.join(outDir, f'{base}_meta.json')
        with open(metaPath, 'w') as f:
            json.dump(meta, f, indent=2)

        ok, buf = cv2.imencode('.tif', tif16)
        tiffB64 = ('data:image/tiff;base64,'
                   + base64.b64encode(buf.tobytes()).decode('ascii')) if ok else None
        self.__logger.info(f'FLIM data saved to {outDir} ({base}_*)')
        return {'status': 'saved', 'directory': outDir,
                'files': [tiffPath, decayPath, metaPath],
                'tiff': tiffB64, 'basename': base}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _toUint8(frame: np.ndarray) -> Optional[np.ndarray]:
        """Normalize an intensity frame to 8-bit with a 'hot' colormap."""
        if not HAS_CV2:
            return None
        f = frame.astype(np.float32)
        maxVal = float(f.max())
        if maxVal <= 0:
            norm = np.zeros(f.shape, dtype=np.uint8)
        else:
            norm = np.clip(f / maxVal * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_HOT)

    @staticmethod
    def _encodePng(img: np.ndarray, maxSize: int) -> Optional[str]:
        if not HAS_CV2:
            return None
        if maxSize and max(img.shape[:2]) > maxSize:
            scale = maxSize / float(max(img.shape[:2]))
            img = cv2.resize(img,
                             (max(1, int(img.shape[1] * scale)),
                              max(1, int(img.shape[0] * scale))),
                             interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode('.png', img)
        if not ok:
            return None
        return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


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
