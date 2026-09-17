from imswitch.imcommon.model import initLogger
from .DetectorManager import (DetectorManager, DetectorAction, DetectorNumberParameter,
                              DetectorListParameter)

# Parameters CameraAV.getPropertyValue() can answer. Everything else is
# ImSwitch-side bookkeeping (preview range, pixel size) and querying it would
# only produce "property does not exist" warnings from the camera wrapper.
_HARDWARE_READABLE_PARAMS = (
    'exposure', 'gain', 'blacklevel', 'exposure_mode', 'frame_rate',
    'frame_number', 'image_width', 'image_height', 'trigger_source',
)
_CAMERA_BACKED_PARAMS = (
    'exposure', 'gain', 'blacklevel', 'exposure_mode', 'frame_rate', 'trigger_source',
)


class AVManager(DetectorManager):
    """ DetectorManager that deals with Allied Vision cameras (VmbPy / Vimba)
    and the parameters for frame extraction from them. Same surface as
    ``HikCamManager``: binning, hardware ROI, auto exposure, frame-rate cap,
    software/external trigger and synchronous snaps.

    Manager properties:

    - ``cameraListIndex`` -- the camera's index in the Allied Vision camera
      list (list indexing starts at 0) or its camera ID string; set this to an
      invalid value, e.g. the string "mock", to load a mocker
    - ``isRGB`` -- force colour (1) or mono (0) output; omit to auto-detect
    - ``binning`` -- startup binning factor (default 1)
    - ``supportedBinnings`` -- list of selectable binning factors (default:
      ``[1, 2, 4, 8]`` limited to what the camera reports)
    - ``avcam`` -- dictionary of camera properties applied at startup
      (``exposure`` [ms], ``gain``, ``blacklevel``, ``frame_rate`` [fps, -1 =
      no cap], ``pixel_format`` [e.g. "Mono8"], ``exposure_mode``)
    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.detectorInfo = detectorInfo
        props = detectorInfo.managerProperties

        cameraId = props.get('cameraListIndex')
        # NOTE: pixel size and flip are owned by PixelCalibrationController and
        # injected via setPixelSizeUm() / setFlipImage() during startup. We only
        # use defaults here so the camera can come up before calibration is loaded.
        pixelSize = 1.0
        flipImage = (False, False)

        self._mockstackpath = props.get('mockstackpath')
        self._mocktype = props.get('mocktype', 'normal')
        # None = let the camera tell us (a colour model lists Bayer/RGB formats)
        isRGB = props.get('isRGB')
        if isRGB is not None:
            isRGB = bool(isRGB)

        try:
            binning = int(props.get('binning', 1))
        except (TypeError, ValueError):
            binning = 1

        avcam = dict(props.get('avcam') or {})
        # Frame rate and pixel format go to the CONSTRUCTOR: the camera has to
        # be brought up with them before the stream configuration is read back.
        try:
            frame_rate = float(avcam.pop('frame_rate', -1))
        except (TypeError, ValueError):
            frame_rate = -1
        pixel_format = avcam.pop('pixel_format', None)

        self._camera = self._getAVObj(cameraId, isRGB, binning, flipImage, frame_rate,
                                      pixel_format)

        for propertyName, propertyValue in avcam.items():
            self._camera.setPropertyValue(propertyName, propertyValue)

        fullShape = (self._camera.SensorWidth, self._camera.SensorHeight)
        model = self._camera.model
        self._running = False
        self._adjustingParameters = False

        self.crop(hpos=0, vpos=0, hsize=fullShape[0], vsize=fullShape[1])

        # Seed the parameter list from the hardware, not from constants
        exposure_min = exposure_max = None
        try:
            hw_exposure = self._camera.get_exposuretime()  # (current, min, max) in µs
            initial_exposure = hw_exposure[0] / 1000 if hw_exposure and hw_exposure[0] is not None else 100
            if hw_exposure and hw_exposure[1] is not None:
                exposure_min = hw_exposure[1] / 1000
            if hw_exposure and hw_exposure[2] is not None:
                exposure_max = hw_exposure[2] / 1000
        except Exception:
            initial_exposure = 100
        gain_min = gain_max = None
        try:
            hw_gain = self._camera.get_gain()  # (current, min, max)
            initial_gain = hw_gain[0] if hw_gain and hw_gain[0] is not None else 1
            gain_min, gain_max = hw_gain[1], hw_gain[2]
        except Exception:
            initial_gain = 1
        try:
            initial_blacklevel = self._camera.getPropertyValue('blacklevel')
            if initial_blacklevel is None or initial_blacklevel is False:
                initial_blacklevel = 0
        except Exception:
            initial_blacklevel = 0
        try:
            initial_frame_rate = self._camera.getPropertyValue('frame_rate')
            if initial_frame_rate is None or initial_frame_rate is False:
                initial_frame_rate = frame_rate
        except Exception:
            initial_frame_rate = frame_rate
        try:
            initial_exposure_mode = self._camera.getPropertyValue('exposure_mode') or 'manual'
        except Exception:
            initial_exposure_mode = 'manual'

        parameters = {
            'exposure': DetectorNumberParameter(group='Misc', value=initial_exposure, valueUnits='ms',
                                                editable=True, valueMin=exposure_min,
                                                valueMax=exposure_max),
            'gain': DetectorNumberParameter(group='Misc', value=initial_gain, valueUnits='arb.u.',
                                            editable=True, valueMin=gain_min, valueMax=gain_max),
            'blacklevel': DetectorNumberParameter(group='Misc', value=initial_blacklevel,
                                                  valueUnits='arb.u.', editable=True),
            'image_width': DetectorNumberParameter(group='Misc', value=fullShape[0], valueUnits='arb.u.',
                                                   editable=False),
            'image_height': DetectorNumberParameter(group='Misc', value=fullShape[1], valueUnits='arb.u.',
                                                    editable=False),
            'frame_rate': DetectorNumberParameter(group='Misc', value=initial_frame_rate,
                                                  valueUnits='fps', editable=True),
            'frame_number': DetectorNumberParameter(group='Misc', value=1, valueUnits='frames',
                                                    editable=False),
            'exposure_mode': DetectorListParameter(group='Misc', value=initial_exposure_mode,
                                                   options=['manual', 'auto', 'once'], editable=True),
            'previewMinValue': DetectorNumberParameter(group='Misc', value=0, valueUnits='arb.u.',
                                                       editable=True),
            'previewMaxValue': DetectorNumberParameter(group='Misc', value=self._getPreviewMaxValue(),
                                                       valueUnits='arb.u.', editable=True),
            'trigger_source': DetectorListParameter(group='Acquisition mode',
                                                    value='Continous',
                                                    options=['Continous',
                                                             'Internal trigger',
                                                             'External trigger'],
                                                    editable=True),
            'Camera pixel size': DetectorNumberParameter(group='Miscellaneous', value=pixelSize,
                                                         valueUnits='µm', editable=True),
        }

        actions = {
            'More properties': DetectorAction(group='Misc',
                                              func=self._camera.openPropertiesGUI)
        }

        try:
            supportedBinnings = [int(b) for b in props['supportedBinnings']]
            if not supportedBinnings:
                raise ValueError('empty supportedBinnings')
        except Exception:
            supportedBinnings = self._defaultSupportedBinnings()
        # The configured startup binning has to be selectable, otherwise the
        # base class rejects it when it applies supportedBinnings[0].
        if binning not in supportedBinnings:
            supportedBinnings.insert(0, binning)

        # The camera constructor already applied the startup binning; keep the
        # base class from re-applying supportedBinnings[0] on the hardware.
        self._initialising = True
        super().__init__(detectorInfo, name, fullShape=fullShape,
                         supportedBinnings=supportedBinnings,
                         model=model, parameters=parameters, actions=actions, croppable=True)
        self._initialising = False

        if isRGB is None:
            # The base class read isRGB from the setup file (absent -> False);
            # follow what the camera actually delivers instead.
            self.setRGB(bool(getattr(self._camera, 'isRGB', False)))

        # DetectorManager.__init__ applies supportedBinnings[0]; make sure the
        # camera ends up on the binning that was requested in the setup file.
        if binning != self.binning:
            self.setBinning(binning)

    def _defaultSupportedBinnings(self):
        """[1, 2, 4, 8] limited to the range the camera reports (mock: all)."""
        candidates = [1, 2, 4, 8]
        if not hasattr(self._camera, 'getBinningRange'):
            return candidates
        try:
            lo, hi = self._camera.getBinningRange()
        except Exception:
            return candidates
        supported = [b for b in candidates if lo <= b <= hi]
        return supported or [1]

    def _getPreviewMaxValue(self):
        """Max preview value from the camera's active pixel-format bit depth."""
        try:
            bits = int(getattr(self._camera, 'bitDepth', 8))
            return 2 ** bits - 1
        except Exception:
            return 255

    def getLatestFrame(self, is_resize=True, returnFrameNumber=False):
        return self._camera.getLast(returnFrameNumber=returnFrameNumber)

    def flushBuffer(self):
        """Drop buffered frames so the next grab is guaranteed post-move/-settle."""
        if hasattr(self._camera, "flushBuffer"):
            self._camera.flushBuffer()

    def snapSync(self, timeout: float = 2.0):
        """Fire a software trigger and return the resulting (post-move) frame.

        Requires software-trigger mode (``setTriggerSource('software')``). Returns
        None if the camera does not support triggered snaps.
        """
        if hasattr(self._camera, "snapSoftwareTrigger"):
            return self._camera.snapSoftwareTrigger(timeout=timeout)
        return None

    def getFrameNumber(self):
        if hasattr(self._camera, "getFrameNumber"):
            return self._camera.getFrameNumber()
        return -1

    def getStreamDiagnostics(self):
        """Camera-side streaming metrics (see avcamera.getStreamDiagnostics)."""
        if hasattr(self._camera, "getStreamDiagnostics"):
            return self._camera.getStreamDiagnostics()
        return {}

    def setParameter(self, name, value):
        """Sets a parameter value and returns the value.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an error will be
        raised."""

        super().setParameter(name, value)

        # Preview min/max only scale the displayed image, there is no camera
        # property behind them (the base class already stored them).
        if name in ('previewMinValue', 'previewMaxValue'):
            return value

        # Same legacy-name normalisation as the base class
        if name.find("posure") > 0 and name != 'exposure_mode':
            name = "exposure"
        if name not in self._DetectorManager__parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        if name not in _CAMERA_BACKED_PARAMS:
            # e.g. 'Camera pixel size': ImSwitch-side bookkeeping only
            return value

        result = self._camera.setPropertyValue(name, value)
        if name == 'frame_rate' and isinstance(result, (int, float)) and not isinstance(result, bool):
            # The camera may clamp/quantise; show what is actually in force
            self._DetectorManager__parameters[name].value = result
            return result
        return value

    def getParameter(self, name):
        """Gets a parameter value and returns the value.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an error will be
        raised."""

        if name not in self.parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        if name in _HARDWARE_READABLE_PARAMS:
            value = self._camera.getPropertyValue(name)
            if value is not None and value is not False:
                return value
        return self.parameters[name].value

    def refreshParameters(self):
        """Re-read the camera-backed parameters (exposure, gain, ...)."""
        return self._refreshParametersFromCamera(_HARDWARE_READABLE_PARAMS)

    def setBinning(self, binning):
        """Apply BinningHorizontal/BinningVertical on the camera and follow the new frame size."""
        super().setBinning(binning)

        if getattr(self, '_initialising', False) or not hasattr(self._camera, 'setBinning'):
            return
        if getattr(self._camera, 'binning', None) == binning:
            # Already applied (e.g. by the camera constructor) – don't restart
            # the stream for a no-op.
            return

        def binningAction():
            self._camera.setBinning(binning)
            # CameraAV.setBinning re-reads WidthMax/HeightMax, which shrink with
            # the binning factor, so the new full frame size is authoritative.
            width = getattr(self._camera, 'SensorWidth', None)
            height = getattr(self._camera, 'SensorHeight', None)
            if width and height:
                self._shape = (width, height)
                self._frameStart = (0, 0)
                self._setFullShape((width, height))

        try:
            self._performSafeCameraAction(binningAction)
        except Exception as e:
            self.__logger.error(f'Failed to set binning {binning}: {e}')

    def setTriggerSource(self, source):
        # update camera safely and mirror value in GUI parameter list
        self.__logger.debug(f'Setting trigger source to {source}')
        ok = self._camera.setTriggerSource(source)
        if ok is not False:
            self.parameters['trigger_source'].value = source
        return ok is not False

    def getChunk(self):
        try:
            return self._camera.getLastChunk()
        except Exception:
            return None

    def flushBuffers(self):
        self._camera.flushBuffer()

    def startAcquisition(self):
        if not self._running:
            self._camera.start_live()
            self._running = True
            self.__logger.debug('startlive')

    def stopAcquisition(self):
        if self._running:
            self._running = False
            self._camera.suspend_live()
            self.__logger.debug('suspendlive')

    def stopAcquisitionForROIChange(self):
        self._running = False
        self._camera.stop_live()
        self.__logger.debug('stoplive')

    def finalize(self) -> None:
        super().finalize()
        self.__logger.debug('Safely disconnecting the camera...')
        self._camera.close()

    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]

    def setPixelSizeUm(self, pixelSizeUm):
        self.parameters['Camera pixel size'].value = pixelSizeUm

    def setFlipImage(self, flipY: bool, flipX: bool):
        """Set flip settings for the camera during runtime (applied per frame)."""
        self._camera.flipImage = (flipY, flipX)
        self.__logger.info(f"Updated flip settings: flipY={flipY}, flipX={flipX}")

    def crop(self, hpos, vpos, hsize, vsize):
        '''
        hpos - horizontal start position of crop window
        vpos - vertical start position of crop window
        hsize - horizontal size of crop window
        vsize - vertical size of crop window
        '''
        def cropAction():
            self.__logger.debug(
                f'{self._camera.model}: crop frame to {hsize}x{vsize} at {hpos},{vpos}.'
            )
            applied = self._camera.setROI(hpos, vpos, hsize, vsize)
            # The camera aligns the ROI to its increments; report what it took
            if applied is not None and len(applied) == 4:
                aHpos, aVpos, aHsize, aVsize = applied
            else:
                aHpos, aVpos, aHsize, aVsize = hpos, vpos, hsize, vsize
            self._shape = (aHsize, aVsize)
            self._frameStart = (aHpos, aVpos)

        try:
            self._performSafeCameraAction(cropAction)
        except Exception as e:
            self.__logger.error(e)

    def _performSafeCameraAction(self, function):
        """ This method is used to change those camera properties that need
        the camera to be idle to be able to be adjusted.
        """
        self._adjustingParameters = True
        wasrunning = self._running
        self.stopAcquisitionForROIChange()
        function()
        if wasrunning:
            self.startAcquisition()
        self._adjustingParameters = False

    def openPropertiesDialog(self):
        self._camera.openPropertiesGUI()

    def sendSoftwareTrigger(self):
        """Send a software trigger to the camera."""
        if self._camera.send_trigger():
            self.__logger.debug('Software trigger sent successfully.')
        else:
            self.__logger.warning('Failed to send software trigger.')

    def getCurrentTriggerType(self):
        """Get the current trigger type of the camera."""
        return self._camera.getTriggerSource()

    def getTriggerTypes(self):
        """Get the available trigger types for the camera."""
        return self._camera.getTriggerTypes()

    def _getAVObj(self, cameraId, isRGB=None, binning=1, flipImage=(False, False),
                  frame_rate=-1, pixel_format=None):
        try:
            from imswitch.imcontrol.model.interfaces.avcamera import CameraAV
            self.__logger.debug(f'Trying to initialize Allied Vision camera {cameraId}')
            camera = CameraAV(cameraId, isRGB=isRGB, binning=binning, flipImage=flipImage,
                              frame_rate=frame_rate, pixel_format=pixel_format)
        except Exception as e:
            self.__logger.error(e)
            self.__logger.warning(f'Failed to initialize AV camera {cameraId}, loading TIS mocker')
            from imswitch.imcontrol.model.interfaces.tiscamera_mock import MockCameraTIS
            camera = MockCameraTIS(mocktype=self._mocktype, mockstackpath=self._mockstackpath,
                                   isRGB=bool(isRGB))

        self.__logger.info(f'Initialized camera, model: {camera.model}')
        return camera

    def closeEvent(self):
        self._camera.close()

    def getCameraStatus(self):
        """ Returns comprehensive Allied Vision camera status information. """
        status = super().getCameraStatus()

        status['cameraType'] = 'AlliedVision'
        status['isMock'] = getattr(self._camera, 'model', '') == 'mock'
        status['isConnected'] = bool(getattr(self._camera, 'is_connected', not status['isMock']))
        status['isAcquiring'] = self._running
        status['isAdjustingParameters'] = self._adjustingParameters

        try:
            camera_params = self._camera.get_camera_parameters()
            if camera_params:
                status['hardwareParameters'] = camera_params
        except Exception as e:
            self.__logger.debug(f"Could not retrieve hardware parameters: {e}")

        try:
            status['currentTriggerSource'] = self._camera.getTriggerSource()
            status['availableTriggerTypes'] = self._camera.getTriggerTypes()
        except Exception as e:
            self.__logger.debug(f"Could not retrieve trigger information: {e}")

        return status

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
