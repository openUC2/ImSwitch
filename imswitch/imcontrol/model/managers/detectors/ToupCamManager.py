from imswitch.imcommon.model import initLogger
from .DetectorManager import DetectorManager, DetectorAction, DetectorNumberParameter, DetectorListParameter, DetectorBooleanParameter

# Parameters the camera can be asked about at any time. Everything else is
# write-only or ImSwitch-side bookkeeping, and querying it would only produce
# "property does not exist" warnings from the camera wrapper.
_HARDWARE_READABLE_PARAMS = (
    'exposure', 'gain', 'blacklevel', 'exposure_mode', 'frame_rate',
    'frame_number', 'image_width', 'image_height', 'trigger_source',
    'temperature', 'pixel_format',
)


class ToupCamManager(DetectorManager):
    """ DetectorManager that deals with ToupTek (Toupcam) cameras and the
    parameters for frame extraction from them.

    Manager properties:

    - ``cameraListIndex`` -- the camera's index in the Toupcam device list
      (list indexing starts at 0); set this to an invalid value, e.g. the
      string "mock" to load a mocker
    - ``toupcam`` -- dictionary of Toupcam camera properties
    - ``supportedBinnings`` -- list of selectable binning factors (default
      ``[1, 2, 3, 4]``); Toupcam applies these as averaged digital binning
    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.detectorInfo = detectorInfo

        cameraId = detectorInfo.managerProperties['cameraListIndex']
        # NOTE: pixel size and flip are owned by PixelCalibrationController and
        # injected via setPixelSizeUm() / setFlipImage() during startup. We only
        # use defaults here so the camera can come up before calibration is loaded.
        pixelSize = 1.0

        try:
            self._mockstackpath = detectorInfo.managerProperties['mockstackpath']
        except:
            self._mockstackpath = None

        try:
            isRGB = detectorInfo.managerProperties['isRGB']
        except:
            isRGB = False

        try:
            self._mocktype = detectorInfo.managerProperties['mocktype']
        except:
            self._mocktype = "normal"

        # Flip is set by PixelCalibrationController via setFlipImage() once the
        # per-detector affine calibration has been loaded from the setup config.
        flipImage = (False, False)

        try:
            binning = detectorInfo.managerProperties['binning']
        except:
            binning = 1
        self._camera = self._getToupcamObj(cameraId, isRGB, binning, flipImage)

        for propertyName, propertyValue in detectorInfo.managerProperties['toupcam'].items():
            self._camera.setPropertyValue(propertyName, propertyValue)

        fullShape = (self._camera.SensorWidth,
                     self._camera.SensorHeight)

        model = self._camera.model
        self._running = False
        self._adjustingParameters = False

        self.crop(hpos=0, vpos=0, hsize=fullShape[0], vsize=fullShape[1])

        # Read actual values from camera hardware instead of using hardcoded defaults
        exposure_min = exposure_max = None
        try:
            hw_exposure = self._camera.get_exposuretime()  # returns (current, min, max) in µs
            # SDK returns µs, UI expects ms → divide by 1000
            initial_exposure = hw_exposure[0] / 1000 if hw_exposure and hw_exposure[0] is not None else 100
            if hw_exposure and hw_exposure[1] is not None:
                exposure_min = hw_exposure[1] / 1000
            if hw_exposure and hw_exposure[2] is not None:
                exposure_max = hw_exposure[2] / 1000
        except Exception:
            initial_exposure = 100
        gain_min = gain_max = None
        try:
            hw_gain = self._camera.get_gain()  # returns (current, min, max) in native percent
            initial_gain = hw_gain[0] if hw_gain and hw_gain[0] is not None else 0
            gain_min, gain_max = hw_gain[1], hw_gain[2]
        except Exception:
            initial_gain = 0

        # Prepare parameters
        parameters = {
            'exposure': DetectorNumberParameter(group='Misc', value=initial_exposure, valueUnits='ms',
                                                editable=True, valueMin=exposure_min,
                                                valueMax=exposure_max),
            'gain': DetectorNumberParameter(group='Misc', value=initial_gain, valueUnits='arb.u.',
                                            editable=True, valueMin=gain_min, valueMax=gain_max),
            'blacklevel': DetectorNumberParameter(group='Misc', value=0, valueUnits='arb.u.',
                                            editable=True),
            'image_width': DetectorNumberParameter(group='Misc', value=fullShape[0], valueUnits='arb.u.',
                        editable=False),
            'image_height': DetectorNumberParameter(group='Misc', value=fullShape[1], valueUnits='arb.u.',
                        editable=False),
            'frame_rate': DetectorNumberParameter(group='Misc', value=-1, valueUnits='fps',
                                    editable=True),
            'frame_number': DetectorNumberParameter(group='Misc', value=1, valueUnits='frames',
                                    editable=False),
            'exposure_mode': DetectorListParameter(group='Misc', value='manual',
                            options=['manual', 'auto'], editable=True),
            'flat_fielding': DetectorBooleanParameter(group='Misc', value=True, editable=True),
            'mode': DetectorBooleanParameter(group='Misc', value=name, editable=False),
            'previewMinValue': DetectorNumberParameter(group='Misc', value=0, valueUnits='arb.u.',
                                    editable=True),
            'previewMaxValue': DetectorNumberParameter(group='Misc', value=self._getPreviewMaxValue(), valueUnits='arb.u.',
                                    editable=True),
            'trigger_source': DetectorListParameter(group='Acquisition mode',
                            value='Continous',
                            options=['Continous',
                                        'Internal trigger',
                                        'External trigger'],
                            editable=True),
            'Camera pixel size': DetectorNumberParameter(group='Miscellaneous', value=pixelSize,
                                                valueUnits='µm', editable=True)
            }

        # Mono cameras can switch the RAW container bit depth at runtime; in RGB
        # mode the SDK always delivers RGB24, so the option is meaningless there.
        if not isRGB:
            parameters['pixel_format'] = DetectorListParameter(
                group='Image format',
                value='mono16' if getattr(self._camera, '_bits', 8) > 8 else 'mono8',
                options=['mono8', 'mono16'], editable=True)

        # TEC-cooled models get temperature control parameters
        if getattr(self._camera, '_hasTEC', False):
            parameters['target_temperature'] = DetectorNumberParameter(
                group='Cooling', value=0, valueUnits='°C', editable=True)
        if getattr(self._camera, '_hasGetTemperature', False):
            parameters['temperature'] = DetectorNumberParameter(
                group='Cooling', value=0, valueUnits='°C', editable=False)
        if getattr(self._camera, '_hasFan', False):
            parameters['fan_speed'] = DetectorNumberParameter(
                group='Cooling', value=-1, valueUnits='arb.u.', editable=True)

        # Prepare actions
        actions = {
            'More properties': DetectorAction(group='Misc',
                                              func=self._camera.openPropertiesGUI)
        }

        try:
            supportedBinnings = [int(b) for b in
                                 detectorInfo.managerProperties['supportedBinnings']]
            if not supportedBinnings:
                raise ValueError('empty supportedBinnings')
        except Exception:
            supportedBinnings = [1, 2, 3, 4]
        # The configured startup binning has to be selectable, otherwise the
        # base class rejects it when it applies supportedBinnings[0].
        if binning not in supportedBinnings:
            supportedBinnings.insert(0, binning)

        super().__init__(detectorInfo, name, fullShape=fullShape,
                         supportedBinnings=supportedBinnings,
                         model=model, parameters=parameters, actions=actions, croppable=True)

        # DetectorManager.__init__ applies supportedBinnings[0]; make sure the
        # camera ends up on the binning that was requested in the setup file.
        if binning != self.binning:
            self.setBinning(binning)

    def _getPreviewMaxValue(self):
        """Return max preview value based on the camera's active bit depth."""
        return getattr(self._camera, 'max_adu', 255)

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
        """Camera-side streaming metrics (see toupcamcamera.getStreamDiagnostics)."""
        if hasattr(self._camera, "getStreamDiagnostics"):
            return self._camera.getStreamDiagnostics()
        return {}

    def getParameterRanges(self) -> dict:
        """Hardware limits for the editable parameters, in UI units.

        Exposure is reported in ms and gain on the UI's 0..23 scale (the driver
        maps that onto the camera's native analog-gain percent range), so the
        values can be used directly to clamp the frontend's inputs.
        """
        ranges = {}
        try:
            _, expMinUs, expMaxUs = self._camera.get_exposuretime()
            if expMinUs is not None and expMaxUs is not None:
                ranges['exposure'] = {'min': expMinUs / 1000.0,
                                      'max': expMaxUs / 1000.0,
                                      'units': 'ms'}
        except Exception as e:
            self.__logger.debug(f"Could not read exposure range: {e}")
        try:
            _, gainMin, gainMax = self._camera.get_gain()
            if gainMin is not None and gainMax is not None:
                ranges['gain'] = {'min': gainMin, 'max': gainMax, 'units': 'arb.u.'}
        except Exception as e:
            self.__logger.debug(f"Could not read gain range: {e}")
        return ranges

    def isLongExposure(self) -> bool:
        """True when the exposure is too long for a useful live stream."""
        if hasattr(self._camera, "isLongExposure"):
            try:
                return bool(self._camera.isLongExposure())
            except Exception:
                return False
        return False

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

        if name not in self._DetectorManager__parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        value = self._camera.setPropertyValue(name, value)

        # Switching the RAW container also changes the full-scale value, so the
        # preview range has to follow it.
        if name == 'pixel_format' and 'previewMaxValue' in self.parameters:
            self.parameters['previewMaxValue'].value = self._getPreviewMaxValue()

        return value

    def refreshParameters(self):
        """Re-read the camera-backed parameters (temperature, exposure, ...)."""
        return self._refreshParametersFromCamera(_HARDWARE_READABLE_PARAMS)

    def setBinning(self, binning):
        """Apply digital binning on the camera and follow the new frame size."""
        super().setBinning(binning)

        if not hasattr(self._camera, 'setBinning'):
            return
        if getattr(self._camera, 'binning', None) == binning:
            # Already applied (e.g. by the camera constructor) – don't restart
            # the stream for a no-op.
            return

        def binningAction():
            self._camera.setBinning(binning)
            # The frame size shrinks with the binning factor; read it back so
            # ROI handling and the shape reported to the UI stay consistent.
            shape = getattr(self._camera, 'shape', None)
            if shape is not None and len(shape) >= 2:
                binnedShape = (shape[1], shape[0])  # camera reports (h, w)
                self._shape = binnedShape
                self._setFullShape(binnedShape)

        try:
            self._performSafeCameraAction(binningAction)
        except Exception as e:
            self.__logger.error(f'Failed to set binning {binning}: {e}')

    def getParameter(self, name):
        """Gets a parameter value and returns the value.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an error will be
        raised."""

        if name not in self.parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        value = self._camera.getPropertyValue(name)
        return value

    def setTriggerSource(self, source):
        # update camera safely and mirror value in GUI parameter list
        self.__logger.debug(f'Setting trigger source to {source}')
        self._camera.setTriggerSource(source)
        self.parameters['trigger_source'].value = source
        return True

    def getChunk(self):
        try:
            return self._camera.getLastChunk()
        except:
            return None

    def flushBuffers(self):
        self._camera.flushBuffer()

    def startAcquisition(self):
        if self._camera.model == "mock":
            self.__logger.debug('We could attempt to reconnect the camera')
            pass

        if not self._running:
            self._camera.start_live()
            self._running = True
            self.__logger.debug('startlive')
            try:
                debug = self._camera.getDiagnostics()
                self.__logger.info(f"Camera diagnostics after starting live: {debug}")
            except Exception as e:
                self.__logger.warning(f"Could not get camera diagnostics: {e}")

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
        """
        Set flip settings for the camera during runtime.

        Args:
            flipY: Whether to flip vertically
            flipX: Whether to flip horizontally
        """
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
            self._camera.setROI(hpos, vpos, hsize, vsize)
            self._shape = (hsize, vsize)
            self._frameStart = (hpos, vpos)
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

    def _getToupcamObj(self, cameraId, isRGB=False, binning=1, flipImage=(False, False)):
        try:
            from imswitch.imcontrol.model.interfaces.toupcamcamera import CameraToupcam
            self.__logger.debug(f'Trying to initialize Toupcam camera {cameraId}')
            camera = CameraToupcam(cameraNo=cameraId, isRGB=isRGB, binning=binning, flipImage=flipImage)
        except Exception as e:
            self.__logger.error(e)
            self.__logger.warning(f'Failed to initialize CameraToupcam {cameraId}, loading TIS mocker')
            from imswitch.imcontrol.model.interfaces.tiscamera_mock import MockCameraTIS
            camera = MockCameraTIS(mocktype=self._mocktype, mockstackpath=self._mockstackpath, isRGB=isRGB)

        self.__logger.info(f'Initialized camera, model: {camera.model}')
        return camera

    def closeEvent(self):
        self._camera.close()

    def getCameraStatus(self):
        """ Returns comprehensive Toupcam camera status information. """
        # Get base status from parent class
        status = super().getCameraStatus()

        # Add Toupcam-specific information
        status['cameraType'] = 'Toupcam'
        status['isMock'] = self._camera.model == "mock"
        status['isConnected'] = getattr(self._camera, 'is_connected', False)
        status['parameterRanges'] = self.getParameterRanges()
        status['isLongExposure'] = self.isLongExposure()

        # Add acquisition status
        status['isAcquiring'] = self._running
        status['isAdjustingParameters'] = self._adjustingParameters

        # Try to get additional camera parameters if available
        try:
            camera_params = self._camera.get_camera_parameters()
            if camera_params:
                status['hardwareParameters'] = camera_params
        except Exception as e:
            self.__logger.debug(f"Could not retrieve hardware parameters: {e}")

        # Add current trigger source if available
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
