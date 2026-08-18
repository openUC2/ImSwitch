
from imswitch.imcommon.model import initLogger
from .DetectorManager import DetectorManager, DetectorAction, DetectorNumberParameter, DetectorListParameter, DetectorBooleanParameter

# Parameters CameraTucsen.getPropertyValue() can answer. Everything else is
# write-only or ImSwitch-side bookkeeping, and querying it would only produce
# "unknown property" warnings from the camera wrapper.
_HARDWARE_READABLE_PARAMS = (
    'exposure', 'gain', 'blacklevel', 'frame_rate', 'trigger_source',
    'image_width', 'image_height',
)


class TucsenCamManager(DetectorManager):
    """ DetectorManager that deals with Tucsen cameras and the
    parameters for frame extraction from them.

    Manager properties:

    - ``cameraListIndex`` -- the camera's index in the Tucsen camera list (list
      indexing starts at 0); set this string to an invalid value, e.g. the
      string "mock" to load a mocker
    - ``tucsencam`` -- dictionary of Tucsen camera properties
    - ``binning`` -- binning factor to start up with (default 1)
    - ``supportedBinnings`` -- selectable binning factors (default ``[1, 2]``);
      the camera switches between its RESOLUTION and SENSITIVE (2x2-combined)
      readout modes, so anything above 2 is not meaningful
    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.detectorInfo = detectorInfo

        try:
            binning = int(detectorInfo.managerProperties['binning'])
        except Exception:
            binning = 1
        cameraId = detectorInfo.managerProperties['cameraListIndex']
        # Pixel size and flip are owned by PixelCalibrationController; the
        # values are injected via setPixelSizeUm() / setFlipImage() at startup
        # and on objective change. Use neutral defaults here.
        pixelSize = 1.0

        try:
            self._mockstackpath = detectorInfo.managerProperties['mockstackpath']
        except:
            self._mockstackpath = None

        try: # FIXME: get that form the real camera
            isRGB = detectorInfo.managerProperties['isRGB']
        except:
            isRGB = False

        try:
            self._mocktype = detectorInfo.managerProperties['mocktype']
        except:
            self._mocktype = "normal"

        flipX = False
        flipY = False

        flipImage = (flipY, flipX)

        self._camera = self._getTucsenObj(cameraId, isRGB, binning, flipImage)

        for propertyName, propertyValue in detectorInfo.managerProperties['tucsencam'].items():
            self._camera.setPropertyValue(propertyName, propertyValue)

        fullShape = (self._camera.SensorWidth, #TODO: This can be zero if loaded from Windows, why?
                     self._camera.SensorHeight)

        model = self._camera.model
        self._running = False
        self._adjustingParameters = False

        # TODO: Not implemented yet
        self.crop(hpos=0, vpos=0, hsize=fullShape[0], vsize=fullShape[1])

        # Read actual values and limits from the camera where it can report them
        exposure_min = exposure_max = None
        try:
            hw_exposure = self._camera.get_exposuretime()  # (current, min, max) in ms
            initial_exposure = hw_exposure[0] if hw_exposure and hw_exposure[0] is not None else 100
            exposure_min, exposure_max = hw_exposure[1], hw_exposure[2]
        except Exception:
            initial_exposure = 100
        gain_min = gain_max = None
        try:
            hw_gain = self._camera.get_gain()  # (current, min, max)
            initial_gain = hw_gain[0] if hw_gain and hw_gain[0] is not None else 1
            gain_min, gain_max = hw_gain[1], hw_gain[2]
        except Exception:
            initial_gain = 1

        # Prepare parameters
        parameters = {
            'exposure': DetectorNumberParameter(group='Misc', value=initial_exposure, valueUnits='ms',
                                                editable=True, valueMin=exposure_min,
                                                valueMax=exposure_max),
            'gain': DetectorNumberParameter(group='Misc', value=initial_gain, valueUnits='arb.u.',
                                            editable=True, valueMin=gain_min, valueMax=gain_max),
            'blacklevel': DetectorNumberParameter(group='Misc', value=100, valueUnits='arb.u.',
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
                            options=['manual', 'auto', 'single'], editable=True),
            'flat_fielding': DetectorBooleanParameter(group='Misc', value=True, editable=True),
            'mode': DetectorBooleanParameter(group='Misc', value=name, editable=False), # auto or manual exposure settings
            'previewMinValue': DetectorNumberParameter(group='Misc', value=0, valueUnits='arb.u.',
                                    editable=True),
            'previewMaxValue': DetectorNumberParameter(group='Misc', value=255, valueUnits='arb.u.',
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
            supportedBinnings = [1, 2]
        # The configured startup binning has to be selectable, otherwise the
        # base class rejects it when it applies supportedBinnings[0].
        if binning not in supportedBinnings:
            supportedBinnings.insert(0, binning)

        # Start the frame grabbing thread
        # self.startAcquisition()
        super().__init__(detectorInfo, name, fullShape=fullShape,
                         supportedBinnings=supportedBinnings,
                         model=model, parameters=parameters, actions=actions, croppable=True)

        # DetectorManager.__init__ applies supportedBinnings[0]; make sure the
        # camera ends up on the binning that was requested in the setup file.
        if binning != self.binning:
            self.setBinning(binning)

    def getLatestFrame(self, is_resize=True, returnFrameNumber=False):
        return self._camera.getLast(returnFrameNumber=returnFrameNumber)

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
        return value

    def getParameter(self, name):
        """Gets a parameter value and returns the value.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an error will be
        raised."""

        if name not in self.parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        value = self._camera.getPropertyValue(name)
        return value

    def refreshParameters(self):
        """Re-read the camera-backed parameters (exposure, gain, ...)."""
        return self._refreshParametersFromCamera(_HARDWARE_READABLE_PARAMS)

    def setBinning(self, binning):
        """Switch the readout mode and follow the resulting frame size.

        Tucsen exposes binning as RESOLUTION (1x) vs SENSITIVE (2x2 combined),
        so the camera re-reports its sensor size after the change.
        """
        super().setBinning(binning)

        if not hasattr(self._camera, 'setBinning'):
            return
        if getattr(self._camera, 'binning', None) == binning:
            # Already applied (e.g. by the camera constructor) – don't restart
            # the stream for a no-op.
            return

        def binningAction():
            self._camera.setBinning(binning)
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
        self._performSafeCameraAction(lambda: self._camera.setTriggerSource(source))
        self.parameters['trigger_source'].value = source

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
        pass

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

    def _getTucsenObj(self, cameraId, isRGB=False, binning=1, flipImage=(False, False)):
        try:
            from imswitch.imcontrol.model.interfaces.tucsencamera import CameraTucsen
            self.__logger.debug(f'Trying to initialize Tucsen camera {cameraId}')
            camera = CameraTucsen(cameraNo=cameraId, isRGB=isRGB, binning=binning, flipImage=flipImage)
        except Exception as e:
            self.__logger.error(e)
            self.__logger.warning(f'Failed to initialize CameraTucsen {cameraId}, loading Tucsen mocker')
            from imswitch.imcontrol.model.interfaces.tucsencamera_mock import MockCameraTucsen
            camera = MockCameraTucsen(cameraNo=cameraId, isRGB=isRGB, binning=binning)

        self.__logger.info(f'Initialized camera, model: {camera.model}')
        return camera

    def closeEvent(self):
        self._camera.close()

    def getCameraStatus(self):
        """ Returns comprehensive Tucsen camera status information. """
        status = super().getCameraStatus()

        status['cameraType'] = 'Tucsen'
        status['isMock'] = self._camera.model == "mock"
        status['isConnected'] = getattr(self._camera, 'is_connected', not status['isMock'])
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
