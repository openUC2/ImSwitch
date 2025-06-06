import numpy as np

from imswitch.imcommon.model import initLogger
from .DetectorManager import DetectorManager, DetectorAction, DetectorNumberParameter, DetectorListParameter


class PCOManager(DetectorManager):
    """ DetectorManager that deals with the PCO cameras and the
    parameters for frame extraction from them.

    Manager properties:

    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)

        binning = 1
        cameraId = detectorInfo.managerProperties['cameraListIndex']
        self._camera = self._getPCOObj(cameraId, binning)

        for propertyName, propertyValue in detectorInfo.managerProperties['PCOcam'].items():
            self._camera.setPropertyValue(propertyName, propertyValue)

        fullShape = (self._camera.SensorWidth,
                     self._camera.SensorHeight)

        model = self._camera.model
        self._running = False
        self._adjustingParameters = False

        # TODO: Not implemented yet
        self.crop(hpos=0, vpos=0, hsize=fullShape[0], vsize=fullShape[1])


        # Prepare parameters
        parameters = {
            'exposure': DetectorNumberParameter(group='Misc', value=50, valueUnits='ms',
                                                editable=True),
            'gain': DetectorNumberParameter(group='Misc', value=1, valueUnits='arb.u.',
                                            editable=True),
            'blacklevel': DetectorNumberParameter(group='Misc', value=100, valueUnits='arb.u.',
                                            editable=True),
            'Width': DetectorNumberParameter(group='Misc', value=fullShape[0], valueUnits='arb.u.',
                        editable=False),
            'Height': DetectorNumberParameter(group='Misc', value=fullShape[1], valueUnits='arb.u.',
                        editable=False),
            'frame_rate': DetectorNumberParameter(group='Misc', value=-1, valueUnits='fps',
                                    editable=False),
            'trigger_source': DetectorListParameter(group='Acquisition mode',
                            value='Continous',
                            options=['Continous',
                                        'Internal trigger',
                                        'External start',
                                        'External control'],
                            editable=True),
            "buffer_size": DetectorNumberParameter(group='Misc', value=100, valueUnits='arb.u.',
                                    editable=True),
            }

        # Prepare actions
        actions = {
            'More properties': DetectorAction(group='Misc',
                                              func=self._camera.openPropertiesGUI)
        }

        super().__init__(detectorInfo, name, fullShape=fullShape, supportedBinnings=[1],
                         model=model, parameters=parameters, actions=actions, croppable=True)

    def getLatestFrame(self, is_save=False):
        return self._camera.getLast()

    def setParameter(self, name, value):
        """Sets a parameter value and returns the value.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an error will be
        raised."""

        super().setParameter(name, value)

        if name not in self._DetectorManager__parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        value = self._camera.setPropertyValue(name, value)
        return value

    def getParameter(self, name):
        """Gets a parameter value and returns the value.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an error will be
        raised."""

        if name not in self._parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        value = self._camera.getPropertyValue(name)
        return value

    def getChunk(self):
        try:
            return self._camera.getLastChunk()
        except Exception as e:
            self.__logger.error(e)
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
            self._camera.stop_live()
            self.__logger.debug('stop_live')

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
        return [1, 1, 1]

    def crop(self, hpos, vpos, hsize, vsize):

        def cropAction():
            self.__logger.debug(
                f'{self._camera.model}: crop frame to {hsize}x{vsize} at {hpos},{vpos}.'
            )
            self._camera.setROI(hpos, vpos, hsize, vsize)
            # TOdO: weird hackaround
            self._shape = (self._camera.SensorWidth//self._camera.binning, self._camera.SensorHeight//self._camera.binning)
            self._frameStart = (hpos, vpos)
            pass
        try:
            self._performSafeCameraAction(cropAction)
        except Exception as e:
            self.__logger.error(e)
            # TODO: unsure if frameStart is needed? Try without.
        # This should be the only place where self.frameStart is changed

        # Only place self.shapes is changed

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

    def _getPCOObj(self, cameraId, binning=1):
        try:
            from imswitch.imcontrol.model.interfaces.pcocamera import CameraPCO
            self.__logger.debug(f'Trying to initialize PCO camera {cameraId}')
            camera = CameraPCO(cameraNo=cameraId, binning=binning)
        except Exception as e:
            self.__logger.debug(e)
            self.__logger.warning(f'Failed to initialize CameraPCO {cameraId}, loading PCO mocker')
            from imswitch.imcontrol.model.interfaces.pco_mock import MockCameraPCO
            camera = MockCameraPCO()

        self.__logger.info(f'Initialized camera, model: {camera.model}')
        return camera

    def closeEvent(self):
        self._camera.close()

    def getFrameId(self):
        return self._camera.getLastFrameId()

    # for simulation only
    def setIlluPatternByID(self, iRot, iPhi):
        self._camera.setIlluPatternByID(iRot, iPhi)

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
