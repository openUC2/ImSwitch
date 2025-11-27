"""
Detector Manager for Raspberry Pi Camera (Picamera2)
Compatible interface with HIK camera manager for seamless integration.
"""

import numpy as np
from imswitch.imcommon.model import initLogger
from .DetectorManager import (
    DetectorManager, 
    DetectorAction, 
    DetectorNumberParameter, 
    DetectorListParameter, 
    DetectorBooleanParameter
)


class Picamera2Manager(DetectorManager):
    """
    DetectorManager for Raspberry Pi camera using Picamera2 library.
    
    Manager properties:
    
    - ``cameraListIndex`` -- camera index (0 for first camera, 1 for second, etc.)
    - ``picamera2`` -- dictionary of camera properties
    - ``cameraEffPixelsize`` -- effective pixel size in micrometers
    - ``isRGB`` -- whether to use RGB mode (True) or mono mode (False)
    - ``resolution`` -- (width, height) tuple for camera resolution
    - ``use_video_mode`` -- use video mode for streaming (default: True)
    - ``mocktype`` -- type of mock camera if using mock ("normal", etc.)
    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.detectorInfo = detectorInfo

        # Get configuration parameters
        binning = detectorInfo.managerProperties.get('binning', 1)
        cameraId = detectorInfo.managerProperties['cameraListIndex']
        
        try:
            pixelSize = detectorInfo.managerProperties['cameraEffPixelsize']  # micrometers
        except KeyError:
            pixelSize = 1.12  # Default for RPi HQ camera
            self.__logger.warning(f"Pixel size not specified, using default: {pixelSize} µm")

        try:
            self._mocktype = detectorInfo.managerProperties['mocktype']
        except KeyError:
            self._mocktype = "normal"

        try:
            isRGB = detectorInfo.managerProperties['isRGB']
        except KeyError:
            isRGB = True
            self.__logger.info("RGB mode not specified, defaulting to True")

        try:
            resolution = tuple(detectorInfo.managerProperties['resolution'])
        except KeyError:
            resolution = (640, 480)
            self.__logger.info(f"Resolution not specified, using default: {resolution}")

        try:
            use_video_mode = detectorInfo.managerProperties['use_video_mode']
        except KeyError:
            use_video_mode = True

        # Get flip settings from configuration
        try:
            flipX = detectorInfo.managerProperties['picamera2']['flipX']
        except KeyError:
            flipX = False

        try:
            flipY = detectorInfo.managerProperties['picamera2']['flipY']
        except KeyError:
            flipY = False

        flipImage = (flipY, flipX)

        # Initialize camera
        self._camera = self._getPicamera2Obj(
            cameraId, 
            isRGB, 
            binning, 
            flipImage,
            resolution,
            use_video_mode
        )

        # Apply additional properties from config
        if 'picamera2' in detectorInfo.managerProperties:
            for propertyName, propertyValue in detectorInfo.managerProperties['picamera2'].items():
                if propertyName not in ['flipX', 'flipY']:
                    self._camera.setPropertyValue(propertyName, propertyValue)

        fullShape = (self._camera.SensorWidth, self._camera.SensorHeight)
        model = self._camera.model
        
        self._running = False
        self._adjustingParameters = False

        # Set initial ROI (full frame)
        self.crop(hpos=0, vpos=0, hsize=fullShape[0], vsize=fullShape[1])

        # Prepare parameters
        parameters = {
            'exposure': DetectorNumberParameter(
                group='Misc', 
                value=10, 
                valueUnits='ms',
                editable=True
            ),
            'gain': DetectorNumberParameter(
                group='Misc', 
                value=1.0, 
                valueUnits='arb.u.',
                editable=True
            ),
            'blacklevel': DetectorNumberParameter(
                group='Misc', 
                value=0, 
                valueUnits='arb.u.',
                editable=False  # Not supported on RPi camera
            ),
            'image_width': DetectorNumberParameter(
                group='Misc', 
                value=fullShape[0], 
                valueUnits='px',
                editable=False
            ),
            'image_height': DetectorNumberParameter(
                group='Misc', 
                value=fullShape[1], 
                valueUnits='px',
                editable=False
            ),
            'frame_rate': DetectorNumberParameter(
                group='Misc', 
                value=30, 
                valueUnits='fps',
                editable=True
            ),
            'frame_number': DetectorNumberParameter(
                group='Misc', 
                value=0, 
                valueUnits='frames',
                editable=False
            ),
            'exposure_mode': DetectorListParameter(
                group='Misc', 
                value='manual',
                options=['manual', 'auto', 'once'], 
                editable=True
            ),
            'flat_fielding': DetectorBooleanParameter(
                group='Misc', 
                value=False, 
                editable=True
            ),
            'mode': DetectorBooleanParameter(
                group='Misc', 
                value=name, 
                editable=False
            ),
            'previewMinValue': DetectorNumberParameter(
                group='Misc', 
                value=0, 
                valueUnits='arb.u.',
                editable=True
            ),
            'previewMaxValue': DetectorNumberParameter(
                group='Misc', 
                value=255, 
                valueUnits='arb.u.',
                editable=True
            ),
            'trigger_source': DetectorListParameter(
                group='Acquisition mode',
                value='Continuous',
                options=['Continuous', 'Software Trigger', 'External Trigger (GPIO)'],
                editable=True
            ),
            'Camera pixel size': DetectorNumberParameter(
                group='Miscellaneous', 
                value=pixelSize,
                valueUnits='µm', 
                editable=True
            )
        }

        # Prepare actions
        actions = {
            'Record Flatfield': DetectorAction(
                group='Misc',
                func=self.recordFlatfieldImage
            )
        }

        super().__init__(
            detectorInfo, 
            name, 
            fullShape=fullShape, 
            supportedBinnings=[1, 2, 4],
            model=model, 
            parameters=parameters, 
            actions=actions, 
            croppable=False  # ROI not fully implemented yet
        )

    def setFlatfieldImage(self, flatfieldImage, isFlatfielding):
        """Set flatfield correction image"""
        self._camera.setFlatfieldImage(flatfieldImage, isFlatfielding)

    def getLatestFrame(self, is_resize=True, returnFrameNumber=False):
        """Get the latest frame from the camera"""
        return self._camera.getLast(returnFrameNumber=returnFrameNumber)

    def setParameter(self, name, value):
        """Set a parameter value"""
        super().setParameter(name, value)

        if name not in self._DetectorManager__parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        value = self._camera.setPropertyValue(name, value)
        return value

    def getParameter(self, name):
        """Get a parameter value"""
        if name not in self._parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        value = self._camera.getPropertyValue(name)
        return value

    def setTriggerSource(self, source):
        """Set trigger source and update GUI parameter"""
        self._performSafeCameraAction(lambda: self._camera.setTriggerSource(source))
        self.parameters['trigger_source'].value = source

    def getChunk(self):
        """Get metadata chunk (not implemented for Picamera2)"""
        try:
            return self._camera.getLastChunk()
        except Exception:
            return None

    def flushBuffers(self):
        """Flush frame buffers"""
        self._camera.flushBuffer()

    def startAcquisition(self):
        """Start camera acquisition"""
        if self._camera.model == "MockPicamera2":
            self.__logger.debug('Using mock camera')

        if not self._running:
            self._camera.start_live()
            self._running = True
            self.__logger.debug('Camera acquisition started')

    def stopAcquisition(self):
        """Stop camera acquisition"""
        if self._running:
            self._running = False
            self._camera.suspend_live()
            self.__logger.debug('Camera acquisition suspended')

    def stopAcquisitionForROIChange(self):
        """Stop acquisition for ROI change"""
        self._running = False
        self._camera.stop_live()
        self.__logger.debug('Camera acquisition stopped for ROI change')

    def finalize(self) -> None:
        """Finalize and close camera"""
        super().finalize()
        self.__logger.debug('Safely disconnecting camera...')
        self._camera.close()

    @property
    def pixelSizeUm(self):
        """Get pixel size in micrometers"""
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]

    def setPixelSizeUm(self, pixelSizeUm):
        """Set pixel size in micrometers"""
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
        """
        Set crop window (ROI).
        
        Args:
            hpos: Horizontal start position
            vpos: Vertical start position
            hsize: Horizontal size
            vsize: Vertical size
        """
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
            self.__logger.error(f"Crop failed: {e}")

    def _performSafeCameraAction(self, function):
        """
        Perform camera action safely (stop/start if needed).
        
        Args:
            function: Function to execute while camera is stopped
        """
        self._adjustingParameters = True
        wasrunning = self._running
        
        if wasrunning:
            self.stopAcquisitionForROIChange()
        
        function()
        
        if wasrunning:
            self.startAcquisition()
        
        self._adjustingParameters = False

    def openPropertiesDialog(self):
        """Open camera properties dialog (not implemented)"""
        self._camera.openPropertiesGUI()

    def sendSoftwareTrigger(self):
        """Send a software trigger to the camera"""
        if self._camera.send_trigger():
            self.__logger.debug('Software trigger sent successfully')
        else:
            self.__logger.warning('Failed to send software trigger')

    def getCurrentTriggerType(self):
        """Get the current trigger type"""
        return self._camera.getTriggerSource()

    def getTriggerTypes(self):
        """Get available trigger types"""
        return self._camera.getTriggerTypes()

    def _getPicamera2Obj(self, cameraId, isRGB=True, binning=1, flipImage=(False, False), 
                         resolution=(640, 480), use_video_mode=True):
        """
        Get camera object (real or mock).
        
        Args:
            cameraId: Camera index
            isRGB: RGB or mono mode
            binning: Binning factor
            flipImage: (flipY, flipX) tuple
            resolution: (width, height) tuple
            use_video_mode: Use video mode
            
        Returns:
            Camera object
        """
        try:
            from imswitch.imcontrol.model.interfaces.picamera2_interface import CameraPicamera2
            
            self.__logger.debug(f'Initializing Picamera2 camera {cameraId}')
            
            camera = CameraPicamera2(
                cameraNo=cameraId,
                isRGB=isRGB,
                binning=binning,
                flipImage=flipImage,
                resolution=resolution,
                use_video_mode=use_video_mode
            )
            
            self.__logger.info(f'Initialized camera: {camera.model}')
            return camera
            
        except Exception as e:
            self.__logger.error(f'Failed to initialize Picamera2: {e}')
            self.__logger.warning('Loading mock camera instead')
            
            from imswitch.imcontrol.model.interfaces.picamera2_interface import MockCameraPicamera2
            
            camera = MockCameraPicamera2(
                isRGB=isRGB,
                resolution=resolution
            )
            
            self.__logger.info(f'Initialized mock camera: {camera.model}')
            return camera

    def closeEvent(self):
        """Handle close event"""
        self._camera.close()

    def recordFlatfieldImage(self):
        """Record flatfield image by averaging multiple frames"""
        self._camera.recordFlatfieldImage()


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
