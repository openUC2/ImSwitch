import cv2
import numpy as np
from platform import system
from imswitch.imcommon.model import initLogger
from .DetectorManager import (
    DetectorManager,
    DetectorNumberParameter,
    DetectorListParameter
)


class ImSwitchRESTCamera(DetectorManager):
    """ DetectorManager that deals with TheImagingSource cameras and the
    parameters for frame extraction from them, now using an MJPEG stream.
    Manager properties:
    - ``cameraListIndex`` -- camera index
    """
    #manager(managedDeviceInfo, managedDeviceName, **lowLevelManagers)
    def __init__(self, detectorInfo, name, parent=None, **lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.detectorInfo = detectorInfo
        self.parent = parent
        # Set default values for parameters like pixel size and exposure
        parameters = {}
        parameters['Camera pixel size'] = DetectorNumberParameter(
            group='Misc',
            value=10,
            valueUnits='um',
            editable=True
        )

        super().__init__(detectorInfo, name, fullShape=(640, 480), supportedBinnings=[1],
                    model="MJPEG Stream", parameters=parameters, croppable=True)

        # Initialize the MJPEG stream from the recording manager
        self._rs232manager = lowLevelManagers['rs232sManager'][detectorInfo.managerProperties['rs232device']]
        self._imswitch_client = self._rs232manager._imswitch_client

        # pull funcitons
        self.startVideoStream = self._imswitch_client.recordingManager.startVideoStream
        self.stopVideoStream = self._imswitch_client.recordingManager.stopVideoStream
        self.getFrame = self._imswitch_client.recordingManager.getVideoFrame

        self._running = False
        self._adjustingParameters = False

    @property
    def pixelSizeUm(self):
        try:
            umxpx = self.parameters['Camera pixel size'].value
        except:
            umxpx = 1
        return [1, umxpx, umxpx]

    def getLatestFrame(self, is_save=False):
        '''
        Get the latest frame from the MJPEG stream and return it as a grayscale image.
        '''
        frame = self.getFrame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return None

    def setParameter(self, name, value):
        '''
        Set parameters like exposure (this method is updated for MJPEG stream case).
        '''
        if name == 'Exposure':
            # Handle exposure changes here (if applicable to the stream)
            if system() == 'Windows':
                # Example for Windows (adjustment logic can vary depending on the API)
                pass
            else:
                # This depends on whether the MJPEG stream supports exposure adjustment
                pass
        super().setParameter(name, value)

    def getChunk(self):
        try:
            return self.getLatestFrame()
        except:
            return None

    def flushBuffers(self):
        pass

    def startAcquisition(self, liveView=False):
        '''
        Start pulling frames from the MJPEG stream.
        '''
        self.startVideoStream()

    def stopAcquisition(self):
        '''
        Stop pulling frames from the MJPEG stream.
        '''
        self.stopVideoStream()

    def finalize(self) -> None:
        self.stopVideoStream()

    def setPixelSizeUm(self, pixelSizeUm):
        self.parameters['Camera pixel size'].value = pixelSizeUm

    def crop(self, hpos, vpos, hsize, vsize):
        '''
        Handle cropping of the frame by adjusting how much of the frame to extract.
        Note that cropping may not be fully supported by the MJPEG stream interface.
        '''
        pass
