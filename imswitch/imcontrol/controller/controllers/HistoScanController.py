import json
import os
import base64
from fastapi import FastAPI, Response, HTTPException
from imswitch import IS_HEADLESS, __file__
from  imswitch.imcontrol.controller.controllers.camera_stage_mapping import OFMStageMapping
from imswitch.imcommon.model import initLogger, ostools
import numpy as np
import time
import tifffile
import threading
from datetime import datetime
import cv2
import numpy as np
from skimage.io import imsave
from scipy.ndimage import gaussian_filter
from collections import deque
import ast
import skimage.transform
import skimage.util
import skimage
import datetime
import numpy as np
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex, Timer
import time
from ..basecontrollers import LiveUpdatedController
from pydantic import BaseModel
from typing import List, Optional, Union
from PIL import Image
import io
from fastapi import Header
import os
from tempfile import TemporaryDirectory
import numpy as np
import zarr
from fastapi.responses import StreamingResponse
import numpy as np
from io import BytesIO
from PIL import Image
isZARR=False

try:
    from ashlarUC2 import utils
    from ashlarUC2.scripts.ashlar import process_images
    IS_ASHLAR_AVAILABLE = True
except Exception as e:
    IS_ASHLAR_AVAILABLE = False

class ScanParameters(object):
    def __init__(self, name="Wellplate", physDimX=164, physDimY=109, physOffsetX=0, physOffsetY=0, imagePath="imswitch/_data/images/WellplateAdapter3Slides.png"):
        self.name = name
        self.physDimX = physDimX*1e3 # mm
        self.physDimY = physDimY*1e3 # mm
        self.physOffsetX = physOffsetX
        self.physOffsetY =  physOffsetY
        self.imagePath = imagePath

from pydantic import BaseModel
from typing import List, Optional, Union, Tuple

class HistoStatus(BaseModel):
    currentPosition: Optional[Tuple[float, float, int, int, int]] = None # X, Y, nX, nY
    ishistoscanRunning: bool = False
    stitchResultAvailable: bool = False
    mScanIndex: int = 0
    mScanCount: int = 0
    currentStepSizeX: Optional[float] = None
    currentStepSizeY: Optional[float] = None
    currentNX: int = 0
    currentNY: int = 0
    currentOverlap: float = 0.75
    currentAshlarStitching: bool = False
    currentAshlarFlipX: bool = False
    currentAshlarFlipY: bool = False
    currentResizeFactor: float = 0.25
    currentIinitialPosX: Optional[float] = None
    currentIinitialPosY: Optional[float] = None
    currentTimeInterval: Optional[float] = None
    currentNtimes: int = 1
    pixelSize: float = 1.0

    @staticmethod
    def from_dict(status_dict: dict) -> "HistoStatus":
        return HistoStatus(**status_dict)

    def to_dict(self) -> dict:
        return self.dict()


class StitchedImageResponse(BaseModel):
    imageList: List[List[float]]
    image: str

class HistoScanController(LiveUpdatedController):
    """Linked to HistoScanWidget."""

    sigImageReceived = Signal()
    sigUpdatePartialImage = Signal()
    sigUpdateLoadingBar = Signal(int, int) # current, total
    sigUpdateScanCoordinatesLayout = Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # check if we are on remote HTTP version
        if kwargs['master'].rs232sManager.getAllDeviceNames()[0]:
            try:
                self.IS_HTTP = True
                httpManager = kwargs['master'].rs232sManager[kwargs['master'].rs232sManager.getAllDeviceNames()[0]]
                self.remoteHistoManager = httpManager._imswitch_client.histoscanManager
            except Exception as e:
                self._logger.error(f"Could not connect to remote histo manager: {e}")
                self.IS_HTTP = False
        else:
            self.IS_HTTP = False
        # read default values from previously loaded config file
        offsetX = self._master.HistoScanManager.offsetX
        offsetY = self._master.HistoScanManager.offsetY
        self.tSettle = 0.05
        self.flipX = True
        self.flipY = True
        self.histoscanTask = None
        self.histoscanStack = np.ones((1,1,1))

        # read offset between cam and microscope from config file in µm
        self.offsetCamMicroscopeX = -2500 #  self._master.HistoScanManager.offsetCamMicroscopeX
        self.offsetCamMicroscopeY = 2500 #  self._master.HistoScanManager.offsetCamMicroscopeY
        self.currentOverlap = 0.85
        self.currentNY = 2
        self.currentNX = 2
        self.currentStepSizeX = None
        self.currentStepSizeY = None
        self.currentNtimes = 1
        self.currentIinitialPosX = 0
        self.currentIinitialPosY = 0
        self.currentTimeInterval = 0
        self.currentAshlarStitching = False
        self.currentAshlarFlipX = False
        self.currentAshlarFlipY = False
        self.currentResizeFactor = 0.25

        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.microscopeDetector = self._master.detectorsManager[allDetectorNames[0]] # FIXME: This is hardcoded, need to be changed through the GUI
        if len(allDetectorNames)>1:
            self.webCamDetector = self._master.detectorsManager[allDetectorNames[1]] # FIXME: HARDCODED NEED TO BE CHANGED
            self.pixelSizeWebcam = self.webCamDetector.pixelSizeUm[-1]
        else:
            self.webCamDetector = None

        # object for stage mapping/calibration
        self.mStageMapper = None

        # some locking mechanisms
        self.ishistoscanRunning = False
        self.ishistoscanRunning = False

        # select lasers and add to gui
        allLaserNames = self._master.lasersManager.getAllDeviceNames()
        if "LED" in allLaserNames:
            self.led = self._master.lasersManager["LED"]
        else:
            self.led = None

        # grab ledmatrix if available
        if len(self._master.LEDMatrixsManager.getAllDeviceNames())>0:
            self.ledMatrix = self._master.LEDMatrixsManager[self._master.LEDMatrixsManager.getAllDeviceNames()[0]]
        else:
            self.ledMatrix = None

        # this is the list of scan coordinates e.g. for a well
        self.scanPositionList = []

        # connect signals
        self.sigImageReceived.connect(self.displayImage)
        self.sigUpdatePartialImage.connect(self.updatePartialImage)
        self.sigUpdateScanCoordinatesLayout.connect(self.setScanCoordinatesLayout)
        self._commChannel.sigUpdateMotorPosition.connect(self.updateAllPositionGUI)
        self._commChannel.sigStartTileBasedTileScanning.connect(self.startHistoScanTileBasedByParameters)
        self._commChannel.sigStopTileBasedTileScanning.connect(self.stophistoscanTilebased)

        self.partialImageCoordinates = (0,0,0,0)
        self.partialHistoscanStack = np.ones((1,1,3))
        self.acceleration = 600000
        self.currentPosition = (0,0)
        self.positionList = []
        self.mScanIndex = 0
        self.initialOverlap = 0.85

        # camera-based scanning coordinates   (select from napari layer)
        self.mCamScanCoordinates = None

        # select stage
        self.stages = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]

        # get flatfield manager
        if hasattr(self._master, "FlatfieldManager"):
            self.flatfieldManager = self._master.FlatfieldManager
        else:
            self.flatfieldManager = None

        # define scan parameter per sample and populate into GUI later
        self.allScanParameters = []
        mFWD = os.path.dirname(os.path.realpath(__file__)).split("imswitch")[0]
        self.allScanParameters.append(ScanParameters("6 Wellplate", 126, 86, 0, 0, mFWD+"imswitch/_data/images/Wellplate6.png"))
        self.allScanParameters.append(ScanParameters("24 Wellplate", 126, 86, 0, 0, mFWD+"imswitch/_data/images/Wellplate24.png"))
        self.allScanParameters.append(ScanParameters("3-Slide Wellplateadapter", 164, 109, 0, 0, mFWD+"imswitch/_data/images/WellplateAdapter3Slides.png"))

        # compute optimal scan step size based on camera resolution and pixel size
        self.bestScanSizeX, self.bestScanSizeY = self.computeOptimalScanStepSize(overlap = self.initialOverlap)

        if not IS_HEADLESS:
            '''
            Set up the GUI
            '''
            self._widget.loadSampleLayout(0, self.allScanParameters)
            self._widget.setOffset(offsetX, offsetY)
            ## update optimal scan parameters for tile-based scan
            try:
                self._widget.setTilebasedScanParameters((self.bestScanSizeX, self.bestScanSizeY))
            except Exception as e:
                self._logger.error(e)

            self._widget.setAvailableIlluSources(allLaserNames)
            self._widget.startButton.clicked.connect(self.startHistoScanCoordinatebased)
            self._widget.stopButton.clicked.connect(self.stophistoscan)
            self._widget.startButton2.clicked.connect(self.starthistoscanTilebased)
            self._widget.stopButton2.clicked.connect(self.stophistoscanTilebased)
            self._widget.sigSliderIlluValueChanged.connect(self.valueIlluChanged)
            self._widget.sigSliderIlluValueChanged.connect(self.valueIlluChanged)
            self._widget.sigGoToPosition.connect(self.goToPosition)
            self._widget.sigCurrentOffset.connect(self.calibrateOffset)
            self._widget.setDefaultSavePath(self._master.HistoScanManager.defaultConfigPath)
            self.sigUpdateLoadingBar.connect(self._widget.setLoadingBarAndText)

            self._widget.startCalibrationButton.clicked.connect(self.startStageMapping)
            self._widget.stopCalibrationButton.clicked.connect(self.stopStageMapping)

            # Image View
            self._widget.resetScanCoordinatesButton.clicked.connect(self.resetScanCoordinates)
            self._widget.getCameraScanCoordinatesButton.clicked.connect(self.getCameraScanCoordinates)
            self._widget.startButton3.clicked.connect(self.starthistoscanCamerabased)
            self._widget.stopButton3.clicked.connect(self.stophistoscanCamerabased)

            # connect to stage mapping
            self._widget.buttonStartCalibration.clicked.connect(self.startStageMappingFromButton)
            self._widget.buttonStopCalibration.clicked.connect(self.stopHistoScanFromButton)

            # on tab click, add the image to the napari viewer
            self._widget.tabWidget.currentChanged.connect(self.onTabChanged)

            # webcam-related parts
            self.isWebcamRunning = False
            self._widget.imageLabel.doubleClicked.connect(self.onDoubleClickWebcam)
            self._widget.imageLabel.dragPosition.connect(self.onDragPositionWebcam)

            # illu settings
            self._widget.buttonTurnOnLED.clicked.connect(self.turnOnLED)
            self._widget.buttonTurnOffLED.clicked.connect(self.turnOffLED)
            self._widget.buttonTurnOnLEDArray.clicked.connect(self.turnOnLEDArray)
            self._widget.buttonTurnOffLEDArray.clicked.connect(self.turnOffLEDArray)

            # set combobox with all samples
            self._widget.setSampleLayouts(self.allScanParameters)
            self._widget.samplePicker.currentIndexChanged.connect(self._widget.loadSampleLayout)


    @APIExport(runOnUIThread=False)
    def getPreviewCameraImage(self, resizeFactor: float=1) -> Response:
        '''
        Taking a snap and return it as a FastAPI Response object.
        detectorName: the name of the detector to take the snap from. If None, take the snap from the first detector.
        resizeFactor: the factor by which to resize the image. If <1, the image will be downscaled, if >1, nothing will happen.
        '''
        # Create a 2D NumPy array representing the image
        frame = self.webCamDetector.getLatestFrame() # X,Y,C, uint8 numpy array
        if frame is None:
            return
        if len(frame.shape)==2:
            frame = np.repeat(frame[:,:,np.newaxis], 3, axis=2)

        # eventually resize image to save bandwidth
        if resizeFactor <1:
            image = cv2.resize(frame, (0,0), fx=resizeFactor, fy=resizeFactor)
        else:
            image = frame
        # using an in-memory image
        if image.dtype != np.uint8:
            # Normalize and convert to uint8 if necessary
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
        im = Image.fromarray(image)

        # save image to an in-memory bytes buffer
        # save image to an in-memory bytes buffer
        with io.BytesIO() as buf:
            im = im.convert('L')  # convert image to 'L' mode
            im.save(buf, format='PNG')
            im_bytes = buf.getvalue()

        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        return Response(im_bytes, headers=headers, media_type='image/png')

    @APIExport()
    def computeOptimalScanStepSize(self, overlap: float = 0.75):
        mFrameSize = (self.microscopeDetector.fullShape[1], self.microscopeDetector.fullShape[0])
        bestScanSizeX = mFrameSize[1]*self.microscopeDetector.pixelSizeUm[-1]*overlap
        bestScanSizeY = mFrameSize[0]*self.microscopeDetector.pixelSizeUm[-1]*overlap
        return (bestScanSizeX, bestScanSizeY)

    def turnOnLED(self):
        if self.led is not None:
            self.led.setEnabled(1)
            self.led.setValue(255)

    def turnOffLED(self):
        if self.led is not None:
            self.led.setEnabled(0)

    def turnOnLEDArray(self):
        if self.ledMatrix is not None:
            self.ledMatrix.setLEDIntensity(intensity=(255,255,255))
            self.ledMatrix.setAll((1,1,1))

    def turnOffLEDArray(self):
        if self.ledMatrix is not None:
            self.ledMatrix.setAll(0)

    def onTabChanged(self, index):
        '''
        Callback, when we click on the tab, we want to add the image to the napari viewer
        '''
        if index == 2:
            # add layer to napari
            self._widget.initShapeLayerNapari()
            self.microscopeDetector.startAcquisition()
            # run image scraper if not started already
            if not self.isWebcamRunning:
                self.timer = Timer(self)
                self.timer.timeout.connect(self.updateFrameWebcam)
                self.timer.start(100)
                self.isWebcamRunning = True

    def updateFrameWebcam(self):
        '''
        Update the webcam image in the dedicated widget periodically to get an overview
        '''
        if self.webCamDetector is None:
            return
        frame = self.webCamDetector.getLatestFrame() # X,Y,C, uint8 numpy array
        if frame is None:
            return
        if len(frame.shape)==2:
            frame = np.repeat(frame[:,:,np.newaxis], 3, axis=2)
        if frame is not None:
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            from PyQt5.QtGui import QImage, QPixmap
            image = QImage(np.uint8(frame.copy()), width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self._widget.imageLabel.setOriginalPixmap(pixmap)

    def resetScanCoordinates(self):
        '''
        reset the shape coordinates in napari
        '''
        # reset shape layer
        self._widget.resetShapeLayerNapari()
        # reset pre-scan image if available
        name = "Histo Prescan"
        self._widget.removeImageNapari(name)

    def onDoubleClickWebcam(self):
        '''
        Callback: when we double click on the webcam image, we want to move the stage to that position
        '''
        # if we double click on the webcam view, we want to move to that position on the plate
        mPositionClicked = self._widget.imageLabel.doubleClickPos.y(), self._widget.imageLabel.doubleClickPos.x()
        # convert to physical coordinates
        #mDimsWebcamFrame = self.webCamDetector.getLatestFrame().shape
        mDimsWebcamFrame = (self._widget.imageLabel.getCurrentImageSize().height(),self._widget.imageLabel.getCurrentImageSize().width())
        mRelativePosToMoveX = -(-mPositionClicked[0]+mDimsWebcamFrame[0]//2)*self.pixelSizeWebcam
        mRelativePosToMoveY = (-mPositionClicked[1]+mDimsWebcamFrame[1]//2)*self.pixelSizeWebcam
        currentPos = self.stages.getPosition()
        mAbsolutePosToMoveX = currentPos["X"]+mRelativePosToMoveX+self.offsetCamMicroscopeX
        mAbsolutePosToMoveY = currentPos["Y"]+mRelativePosToMoveY+self.offsetCamMicroscopeY
        self.goToPosition(mAbsolutePosToMoveX,mAbsolutePosToMoveY)

    def onDragPositionWebcam(self, start, end):
        '''
        Callback: when we drag the mouse on the webcam image, we want to move the stage to that position
        '''
        print(f"Dragged from {start} to {end}")
        if start is None or self._widget.imageLabel.currentRect is None:
            return
        # use the coordinates for the stage scan
        # 1. retreive the coordinates on the canvas
        minPosX = np.min([start.x(), end.x()])
        maxPosX = np.max([start.x(), end.x()])
        minPosY = np.min([start.y(), end.y()])
        maxPosY = np.max([start.y(), end.y()])

        # 2. compute scan positions
        currentPos = self.stages.getPosition()
        mDimsWebcamFrame = (self._widget.imageLabel.getCurrentImageSize().height(),self._widget.imageLabel.getCurrentImageSize().width())
        minPosXReal = currentPos["X"]-(-minPosX+mDimsWebcamFrame[0]//2)*self.pixelSizeWebcam + self.offsetCamMicroscopeX
        maxPosXReal = currentPos["X"]-(-maxPosX+mDimsWebcamFrame[0]//2)*self.pixelSizeWebcam + self.offsetCamMicroscopeX
        minPosYReal = currentPos["Y"]+(mDimsWebcamFrame[1]//2-maxPosY)*self.pixelSizeWebcam + self.offsetCamMicroscopeY
        maxPosYReal = currentPos["Y"]+(mDimsWebcamFrame[1]//2-minPosY)*self.pixelSizeWebcam + self.offsetCamMicroscopeY

        # 3. get microscope camera parameters
        mFrame = self.microscopeDetector.getLatestFrame()
        pixelSizeMicroscopeDetector = self.microscopeDetector.pixelSizeUm[-1]
        NpixX, NpixY = mFrame.shape[1], mFrame.shape[0]

        # starting the snake scan
        # Calculate the size of the area each image covers
        img_width = NpixX * pixelSizeMicroscopeDetector
        img_height = NpixY * pixelSizeMicroscopeDetector

        # compute snake scan coordinates
        mOverlap = 0.75
        self.mCamScanCoordinates = self.generate_snake_scan_coordinates(minPosXReal, minPosYReal, maxPosXReal, maxPosYReal, img_width, img_height, mOverlap)
        nTilesX = int((maxPosXReal-minPosXReal)/(img_width*mOverlap))
        nTilesY = int((maxPosYReal-minPosYReal)/(img_height*mOverlap))
        self._widget.setCameraScanParameters(nTilesX, nTilesY, minPosX, maxPosX, minPosY, maxPosY)

        return self.mCamScanCoordinates


    def getCameraScanCoordinates(self):
        ''' retreive the coordinates of the shape layer in napari and compute the
        min/max positions for X/Y to provide the snake-scan coordinates

        As of now: No error handling:
        A rect. shape in a shape Layer will provide e.g.:
        array([[ 299.5774541 , -157.22546457],
       [ 299.5774541 ,  160.6666534 ],
       [ 692.26771747,  160.6666534 ],
       [ 692.26771747, -157.22546457]])
        '''
        mCoordinates = self._widget.getCoordinatesShapeLayerNapari()[0]
        maxPosX = np.max(mCoordinates[:,0])
        minPosX = np.min(mCoordinates[:,0])
        maxPosY = np.max(mCoordinates[:,1])
        minPosY = np.min(mCoordinates[:,1])

        # get number of pixels in X/Y
        mFrame = self.microscopeDetector.getLatestFrame()
        NpixX, NpixY = mFrame.shape[1], mFrame.shape[0]

        # set frame as reference in napari
        isRGB = mFrame.shape[-1]==3 # most likely True!
        name = "Histo Prescan"
        pixelsize = self.microscopeDetector.pixelSizeUm[-1]
        self._widget.setImageNapari(mFrame, colormap="gray", isRGB=isRGB, name=name, pixelsize=(pixelsize,pixelsize), translation=(0,0))

        # starting the snake scan
        # Calculate the size of the area each image covers
        img_width = NpixX * self.microscopeDetector.pixelSizeUm[-1]
        img_height = NpixY * self.microscopeDetector.pixelSizeUm[-1]

        # compute snake scan coordinates
        mOverlap = 0.75
        self.mCamScanCoordinates = self.generate_snake_scan_coordinates(minPosX, minPosY, maxPosX, maxPosY, img_width, img_height, mOverlap)
        nTilesX = int((maxPosX-minPosX)/(img_width*mOverlap))
        nTilesY = int((maxPosY-minPosY)/(img_height*mOverlap))
        self._widget.setCameraScanParameters(nTilesX, nTilesY, minPosX, maxPosX, minPosY, maxPosY)

    @APIExport(runOnUIThread=False)
    def fetchStageMap(self, resizeFactor:float=1, mapID:int=0):
        '''return the image that represents the stage mapping'''

        # Create a 2D NumPy array representing the image
        image = cv2.imread(self.allScanParameters[mapID].imagePath)

        # eventually resize image to save bandwidth
        if resizeFactor <1:
            image = self.resizeImage(image, resizeFactor)

        # using an in-memory image
        im = Image.fromarray(image)

        # save image to an in-memory bytes buffer
        # save image to an in-memory bytes buffer
        with io.BytesIO() as buf:
            im = im.convert('L')  # convert image to 'L' mode
            im.save(buf, format='PNG')
            im_bytes = buf.getvalue()

        headers = {'Content-Disposition': 'inline; filename="test.png"'}
        return Response(im_bytes, headers=headers, media_type='image/png')

    @APIExport(runOnUIThread=False)
    def getSampleLayoutFilePaths(self):
        # return the paths of the sample layouts
        # images are provided via imswitchserver
        _baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_data')
        images_dir =  os.path.join(_baseDataFilesDir, 'images')
        # create list of all image files in folder and subfolders
        image_files = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.json')):
                    image_files.append(os.path.join(root.split("_data/")[-1], file))
        return image_files

    def computeScanCoordinatesWellplate(self, samplelayoutfilepath:str) -> list:
        '''load json configurtion file and compute the scan coordinates'''
        #%% compute the positions of wells in a wellplate
        # read in the json file with the coordinates
        pixelsize_eff = self.microscopeDetector.pixelSizeUm[-1] # um from camera
        overlap = self.currentOverlap # e.g. 25% overlap
        n_pix_x, n_pix_y = self.microscopeDetector._camera.SensorWidth, self.microscopeDetector._camera.SensorHeight
        with open(samplelayoutfilepath) as f:
            data = json.load(f)
        n_pix_x, n_pix_y = 4000,3000
        # iterate over all wells and compute the positions
        well_positions = []
        pixelToMMFactor = data['ScanParameters']['pixelImageY']/data['ScanParameters']['physDimY']
        fovX = n_pix_x*pixelsize_eff*(overlap)/1e3
        fovY = n_pix_y*pixelsize_eff*(overlap)/1e3
        radius = data['ScanParameters']['well_radius'] # mm
        fov_physical_x = pixelsize_eff*n_pix_x*(overlap)/1e3
        fov_physical_y = pixelsize_eff*n_pix_y*(overlap)/1e3
        # compute positions of radius
        n_tiles_x = int(2*radius/fov_physical_x) # number of pixels in the radius
        n_tiles_y = int(2*radius/fov_physical_y) # number of pixels in the radius

        # % create xx/yy meshgrid
        xx,yy = np.meshgrid(fov_physical_x*np.arange(-n_tiles_x//2,n_tiles_x//2)+1,fov_physical_y*np.arange(-n_tiles_y//2,n_tiles_y//2)+1)
        circle = ((xx)**2+(yy)**2) < radius**2
        well_scan_locations = (xx[circle].flatten(),yy[circle].flatten())
        well_positions = []
        DEBUG=1
        for well in data['ScanParameters']['wells']:
            center_x, center_y = well['positionX'], well['positionY']
            well_positions.append((((well_scan_locations[0]+center_x)), ((well_scan_locations[1]+center_y))))
            # for debugging:
            if DEBUG:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')

                plt.plot(well_scan_locations[0]+center_x,well_scan_locations[1]+center_y,'r.')
        if DEBUG:
            plt.savefig("well_positions.png")
            plt.close()
        well_positions = np.array(well_positions)
        well_positions_list = well_positions.tolist()
        return {"shape": list(well_positions.shape), "data": well_positions_list, "units": "um", "pixelToMMFactor": pixelToMMFactor, "fovX": fovX, "fovY": fovY, "pixelImageX":  data['ScanParameters']['pixelImageX'], "pixelImageY":  data['ScanParameters']['pixelImageY']}

    @APIExport(runOnUIThread=False)
    def setActiveSampleLayoutFilePath(self, filePath:str):
        # set the active sample layout file path
        if filePath in self.getSampleLayoutFilePaths():
            _baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_data')
            filePath =  os.path.join(_baseDataFilesDir, filePath)
            # check if file exists
            if os.path.exists(filePath):
                self.activeSampleLayoutFilePath = filePath
                mPositionList = self.computeScanCoordinatesWellplate(filePath)
                # transfer the coordinates to the GUI
                self.sigUpdateScanCoordinatesLayout.emit(mPositionList)
                return {"coordinates":mPositionList}
        return None


    @APIExport(runOnUIThread=False)
    def process_list(self):
        try:

            # List of lists
            data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

            # Encode it into JSON
            json_data = json.dumps(data)
            # Decode JSON-encoded list of lists
            decoded_data = json.loads(data)
            return {"received_data": decoded_data, "status": "success"}
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format"}

    @APIExport(runOnUIThread=False)
    def startStageMapping(self, mumPerStep: int=1, calibFilePath: str = "calibFile.json") -> str:
        self.stageMappingResult = None
        if not self.ishistoscanRunning or not self.ishistoscanRunning:
            self.ishistoscanRunning = True
            pixelSize = self.microscopeDetector.pixelSizeUm[-1] # µm
            # mumPerStep = 1 # µm
            try:
                self.mStageMapper = OFMStageMapping.OFMStageScanClass(self, calibration_file_path=calibFilePath, effPixelsize=pixelSize, stageStepSize=mumPerStep,
                                                                IS_CLIENT=False, mDetector=self.microscopeDetector, mStage=self.stages)
                def launchStageMappingBackground():
                    self.stageMappingResult = self.mStageMapper.calibrate_xy(return_backlash_data=0)
                    if self.stageMappingResult is bool:
                        self._logger.error("Calibration failed")
                        self._widget.sigStageMappingComplete.emit(None, None, False)
                    print(f"Calibration result:")
                    for k, v in self.stageMappingResult.items():
                        print(f"    {k}:")
                        for l, w in v.items():
                            if len(str(w)) < 50:
                                print(f"        {l}: {w}")
                            else:
                                print(f"        {l}: too long to print")
                    image_to_stage_displacement = self.stageMappingResult["camera_stage_mapping_calibration"]["image_to_stage_displacement"]
                    backlash_vector = self.stageMappingResult["camera_stage_mapping_calibration"]["backlash_vector"]
                    self._widget.sigStageMappingComplete.emit(image_to_stage_displacement, backlash_vector, True)
                threading.Thread(target=launchStageMappingBackground).start()
                '''
                The result:
                mData["camera_stage_mapping_calibration"]["image_to_stage_displacement"]=
                array([[ 0.        , -1.00135997],
                    [-1.00135997,  0.        ]])

                mData["camera_stage_mapping_calibration"]["backlash_vector"]=
                array([ 0.,  0.,  0.])

                From Richard:
                """Combine X and Y calibrations

                This uses the output from :func:`.calibrate_backlash_1d`, run at least
                twice with orthogonal (or at least different) `direction` parameters.
                The resulting 2x2 transformation matrix should map from image
                to stage coordinates.  Currently, the backlash estimate given
                by this function is only really trustworthy if you've supplied
                two orthogonal calibrations - that will usually be the case.

                Returns
                -------
                dict
                    A dictionary of the resulting calibration, including:

                    * **image_to_stage_displacement:** (`numpy.ndarray`) - a 2x2 matrix mapping
                    image displacement to stage displacement
                    * **backlash_vector:** (`numpy.ndarray`) - representing the estimated
                    backlash in each direction
                    * **backlash:** (`number`) - the highest element of `backlash_vector`
                """

                '''
            except Exception as e:
                self._logger.error(e)
                self.stageMappingResult = ((0,0),(0,0))
            self.ishistoscanRunning = False
            return self.ishistoscanRunning
        else:
            return "busy"

    def startStageMappingFromButton(self):
        self.positionBeforeStageMapping = self.stages.getPosition()
        self.isStopStageMapping = False
        self._widget.buttonStartCalibration.setEnabled(False)
        self.mStageMappingThread = threading.Thread(target=self.startStageMapping)
        self.mStageMappingThread.start()

    def stopHistoScanFromButton(self):
        self._widget.buttonStartCalibration.setEnabled(True)
        self.isStopStageMapping = True
        try:
            # move back to the position before the stage mapping
            del self.mStageMappingThread
            self.stages.move(value=(self.positionBeforeStageMapping["X"], self.positionBeforeStageMapping["Y"]), axis="XY", is_absolute=True, is_blocking=False, acceleration=(self.acceleration,self.acceleration))
        except Exception as e:
            self._logger.error(e)

    @APIExport()
    def stopStageMapping(self):
        self.mStageMapper.stop()

    def starthistoscanCamerabased(self):
        '''
        start a camera scan
        '''
        if self.mCamScanCoordinates is None:
            return
        self.turnOffLEDArray()
        self.turnOnLED()
        # update GUI elements
        self._widget.startButton3.setEnabled(False)
        self._widget.stopButton3.setEnabled(True)
        self._widget.startButton3.setText("Running")
        self._widget.stopButton3.setText("Stop")
        self._widget.stopButton3.setStyleSheet("background-color: red")
        self._widget.startButton3.setStyleSheet("background-color: green")
        initialPosition = self.stages.getPosition()
        minPosX = np.min(self.mCamScanCoordinates, axis=0)[0]
        maxPosX = np.max(self.mCamScanCoordinates, axis=0)[0]
        minPosY = np.min(self.mCamScanCoordinates, axis=0)[1]
        maxPosY = np.max(self.mCamScanCoordinates, axis=0)[1]

        nTimes = 1
        tPeriod = 0

        self.startStageScanning(minPosX=minPosX, minPosY=minPosY, maxPosX=maxPosX, maxPosY=maxPosY, positionList=self.mCamScanCoordinates, nTimes=nTimes, tPeriod=tPeriod)


    def stophistoscanCamerabased(self):
        '''
        stop a camera scan
        '''
        #self.turnOnLEDArray()
        #self.turnOffLED()

        self.ishistoscanRunning = False
        self._logger.debug("histoscan scanning stopped.")
        if IS_HEADLESS: return
        self._widget.startButton3.setEnabled(True)
        self._widget.stopButton3.setEnabled(False)
        self._widget.startButton3.setText("Start")
        self._widget.stopButton3.setText("Stopped")
        self._widget.stopButton3.setStyleSheet("background-color: green")
        self._widget.startButton3.setStyleSheet("background-color: red")

    def updateAllPositionGUI(self):
        allPositions = self.stages.position
        if not IS_HEADLESS: self._widget.updateBoxPosition(allPositions["X"], allPositions["Y"])

    def goToPosition(self, posX, posY):
        # {"task":"/motor_act",     "motor":     {         "steppers": [             { "stepperid": 1, "position": -1000, "speed": 30000, "isabs": 0, "isaccel":1, "isen":0, "accel":500000}     ]}}
        currentPosition = self.stages.getPosition()
        self.stages.move(value=(posX,posY), axis="XY", is_absolute=True, is_blocking=False, acceleration=(self.acceleration,self.acceleration))
        self._commChannel.sigUpdateMotorPosition.emit()
        newPosition = self.stages.getPosition()
        if currentPosition["X"]==newPosition["X"] and currentPosition["Y"]==newPosition["Y"]:
            self._logger.error("Could not move to position - check if coordinates are within the allowed range or if the stage is homed properly.")

    def setScanCoordinatesLayout(self, mPositionList):
        # set the scan coordinates in the GUI
        self.scanPositionList = mPositionList

    def displayImage(self):
        # a bit weird, but we cannot update outside the main thread
        if IS_HEADLESS: return
        name = self.histoScanStackName
        # subsample stack
        isRGB = self.histoscanStack.shape[-1]==3
        self._widget.setImageNapari(self.histoscanStack, colormap="gray", isRGB=isRGB, name=name, pixelsize=(1,1), translation=(0,0))

    def updatePartialImage(self):
        # a bit weird, but we cannot update outside the main thread
        name = self.histoScanStackName
        # subsample stack
        isRGB = self.histoscanStack.shape[-1]==3
        # coordinates: (x,y,w,h)
        self._widget.updatePartialImageNapari(im=np.uint16(self.partialHistoscanStack ),
                                              coords=self.partialImageCoordinates,
                                              name=name)

    def valueIlluChanged(self):
        illuSource = self._widget.getIlluminationSource()
        illuValue = self._widget.illuminationSlider.value()
        self._master.lasersManager
        if not self._master.lasersManager[illuSource].enabled:
            self._master.lasersManager[illuSource].setEnabled(1)

        illuValue = illuValue/100*self._master.lasersManager[illuSource].valueRangeMax
        self._master.lasersManager[illuSource].setValue(illuValue)

    def calibrateOffset(self):
        # move to a known position and click in the
        # 1. retreive the coordinates on the canvas
        clickedCoordinates = self._widget.ScanSelectViewWidget.clickedCoordinates
        # 2. measure the stage coordinates that relate to the clicked coordintes
        self.stageinitialPosition = self.stages.getPosition()
        # true position:
        initX = self.stageinitialPosition["X"]
        initY = self.stageinitialPosition["Y"]
        initZ = self.stageinitialPosition["Z"]

        # compute the differences
        offsetX =  initX - clickedCoordinates[0]
        offsetY =  initY - clickedCoordinates[1]
        self._logger.debug("Offset coordinates in X/Y"+str(offsetX)+" / "+str(offsetY))

        # now we need to calculate the offset here
        self._master.HistoScanManager.writeConfig({"offsetX":offsetX, "offsetY":offsetY})
        self._widget.ScanSelectViewWidget.setOffset(offsetX,offsetY)

    def startHistoScanCoordinatebased(self):
        '''
        Start an XY scan that was triggered from the Figure-based Scan Tab
        '''
        minPosX = self._widget.getMinPositionX()
        maxPosX = self._widget.getMaxPositionX()
        minPosY = self._widget.getMinPositionY()
        maxPosY = self._widget.getMaxPositionY()
        nTimes = self._widget.getNTimesScan()
        tPeriod = self._widget.getTPeriodScan()
        self._widget.startButton.setEnabled(False)
        self._widget.stopButton.setEnabled(True)
        self._widget.startButton.setText("Running")
        self._widget.stopButton.setText("Stop")
        self._widget.stopButton.setStyleSheet("background-color: red")
        self._widget.startButton.setStyleSheet("background-color: green")
        overlap = 0.75
        # start stage scanning without provision of stage coordinate list
        self.startStageScanning(minPosX, maxPosX, minPosY, maxPosY, overlap, nTimes, tPeriod)

    def starthistoscanTilebased(self):
        numberTilesX, numberTilesY = self._widget.getNumberTiles()
        stepSizeX, stepSizeY = self._widget.getStepSize()
        nTimes = self._widget.getNTimesScan()
        tPeriod = self._widget.getTPeriodScan()
        resizeFactor = self._widget.getResizeFactor()
        self._widget.startButton2.setEnabled(False)
        self._widget.stopButton2.setEnabled(True)
        self._widget.startButton2.setText("Running")
        self._widget.stopButton2.setText("Stop")
        self._widget.stopButton2.setStyleSheet("background-color: red")
        self._widget.startButton2.setStyleSheet("background-color: green")
        initialPosition = self.stages.getPosition()
        initPosX = initialPosition["X"]
        initPosY = initialPosition["Y"]
        # check if we want to stitch the images
        isStitchAshlar = self._widget.stitchAshlarCheckBoxTileBased.isChecked()
        isStitchAshlarFlipX = self._widget.stitchAshlarFlipXCheckBoxTileBased.isChecked()
        isStitchAshlarFlipY = self._widget.stitchAshlarFlipYCheckBoxTileBased.isChecked()

        self.startHistoScanTileBasedByParameters(numberTilesX, numberTilesY, stepSizeX, stepSizeY, nTimes, tPeriod, initPosX, initPosY,
                                                 isStitchAshlar, isStitchAshlarFlipX, isStitchAshlarFlipY, resizeFactor=resizeFactor)


    @APIExport()
    def stopHistoScan(self):
        self.ishistoscanRunning = False
        if IS_HEADLESS:
            self._widget.startButton.setEnabled(True)
            self._widget.stopButton.setEnabled(False)
            self._widget.startButton.setText("Start")
            self._widget.stopButton.setText("Stopped")
            self._widget.stopButton.setStyleSheet("background-color: green")
            self._widget.startButton.setStyleSheet("background-color: red")
            self._logger.debug("histoscan scanning stopped.")



    @APIExport()
    def startHistoScanTileBasedByParameters(self, numberTilesX:int=2, numberTilesY:int=2, stepSizeX:int=100, stepSizeY:int=100,
                                            nTimes:int=1, tPeriod:int=1, initPosX:Optional[Union[int, str]] = None, initPosY:Optional[Union[int, str]] = None,
                                            isStitchAshlar:bool=False, isStitchAshlarFlipX:bool=False, isStitchAshlarFlipY:bool=False, resizeFactor:float=0.25,
                                            overlap:float=0.75):
        def computePositionList(numberTilesX, numberTilesY, stepSizeX, stepSizeY, initPosX, initPosY):
            positionList = []
            for ix in range(numberTilesX):
                if ix % 2 == 0:  # X-Position ist gerade
                    rangeY = range(numberTilesY)
                else:  # X-Position ist ungerade
                    rangeY = range(numberTilesY - 1, -1, -1)
                for iy in rangeY:
                    positionList.append((ix*stepSizeX+initPosX-numberTilesX//2*stepSizeX, iy*stepSizeY+initPosY-numberTilesY//2*stepSizeY, ix, iy))
            return positionList
        # compute optimal step size if not provided
        if stepSizeX<=0 or stepSizeX is None:
            stepSizeX, _ = self.computeOptimalScanStepSize()
        if stepSizeY<=0 or stepSizeY is None:
            _, stepSizeY = self.computeOptimalScanStepSize()
        if initPosX is None or type(initPosX)==str :
            initPosX = self.stages.getPosition()["X"]
        if initPosY is None or type(initPosY)==str:
            initPosY = self.stages.getPosition()["Y"]

        # assign parameters for status feedback
        self.currentOverlap = overlap
        self.currentNX = numberTilesX
        self.currentNY = numberTilesY
        self.currentStepSizeX = stepSizeX
        self.currentStepSizeY = stepSizeY
        self.currentNtimes = nTimes
        self.currentIinitialPosX = initPosX
        self.currentIinitialPosY = initPosY
        self.currentTimeInterval = tPeriod
        self.currentAshlarStitching = isStitchAshlar
        self.currentAshlarFlipX = isStitchAshlarFlipX
        self.currentAshlarFlipY = isStitchAshlarFlipY
        self.currentResizeFactor = resizeFactor

        # compute the scan-grid
        positionList = computePositionList(numberTilesX, numberTilesY, stepSizeX, stepSizeY, initPosX, initPosY)
        minPosX = np.min(positionList, axis=0)[0]
        maxPosX = np.max(positionList, axis=0)[0]
        minPosY = np.min(positionList, axis=0)[1]
        maxPosY = np.max(positionList, axis=0)[1]

        # start stage scanning with positionlist
        #def startHistoScanTileBasedByParameters(self, numberTilesX, numberTilesY, stepSizeX, stepSizeY, initPosX, initPosY, nTimes=1, tPeriod=1):
        if self.IS_HTTP:
            self.startStageScanning = self.remoteHistoManager.startStageScanning
        self.startStageScanning(minPosX=minPosX, minPosY=minPosY, maxPosX=maxPosX, maxPosY=maxPosY, positionList=positionList, nTimes=nTimes, tPeriod=tPeriod,
                                isStitchAshlar=isStitchAshlar, isStitchAshlarFlipX=isStitchAshlarFlipX, isStitchAshlarFlipY=isStitchAshlarFlipY,
                                resizeFactor=resizeFactor)

    def stophistoscanMain(self):
        if IS_HEADLESS: return
        self._widget.startButton.setEnabled(True)
        self._widget.stopButton.setEnabled(False)
        self._widget.startButton.setText("Start")
        self._widget.stopButton.setText("Stopped")
        self._widget.stopButton.setStyleSheet("background-color: green")
        self._widget.startButton.setStyleSheet("background-color: red")
        self._logger.debug("histoscan scanning stopped.")

    def stophistoscanTilebased(self):
        if IS_HEADLESS: return
        self.ishistoscanRunning = False
        self._widget.startButton2.setEnabled(True)
        self._widget.stopButton2.setEnabled(False)
        self._widget.startButton2.setText("Start")
        self._widget.stopButton2.setText("Stopped")
        self._widget.stopButton2.setStyleSheet("background-color: green")
        self._widget.startButton2.setStyleSheet("background-color: red")
        self._logger.debug("histoscan scanning stopped.")

    @APIExport()
    def startStageScanningWellplatePositionlistbased(self, wells:str='1,2'):
        '''
        first compute the scan positions based on the wellplate layout and then start the scan
        This has to be done by selecting the json file that holds the parameters and coordinates
        '''
        if self.scanPositionList is None or len(self.scanPositionList)==0:
            return
        if wells == []:
            return
        if wells.find(",")>0:
            wells = wells.split(",")
            # convert wells to indices
            wells = [int(w)-1 for w in wells]
        else:
            wells = int(wells)

        # return the lists based on the selected wells
        positionList = []
        for well in wells:
            positionList.append(self.scanPositionList['data'][well])

        # merge the lists along the 0th axis
        positionList = np.hstack(positionList)

        self.startStageScanningPositionlistbased(positionList = self.scanPositionList['data'][well], nTimes=1, tPeriod=0)

    @APIExport()
    def startStageScanningPositionlistbased(self, positionList: Union[str, List], nTimes: int = 1, tPeriod: int = 0):
        '''
        Start a stage scanning based on a list of positions
        positionList: list of tuples with X/Y positions (e.g. "[(10, 10, 100), (100, 100, 100)]")
        nTimes: number of times to repeat the scan
        tPeriod: time between scans
        '''
        if type(positionList)==str:
            positionList = np.array(ast.literal_eval(positionList))
        maxPosX = np.max(positionList[:,0])
        minPosX = np.min(positionList[:,0])
        maxPosY = np.max(positionList[:,1])
        minPosY = np.min(positionList[:,1])
        return self.startStageScanning(minPosX=minPosX, maxPosX=maxPosX, minPosY=minPosY, maxPosY=maxPosY, overlap=None,
                                nTimes=nTimes, tPeriod=tPeriod, positionList=positionList)

    @APIExport()
    def startStageScanning(self, minPosX:float=None, maxPosX:float=None, minPosY:float=None, maxPosY:float=None,
                           overlap:float=None, nTimes:int=1, tPeriod:int=0, positionList: Optional[Union[list, str]] = None,
                           isStitchAshlar:bool=False, isStitchAshlarFlipX:bool=False, isStitchAshlarFlipY:bool=False,
                           resizeFactor=0.25):
        '''
        Start A stage scanning based on the provided parameters (position list will be generated based on the min/max positions)
        minPosX: minimum X position
        maxPosX: maximum X position
        minPosY: minimum Y position
        maxPosY: maximum Y position
        overlap: overlap between the images
        nTimes: number of times to repeat the scan
        tPeriod: time between scans
        positionList: list of tuples with X/Y positions - if provided, the min/max positions are ignored
        isStitchAshlar: stitch the images using ashlar, default is numpy based stitching (tiling)
        isStitchAshlarFlipX: flip the images in X direction
        isStitchAshlarFlipY: flip the images in Y direction
        resizeFactor: resize the images

        '''
        if not self.ishistoscanRunning:
            self.ishistoscanRunning = True
            if self.histoscanTask is not None:
                self.histoscanTask.join()
                del self.histoscanTask
            # Launch the XY scan in the background
            self.histoscanTask = threading.Thread(target=self.histoscanThread, args=(minPosX, maxPosX, minPosY,
                                                                                     maxPosY, overlap, nTimes, tPeriod, positionList,
                                                                                     isStitchAshlarFlipX, isStitchAshlarFlipY, 0.05,
                                                                                     isStitchAshlar, resizeFactor))
            self.histoscanTask.start()

    def generate_snake_scan_coordinates(self, posXmin, posYmin, posXmax, posYmax, img_width, img_height, overlap):
        # Calculate the number of steps in x and y directions
        steps_x = int((posXmax - posXmin) / (img_width*overlap))
        steps_y = int((posYmax - posYmin) / (img_height*overlap))

        coordinates = []

        # Loop over the positions in a snake pattern
        for y in range(steps_y):
            if y % 2 == 0:  # Even rows: left to right
                for x in range(steps_x):
                    coordinates.append((posXmin + x * img_width *overlap, posYmin + y * img_height *overlap), x, y)
            else:  # Odd rows: right to left
                for x in range(steps_x - 1, -1, -1):  # Starting from the last position, moving backwards
                    coordinates.append((posXmin + x * img_width *overlap, posYmin + y * img_height *overlap), x, y)

        return coordinates
    '''
    @APIExport()
    def getHistoStatus(self) -> dict:
        statusDict = {}
        statusDict["currentPosition"] = self.currentPosition
        statusDict["ishistoscanRunning"] = bool(self.ishistoscanRunning)
        statusDict["stitchResultAvailable"] = bool(self.histoscanStack is not None)
        statusDict["mScanIndex"] = self.mScanIndex
        statusDict["mScanCount"] = len(self.positionList)
        statusDict["currentStepSizeX"] = self.currentStepSizeX if self.currentStepSizeX is not None else self.bestScanSizeX
        statusDict["currentStepSizeY"] = self.currentStepSizeY if self.currentStepSizeY is not None else self.bestScanSizeY
        statusDict["currentNX"] = self.currentNX
        statusDict["currentNY"] = self.currentNY
        statusDict["currentOverlap"] = self.currentOverlap
        statusDict["currentAshlarStitching"] = self.currentAshlarStitching
        statusDict["currentAshlarFlipX"] = self.currentAshlarFlipX
        statusDict["currentAshlarFlipY"] = self.currentAshlarFlipY
        statusDict["currentResizeFactor"] = self.currentResizeFactor
        mCurrentPositions = self.stages.getPosition()
        statusDict["currentIinitialPosX"] = mCurrentPositions["X"]
        statusDict["currentIinitialPosY"] = mCurrentPositions["Y"]
        statusDict["currentTimeInterval"] = self.currentTimeInterval
        statusDict["currentNtimes"] = self.currentNtimes
        #statusDict["currentIlluSource"] = self.currentIlluSource
        #statusDict["currentIlluValue"] = self.currentIlluValue
        statusDict["pixelSize"] = self.microscopeDetector.pixelSizeUm[-1]

        #statusDict["positionList"] = self.positionList
        return statusDict

    @APIExport()
    def setHistoStatus(self, statusDict:dict) -> bool:
        if "currentPosition" in statusDict:
            self.currentPosition = statusDict["currentPosition"]
        if "ishistoscanRunning" in statusDict:
            self.ishistoscanRunning = statusDict["ishistoscanRunning"]
        if "stitchResultAvailable" in statusDict:
            self.stitchResultAvailable = statusDict["stitchResultAvailable"]
        if "mScanIndex" in statusDict:
            self.mScanIndex = statusDict["mScanIndex"]
        if "mScanCount" in statusDict:
            self.mScanCount = statusDict["mScanCount"]
        if "currentStepSizeX" in statusDict:
            self.currentStepSizeX = statusDict["currentStepSizeX"]
        if "currentStepSizeY" in statusDict:
            self.currentStepSizeY = statusDict["currentStepSizeY"]
        if "currentNX" in statusDict:
            self.currentNX = statusDict["currentNX"]
        if "currentNY" in statusDict:
            self.currentNY = statusDict["currentNY"]
        if "currentOverlap" in statusDict:
            self.currentOverlap = statusDict["currentOverlap"]
        if "currentAshlarStitching" in statusDict:
            self.currentAshlarStitching = statusDict["currentAshlarStitching"]
        if "currentAshlarFlipX" in statusDict:
            self.currentAshlarFlipX = statusDict["currentAshlarFlipX"]
        if "currentAshlarFlipY" in statusDict:
            self.currentAshlarFlipY = statusDict["currentAshlarFlipY"]
        if "currentResizeFactor" in statusDict:
            self.currentResizeFactor = statusDict["currentResizeFactor"]
        if "currentIinitialPosX" in statusDict:
            self.currentIinitialPosX = statusDict["currentIinitialPosX"]
        if "currentIinitialPosY" in statusDict:
            self.currentIinitialPosY = statusDict["currentIinitialPosY"]
        if "currentTimeInterval" in statusDict:
            self.currentTimeInterval = statusDict["currentTimeInterval"]
        if "currentNtimes" in statusDict:
            self.currentNtimes = statusDict["currentNtimes"]
        #if "currentIlluSource" in statusDict:
        #    self.currentIlluSource = statusDict["currentIlluSource"]

    '''

    @APIExport()
    def getHistoStatus(self) -> HistoStatus:
        # ensure currentPosition is a tuple of 4 elements
        if not isinstance(self.currentPosition, tuple) or len(self.currentPosition)!=4:
            self.currentPosition = (self.currentPosition[0], self.currentPosition[1],0,0,0) #
        statusDict = {
            "currentPosition": self.currentPosition,
            "ishistoscanRunning": bool(self.ishistoscanRunning),
            "stitchResultAvailable": bool(self.histoscanStack is not None),
            "mScanIndex": self.mScanIndex,
            "mScanCount": len(self.positionList),
            "currentStepSizeX": self.currentStepSizeX or self.bestScanSizeX,
            "currentStepSizeY": self.currentStepSizeY or self.bestScanSizeY,
            "currentNX": self.currentNX,
            "currentNY": self.currentNY,
            "currentOverlap": self.currentOverlap,
            "currentAshlarStitching": self.currentAshlarStitching,
            "currentAshlarFlipX": self.currentAshlarFlipX,
            "currentAshlarFlipY": self.currentAshlarFlipY,
            "currentResizeFactor": self.currentResizeFactor,
            "currentIinitialPosX": self.currentIinitialPosX,
            "currentIinitialPosY": self.currentIinitialPosY,
            "currentTimeInterval": self.currentTimeInterval,
            "currentNtimes": self.currentNtimes,
            "pixelSize": self.microscopeDetector.pixelSizeUm[-1],
        }
        return HistoStatus.from_dict(statusDict)

    @APIExport()
    def setHistoStatus(self, status: HistoStatus) -> bool:
        self.currentPosition = status.currentPosition
        self.ishistoscanRunning = status.ishistoscanRunning
        self.stitchResultAvailable = status.stitchResultAvailable
        self.mScanIndex = status.mScanIndex
        self.currentStepSizeX = status.currentStepSizeX
        self.currentStepSizeY = status.currentStepSizeY
        self.currentNX = status.currentNX
        self.currentNY = status.currentNY
        self.currentOverlap = status.currentOverlap
        self.currentAshlarStitching = status.currentAshlarStitching
        self.currentAshlarFlipX = status.currentAshlarFlipX
        self.currentAshlarFlipY = status.currentAshlarFlipY
        self.currentResizeFactor = status.currentResizeFactor
        self.currentIinitialPosX = status.currentIinitialPosX
        self.currentIinitialPosY = status.currentIinitialPosY
        self.currentTimeInterval = status.currentTimeInterval
        self.currentNtimes = status.currentNtimes
        return True

    @APIExport()
    def getLastStitchedRawList(self) -> StitchedImageResponse:
        histoscanStack = self.histoscanStack.copy()
        self._logger.debug(f"Shape of histoscanStack: {histoscanStack.shape}")
        if histoscanStack is not None and len(histoscanStack.shape)>1:
            # if size of image exceeds 1000x1000, we need to resize it
            if histoscanStack.shape[0]>1000 or histoscanStack.shape[1]>1000:
                histoscanStack = cv2.resize(histoscanStack, (0,0), fx=0.25, fy=0.25)

            #if not len(histoscanStack.shape)==3:
            #    histoscanStack = np.repeat(histoscanStack[:,:,np.newaxis], 3, axis=2)

            _, buffer = cv2.imencode('.png', histoscanStack)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return StitchedImageResponse(imageList=histoscanStack.tolist(), image=image_base64)
        else:
            raise HTTPException(status_code=404, detail="No image found")

    @APIExport()
    def getLastStitchedImage(self) -> Response:
        histoscanStack = self.histoscanStack.copy()
        if histoscanStack is not None and len(histoscanStack.shape)>1:
            #_, buffer = cv2.imencode('.png', histoscanStack)
            #image_base64 = base64.b64encode(buffer).decode('utf-8')
            #return image_base64

            # using an in-memory image
            im = Image.fromarray(histoscanStack)

            # save image to an in-memory bytes buffer
            # save image to an in-memory bytes buffer
            with io.BytesIO() as buf:
                im = im.convert('L')  # convert image to 'L' mode
                im.save(buf, format='PNG')
                im_bytes = buf.getvalue()

            headers = {'Content-Disposition': 'inline; filename="histo.png"'}
            return Response(im_bytes, headers=headers, media_type='image/png')


        else:
            raise HTTPException(status_code=404, detail="No image found")


    def histoscanThread(self, minPosX, maxPosX, minPosY, maxPosY, overlap=0.75, nTimes=1,
                        tPeriod=0, positionList=None,
                        flipX=False, flipY=False, tSettle=0.05,
                        isStitchAshlar=False, resizeFactor=0.25):
        self._logger.debug("histoscan thread started.")

        initialPosition = self.stages.getPosition()
        initPosX = initialPosition["X"]
        initPosY = initialPosition["Y"]
        if not self.microscopeDetector._running: self.microscopeDetector.startAcquisition()

        # now start acquiring images and move the stage in Background
        mFrame = self.microscopeDetector.getLatestFrame()
        NpixX, NpixY = mFrame.shape[1], mFrame.shape[0]

        # starting the snake scan
        # Calculate the size of the area each image covers
        img_width = NpixX * self.microscopeDetector.pixelSizeUm[-1]
        img_height = NpixY * self.microscopeDetector.pixelSizeUm[-1]
        image_dims = (img_width, img_height)
        # precompute the position list in advance
        if positionList is None:
            positionList = self.generate_snake_scan_coordinates(minPosX, minPosY, maxPosX, maxPosY, img_width, img_height, overlap)
            nStepsX = int((maxPosX - minPosX) / (img_width*overlap))
            nStepsY = int((maxPosY - maxPosY) / (img_height*overlap))
        else:
            nStepsX = np.ptp(np.array(positionList)[:, 2])+1
            nStepsY = np.ptp(np.array(positionList)[:, 3])+1

        maxPosPixY = int((maxPosY-minPosY)/self.microscopeDetector.pixelSizeUm[-1])
        maxPosPixX = int((maxPosX-minPosX)/self.microscopeDetector.pixelSizeUm[-1])

        # are we RGB or monochrome?
        if len(mFrame.shape)==2:
            nChannels = 1
        else:
            nChannels = mFrame.shape[-1]

        # perform timelapse imaging
        for i in range(nTimes):
            tz = datetime.timezone.utc
            ft = "%Y-%m-%dT%H_%M_%S"
            HistoDate = datetime.datetime.now(tz=tz).strftime(ft)
            file_name = "test_"+HistoDate
            if isZARR:
                extension = ".zarr"
            else:
                extension = ".ome.tif"

            if IS_HEADLESS:
                filePath = self.getSaveFilePath(date=HistoDate,
                                        filename=file_name,
                                        extension=extension)
                folder = os.path.dirname(filePath)
            else: folder = self._widget.getDefaulSavePath()
            t0 = time.time()

            # create a new image stitcher
            if self.flatfieldManager is not None:
                flatfieldImage = self.flatfieldManager.getFlatfieldImage()
            else:
                flatfieldImage = None
            stitcher = ImageStitcher(self, origin_coords=(0,0), max_coords=(maxPosPixX, maxPosPixY), folder=folder, image_dims=image_dims,
                                     nChannels=nChannels, file_name=file_name, extension=extension, flatfieldImage=flatfieldImage,
                                     flipX=flipX, flipY=flipY, isStitchAshlar=isStitchAshlar, pixel_size=self.microscopeDetector.pixelSizeUm[1],
                                     resolution_scale=resizeFactor, tile_shape=(1, nStepsX, nStepsY, NpixY, NpixX))

            # move to the first position
            self.stages.move(value=positionList[0], axis="XY", is_absolute=True, is_blocking=True, acceleration=(self.acceleration,self.acceleration))
            # move to all coordinates and take an image

            # we try an alternative way to move the stage and take images:
            # We move the stage in the background from min to max X and take
            # images in the foreground everytime the stage is in the region where there is a frame due
            if 0:
                self.stages.move(value=(minPosX, minPosY), axis="XY", is_absolute=True, is_blocking=True)

                # now we need to move to max X and take images in the foreground everytime the stage is in the region where there is a frame due
                self.stages.move(value=maxPosX, axis="X", is_absolute=True, is_blocking=False)
                stepSizeX = positionList[1][0]-positionList[0][0]
                lastStagePositionX = self.stages.getPosition()["X"]
                running=1
                while running:
                    self.currentPosX = self.stages.getPosition()["X"]
                    if self.currentPosX-lastStagePositionX > stepSizeX:
                        print("Taking image")
                        mFrame = self.microscopeDetector.getLatestFrame()
                        import tifffile as tif
                        tif.imwrite("test.tif", mFrame, append=True)

                        lastStagePositionX = self.currentPosX

            # Scan over all positions in XY
            for mIndex, iPos in enumerate(positionList):
                # update the loading bar
                self.currentPosition = iPos # use for status updates in the GUI
                self.positionList = positionList
                self.mScanIndex = mIndex
                self.sigUpdateLoadingBar.emit(self.mScanIndex, len(self.positionList))

                try:
                    if not self.ishistoscanRunning:
                        break
                    self.stages.move(value=self.currentPosition, axis="XY", is_absolute=True, is_blocking=True, acceleration=(self.acceleration,self.acceleration))
                    time.sleep(self.tSettle)

                    # always mmake sure we get a frame that is not the same as the one with illumination off eventually
                    timeoutFrameRequest = 1 # seconds # TODO: Make dependent on exposure time
                    cTime = time.time()
                    frameSync=3
                    lastFrameNumber=-1
                    while(1):
                        # get frame and frame number to get one that is newer than the one with illumination off eventually
                        mFrame, currentFrameNumber = self.microscopeDetector.getLatestFrame(returnFrameNumber=True)
                        if lastFrameNumber==-1:
                            # first round
                            lastFrameNumber = currentFrameNumber
                        if time.time()-cTime> timeoutFrameRequest:
                            # in case exposure time is too long we need break at one point
                            break
                        if currentFrameNumber <= lastFrameNumber+frameSync:
                            time.sleep(0.01) # off-load CPU
                        else:
                            break

                    def addImage(mFrame, positionList):
                        metadata = {'Pixels': {
                            'PhysicalSizeX': self.microscopeDetector.pixelSizeUm[-1],
                            'PhysicalSizeXUnit': 'µm',
                            'PhysicalSizeY': self.microscopeDetector.pixelSizeUm[-1],
                            'PhysicalSizeYUnit': 'µm'},

                            'Plane': {
                                'PositionX': positionList[0],
                                'PositionY': positionList[1],
                                'IndexX': positionList[2],
                                'IndexY': positionList[3]
                        }, }
                        self._commChannel.sigUpdateMotorPosition.emit(list(positionList))
                        posY_pix_value = (float(positionList[1])-minPosY)/self.microscopeDetector.pixelSizeUm[-1]
                        posX_pix_value = (float(positionList[0])-minPosX)/self.microscopeDetector.pixelSizeUm[-1]
                        iPosPix = (posX_pix_value, posY_pix_value)
                        stitcher.add_image(np.copy(mFrame), np.copy(iPosPix), metadata.copy())
                    threading.Thread(target=addImage, args=(mFrame,iPos)).start()

                except Exception as e:
                    self._logger.error(e)

            # wait until we go for the next timelapse
            while 1:
                if time.time()-t0 > tPeriod:
                    break
                if not self.ishistoscanRunning:
                    return
                time.sleep(.1)
        # return to initial position
        self.stages.move(value=(initPosX,initPosY), axis="XY", is_absolute=True, is_blocking=False, acceleration=(self.acceleration,self.acceleration))
        self._commChannel.sigUpdateMotorPosition.emit()

        # move back to initial position
        self.stophistoscan()
        if isStitchAshlar and IS_ASHLAR_AVAILABLE:
            mTileList, mPositionList = stitcher.get_tile_list()
            self._commChannel.sigOnResultTileBasedTileScanning(mTileList, np.array(mPositionList))

        # get stitched result
        def getStitchedResult():
            # display and save result
            mDate = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            largeImage = stitcher.get_stitched_image()
            dirPath  = os.path.join(dirtools.UserFileDirs.Root, 'recordings', mDate)
            os.makedirs(dirPath, exist_ok=True)
            tifffile.imwrite(os.path.join(dirPath, "stitchedImage.tif"), largeImage, append=False)
            self.setImageForDisplay(largeImage, "histoscanStitch"+mDate)
        threading.Thread(target=getStitchedResult).start()

    def getSaveFilePath(self, date, filename, extension):
        mFilename =  f"{date}_{filename}.{extension}"
        dirPath  = os.path.join(dirtools.UserFileDirs.Data, 'recordings', date)
        newPath = os.path.join(dirPath,mFilename)

        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        return newPath

    def valueIlluChanged(self):
        illuSource = self._widget.getIlluminationSource()
        illuValue = self._widget.illuminationSlider.value()
        self._master.lasersManager
        if not self._master.lasersManager[illuSource].enabled:
            self._master.lasersManager[illuSource].setEnabled(1)

        illuValue = illuValue/100*self._master.lasersManager[illuSource].valueRangeMax
        self._master.lasersManager[illuSource].setValue(illuValue)

    def setImageForDisplay(self, image, name):
        self.histoScanStackName = name
        self.histoscanStack = image
        self.sigImageReceived.emit()

    def setPartialImageForDisplay(self, image, coordinates, name):
        # coordinates: (x,y,w,h)
        self.partialImageCoordinates = coordinates
        self.partialHistoscanStack = image
        self.histoscanStack = image
        self.sigUpdatePartialImage.emit()

    def stophistoscan(self):
        # update GUI elements
        self.ishistoscanRunning = False
        self._logger.debug("histoscan scanning stopped.")

        # other tabs
        self.stophistoscanMain()
        self.stophistoscanTilebased()
        self.stophistoscanCamerabased()


    # Assuming a tile size of 256 pixels and a known pixel size in microns:
    # pixel_size_microns = <microns per pixel at zoom level 0>
    # stage_width_microns, stage_height_microns = total size of the sample in microns
    # max_zoom_levels = number of zoom levels you want (e.g., 6, 7, etc.)

    def compute_tile_coordinates(self, stage_extent, tile_size, zoom_level, x_tile, y_tile):
        # Stage extent
        min_x, min_y, max_x, max_y = stage_extent

        # Resolution at zoom level 0
        base_resolution = (max_x - min_x) / tile_size

        # Resolution at the current zoom level
        resolution = base_resolution / (2 ** zoom_level)

        # Physical dimensions of the tile
        tile_width = tile_size * resolution
        tile_height = tile_size * resolution

        # Calculate physical coordinates of the tile
        x_min = min_x + x_tile * tile_width
        y_min = min_y + y_tile * tile_height
        x_max = x_min + tile_width
        y_max = y_min + tile_height

        # Calculate tile center
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        return {
            "tile_extent": [x_min, y_min, x_max, y_max],
            "tile_size": [tile_width, tile_height],
            "tile_center": [x_center, y_center],
            "resolution": resolution,
            "xy_tile": [x_tile, y_tile],
            "zoom_level": zoom_level,
        }

    # For a request like: GET /microscope/tiles/{z}/{x}/{y}.png
    # Convert (z, x, y) to an image tile from the mosaic or request a stage move for capturing.
    @APIExport()
    def get_tile(self, z:float, x:float, y:float):
        """
        1) Determine lens from zoom level z
        2) Switch lens (set pixel_size, do stage's internal configuration)
        3) Calculate physical stage area from (x,y,z)
        4) Move stage to cover that region (coarse or fine)
        5) Acquire an image from the camera
        6) Crop/scale image to 256x256 tile
        7) Return PNG
        """
        LENS_CONFIG = {
            '4x':  (1.6, (2048, 2048)),
            '10x': (0.64, (2048, 2048)),
            '20x': (0.32, (2048, 2048)),
        }
        # The range of stage motion in microns
        STAGE_LIMIT_X = 1024  # 0 <= X <= 100000 mm
        STAGE_LIMIT_Y = 1024  # 0 <= Y <= 100000 mm
        stage_extent = (0, 0, STAGE_LIMIT_X, STAGE_LIMIT_Y)
        tile_size = 256  # 256x256 pixel tiles
        zoom_level = int(z)
        x_tile = int(x)
        y_tile = int(y)
        mPixelSizeEff = self.microscopeDetector.pixelSizeUm[-1]
        mPixelX, mPixelY = 400,300
        x_tile_pos = x_tile*mPixelSizeEff*mPixelX
        y_tile_pos = y_tile*mPixelSizeEff*mPixelY
        #mCoordinates = self.compute_tile_coordinates(stage_extent, tile_size, zoom_level, x_tile, y_tile)
        #print(mCoordinates)

        # Move the stage (absolute movement)
        print("PosX: ", x_tile_pos+tile_size//2, "PosY: ", y_tile_pos+tile_size//2)
        self.stages.move(value=(x_tile_pos+tile_size//2, y_tile_pos+tile_size//2), axis="XY", is_absolute=True, is_blocking=True)
        mFrame = self.microscopeDetector.getLatestFrame() # NxM array

        if len(mFrame.shape)==2:
            mFrame = np.repeat(mFrame[:,:,np.newaxis], 3, axis=2)

        # crop center region of the image
        #mFrame = mFrame[int(mPixelY/2-tile_size/2):int(mPixelY/2+tile_size/2), int(mPixelX/2-tile_size/2):int(mPixelX/2+tile_size/2)]
        # flip in X
        if self.flipX: mFrame = np.flip(mFrame, axis=1)
        # flip in Y
        if self.flipY: mFrame = np.flip(mFrame, axis=0)

        # Convert to PNG
        try:
            img = Image.fromarray(np.uint8(mFrame), 'RGB')
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            return StreamingResponse(buffer , media_type="image/png")
        except Exception as e:
            return {"error": str(e)}





class ImageStitcher:

    def __init__(self, parent, origin_coords, max_coords,  folder, file_name, extension,
                 resolution_scale=.25, nChannels = 3, flatfieldImage=None, image_dims=None,
                 flipX=True, flipY=True, isStitchAshlar=False, pixel_size = -1,
                 tile_shape=(1,1), dtype=np.uint16):
        # Initial min and max coordinates
        self._parent = parent
        self.isStitchAshlar = isStitchAshlar
        self.flipX = flipX
        self.flipY = flipY
        self.pixel_size = pixel_size
        self.tile_shape = tile_shape[-2:] # tile_shape -> (channels, rows, columns, height, width)
        self.grid_shape = tile_shape[1:3] # (rows, columns)
        self.dtype = dtype

        # determine write location
        self.file_name = file_name
        self.file_path = os.sep.join([folder, file_name + extension])

        # Queue to hold incoming images
        self.queue = deque()

        # Thread lock for thread safety
        self.lock = threading.Lock()

        # Start a background thread for processing the queue
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.isRunning = True
        self.processing_thread.start()

        # differentiate between ASHLAR and simple memory based stitching
        if self.isStitchAshlar and IS_ASHLAR_AVAILABLE:
            self.ashlarImageList = []
            self.ashlarPositionList = []
        else:
            self.resolution_scale = resolution_scale                        # how much we want to downscale the result to save memory?
            self.origin_coords = np.int32(np.array(origin_coords))          # origin coordinate of the stage (e.g. x=0, y=0)
            image_width, image_height = image_dims[0], image_dims[1]        # physical size of the image in microns
            size = np.array((max_coords[1]+image_height,max_coords[0]+image_width))/pixel_size # size of the final image that contains all tiles in microns
            mshape = np.int32(np.ceil(size)*self.resolution_scale*pixel_size)          # size of the final image in pixels (i.e. canvas)
            self.stitched_image = np.zeros(mshape.T, dtype=np.uint16)       # create a canvas for the stitched image


    def process_ashlar(self, arrays, position_list, pixel_size, output_filename='ashlar_output_numpy.tif', maximum_shift_microns=10, flip_x=False, flip_y=False):
        '''
        install from here: https://github.com/openUC2/ashlar
        arrays => 4d numpy array (n_images, n_channels, height, width)
        position_list => list of tuples with (x,y) positions
        pixel_size => pixel size in microns
        '''
        print("Stitching tiles with ashlar..")
        # Process numpy arrays
        if len(arrays[0].shape)>4:
            print("We can do Monochrome for now only")
            arrays = [np.mean(arrays[0], axis=-1)]
        process_images(filepaths=arrays,
                        output=output_filename,
                        align_channel=0,
                        flip_x=flip_x,
                        flip_y=not flip_y,#make it compatible with the non-ashlar version
                        flip_mosaic_y=False,
                        flip_mosaic_x=False,
                        output_channels=None,
                        maximum_shift=maximum_shift_microns,
                        stitch_alpha=0.01,
                        maximum_error=None,
                        filter_sigma=0,
                        filename_format='cycle_{cycle}_channel_{channel}.tif',
                        pyramid=False,
                        tile_size=1024,
                        ffp=None,
                        dfp=None,
                        barrel_correction=0,
                        plates=False,
                        quiet=False,
                        position_list=position_list,
                        pixel_size=pixel_size)

    def add_image(self, img, coords, metadata):
        '''
        Add an image to the queue for processing
        img - 2/3D numpy array (grayscale or RGB)
        coords - tuple of (x, y) stage coordinates in pixels
        metadata - dictionary of metadata
        '''
        with self.lock:
            self.queue.append((img, coords, metadata))

    def _process_queue(self):
        #https://forum.image.sc/t/python-tifffile-ome-full-metadata-support/56526/11?u=beniroquai
        with tifffile.TiffWriter(self.file_path, bigtiff=True, append=True) as tif:
            while self.isRunning:
                if not self.queue:
                    time.sleep(.02) # unload CPU
                    continue
                img, coords, metadata = self.queue.popleft()
                
                # flip image if needed
                if self.flipX:
                    img = np.fliplr(img)
                if self.flipY:
                    img = np.flipud(img)
                #
                
                self._place_on_canvas(img, coords)
                # write image to disk
                #metadata e.g. {"Pixels": {"PhysicalSizeX": 0.2, "PhysicalSizeXUnit": "\\u00b5m", "PhysicalSizeY": 0.2, "PhysicalSizeYUnit": "\\u00b5m"}, "Plane": {"PositionX": -100, "PositionY": -100, "IndexX": 0, "IndexY": 0}}
                tif.write(data=img, metadata=metadata)
                
    def _place_on_canvas(self, img, coords):
        if self.isStitchAshlar and IS_ASHLAR_AVAILABLE:
            # in case we want to process it with ASHLAR later on
            self.ashlarImageList.append(img)
            self.ashlarPositionList.append(coords)
        else:
            # subsample the image
            coords = np.flip(coords) # YX
            img = skimage.transform.rescale(img, self.resolution_scale, anti_aliasing=False)
            if len(img.shape)==3: img = np.mean(img, axis=-1)  # RGB
            if img.dtype != np.float64 or img.dtype != np.float32:
                img = np.abs(img)*(img<1)
            img = skimage.img_as_uint(img)
            # Round position so paste will skip the expensive subpixel shift.
            pos = np.round((coords-self.origin_coords)*self.resolution_scale)
            utils.paste(self.stitched_image, img, pos, np.maximum)

    def get_tile_list(self):
        '''
        return the list of unstitched images and their positions
        '''
        return self.ashlarImageList, self.ashlarPositionList

    def get_stitched_image(self):
        # introduce a little delay to free the queue ? # TODO
        time.sleep(0.5)
        if self.isStitchAshlar and IS_ASHLAR_AVAILABLE:
            # convert the image and positionlist
            arrays = [np.expand_dims(np.array(self.ashlarImageList),1)]  # (num_images, num_channels, height, width)
            position_list = np.array(self.ashlarPositionList)
            self.process_ashlar(arrays, position_list, self.pixel_size, output_filename=self.file_path, maximum_shift_microns=100, flip_x=self.flipX, flip_y=self.flipY)
            # reload the image
            stitched = tifffile.imread(self.file_path)
            return stitched
        else:
            # Normalize by the weight image to get the final result
            stitched = self.stitched_image.copy()
            '''
            if len(stitched.shape)>2:
                stitched = stitched/np.max(stitched)
                stitched = np.uint8(stitched*255)
            '''
            self.isRunning = False
            return stitched

    def save_stitched_image(self, filename):
        stitched = self.get_stitched_image()
        imsave(filename, stitched)



class MovementController:
    def __init__(self, stages):
        self.stages = stages
        self.target_reached = False
        self.target_position = None
        self.axis = None


    def move_to_position(self, minPos, axis, speed, is_absolute):
        self.target_position = minPos
        self.speed = speed
        self.is_absolute = is_absolute
        self.axis = axis
        thread = threading.Thread(target=self._move)
        thread.start()

    def _move(self):
        self.target_reached = False
        self.stages.move(value=self.target_position, axis=self.axis, speed=self.speed, is_absolute=self.is_absolute, is_blocking=True)
        self.target_reached = True

    def is_target_reached(self):
        return self.target_reached



