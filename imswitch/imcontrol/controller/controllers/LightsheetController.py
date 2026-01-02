import os
import threading
from datetime import datetime
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as ndi
import scipy.signal as signal
import skimage.transform as transform
import tifffile as tif
from imswitch import IS_HEADLESS
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex, Timer
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from skimage.registration import phase_cross_correlation
from ..basecontrollers import ImConWidgetController
from fastapi.responses import FileResponse
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum

# import OME-Zarr writer
from imswitch.imcontrol.controller.controllers.experiment_controller.single_multiscale_zarr_data_source import SingleMultiscaleZarrWriter



# ============================================================================
# Pydantic models for API
# ============================================================================

class ScanMode(str, Enum):
    """Scan mode enumeration for lightsheet acquisition"""
    CONTINUOUS = "continuous"  # Original fast scan mode - stage moves continuously
    STEP_ACQUIRE = "step_acquire"  # Go-Stop-Acquire mode - stage moves, stops, acquires

class StorageFormat(str, Enum):
    """Storage format options for lightsheet data"""
    TIFF = "tiff"
    OME_ZARR = "ome_zarr"
    BOTH = "both"

class LightsheetScanParameters(BaseModel):
    """Parameters for lightsheet scan configuration"""
    minPos: float = Field(-500, description="Minimum position for scan axis")
    maxPos: float = Field(500, description="Maximum position for scan axis")
    stepSize: float = Field(10, description="Step size for step-acquire mode (µm)")
    speed: float = Field(1000, description="Speed for continuous mode")
    axis: str = Field("A", description="Scan axis (A, X, Y, Z)")
    illuSource: str = Field("", description="Illumination source name")
    illuValue: float = Field(512, description="Illumination intensity value")
    scanMode: ScanMode = Field(ScanMode.CONTINUOUS, description="Scan mode")
    storageFormat: StorageFormat = Field(StorageFormat.OME_ZARR, description="Storage format")
    experimentName: str = Field("lightsheet_scan", description="Experiment name for file naming")
    currentPosition: float = Field(0, description="Current position of the scan axis")

class LightsheetScanStatus(BaseModel):
    """Status information for lightsheet scan"""
    isRunning: bool = False
    scanMode: Optional[str] = None
    currentPosition: float = 0
    totalPositions: int = 0
    currentFrame: int = 0
    progress: float = 0.0
    zarrPath: Optional[str] = None
    tiffPath: Optional[str] = None
    errorMessage: Optional[str] = None


class LightsheetController(ImConWidgetController):
    """Linked to LightsheetWidget."""
    sigImageReceived = Signal(np.ndarray)
    sigSliderIlluValueChanged = Signal(float)
    sigScanStatusUpdate = Signal(dict)  # Signal for scan status updates
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        self.lightsheetTask = None
        self.lightsheetStack = np.ones((1,1,1))
        self.mFilePath = None
        self.mZarrPath = None  # Path to OME-Zarr store
        
        # Scan state
        self._scanStatus = LightsheetScanStatus()
        self._currentScanParams: Optional[LightsheetScanParameters] = None
        self._zarrWriter: Optional[SingleMultiscaleZarrWriter] = None
        
        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]

        # select lasers and add to gui
        self.lasers = self._master.lasersManager.getAllDeviceNames()
        self.laser = self.lasers[0]
        self.stageName = self._master.positionersManager.getAllDeviceNames()[0]
        self.stages = self._master.positionersManager[self.stageName]
        self.isLightsheetRunning = False
        
         

        # connect signals
        self.sigImageReceived.connect(self.displayImage)
        self._commChannel.sigStartLightSheet.connect(self.performScanningRecording)
        self._commChannel.sigStopLightSheet.connect(self.stopLightsheet)
        self._commChannel.sigUpdateMotorPosition.connect(self.updateAllPositionGUI)

        if IS_HEADLESS:
            return
        self._widget.startButton.clicked.connect(self.startLightsheet)
        self._widget.stopButton.clicked.connect(self.stopLightsheet)
        self._widget.setAvailableIlluSources(self.lasers)
        self._widget.setAvailableStageAxes(self.stages.axes)
        self._widget.sigSliderIlluValueChanged.connect(self.valueIlluChanged)

        # Connect all GUI elements from the SCAN tab
        self._widget.button_scan_xyz_start.clicked.connect(self.onButtonScanStart)
        self._widget.button_scan_xyz_stop.clicked.connect(self.onButtonScanStop)
        self._widget.buttonXY_up.clicked.connect(self.onButtonXYUp)
        self._widget.buttonXY_down.clicked.connect(self.onButtonXYDown)
        self._widget.buttonXY_left.clicked.connect(self.onButtonXYLeft)
        self._widget.buttonXY_right.clicked.connect(self.onButtonXYRight)
        self._widget.buttonXY_zero.clicked.connect(self.onButtonXYZero)
        self._widget.buttonFocus_up.clicked.connect(self.onButtonFocusUp)
        self._widget.buttonFocus_down.clicked.connect(self.onButtonFocusDown)
        self._widget.buttonSample_fwd.clicked.connect(self.onButtonSampleForward)
        self._widget.buttonSample_bwd.clicked.connect(self.onButtonSampleBackward)
        self._widget.buttonFocus_zero.clicked.connect(self.onButtonFocusZero)
        self._widget.button_scan_x_min_snap.clicked.connect(self.onButtonScanXMin)
        self._widget.button_scan_x_max_snap.clicked.connect(self.onButtonScanXMax)
        self._widget.button_scan_y_min_snap.clicked.connect(self.onButtonScanYMin)
        self._widget.button_scan_y_max_snap.clicked.connect(self.onButtonScanYMax)
        self._widget.button_scan_z_min_snap.clicked.connect(self.onButtonScanZMin)
        self._widget.button_scan_z_max_snap.clicked.connect(self.onButtonScanZMax)

        #self._widget.buttonRotation_minus.clicked.connect(self.onButtonRotationMinus)
        #self._widget.buttonRotation_plus.clicked.connect(self.onButtonRotationPlus)
        #self._widget.buttonRotation_zero.clicked.connect(self.onButtonRotationZero)

    # Event handler methods
    def onButtonScanStart(self):
        if IS_HEADLESS: # TODO: implement headless parameters
            return
        mScanParams = self._widget.get_scan_parameters()
        # returns => (self.scan_x_min[1].value(), self.scan_x_max[1].value(), self.scan_y_min[1].value(), self.scan_y_max[1].value(),
        # self.scan_z_min[1].value(), self.scan_z_max[1].value(), self.scan_overlap[1].value())

        # compute the spacing between xy tiles
        nPixels = self.detector.shape[0]
        pixelSize = self.detector.pixelSizeUm[-1]
        scanOverlap = mScanParams['overlap']*0.01

        # compute the number of pixels to move in x and y direction
        xrange = mScanParams['x_max']-mScanParams['x_min']
        yrange = mScanParams['y_max']-mScanParams['y_min']
        xSpacing = nPixels*pixelSize*(1-scanOverlap)
        ySpacing = nPixels*pixelSize*(1-scanOverlap)
        nTilesX = int(np.ceil(xrange/xSpacing))
        nTilesY = int(np.ceil(yrange/ySpacing))

        # compute x and y scan positions
        xyPositions = []
        for i in range(nTilesX):
            for j in range(nTilesY):
                xPosition = mScanParams['x_min']+i*xSpacing
                yPosition = mScanParams['y_min']+j*ySpacing
                xyPositions.append((int(xPosition), int(yPosition)))

        # perform the scanning in the background
        def performScanning(xyPositions, zMin, zMax, speed, axis, illuSource, illuValue):
            if not self.isLightsheetRunning:
                self.isLightsheetRunning = True
                for x, y in xyPositions:
                    self.lightsheetThread(zMin, zMax, x, y, speed, axis, illuSource, illuValue)
                self.isLightsheetRunning = False
        self._logger.info("Scan started")
        mThread = threading.Thread(target=performScanning, args=(xyPositions, mScanParams['z_min'], mScanParams['z_max'], mScanParams['speed'], mScanParams['stage_axis'], mScanParams['illu_source'], mScanParams['illu_value']))
        mThread.start()



    def onButtonScanStop(self):
        self._logger.debug("Scan stopped")
        self.isLightsheetRunning = False

    def onButtonXYUp(self):
        if IS_HEADLESS:  # TODO: implement headless parameters
            return
        mStepsizeXY,_ = self._widget.get_step_size_xy_zf()
        self._master.positionersManager.execOn(self.stageName, lambda c: c.move(axis="X", value=mStepsizeXY, is_absolute=False, is_blocking=False))

    def onButtonXYDown(self):
        mStepsizeXY,_ = self._widget.get_step_size_xy_zf()
        self.stages.move(value=-mStepsizeXY, axis="X", is_absolute=False, is_blocking=False)

    def onButtonXYLeft(self):
        mStepsizeXY,_ = self._widget.get_step_size_xy_zf()
        self.stages.move(value=mStepsizeXY, axis="Y", is_absolute=False, is_blocking=False)

    def onButtonXYRight(self):
        mStepsizeXY,_ = self._widget.get_step_size_xy_zf()
        self.stages.move(value=-mStepsizeXY, axis="Y", is_absolute=False, is_blocking=False)

    def onButtonXYZero(self):
        print("XY position reset")

    def onButtonFocusUp(self):
        _, mStepsizeXY = self._widget.get_step_size_xy_zf()
        self.stages.move(value=mStepsizeXY, axis="Z", is_absolute=False, is_blocking=False)

    def onButtonFocusDown(self):
        _, mStepsizeXY = self._widget.get_step_size_xy_zf()
        self.stages.move(value=-mStepsizeXY, axis="Z", is_absolute=False, is_blocking=False)

    def onButtonSampleForward(self):
        _, mStepsizeXY = self._widget.get_step_size_xy_zf()
        self.stages.move(value=mStepsizeXY, axis="A", is_absolute=False, is_blocking=False)

    def onButtonSampleBackward(self):
        _, mStepsizeXY = self._widget.get_step_size_xy_zf()
        self.stages.move(value=-mStepsizeXY, axis="A", is_absolute=False, is_blocking=False)

    def onButtonFocusZero(self):
        print("Focus position reset")

    def onButtonRotationMinus(self):
        print("Rotation decreased")

    def onButtonRotationPlus(self):
        print("Rotation increased")

    def onButtonRotationZero(self):
        print("Rotation reset")

    def getPositionByAxis(self, axis):
        allPositions = self.stages.getPosition()
        return allPositions[axis]

    def onButtonScanXMin(self):
        mPosition = self.getPositionByAxis("X")
        self._widget.set_scan_x_min(mPosition)

    def onButtonScanXMax(self):
        mPosition = self.getPositionByAxis("X")
        self._widget.set_scan_x_max(mPosition)

    def onButtonScanYMin(self):
        mPosition = self.getPositionByAxis("Y")
        self._widget.set_scan_y_min(mPosition)

    def onButtonScanYMax(self):
        mPosition = self.getPositionByAxis("Y")
        self._widget.set_scan_y_max(mPosition)

    def onButtonScanZMin(self):
        mPosition = self.getPositionByAxis("Z")
        self._widget.set_scan_z_min(mPosition)

    def onButtonScanZMax(self):
        mPosition = self.getPositionByAxis("Z")
        self._widget.set_scan_z_max(mPosition)

    def updateAllPositionGUI(self):
        #allPositions = self.stages.getPosition() # TODO: Necesllary?
        #mPositionsXYZ = (allPositions["X"], allPositions["Y"], allPositions["Z"])
        #self._widget.updatePosition(mPositionsXYZ)
        #self._commChannel.sigUpdateMotorPosition.emit()
        pass
        #TODO: This needs an update!

    def displayImage(self, lightsheetStack):
        # a bit weird, but we cannot update outside the main thread
        if IS_HEADLESS:
            return
        name = "Lightsheet Stack"
        # subsample stack
        # if the stack is too large, we have to subsample it
        if lightsheetStack.shape[0] > 200:
            subsample = 10
            lightsheetStack = lightsheetStack[::subsample,:,:]
        if not IS_HEADLESS:
            return self._widget.setImage(np.uint16(lightsheetStack ), colormap="gray", name=name, pixelsize=(20,1,1), translation=(0,0,0))

    def valueIlluChanged(self):
        if IS_HEADLESS:
            return
        illuSource = self._widget.getIlluminationSource()
        illuValue = self._widget.illuminationSlider.value()
        self._master.lasersManager
        if not self._master.lasersManager[illuSource].enabled:
            self._master.lasersManager[illuSource].setEnabled(1)

        illuValue = illuValue/100*self._master.lasersManager[illuSource].valueRangeMax
        self._master.lasersManager[illuSource].setValue(illuValue)

    def startLightsheet(self):
        if IS_HEADLESS:
            return
        minPos = self._widget.getMinPosition()
        maxPos = self._widget.getMaxPosition()
        speed = self._widget.getSpeed()
        illuSource = self._widget.getIlluminationSource()
        stageAxis = self._widget.getStageAxis()

        self._widget.startButton.setEnabled(False)
        self._widget.stopButton.setEnabled(True)
        self._widget.startButton.setText("Running")
        self._widget.stopButton.setText("Stop")
        self._widget.stopButton.setStyleSheet("background-color: red")
        self._widget.startButton.setStyleSheet("background-color: green")

        self.performScanningRecording(minPos, maxPos, speed, stageAxis, illuSource, 0)

    @APIExport()
    def setGalvo(self, channel:int=1, frequency:float=10, offset:float=0, amplitude:float=1, clk_div:int=0, phase:int=0, invert:int=1):
        '''Sets the galvo parameters for the lightsheet.'''
        if not self.isLightsheetRunning:
            try:
                self._master.lasersManager[self.laser].setGalvo(channel=channel, frequency=frequency, offset=offset, amplitude=amplitude, clk_div=clk_div, phase=phase, invert=invert)
                self._logger.info(f"Set galvo parameters: channel={channel}, frequency={frequency}, offset={offset}, amplitude={amplitude}, clk_div={clk_div}, phase={phase}, invert={invert}")
            except Exception as e:
                self._logger.error(f"Error setting galvo parameters: {e}")
        else:
            self._logger.warning("Cannot set galvo parameters while lightsheet is running.")
            
            
    @APIExport()
    def performScanningRecording(self, minPos:int=0, maxPos:int=1000, speed:int=1000, axis:str="A", illusource:int=-1, illuvalue:int=512):
        if not self.isLightsheetRunning:

            # check parameters
            if axis not in ("A", "X", "Y", "Z"):
                axis = "A"
            # use default illumination source if not selectd
            if illusource is None or illusource==-1 or illusource not in self._master.lasersManager.getAllDeviceNames():
                illusource = self._master.lasersManager.getAllDeviceNames()[0]

            #initialPosition = self.stages.getPosition()[axis]

            self.isLightsheetRunning = True
            if self.lightsheetTask is not None:
                self.lightsheetTask.join()
                del self.lightsheetTask
            self.lightsheetTask = threading.Thread(target=self.lightsheetThread, args=(minPos, maxPos, None, None, speed, axis, illusource, illuvalue))
            self.lightsheetTask.start()

    @APIExport()
    def returnLastLightsheetStackPath(self) -> str:
        '''Returns the path of the last saved lightsheet stack.'''
        if self.mFilePath is not None:
            return self.mFilePath
        else:
            return "No stack available yet"

    def lightsheetThread(self, minPosZ, maxPosZ, posX=None, posY=None, speed=10000, axis="A", illusource=None, illuvalue=None, isSave=True):
        '''Performs a lightsheet scan.'''
        self._logger.debug("Lightsheet thread started.")
        # TODO Have button for is save
        if posX is not None:
            self.stages.move(value=posX, axis="X", is_absolute=True, is_blocking=True)
        if posY is not None:
            self.stages.move(value=posY, axis="Y", is_absolute=True, is_blocking=True)
        self.detector.startAcquisition()
        # move to minPos
        self.stages.move(value=minPosZ, axis=axis, is_absolute=False, is_blocking=True)
        time.sleep(1)
        # now start acquiring images and move the stage in Background
        controller = MovementController(self.stages)
        controller.move_to_position(maxPosZ+np.abs(minPosZ), axis, speed, is_absolute=False)

        iFrame = 0
        allFrames = []
        while self.isLightsheetRunning:
            # Todo: Need to ensure thatwe have the right pattern displayed and the buffer is free - this heavily depends on the exposure time..
            mFrame = None
            lastFrameNumber = -1
            timeoutFrameRequest = .3 # seconds # TODO: Make dependent on exposure time
            cTime = time.time()
            frameSync = 2
            while(1):
                # get frame and frame number to get one that is newer than the one with illumination off eventually
                mFrame, currentFrameNumber = self.detector.getLatestFrame(returnFrameNumber=True)
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

            if mFrame is not None and mFrame.shape[0] != 0:
                allFrames.append(mFrame.copy())
            if controller.is_target_reached():
                break

            iFrame += 1
            # self._logger.debug(iFrame)


        # move back to initial position
        self.stages.move(value=-maxPosZ, axis=axis, is_absolute=False, is_blocking=True)

        # do something with the frames
        def displayAndSaveImageStack(isSave):
            # retreive positions and store the data if necessary
            pixelSizeZ = (maxPosZ-minPosZ)/len(allFrames)
            pixelSizeXY = self.detector.pixelSizeUm[-1]
            allPositions = self.stages.getPosition()
            posX = allPositions["X"]
            posY = allPositions["Y"]
            posZ = allPositions["Z"]
            if isSave:
                # save image stack with metadata
                mDate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                mExtension = "tif"
                mFileName = "lightsheet_stack_x_{posX}_y_{posY}_z_{posZ}_pz_{pixelSizeZ}_pxy_{pixelSizeXY}"
                self.mFilePath = self.getSaveFilePath(mDate, mFileName, mExtension)
                self._logger.info(f"Saving lightsheet stack to {self.mFilePath}")
                tif.imwrite(self.mFilePath, self.lightsheetStack)

        if len(allFrames) == 0:
            self._logger.error("No frames captured.")
            return
        self.lightsheetStack = np.array(allFrames).copy()
        saveImageThread = threading.Thread(target=displayAndSaveImageStack, args =(isSave,))
        saveImageThread.start()
        self.stopLightsheet()
        if not IS_HEADLESS: self.sigImageReceived.emit(self.lightsheetStack)

    def getSaveFilePath(self, date, filename, extension):
        mFilename =  f"{date}_{filename}.{extension}"
        dirPath  = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), 'LightSheet', date)
        newPath = os.path.join(dirPath,mFilename)

        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        return newPath

    @APIExport()
    def getLatestLightsheetStackAsTif(self):
        """
        If there is a lightsheet stack available, return it as a TIFF file for download.
        """
        if self.mFilePath is not None and os.path.exists(self.mFilePath):
            # Return the file as a response for download
            return FileResponse(
                path=self.mFilePath,
                media_type="application/octet-stream",
                filename=os.path.basename(self.mFilePath)
            )
        else:
            # Return an error message if the file is not available
            return {"error": "No lightsheet stack available"}

    @APIExport()
    def getIsLightsheetRunning(self):
        return self.isLightsheetRunning

    def stopLightsheet(self):
        self.isLightsheetRunning = False
        self._scanStatus.isRunning = False
        self._emitScanStatus()
        
        # Close zarr writer if open
        if self._zarrWriter is not None:
            try:
                self._zarrWriter.close()
            except Exception as e:
                self._logger.error(f"Error closing zarr writer: {e}")
            self._zarrWriter = None
        
        if IS_HEADLESS:
            return
        self._widget.startButton.setEnabled(True)
        self._widget.stopButton.setEnabled(False)
        self._widget.illuminationSlider.setValue(0)
        illuSource = self._widget.getIlluminationSource()
        if not self._master.lasersManager[illuSource].enabled:
            self._master.lasersManager[illuSource].setEnabled(0)
        self._widget.startButton.setText("Start")
        self._widget.stopButton.setText("Stopped")
        self._widget.stopButton.setStyleSheet("background-color: green")
        self._widget.startButton.setStyleSheet("background-color: red")
        self._logger.debug("Lightsheet scanning stopped.")

    # ========================================================================
    # New Go-Stop-Acquire Mode with OME-Zarr support
    # ========================================================================

    def _emitScanStatus(self):
        """Emit scan status via socket and signal"""
        status_dict = self._scanStatus.model_dump()
        self.sigScanStatusUpdate.emit(status_dict)

    @APIExport()
    def getScanStatus(self) -> dict:
        """Get current scan status including progress and file paths."""
        return self._scanStatus.model_dump()

    @APIExport()
    def getAvailableScanModes(self) -> list:
        """Get available scan modes."""
        return [mode.value for mode in ScanMode]

    @APIExport()
    def getAvailableStorageFormats(self) -> list:
        """Get available storage formats."""
        formats = [StorageFormat.TIFF.value]
        formats.append(StorageFormat.OME_ZARR.value)
        formats.append(StorageFormat.BOTH.value)
        return formats

    @APIExport()
    def startStepAcquireScan(
        self,
        minPos: float = -500,
        maxPos: float = 500,
        stepSize: float = 10,
        axis: str = "A",
        illuSource: str = "",
        illuValue: float = 512,
        storageFormat: str = "ome_zarr",
        experimentName: str = "lightsheet_scan"
    ) -> dict:
        """
        Start a Go-Stop-Acquire scan with configurable storage format.
        
        This mode moves the stage in discrete steps, stops, acquires an image,
        then moves to the next position. Suitable for high-quality Z-stacks.
        
        Args:
            minPos: Start position in µm
            maxPos: End position in µm
            stepSize: Step size between acquisitions in µm
            axis: Scan axis (A, X, Y, Z)
            illuSource: Illumination source name
            illuValue: Illumination intensity
            storageFormat: Storage format (tiff, ome_zarr, both)
            experimentName: Name for the experiment/files
            
        Returns:
            dict: Status information including file paths
        """
        if self.isLightsheetRunning:
            return {"error": "Scan already running", "success": False}

        # Validate parameters
        if axis not in ("A", "X", "Y", "Z"):
            axis = "A"
        
        # Use default illumination source if not selected
        if not illuSource or illuSource == "-1":
            illuSource = self._master.lasersManager.getAllDeviceNames()[0]

        # Parse storage format
        try:
            storage = StorageFormat(storageFormat.lower())
        except ValueError:
            storage = StorageFormat.OME_ZARR

        # get current position from device:
        currentPosition = self.stages.getPosition()[axis]

        # Create scan parameters
        params = LightsheetScanParameters(
            minPos=currentPosition + minPos, # TODO: we should make these parameters relative to the current position
            maxPos=currentPosition + maxPos,# TODO: we should make these parameters relative to the current position
            currentPosition=currentPosition,
            stepSize=stepSize,
            axis=axis,
            illuSource=illuSource,
            illuValue=illuValue,
            scanMode=ScanMode.STEP_ACQUIRE,
            storageFormat=storage,
            experimentName=experimentName
        )
        
        self._currentScanParams = params
        self.isLightsheetRunning = True
        
        # Calculate total positions
        totalSteps = int(abs(maxPos - minPos) / stepSize) + 1
        
        # Initialize scan status
        self._scanStatus = LightsheetScanStatus(
            isRunning=True,
            scanMode=ScanMode.STEP_ACQUIRE.value,
            currentPosition=minPos,
            totalPositions=totalSteps,
            currentFrame=0,
            progress=0.0
        )
        self._emitScanStatus()
        
        # Start scan in background thread
        if self.lightsheetTask is not None:
            try:
                self.lightsheetTask.join(timeout=1)
            except:
                pass
        
        self.lightsheetTask = threading.Thread(
            target=self._stepAcquireThread,
            args=(params, totalSteps)
        )
        self.lightsheetTask.start()
        
        return {
            "success": True,
            "message": "Step-acquire scan started",
            "totalPositions": totalSteps,
            "scanMode": ScanMode.STEP_ACQUIRE.value,
            "storageFormat": storage.value
        }

    def _stepAcquireThread(self, params: LightsheetScanParameters, totalSteps: int):
        """Background thread for step-acquire scanning."""
        self._logger.info(f"Starting step-acquire scan: {params.minPos} to {params.maxPos}, step={params.stepSize}")
        
        allFrames = []
        mDate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Setup storage paths
        dirPath = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), 'LightSheet', mDate)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        
        # Initialize OME-Zarr writer if needed
        zarrPath = None
        if params.storageFormat in (StorageFormat.OME_ZARR, StorageFormat.BOTH):
            zarrPath = os.path.join(dirPath, f"{params.experimentName}.zarr")
            self.mZarrPath = zarrPath
            self._scanStatus.zarrPath = zarrPath
            
            # Get image dimensions
            testFrame = self.detector.getLatestFrame()
            if testFrame is not None:
                imgHeight, imgWidth = testFrame.shape[:2]
            else:
                imgHeight, imgWidth = 512, 512
            
            # Initialize zarr writer
            self._zarrWriter = SingleMultiscaleZarrWriter(zarrPath, mode="w")
            self._zarrWriter.set_metadata(
                t=1,  # single timepoint
                c=1,  # single channel
                z=totalSteps,
                bigY=imgHeight,
                bigX=imgWidth,
                dtype=np.uint16
            )
            self._zarrWriter.open_store()
            self._logger.info(f"OME-Zarr store opened: {zarrPath}")

        try:
            # Turn on illumination
            if params.illuSource and params.illuSource in self._master.lasersManager.getAllDeviceNames():
                laser = self._master.lasersManager[params.illuSource]
                if not laser.enabled:
                    laser.setEnabled(True)
                maxValue = laser.valueRangeMax
                scaledValue = (params.illuValue / 1024) * maxValue
                laser.setValue(scaledValue)

            # Start detector acquisition
            self.detector.startAcquisition()
            
            # Move to start position
            self.stages.move(value=params.minPos, axis=params.axis, is_absolute=True, is_blocking=True)
            time.sleep(0.2)  # Settling time

            # Calculate positions
            positions = np.arange(params.minPos, params.maxPos + params.stepSize/2, params.stepSize)
            
            for idx, pos in enumerate(positions):
                if not self.isLightsheetRunning:
                    self._logger.info("Scan aborted by user")
                    break
                
                # Move to position (Go)
                self.stages.move(value=pos, axis=params.axis, is_absolute=True, is_blocking=True)
                time.sleep(0.05)  # Short settling time (Stop)
                
                # Acquire frame
                frame = None
                for attempt in range(2):
                    frame = self.detector.getLatestFrame()
                    if frame is not None and frame.shape[0] > 0:
                        break
                    time.sleep(0.02)
                
                if frame is not None:
                    allFrames.append(frame.copy())
                    
                    # Write to Zarr if enabled
                    if self._zarrWriter is not None:
                        try:
                            # Convert to 2D if needed (take first channel if RGB)
                            if len(frame.shape) == 3:
                                frame_2d = frame[:, :, 0]
                            else:
                                frame_2d = frame
                            self._zarrWriter.write_tile(
                                tile=frame_2d.astype(np.uint16),
                                t=0,
                                c=0,
                                z=idx,
                                y_start=0,
                                x_start=0
                            )
                        except Exception as e:
                            self._logger.error(f"Error writing to Zarr: {e}")
                
                # Update status
                self._scanStatus.currentFrame = idx + 1
                self._scanStatus.currentPosition = pos
                self._scanStatus.progress = (idx + 1) / totalSteps * 100
                self._emitScanStatus()
                
                self._logger.debug(f"Frame {idx+1}/{totalSteps} at position {pos}")

            # Save as TIFF if needed
            if len(allFrames) > 0:
                self.lightsheetStack = np.array(allFrames)
                
                if params.storageFormat in (StorageFormat.TIFF, StorageFormat.BOTH):
                    tiffPath = os.path.join(dirPath, f"{params.experimentName}.tif")
                    self.mFilePath = tiffPath
                    self._scanStatus.tiffPath = tiffPath
                    tif.imwrite(tiffPath, self.lightsheetStack)
                    self._logger.info(f"TIFF stack saved: {tiffPath}")

            # Close Zarr writer
            if self._zarrWriter is not None:
                self._zarrWriter.close()
                self._zarrWriter = None
                self._logger.info(f"OME-Zarr store closed: {zarrPath}")

            # Turn off illumination
            if params.illuSource and params.illuSource in self._master.lasersManager.getAllDeviceNames():
                laser = self._master.lasersManager[params.illuSource]
                laser.setValue(0)
                
            # move back to initial position
            self.stages.move(value=params.currentPosition, axis=params.axis, is_absolute=True, is_blocking=False)

        except Exception as e:
            self._logger.error(f"Error during step-acquire scan: {e}")
            self._scanStatus.errorMessage = str(e)
        
        finally:
            self._scanStatus.isRunning = False
            self._scanStatus.progress = 100.0
            self._emitScanStatus()
            self.isLightsheetRunning = False
            
            # Emit final stack to GUI if available
            if len(allFrames) > 0 and not IS_HEADLESS:
                self.sigImageReceived.emit(self.lightsheetStack)

    @APIExport()
    def getLatestZarrPath(self) -> dict:
        """Get the path to the latest OME-Zarr store for visualization."""
        if self.mZarrPath and os.path.exists(self.mZarrPath):
            # Return path relative to data directory for frontend access
            dataPath = str(dirtools.UserFileDirs.getValidatedDataPath())
            relPath = self.mZarrPath.replace(dataPath, '').lstrip(os.sep)
            return {
                "zarrPath": f"/data/{relPath.replace(os.sep, '/')}",
                "absolutePath": self.mZarrPath,
                "exists": True
            }
        return {"zarrPath": None, "exists": False}

    @APIExport()
    def startContinuousScanWithZarr(
        self,
        minPos: float = -500,
        maxPos: float = 500,
        speed: float = 1000,
        axis: str = "A",
        illuSource: str = "",
        illuValue: float = 512,
        storageFormat: str = "ome_zarr",
        experimentName: str = "lightsheet_continuous"
    ) -> dict:
        """
        Start a continuous scan with optional OME-Zarr storage.
        
        This is an enhanced version of the original continuous scan that
        supports OME-Zarr output alongside TIFF.
        
        Args:
            minPos: Start position in µm (relative)
            maxPos: End position in µm (relative)
            speed: Stage movement speed
            axis: Scan axis (A, X, Y, Z)
            illuSource: Illumination source name
            illuValue: Illumination intensity
            storageFormat: Storage format (tiff, ome_zarr, both)
            experimentName: Name for the experiment/files
            
        Returns:
            dict: Status information
        """
        if self.isLightsheetRunning:
            return {"error": "Scan already running", "success": False}
        # Parse storage format
        try:
            storage = StorageFormat(storageFormat.lower())
        except ValueError:
            storage = StorageFormat.TIFF
        # current position from device:
        currentPosition = self.stages.getPosition()[axis]
        params = LightsheetScanParameters(
            minPos=minPos + currentPosition, # TODO: we should make these parameters relative to the current position
            maxPos=maxPos + currentPosition,# TODO: we should make these parameters relative to the current position
            currentPosition=currentPosition,
            speed=speed,
            axis=axis,
            illuSource=illuSource,
            illuValue=illuValue,
            scanMode=ScanMode.CONTINUOUS,
            storageFormat=storage,
            experimentName=experimentName
        )
        
        self._currentScanParams = params
        self.isLightsheetRunning = True
        self._scanStatus = LightsheetScanStatus(
            isRunning=True,
            scanMode=ScanMode.CONTINUOUS.value,
            currentPosition=minPos,
            progress=0.0
        )
        self._emitScanStatus()
        
        # Start original lightsheet thread with storage format handling
        if self.lightsheetTask is not None:
            try:
                self.lightsheetTask.join(timeout=1)
            except:
                pass
        
        self.lightsheetTask = threading.Thread(
            target=self._continuousScanWithZarrThread,
            args=(params,)
        )
        self.lightsheetTask.start()
        
        return {
            "success": True,
            "message": "Continuous scan with Zarr support started",
            "scanMode": ScanMode.CONTINUOUS.value,
            "storageFormat": storage.value
        }

    def _continuousScanWithZarrThread(self, params: LightsheetScanParameters):
        """Enhanced continuous scan thread with OME-Zarr support."""
        self._logger.info(f"Starting continuous scan: {params.minPos} to {params.maxPos}")
        
        allFrames = []
        mDate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Setup storage
        dirPath = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), 'LightSheet', mDate)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        try:
            # Turn on illumination
            if params.illuSource and params.illuSource in self._master.lasersManager.getAllDeviceNames():
                laser = self._master.lasersManager[params.illuSource]
                if not laser.enabled:
                    laser.setEnabled(True)
                maxValue = laser.valueRangeMax
                scaledValue = (params.illuValue / 1024) * maxValue
                laser.setValue(scaledValue)

            self.detector.startAcquisition()
            
            # Move to start
            self.stages.move(value=params.minPos, axis=params.axis, is_absolute=True, is_blocking=True)
            time.sleep(0.5)
            
            # Start continuous movement
            controller = MovementController(self.stages)
            totalDistance = params.maxPos - params.minPos
            controller.move_to_position(totalDistance, params.axis, params.speed, is_absolute=False)
            
            frameCount = 0
            while self.isLightsheetRunning and not controller.is_target_reached():
                frame = self.detector.getLatestFrame()
                if frame is not None and frame.shape[0] > 0:
                    allFrames.append(frame.copy())
                    frameCount += 1
                    
                    # Update status
                    currentPos = self.stages.getPosition().get(params.axis, params.minPos)
                    self._scanStatus.currentFrame = frameCount
                    self._scanStatus.currentPosition = currentPos
                    progress = abs(currentPos - params.minPos) / abs(totalDistance) * 100
                    self._scanStatus.progress = min(progress, 100)
                    self._emitScanStatus()
                
                time.sleep(0.01)  # Small delay to prevent CPU overload

            # Move back
            self.stages.move(value=-totalDistance, axis=params.axis, is_absolute=False, is_blocking=True)

            # Save data
            if len(allFrames) > 0:
                self.lightsheetStack = np.array(allFrames)
                
                # Save TIFF if needed
                if params.storageFormat in (StorageFormat.TIFF, StorageFormat.BOTH):
                    tiffPath = os.path.join(dirPath, f"{params.experimentName}.tif")
                    self.mFilePath = tiffPath
                    self._scanStatus.tiffPath = tiffPath
                    tif.imwrite(tiffPath, self.lightsheetStack)
                    self._logger.info(f"TIFF saved: {tiffPath}")
                
                # Save Zarr if needed
                if params.storageFormat in (StorageFormat.OME_ZARR, StorageFormat.BOTH):
                    zarrPath = os.path.join(dirPath, f"{params.experimentName}.zarr")
                    self.mZarrPath = zarrPath
                    self._scanStatus.zarrPath = zarrPath
                    
                    imgHeight, imgWidth = allFrames[0].shape[:2]
                    writer = SingleMultiscaleZarrWriter(zarrPath, mode="w")
                    writer.set_metadata(t=1, c=1, z=len(allFrames), bigY=imgHeight, bigX=imgWidth, dtype=np.uint16)
                    writer.open_store()
                    
                    for idx, frame in enumerate(allFrames):
                        frame_2d = frame[:, :, 0] if len(frame.shape) == 3 else frame
                        writer.write_tile(frame_2d.astype(np.uint16), t=0, c=0, z=idx, y_start=0, x_start=0)
                    
                    writer.close()
                    self._logger.info(f"Zarr saved: {zarrPath}")

            # Turn off illumination
            if params.illuSource and params.illuSource in self._master.lasersManager.getAllDeviceNames():
                laser = self._master.lasersManager[params.illuSource]
                laser.setValue(0)

        except Exception as e:
            self._logger.error(f"Error in continuous scan: {e}")
            self._scanStatus.errorMessage = str(e)
        
        finally:
            self._scanStatus.isRunning = False
            self._scanStatus.progress = 100.0
            self._emitScanStatus()
            self.isLightsheetRunning = False
            
            if len(allFrames) > 0 and not IS_HEADLESS:
                self.sigImageReceived.emit(self.lightsheetStack)


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

# Copyright (C) 2020-2024 ImSwitch developers
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
