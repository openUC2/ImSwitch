import os
import threading
from datetime import datetime
import time
import cv2
import numpy as np
import tifffile as tif
from imswitch import IS_HEADLESS
from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from fastapi.responses import FileResponse
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

# import OME-Zarr writer
from imswitch.imcontrol.controller.controllers.experiment_controller.ome_writer import OMEWriter, OMEWriterConfig
from imswitch.imcontrol.controller.controllers.experiment_controller.SingleTiffWriter import SingleTiffWriter
from dataclasses import dataclass



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
        self._omeWriter: Optional[OMEWriter] = None
        self._tiffWriter: Optional[SingleTiffWriter] = None

        # Observation camera streaming (similar to PixelCalibrationController)
        self.observationStreamRunning = False
        self.observationStreamStarted = False
        self.observationStreamQueue = None

        # Try to get observation camera if available
        try:
            # Look for an observation camera (often named differently from main detector)
            allCameras = self._master.detectorsManager.getAllDeviceNames()
            # Try to find observation camera (you may need to adjust this based on your setup)
            self.observationCamera = None
            for camName in allCameras:
                if "observation" in camName.lower() or "overview" in camName.lower():
                    self.observationCamera = self._master.detectorsManager[camName]
                    self._logger.info(f"Found observation camera: {camName}")
                    break
        except Exception as e:
            self._logger.warning(f"No observation camera found: {e}")
            self.observationCamera = None

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
        """
        [DEPRECATED] Use startContinuousScanWithZarr() instead.
        
        This method is kept for backward compatibility but internally redirects 
        to the new startContinuousScanWithZarr() which supports OME-Zarr storage.
        """
        self._logger.warning("performScanningRecording is deprecated. Use startContinuousScanWithZarr instead.")

        # check parameters
        if axis not in ("A", "X", "Y", "Z"):
            axis = "A"
        # use default illumination source if not selected
        if illusource is None or illusource==-1 or illusource not in self._master.lasersManager.getAllDeviceNames():
            illusource = self._master.lasersManager.getAllDeviceNames()[0]

        # Redirect to new method with default TIFF storage for backward compatibility
        return self.startContinuousScanWithZarr(
            minPos=minPos,
            maxPos=maxPos,
            speed=speed,
            axis=axis,
            illuSource=illusource,
            illuValue=illuvalue,
            storageFormat="tiff",  # Use TIFF for backward compatibility
            experimentName="lightsheet_scan_legacy"
        )

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
        self._scanStatus.progress = 100.0
        self._emitScanStatus()

        # Close OME writer if open
        if self._omeWriter is not None:
            try:
                self._omeWriter.finalize()
            except Exception as e:
                self._logger.error(f"Error finalizing OME writer: {e}")
            self._omeWriter = None

        # Close tiff writer if open
        if self._tiffWriter is not None:
            try:
                self._tiffWriter.close()
            except Exception as e:
                self._logger.error(f"Error closing tiff writer: {e}")
            self._tiffWriter = None

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
        """Background thread for step-acquire scanning with immediate frame writing."""
        self._logger.info(f"Starting step-acquire scan: {params.minPos} to {params.maxPos}, step={params.stepSize}")

        mDate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Setup storage paths
        dirPath = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), 'LightSheet', mDate)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # Initialize writers based on storage format
        zarrPath = None
        tiffPath = None

        # Initialize OME-Zarr writer if needed
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

            # Initialize OME writer with proper config
            from imswitch.imcontrol.controller.controllers.experiment_controller.ome_writer import OMEWriter, OMEWriterConfig
            from imswitch.imcontrol.controller.controllers.experiment_controller.experiment_mode_base import OMEFileStorePaths
            
            # Create file paths structure - base_dir should be WITHOUT .zarr extension
            # OMEFileStorePaths automatically adds .ome.zarr to create zarr_dir
            base_name = params.experimentName
            base_path = os.path.join(dirPath, base_name)
            file_paths = OMEFileStorePaths(base_dir=base_path)
            
            # Update zarrPath to match what OMEFileStorePaths creates
            zarrPath = file_paths.zarr_dir
            self.mZarrPath = zarrPath
            self._scanStatus.zarrPath = zarrPath
            currentPosition = self.stages.getPosition()
            # Create OME writer config
            ome_config = OMEWriterConfig(
                write_tiff=False,
                write_zarr=True,
                n_time_points=1,
                n_z_planes=totalSteps,
                n_channels=1,
                pixel_size=self.detector.pixelSizeUm[-1],
                pixel_size_z=abs(params.stepSize),
                channel_names=["Lightsheet"],
                channel_colors=["00FF00"],
                x_start=currentPosition["X"],
                y_start=currentPosition["Y"],
                z_start=currentPosition[params.axis],
            )
            
            # Initialize OME writer
            self._omeWriter = OMEWriter(
                file_paths=file_paths,
                tile_shape=(imgHeight, imgWidth),
                grid_shape=(1, 1),  # Single tile
                grid_geometry=(0, 0, 0, 0),
                config=ome_config,
                logger=self._logger
            )
            self._logger.info(f"OME-Zarr writer initialized: {zarrPath}")

        # Initialize TIFF writer if needed
        if params.storageFormat in (StorageFormat.TIFF, StorageFormat.BOTH):
            tiffPath = os.path.join(dirPath, f"{params.experimentName}.tif")
            self.mFilePath = tiffPath
            self._scanStatus.tiffPath = tiffPath
            self._tiffWriter = SingleTiffWriter(tiffPath, bigtiff=True)
            self._logger.info(f"TIFF writer initialized: {tiffPath}")

        try:
            # Turn on illumination
            if params.illuSource and params.illuSource in self._master.lasersManager.getAllDeviceNames():
                laser = self._master.lasersManager[params.illuSource]
                if not laser.enabled:
                    laser.setEnabled(True)
                maxValue = laser.valueRangeMax
                scaledValue = (params.illuValue / 1024) * maxValue
                laser.setValue(scaledValue)
                laser.setEnabled(True)
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
                    # Convert to 2D if needed (take first channel if RGB)
                    if len(frame.shape) == 3:
                        frame_2d = frame[:, :, 0]
                    else:
                        frame_2d = frame

                    # Write to Zarr immediately if enabled
                    if self._omeWriter is not None:
                        try:
                            # Create metadata dict for OME writer - must include all required fields
                            frame_metadata = {
                                "x": 0,
                                "y": 0,
                                "z": pos,  # Use physical position from axis, not index
                                "tile_index": 0,
                                "time_index": 0,
                                "z_index": idx,
                                "channel_index": 0,
                                "runningNumber": idx,  # Required by OMEWriter
                                "illuminationChannel": params.illuSource if params.illuSource else "Default",
                                "illuminationValue": params.illuValue
                            }
                            self._omeWriter.write_frame(frame_2d.astype(np.uint16), frame_metadata)
                        except Exception as e:
                            self._logger.error(f"Error writing to Zarr: {e}")

                    # Write to TIFF immediately if enabled
                    if self._tiffWriter is not None:
                        try:
                            metadata = {
                                "pixel_size": 1.0,  # TODO: Get from config
                                "x": pos,
                                "y": 0,
                                "z": idx
                            }
                            self._tiffWriter.add_image(frame_2d.astype(np.uint16), metadata)
                        except Exception as e:
                            self._logger.error(f"Error writing to TIFF: {e}")


                # Update status
                self._scanStatus.currentFrame = idx + 1
                self._scanStatus.currentPosition = pos
                self._scanStatus.progress = (idx + 1) / totalSteps * 100
                self._emitScanStatus()

                self._logger.debug(f"Frame {idx+1}/{totalSteps} at position {pos}")


            # Close writers
            if self._omeWriter is not None:
                self._omeWriter.finalize()
                self._omeWriter = None
                self._logger.info(f"OME writer finalized: {zarrPath}")

            if self._tiffWriter is not None:
                self._tiffWriter.close()
                self._tiffWriter = None
                self._logger.info(f"TIFF writer closed: {tiffPath}")

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
        """Enhanced continuous scan thread with immediate writing (memory-efficient)."""
        self._logger.info(f"Starting continuous scan: {params.minPos} to {params.maxPos}")

        mDate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Setup storage
        dirPath = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), 'LightSheet', mDate)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # Initialize writers based on storage format
        zarrPath = None
        tiffPath = None
        frameCount = 0

        # For keeping last frames for GUI preview
        last_frames_preview = []

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

            # Get initial frame for writer initialization
            initFrame = None
            for _ in range(5):
                initFrame = self.detector.getLatestFrame()
                if initFrame is not None and initFrame.shape[0] > 0:
                    break
                time.sleep(0.1)

            if initFrame is None:
                raise Exception("Could not acquire initial frame for writer initialization")

            imgHeight, imgWidth = initFrame.shape[:2]

            # Initialize TIFF writer if needed
            if params.storageFormat in (StorageFormat.TIFF, StorageFormat.BOTH):
                tiffPath = os.path.join(dirPath, f"{params.experimentName}.tif")
                self.mFilePath = tiffPath
                self._scanStatus.tiffPath = tiffPath
                self._tiffWriter = SingleTiffWriter(tiffPath, bigtiff=True)
                self._logger.info(f"TIFF writer initialized: {tiffPath}")

            # Start continuous movement
            controller = MovementController(self.stages)
            totalDistance = params.maxPos - params.minPos
            controller.move_to_position(totalDistance, params.axis, params.speed, is_absolute=False)

            while self.isLightsheetRunning and not controller.is_target_reached():
                frame = self.detector.getLatestFrame()
                if frame is not None and frame.shape[0] > 0:
                    # Convert to 2D if needed
                    if len(frame.shape) == 3:
                        frame_2d = frame[:, :, 0]
                    else:
                        frame_2d = frame

                    # Write to TIFF immediately if enabled
                    if self._tiffWriter is not None:
                        try:
                            currentPos = self.stages.getPosition().get(params.axis, params.minPos)
                            metadata = {
                                "pixel_size": 1.0,
                                "x": currentPos,
                                "y": 0,
                                "z": frameCount
                            }
                            self._tiffWriter.add_image(frame_2d.astype(np.uint16), metadata)
                        except Exception as e:
                            self._logger.error(f"Error writing to TIFF: {e}")

                    frameCount += 1

                    # Keep last frames for preview (limit memory)
                    if len(last_frames_preview) < 10:
                        last_frames_preview.append(frame.copy())
                    else:
                        last_frames_preview.pop(0)
                        last_frames_preview.append(frame.copy())

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

            # Close TIFF writer
            if self._tiffWriter is not None:
                self._tiffWriter.close()
                self._tiffWriter = None
                self._logger.info(f"TIFF writer closed: {tiffPath}")

            # Save Zarr if needed (post-processing from TIFF or direct write)
            # Note: For continuous scan, we write TIFF first then can convert to Zarr if needed
            if params.storageFormat in (StorageFormat.OME_ZARR, StorageFormat.BOTH):
                # If we have TIFF, we can read and convert using OME writer
                if tiffPath and os.path.exists(tiffPath):
                    self._logger.info("Converting TIFF to Zarr using OME writer...")
                    try:
                        from imswitch.imcontrol.controller.controllers.experiment_controller.ome_writer import OMEWriter, OMEWriterConfig
                        from imswitch.imcontrol.controller.controllers.experiment_controller.experiment_mode_base import OMEFileStorePaths
                        
                        tiff_data = tif.imread(tiffPath)
                        if len(tiff_data.shape) == 2:
                            tiff_data = tiff_data[np.newaxis, ...]  # Add Z dimension

                        # Create file paths structure - base_dir without .zarr extension
                        base_name = params.experimentName
                        base_path = os.path.join(dirPath, base_name)
                        file_paths = OMEFileStorePaths(base_dir=base_path)
                        
                        # Update zarrPath to match what OMEFileStorePaths creates
                        zarrPath = file_paths.zarr_dir
                        self.mZarrPath = zarrPath
                        self._scanStatus.zarrPath = zarrPath
                        
                        # Calculate step size from distance and frame count
                        totalDistance = abs(params.maxPos - params.minPos)
                        step_size_z = totalDistance / max(1, tiff_data.shape[0] - 1) if tiff_data.shape[0] > 1 else 1.0
                        
                        # Create OME writer config
                        ome_config = OMEWriterConfig(
                            write_tiff=False,
                            write_zarr=True,
                            n_time_points=1,
                            n_z_planes=tiff_data.shape[0],
                            n_channels=1,
                            pixel_size=1.0,
                            pixel_size_z=step_size_z,
                            channel_names=["Lightsheet"],
                            channel_colors=["00FF00"],
                            x_start=0.0,
                            y_start=0.0,
                            z_start=params.minPos
                        )
                        
                        # Initialize OME writer
                        ome_writer = OMEWriter(
                            file_paths=file_paths,
                            tile_shape=(tiff_data.shape[1], tiff_data.shape[2]),
                            grid_shape=(1, 1),
                            grid_geometry=(0, 0, 0, 0),
                            config=ome_config,
                            logger=self._logger
                        )

                        for idx in range(tiff_data.shape[0]):
                            # Calculate physical z position for each frame
                            z_physical = params.minPos + (idx * step_size_z)
                            frame_metadata = {
                                "x": 0,
                                "y": 0,
                                "z": z_physical,
                                "tile_index": 0,
                                "time_index": 0,
                                "z_index": idx,
                                "channel_index": 0,
                                "runningNumber": idx,  # Required by OMEWriter
                                "illuminationChannel": params.illuSource if params.illuSource else "Default",
                                "illuminationValue": params.illuValue
                            }
                            ome_writer.write_frame(tiff_data[idx].astype(np.uint16), frame_metadata)

                        ome_writer.finalize()
                        self._logger.info(f"Zarr saved using OME writer: {zarrPath}")
                    except Exception as e:
                        self._logger.error(f"Error converting to Zarr: {e}")

            # Update preview stack
            if len(last_frames_preview) > 0:
                self.lightsheetStack = np.array(last_frames_preview)

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

            if len(last_frames_preview) > 0 and not IS_HEADLESS:
                self.sigImageReceived.emit(self.lightsheetStack)

    # ========================================================================
    # Observation Camera Streaming (for sample positioning visualization)
    # ========================================================================

    @APIExport()
    def stopObservationStream(self):
        """Stop the observation camera MJPEG stream."""
        self.observationStreamRunning = False
        self.observationStreamStarted = False
        self.observationStreamQueue = None

    def startObservationStream(self):
        """
        Background thread that converts observation camera frames to JPEG and queues them.
        """

        if self.observationCamera is None:
            self._logger.error("Observation camera not available")
            return

        # Wait for first valid frame (up to 2s); fall back to black frame
        deadline = time.time() + 2.0
        output_frame = None
        while self.observationStreamRunning and output_frame is None and time.time() < deadline:
            try:
                output_frame = self.observationCamera.getLatestFrame()
            except Exception:
                output_frame = None
            if output_frame is None:
                time.sleep(0.05)

        if output_frame is None:
            # Default black frame if nothing available (grayscale)
            output_frame = np.zeros((480, 640), dtype=np.uint8)

        # Adaptive resize: Keep frames below 640x480
        try:
            if output_frame.shape[0] > 640 or output_frame.shape[1] > 480:
                everyNthsPixel = int(
                    np.min(
                        [
                            max(1, output_frame.shape[0] // 480),
                            max(1, output_frame.shape[1] // 640),
                        ]
                    )
                )
            else:
                everyNthsPixel = 1
        except Exception:
            everyNthsPixel = 1

        try:
            while self.observationStreamRunning:
                output_frame = self.observationCamera.getLatestFrame()
                if output_frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    # Downsample if needed
                    output_frame = output_frame[::everyNthsPixel, ::everyNthsPixel]
                except Exception:
                    output_frame = np.zeros((480, 640), dtype=np.uint8)

                # Convert grayscale to BGR if needed (for consistent processing)
                if len(output_frame.shape) == 2:
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

                # Ensure uint8 image for JPEG; normalize if needed
                if output_frame.dtype != np.uint8:
                    try:
                        vmin = float(np.min(output_frame))
                        vmax = float(np.max(output_frame))
                        if vmax > vmin:
                            output_frame = (
                                (output_frame - vmin) / (vmax - vmin) * 255.0
                            ).astype(np.uint8)
                        else:
                            output_frame = np.zeros_like(output_frame, dtype=np.uint8)
                    except Exception:
                        output_frame = np.zeros_like(output_frame, dtype=np.uint8)

                # JPEG compression
                quality = 90  # Quality level (0-100)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                flag, encodedImage = cv2.imencode(".jpg", output_frame, encode_params)
                if not flag:
                    continue

                # Put raw JPEG bytes into queue; avoid blocking forever if queue is full
                try:
                    self.observationStreamQueue.put(encodedImage.tobytes(), timeout=0.5)
                except Exception:
                    # Drop frame if queue is full or unavailable
                    pass

                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            self._logger.error(f"Observation stream error: {e}", exc_info=True)
            self.observationStreamRunning = False

    def observationStreamer(self):
        """
        Generator that yields JPEG frames from the queue.
        Starts the background streaming thread if not already running.
        """
        import queue

        # Start the streaming worker thread once and create a thread-safe queue
        if not self.observationStreamStarted:
            import threading

            self.observationStreamQueue = queue.Queue(maxsize=10)
            self.observationStreamRunning = True
            self.observationStreamStarted = True
            t = threading.Thread(target=self.startObservationStream, daemon=True)
            t.start()

        try:
            while self.observationStreamRunning:
                try:
                    # Use timeout to allow graceful shutdown
                    jpeg_bytes = self.observationStreamQueue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Build proper MJPEG part with Content-Length for better client compatibility
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii")
                )
                yield header + jpeg_bytes + b"\r\n"
        except GeneratorExit:
            self._logger.debug("Observation stream connection closed by client.")
            self.stopObservationStream()

    @APIExport(runOnUIThread=False)
    def observationStream(self, startStream: bool = True):
        """
        Get MJPEG stream from observation camera for sample positioning.
        
        Args:
            startStream: Whether to start the stream (default True)
            
        Returns:
            StreamingResponse with multipart/x-mixed-replace for MJPEG stream
        """
        if not startStream:
            self.stopObservationStream()
            return {"status": "success", "message": "stream stopped"}

        if self.observationCamera is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=409, detail="Observation camera not available")

        from fastapi.responses import StreamingResponse
        headers = {
            # Disable buffering and caching to reduce latency
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }

        return StreamingResponse(
            self.observationStreamer(),
            media_type="multipart/x-mixed-replace;boundary=frame",
            headers=headers,
        )


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
