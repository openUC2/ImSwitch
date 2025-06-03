from datetime import datetime
import time
from fastapi import HTTPException
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as signal
import skimage.transform as transform
import tifffile as tif
from pydantic import BaseModel
from typing import Any, List, Optional, Dict, Any, Union
import os 
import uuid
import os
import time
import threading
import collections
import tifffile as tif
from pathlib import Path


from imswitch.imcommon.framework import Signal
from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext, WorkflowStep, WorkflowsManager
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from pydantic import BaseModel
import numpy as np

try:
    from ashlarUC2 import utils
    from ashlarUC2.scripts.ashlar import process_images
    IS_ASHLAR_AVAILABLE = True
except Exception as e:
    IS_ASHLAR_AVAILABLE = False
    
# Attempt to use OME-Zarr
try:
    from imswitch.imcontrol.controller.controllers.experiment_controller.zarr_data_source import MinimalZarrDataSource
    from imswitch.imcontrol.controller.controllers.experiment_controller.single_multiscale_zarr_data_source import SingleMultiscaleZarrWriter
    IS_OMEZARR_AVAILABLE = False # TODO: True
except Exception as e:
    IS_OMEZARR_AVAILABLE = False

from imswitch.imcontrol.controller.controllers.experiment_controller.OmeTiffStitcher import OmeTiffStitcher

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict
import uuid

# -----------------------------------------------------------
# Reuse the existing sub-models:
# -----------------------------------------------------------
class NeighborPoint(BaseModel):
    x: float
    y: float
    iX: int
    iY: int

class Point(BaseModel):
    id: uuid.UUID
    name: str
    x: float
    y: float
    iX: int = 0
    iY: int = 0
    neighborPointList: List[NeighborPoint]

class ParameterValue(BaseModel):
    illumination: Union[List[str], str] = None # X, Y, nX, nY
    illuIntensities: Union[List[Optional[int]], Optional[int]] = None 
    brightfield: bool = 0,
    darkfield: bool = 0, 
    differentialPhaseContrast: bool = 0,
    timeLapsePeriod: float
    numberOfImages: int
    autoFocus: bool
    autoFocusMin: float
    autoFocusMax: float
    autoFocusStepSize: float
    zStack: bool
    zStackMin: float
    zStackMax: float
    zStackStepSize: Union[List[float], float] = 1.
    exposureTimes: Union[List[float], float] = None
    gains: Union[List[float], float] = None
    resortPointListToSnakeCoordinates: bool = True
    speed: float = 20000.0
    performanceMode: bool = False

class Experiment(BaseModel):
    # From your old "Experiment" BaseModel:
    name: str
    parameterValue: ParameterValue
    pointList: List[Point]

    # From your old "ExperimentModel":
    number_z_steps: int = Field(0, description="Number of Z slices")
    timepoints: int = Field(1, description="Number of timepoints for time-lapse")
    
    # -----------------------------------------------------------
    # A helper to produce the "configuration" dict 
    # -----------------------------------------------------------
    def to_configuration(self) -> dict:
        """
        Convert this Experiment into a dict structure that your Zarr writer or
        scanning logic can easily consume.
        """
        config = {
            "experiment": {
                "MicroscopeState": {
                    "number_z_steps": self.number_z_steps,
                    "timepoints": self.timepoints,
                },
                # TODO: Complete it again
            },
        }
        return config
    
    
class ExperimentWorkflowParams(BaseModel):
    """Parameters for the experiment workflow."""


    # Illumination parameters
    illuSources: List[str] = Field(default_factory=list, description="List of illumination sources")
    illuSourceMinIntensities: List[float] = Field(default_factory=list, description="Minimum intensities for each source")
    illuSourceMaxIntensities: List[float] = Field(default_factory=list, description="Maximum intensities for each source")
    illuIntensities: List[float] = Field(default_factory=list, description="Intensities for each source")

    # Camera parameters
    exposureTimes: List[float] = Field(default_factory=list, description="Exposure times for each source")
    gains: List[float] = Field(default_factory=list, description="gains settings for each source")

    # Feature toggles
    isDPCpossible: bool = Field(False, description="Whether DPC is possible")
    isDarkfieldpossible: bool = Field(False, description="Whether darkfield is possible")

    # timelapse parameters 
    timeLapsePeriodMin: float = Field(0, description="Minimum time for a timelapse series")
    timeLapsePeriodMax: float = Field(100000000, description="Maximum time for a timelapse series in seconds")
    numberOfImagesMin: int = Field(0, description="Minimum time for a timelapse series")
    numberOfImagesMax: int = Field(0, description="Minimum time for a timelapse series")
    autofocusMinFocusPosition: float = Field(-10000, description="Minimum autofocus position")
    autofocusMaxFocusPosition: float = Field(10000, description="Maximum autofocus position")
    autofocusStepSizeMin: float = Field(1, description="Minimum autofocus position")
    autofocusStepSizeMax: float = Field(1000, description="Maximum autofocus position")
    zStackMinFocusPosition: float = Field(0, description="Minimum Z-stack position")
    zStackMaxFocusPosition: float = Field(10000, description="Maximum Z-stack position")
    zStackStepSizeMin: float = Field(1, description="Minimum Z-stack position")
    zStackStepSizeMax: float = Field(1000, description="Maximum Z-stack position")
    performanceMode: bool = Field(False, description="Whether to use performance mode for the experiment - this would be executing the scan on the Cpp hardware directly, not on the Python side.")
    
      

class ExperimentController(ImConWidgetController):
    """Linked to ExperimentWidget."""

    sigExperimentWorkflowUpdate = Signal()
    sigExperimentImageUpdate = Signal(str, np.ndarray, bool, list, bool)  # (detectorName, image, init, scale, isCurrentDetector)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # initialize variables
        self.tWait = 0.1
        self.workflow_manager = WorkflowsManager()
        self.mFilePaths = []
        self.isPreview = False
        
        # set default values
        self.SPEED_Y_default = 20000
        self.SPEED_X_default = 20000
        self.SPEED_Z_default = 20000
        self.ACCELERATION = 500000
        
        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.mDetector = self._master.detectorsManager[allDetectorNames[0]]
        self.isRGB = self.mDetector._camera.isRGB
        self.detectorPixelSize = self.mDetector.pixelSizeUm
        
        # select lasers
        self.allIlluNames = self._master.lasersManager.getAllDeviceNames()+ self._master.LEDMatrixsManager.getAllDeviceNames()
        self.availableIlliminations = []
        for iDevice in self.allIlluNames:
            try:
                # laser maanger
                self.availableIlliminations.append(self._master.lasersManager[iDevice])
            except:
                # lexmatrix manager
                self.availableIlliminations.append(self._master.LEDMatrixsManager[iDevice])
              
        # select stage
        self.allPositionerNames = self._master.positionersManager.getAllDeviceNames()[0]
        try:
            self.mStage = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]
        except:
            self.mStage = None
            
        # stop if some external signal (e.g. memory full is triggered)
        self._commChannel.sigExperimentStop.connect(self.stopExperiment)
            
        # TODO: Adjust parameters 
        # define changeable Experiment parameters as ExperimentWorkflowParams
        self.ExperimentParams = ExperimentWorkflowParams()
        self.ExperimentParams.illuSources = self.allIlluNames
        self.ExperimentParams.illuSourceMinIntensities = [0]*len(self.allIlluNames)
        self.ExperimentParams.illuSourceMaxIntensities = [1023]*len(self.allIlluNames)
        self.ExperimentParams.illuIntensities = [0]*len(self.allIlluNames)
        self.ExperimentParams.exposureTimes = 1000000 # TODO: FIXME
        self.ExperimentParams.gains = [23] # TODO: FIXME
        self.ExperimentParams.isDPCpossible = False
        self.ExperimentParams.isDarkfieldpossible = False
        self.ExperimentParams.performanceMode = False
        
        '''
        For Fast Scanning - Performance Mode -> Parameters will be sent to the hardware directly
        requires hardware triggering
        '''
        # where to dump the TIFFs ----------------------------------------------
        save_dir = dirtools.UserFileDirs.Data
        self.save_dir  = os.path.join(save_dir, "ExperimentController")
        os.mkdir(self.save_dir) if not os.path.exists(self.save_dir) else None

        # writer thread control -------------------------------------------------
        self._writer_thread   = None
        self._stop_writer_evt = threading.Event()
        
        # fast stage scanning parameters ----------------------------------------
        self.fastStageScanIsRunning = False

        
    @APIExport(requestType="GET")
    def getHardwareParameters(self):
        return self.ExperimentParams

    def get_num_xy_steps(self, pointList):
        # we don't consider the center point as this .. well in the center
        if len(pointList) == 0:
            return 1,1
        all_iX = []
        all_iY = []
        for point in pointList:
            all_iX.append(point.iX)
            all_iY.append(point.iY)
        min_iX, max_iX = min(all_iX), max(all_iX)
        min_iY, max_iY = min(all_iY), max(all_iY)

        num_x_steps = (max_iX - min_iX) + 1
        num_y_steps = (max_iY - min_iY) + 1

        return num_x_steps, num_y_steps

    def generate_snake_tiles(self, mExperiment):
        tiles = []
        for iCenter, centerPoint in enumerate(mExperiment.pointList):
            # Collect central and neighbour points (without duplicating the center)
            allPoints = [(n.x, n.y) for n in centerPoint.neighborPointList]
            # Sort by y then by x (i.e., raster order)
            allPoints.sort(key=lambda coords: (coords[1], coords[0]))

            num_x_steps, num_y_steps = self.get_num_xy_steps(centerPoint.neighborPointList)
            allPointsSnake = [0] * (num_x_steps * num_y_steps)
            iTile = 0
            for iY in range(num_y_steps):
                for iX in range(num_x_steps):
                    if iY % 2 == 1 and num_x_steps != 1:
                        mIdex = iY * num_x_steps + num_x_steps - 1 - iX
                    else:
                        mIdex = iTile
                    if len(allPointsSnake) <= mIdex or len(allPoints) <= iTile: 
                        # remove that index from allPointsSnake
                        allPointsSnake[mIdex] = None
                        continue
                    allPointsSnake[mIdex] = {
                        "iterator": iTile,
                        "centerIndex": iCenter,
                        "iX": iX,
                        "iY": iY,
                        "x": allPoints[iTile][0],
                        "y": allPoints[iTile][1],
                    }
                    iTile += 1
            tiles.append(allPointsSnake)
        return tiles

    @APIExport()
    def getLastFilePathsList(self) -> List[str]:
        """
        Returns the last file paths list.
        """
        return self.mFilePaths
        
        

    @APIExport(requestType="POST")
    def startWellplateExperiment(self, mExperiment: Experiment):
        # Extract key parameters
        exp_name = mExperiment.name
        p = mExperiment.parameterValue
        
        # Timelapse-related
        nTimes = p.numberOfImages
        tPeriod = p.timeLapsePeriod
        
        # Z-steps -related
        nZSteps = int((mExperiment.parameterValue.zStackMax-mExperiment.parameterValue.zStackMin)//mExperiment.parameterValue.zStackStepSize)+1
        isZStack = p.zStack
        zStackMin = p.zStackMin
        zStackMax = p.zStackMax
        zStackStepSize = p.zStackStepSize
        
        # Illumination-related
        illuSources = p.illumination
        illuminationIntensites = p.illuIntensities
        if type(illuminationIntensites) is not List  and type(illuminationIntensites) is not list: illuminationIntensites = [p.illuIntensities]
        if type(illuSources) is not List  and type(illuSources) is not list: illuSources = [p.illumination]
        isDarkfield = p.darkfield
        isBrightfield = p.brightfield
        isDPC = p.differentialPhaseContrast
        
        # check if we want to use performance mode
        performanceMode = p.performanceMode
        
        # camera-related
        gains = p.gains
        exposures = p.exposureTimes
        if p.speed <= 0:
            self.SPEED_X = self.SPEED_X_default
            self.SPEED_Y = self.SPEED_Y_default
            self.SPEED_Z = self.SPEED_Z_default
        else: 
            self.SPEED_X = p.speed
            self.SPEED_Y = p.speed
            self.SPEED_Z = p.speed
        
        # Autofocus Related
        isAutoFocus = p.autoFocus
        autofocusMax = p.autoFocusMax
        autofocusMin = p.autoFocusMin
        autofocusStepSize = p.autoFocusStepSize
        
        # pre-check gains/exposures  if they are lists and have same lengths as illuminationsources
        if type(gains) is not List and type(gains) is not list: gains = [gains]
        if type(exposures) is not List and type(exposures) is not list: exposures = [exposures]
        if len(gains) != len(illuSources): gains = [-1]*len(illuSources)
        if len(exposures) != len(illuSources): exposures = [exposures[0]]*len(illuSources)


        # Check if another workflow is running
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        # Start the detector if not already running
        if not self.mDetector._running:
            self.mDetector.startAcquisition()

        # Generate the list of points to scan based on snake scan
        if p.resortPointListToSnakeCoordinates:
            pass # TODO: we need an alternative case 
        snake_tiles = self.generate_snake_tiles(mExperiment)
        # remove none values from all_points list 
        snake_tiles = [[pt for pt in tile if pt is not None] for tile in snake_tiles]
            
        # Generate Z-positions 
        currentZ = self.mStage.getPosition()["Z"]
        if isZStack:
            z_positions = np.arange(zStackMin, zStackMax + zStackStepSize, zStackStepSize) + currentZ
        else:
            z_positions = [currentZ]  # Get current Z position
        minX, maxX, minY, maxY, diffX, diffY = self.computeScanRanges(snake_tiles)
        mPixelSize = self.detectorPixelSize[-1]  # Pixel size in µm

        # Prepare OME-Zarr writer (new)
        # fill ome model
        if IS_OMEZARR_AVAILABLE:
            ome_store = None
            zarr_path = os.path.join(dirPath, mFileName + ".ome.zarr")
            self._logger.debug(f"OME-Zarr path: {zarr_path}")
            if 0:
                ome_store = MinimalZarrDataSource(file_name=zarr_path, mode="w")
                # Configure it from your model
                mExperiment.x_pixels, mExperiment.y_pixels=self.mDetector._shape[-2],self.mDetector._shape[-1]
                mExperiment.number_z_steps = nZSteps
                mExperiment.timepoints = nTimes
                exp_config = mExperiment.to_configuration()
                ome_store.set_metadata_from_configuration_experiment(exp_config)
            else:
                ome_store = SingleMultiscaleZarrWriter(zarr_path, "w")
                # compute max coordinates for x/y
                fovX = int(maxX - minX + diffX)
                fovY = int(maxY - minY + diffY)
                
                ome_store.set_metadata(t=nTimes, c=1, z=nZSteps, bigY=fovY, bigX=fovX)
                ome_store.open_store()

            
        else:
            self._logger.error("OME-ZARR not available or not installed.")

        workflowSteps = []
        step_id = 0

        for t in range(nTimes):
            # Prepare TIFF writer
            timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drivePath = dirtools.UserFileDirs.Data
            dirPath = os.path.join(drivePath, 'recordings', timeStamp)
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            # if performanceMode is True, we will execute on the Hardware directly
            if performanceMode:
                self._logger.debug("Performance mode is enabled. Executing on hardware directly.")
                for snake_tile in snake_tiles:
                    # we need to wait if there is another fast stage scan running
                    while self.fastStageScanIsRunning:
                        self._logger.debug("Waiting for fast stage scan to finish...")
                        time.sleep(0.1) # TODO: Probably we want to do that in a thread since we need to return http response here?
                    # need to compute the pos/net dx and dy and center pos as well as number of images in X / Y 
                    xStart, xEnd, yStart, yEnd, xStep, yStep = self.computeScanRanges([snake_tile])
                    nx, ny = int((xEnd-xStart)//xStep)+1, int((yEnd-yStart)//yStep)+1
                    if len(illuminationIntensites) == 1: illumination0 = illuminationIntensites[0]
                    else: illumination0 = illuminationIntensites[0] if len(illuminationIntensites) > 0 else None
                    if len(illuminationIntensites) == 2: illumination1 = illuminationIntensites[1]
                    else: illumination1 = illuminationIntensites[1] if len(illuminationIntensites) > 1 else None
                    if len(illuminationIntensites) == 3: illumination2 = illuminationIntensites[2]
                    else: illumination2 = illuminationIntensites[2] if len(illuminationIntensites) > 2 else None
                    if len(illuminationIntensites) == 4: illumination3 = illuminationIntensites[3]
                    else: illumination3 = illuminationIntensites[3] if len(illuminationIntensites) > 3 else None
                    led_index = next((i for i, item in enumerate(mExperiment.parameterValue.illumination) if "led" in item.lower()), None)
                    if led_index is not None:
                        # If "LED" is found, get the intensity and limit it to 255
                        led = min(mExperiment.parameterValue.illuIntensities[led_index], 255)
                    else:
                        # Default value if "LED" is not found
                        led = 0
                    if nx>100 or ny>100:
                        self._logger.error("Too many points in X/Y direction. Please reduce the number of points.")
                        raise HTTPException(status_code=400, detail="Too many points in X/Y direction. Please reduce the number of points.")
                    # move to inital position first
                    # xStart/ySTart => 0 means we start from that position 
                    self.startFastStageScanAcquisition(xstart=xStart, xstep=xStep, nx=nx,
                                                        ystart=yStart, ystep=yStep, ny=ny,      
                                                        tsettle=90, tExposure=50, # TODO: make these parameters adjustable
                                                        illumination0=illumination0, illumination1=illumination1,
                                                        illumination2=illumination2, illumination3=illumination3, led=led)
                    # we need to wait until the acquisition is done so that we can start the next one
                return 
            mFilePaths = []
            tiff_writers = []
            for index, experiments_ in enumerate(mExperiment.pointList):
                experimentName = experiments_.name
                mFileName = f"{timeStamp}_{exp_name}"
                mFilePath = os.path.join(dirPath, mFileName + str(index) + "_" + experimentName + "_" + ".ome.tif")
                mFilePaths.append(mFilePath)
                self._logger.debug(f"OME-TIFF path: {mFilePath}")
                #tiff_writer = tif.TiffWriter(mFilePath)
            
                # Create a new OmeTiffStitcher instance for stitching tiles
                tiff_writer = OmeTiffStitcher(mFilePath)
                tiff_writers.append(tiff_writer)


            # Loop over each tile (each central point and its neighbors)
            for position_center_index, tile in enumerate(snake_tiles):
                
                # start storer - store one center point per file # TODO: How about time series?
                self.start_tiff_writer(tiff_writers=tiff_writers, tiff_index=position_center_index)
            
                # iterate over positions     
                for mIndex, mPoint in enumerate(tile):
                    try:
                        name = f"Move to point {mPoint['iterator']}"
                    except Exception:
                        name = f"Move to point {mPoint['x']}, {mPoint['y']}"
                    
                    workflowSteps.append(WorkflowStep(
                        name=name,
                        step_id=step_id,
                        main_func=self.move_stage_xy,
                        main_params={"posX": mPoint["x"], "posY": mPoint["y"], "relative": False},
                    ))
                    
                    # iterate over z-steps
                    for indexZ, iZ in enumerate(z_positions):
                    
                        #move to Z position - but only if we have more than one z position
                        if len(z_positions) > 1 or (len(z_positions) == 1 and mIndex == 0):
                            workflowSteps.append(WorkflowStep(
                                name="Move to Z position",
                                step_id=step_id,
                                main_func=self.move_stage_z,
                                main_params={"posZ": iZ, "relative": False},
                                pre_funcs=[self.wait_time],
                                pre_params={"seconds": 0.1},
                            ))
                            
                        step_id += 1
                        for illuIndex, illuSource in enumerate(illuSources):
                            illuIntensity = illuminationIntensites[illuIndex-1]
                            if illuIntensity <= 0: continue
                            
                            # Turn on illumination - if we have only one source, we can skip this step after the first stop of mIndex
                            if sum(np.array(illuminationIntensites)>0) > 1 or  ( mIndex == 0):
                                workflowSteps.append(WorkflowStep(
                                    name="Turn on illumination",
                                    step_id=step_id,
                                    main_func=self.set_laser_power,
                                    main_params={"power": illuIntensity, "channel": illuSource},
                                    post_funcs=[self.wait_time],
                                    post_params={"seconds": 0.05},
                                ))
                                step_id += 1

                        # Acquire frame
                        isPreview = self.isPreview
                        '''
                        workflowSteps.append(WorkflowStep(
                            name="Acquire frame",
                            step_id=step_id,
                            main_func=self.acquire_frame,
                            main_params={"channel": "Mono"},
                            post_funcs=[self.save_frame_tiff, self.add_image_to_canvas] if not isPreview else [self.append_frame_to_stack, self.add_image_to_canvas],
                            pre_funcs=[self.set_exposure_time_gain],
                            pre_params={"exposure_time": exposure, "gain": gain},
                            # Hier übergeben wir posX, posY an das Metadata-Dict
                            post_params={"posX": mPoint["x"], "posY": mPoint["y"], "minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY},
                        ))
                        '''

                        workflowSteps.append(WorkflowStep(
                            name="Acquire frame",
                            step_id=step_id,
                            main_func=self.acquire_frame,
                            main_params={"channel": "Mono"},
                            post_funcs=[self.save_frame_tiff, self.save_frame_ome_zarr, self.add_image_to_canvas],
                            pre_funcs=[self.set_exposure_time_gain],
                            pre_params={"exposure_time": exposures[illuIndex], "gain": gains[illuIndex]},
                            post_params={
                                "posX": mPoint["x"],
                                "posY": mPoint["y"],
                                "posZ": 0, # TODO: Add Z position if needed
                                "iX": mPoint["iX"],
                                "iY": mPoint["iY"], 
                                "pixel_size": mPixelSize,
                                "minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY,
                                "channel": illuSource,
                                "time_index": t,       # or whatever loop index
                                "tile_index": mIndex,   # or snake-tile index
                                "position_center_index": position_center_index,
                                "isPreview": isPreview,
                            },
                        ))
                        step_id += 1

                        # Turn off illumination
                        if len(illuminationIntensites) > 1 and sum(np.array(illuminationIntensites)>0)>1: # TODO: Is htis the right approach?
                            workflowSteps.append(WorkflowStep(
                                name="Turn off illumination",
                                step_id=step_id,
                                main_func=self.set_laser_power,
                                main_params={"power": 0, "channel": illuSource},
                            ))
                            step_id += 1

                # (Optional) Perform autofocus once per loop, if enabled
                if isAutoFocus:
                    workflowSteps.append(WorkflowStep(
                        name="Autofocus",
                        step_id=step_id,
                        main_func=self.autofocus,
                        main_params={"minZ": autofocusMin, "maxZ": autofocusMax, "stepSize": autofocusStepSize},
                    ))
                    step_id += 1

                step_id += 1

                # Close TIFF writer at the end
                workflowSteps.append(WorkflowStep(
                    name="Close TIFF writer",
                    step_id=step_id,
                    main_func=self.close_tiff_writer,
                    main_params={"tiff_writers": tiff_writers, "tiff_index": position_center_index},
                ))
            
            # Add a wait period between each loop
            workflowSteps.append(WorkflowStep(
                name="Wait for next frame",
                step_id=step_id,
                main_func=self.dummy_main_func,
                main_params={},
                pre_funcs=[self.wait_time],
                pre_params={"seconds": 0.01}
            ))

        if IS_OMEZARR_AVAILABLE:
            # Close OME-Zarr store at the end (new)
            workflowSteps.append(WorkflowStep(
                name="Close OME-Zarr store",
                step_id=step_id,
                main_func=self.close_ome_zarr_store,
                main_params={"omezarr_store": ome_store},
            ))
            step_id += 1
        
        # Emit final canvas
        if self.isPreview:
            workflowSteps.append(WorkflowStep(
                name="Emit Final Canvas",
                step_id=step_id,
                main_func=self.dummy_main_func,
                main_params={},
                pre_funcs=[self.emit_final_canvas],
                pre_params={}
            ))
            step_id += 1

        # Final step: mark done
        workflowSteps.append(WorkflowStep(
            name="Done",
            step_id=step_id,
            main_func=self.dummy_main_func,
            main_params={},
            pre_funcs=[self.wait_time],
            pre_params={"seconds": 0.1}
        ))
        
        # turn off all illuminations
        for illuIndex, illuSource in enumerate(illuSources):
            illuIntensity = illuminationIntensites[illuIndex-1]
            if illuIntensity <= 0: 
                continue
            # Turn off illumination
            workflowSteps.append(WorkflowStep(
                name="Turn off illumination",
                step_id=step_id,
                main_func=self.set_laser_power,
                main_params={"power": 0, "channel": illuSource},
            ))
            
        def sendProgress(payload):
            self.sigExperimentWorkflowUpdate.emit(payload)

        # Create workflow and context
        wf = Workflow(workflowSteps, self.workflow_manager)
        context = WorkflowContext()
        # Set metadata (optional – store whatever data you want)
        context.set_metadata("experimentName", exp_name)
        context.set_metadata("nTimes", nTimes)
        context.set_metadata("tPeriod", tPeriod)

        context.set_object("tiff_writers", tiff_writers)
        if IS_OMEZARR_AVAILABLE:
            context.set_object("omezarr_store", ome_store)

        '''
        image_width, image_height = image_dims[0], image_dims[1]        # physical size of the image in microns
        size = np.array((max_coords[1]+image_height,max_coords[0]+image_width))/pixel_size # size of the final image that contains all tiles in microns
        mshape = np.int32(np.ceil(size)*self.resolution_scale*pixel_size)          # size of the final image in pixels (i.e. canvas)
        self.stitched_image = np.zeros(mshape.T, dtype=np.uint16)       # create a canvas for the stitched image
        ''' 
        
        def compute_canvas_dimensions(minX, maxX, minY, maxY, diffX, diffY, pixelSize):
            width_pixels = int(np.ceil((maxX - minX + diffX) / pixelSize))
            height_pixels = int(np.ceil((maxY - minY + diffY) / pixelSize))
            return width_pixels, height_pixels
        canvas_width, canvas_height = compute_canvas_dimensions(minX, maxX, minY, maxY, diffX, diffY,  mPixelSize)

        if self.isPreview:
            if self.mDetector._isRGB:
                canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            else:
                canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint16) # numpy assigns y,x,z
            context.set_object("canvas", canvas)
        context.on("progress", sendProgress)
        context.on("rgb_stack", sendProgress)
        context.on("rgb_stack", sendProgress)

        # Start the workflow
        self.workflow_manager.start_workflow(wf, context)

        self.mFilePaths = mFilePaths
        return {"status": "running", "store_path": mFilePaths}

    def computeScanRanges(self, snake_tiles):
        # Flatten all point dictionaries from all tiles to compute scan range
        all_points = [pt for tile in snake_tiles for pt in tile]
        minX = min(pt["x"] for pt in all_points)
        maxX = max(pt["x"] for pt in all_points)
        minY = min(pt["y"] for pt in all_points)
        maxY = max(pt["y"] for pt in all_points)
        # compute step between two adjacent points in X/Y
        diffX = np.diff(np.unique([pt["x"] for pt in all_points])).min()
        diffY = np.diff(np.unique([pt["y"] for pt in all_points])).min()
        return minX, maxX, minY, maxY, diffX, diffY




    ########################################
    # Hardware-related functions
    ########################################
    def acquire_frame(self, channel: str):
        self._logger.debug(f"Acquiring frame on channel {channel}")
        mFrame = self.mDetector.getLatestFrame()
        return mFrame
    
    def set_exposure_time_gain(self, exposure_time: float, gain: float, context: WorkflowContext, metadata: Dict[str, Any]):
        if gain and gain >=0:
            self._commChannel.sharedAttrs.sigAttributeSet(['Detector', None, None, "gain"], gain)  # [category, detectorname, ROI1, ROI2] attribute, value
            self._logger.debug(f"Setting gain to {gain}")
        if exposure_time and exposure_time >0:
            self._commChannel.sharedAttrs.sigAttributeSet(['Detector', None, None, "exposureTime"],exposure_time) # category, detectorname, attribute, value
            self._logger.debug(f"Setting exposure time to {exposure_time}")
        
    def dummy_main_func(self):
        self._logger.debug("Dummy main function called")
        return True
 
    def autofocus(self, minZ: float=0, maxZ: float=0, stepSize: float=0):
        self._logger.debug("Performing autofocus... with parameters minZ, maxZ, stepSize: %s, %s, %s", minZ, maxZ, stepSize)
        # TODO: Connect this to the Autofocus Function
        
    def wait_time(self, seconds: int, context: WorkflowContext, metadata: Dict[str, Any]):
        import time
        time.sleep(seconds)

    def save_frame_tiff(self, context: WorkflowContext, metadata: Dict[str, Any], **kwargs):
        # Retrieve the TIFF writer and write the tile
        tiff_writers = context.get_object("tiff_writers")
        if tiff_writers is None:
            self._logger.debug("No TIFF writer found in context!")
            return
        
        
        # get latest image from the camera 
        img = metadata["result"]

        # get metadata
        posX = kwargs.get("posX", 0)
        posY = kwargs.get("posY", 0)
        posZ = kwargs.get("posZ", 0)
        channel_str = kwargs.get("channel", "Mono")
        position_center_index = kwargs.get("position_center_index", 0)
        time_stamp = kwargs.get("time_index", 0)
        iX = kwargs.get("iX", 0)
        iY = kwargs.get("iY", 0)
        pixel_size = kwargs.get("pixel_size", 1.0)  # 1 micron per pixel
        # get tiff writer
        tiff_writer = tiff_writers[position_center_index]
        try:
            tiff_writer.add_image(
                image=img,
                position_x=posX,
                position_y=posY,
                index_x=iX,
                index_y=iY,
                pixel_size=pixel_size
            )
            metadata["frame_saved"] = True
        except Exception as e:
            self._logger.error(f"Error saving TIFF: {e}")
            metadata["frame_saved"] = False

        '''
        if tiff_writer is None:
            self._logger.debug("No TIFF writer found in context!")
            return
        img = metadata["result"]
        # append the image to the tiff file
        try:
            tiff_writer.write(img)
            metadata["frame_saved"] = True
        except Exception as e:
            self._logger.error(f"Error saving TIFF: {e}")
            metadata["frame_saved"] = False
        '''
        
    def close_ome_zarr_store(self, omezarr_store):
        # If you need to do anything special (like flush) for the store, do it here.
        # Otherwise, Zarr’s FS-store or disk-store typically closes on its own.
        # This function can be effectively a no-op if you do not require extra steps.
        try:
            if omezarr_store:
                omezarr_store.close()
            else:
                self._logger.debug("OME-Zarr store not found in context.")
            return
        except Exception as e:
            self._logger.error(f"Error closing OME-Zarr store: {e}")
            raise e
    
    def save_frame_ome_zarr(self, context: Dict[str, Any], metadata: Dict[str, Any], **kwargs):
        """
        Saves a single frame (tile) to an OME-Zarr store, handling coordinate transformation mismatch.

        Args:
            context: A dictionary containing the OME-Zarr store and other relevant data.
            metadata: A dictionary containing the image data and other metadata.
            **kwargs: Additional keyword arguments, including tile position, channel, etc.
        """
        if not IS_OMEZARR_AVAILABLE:
            # self._logger.error("OME-Zarr is not available.")
            return
        omeZarrStore = context.get_object("omezarr_store")
        if omeZarrStore is None:
            raise ValueError("OME-Zarr store not found in context.")

        img = metadata.get("result")
        if img is None:
            return

        posX = kwargs.get("posX", 0)
        posY = kwargs.get("posY", 0)
        posZ = kwargs.get("posZ", 0)
        channel_str = kwargs.get("channel", "Mono")
        
        # 3) Write the frame with stage coords:
        if 0:
            omeZarrStore.write(img, x=posX, y=posY, z=posZ)
        else:
            omeZarrStore.write_tile(img, t=0, c=0, z=0, y_start=posY, x_start=posX)

        time.sleep(0.01)

    def add_image_to_canvas(self, context: WorkflowContext, metadata: Dict[str, Any], emitCurrentProgress=False, **kwargs):
        # Retrieve the canvas and add the image
        isPreview = kwargs.get("isPreview", False)
        if isPreview:
            self._logger.debug("Preview mode is enabled. Skipping canvas update.")
            return
        canvas = context.get_object("canvas")
        if canvas is None:
            self._logger.debug("No canvas found in context!")
            return        
        img = metadata["result"] # After acquire_frame runs, its return (the frame) goes into metadata["result"]. Then each post-func—save_frame_tiff, add_image_to_canvas, etc.—can read it from metadata["result"].
        if img is None:
            self._logger.debug("No image found in metadata!")
            return
        posX, posY = kwargs['posX'], kwargs['posY']
        minX,minY,maxX,maxY = kwargs['minX'], kwargs['minY'], kwargs['maxX'], kwargs['maxY']
        mPixelSize = self.detectorPixelSize
        posPixels = ((((posY-minY)/mPixelSize[-1]), (posX-minX)/mPixelSize[-1]))
        utils.paste(canvas, img, posPixels, np.maximum)
        context.set_object("canvas", canvas)
        metadata["frame_saved"] = True
        if emitCurrentProgress:
            #Signal(str, np.ndarray, bool, list, bool)  # (detectorName, image, init, scale, isCurrentDetector)
            def mSendCanvas(canvas, init, scale):
                mCanvas = np.copy(canvas)
                self.sigExperimentImageUpdate.emit("canvas", mCanvas, init, scale, True)
            import threading
            threading.Thread(target=mSendCanvas, args=(canvas, False, 1)).start()

    def emit_final_canvas(self, context: WorkflowContext, metadata: Dict[str, Any]):
        final_canvas = context.get_object("canvas")
        if final_canvas is not None:
            self.sigExperimentImageUpdate.emit("canvas", final_canvas, True, 1, 0)  
            tif.imwrite("final_canvas.tif", final_canvas)  
        else:
            print("No final canvas found!")
            
    def append_frame_to_stack(self, context: WorkflowContext, metadata: Dict[str, Any]):
        # Retrieve the stack writer and append the frame
        # if we have one stack "full", we need to reset the stack
        stack_writer = context.get_object("stack_writer")
        if stack_writer is None:
            self._logger.debug("No stack writer found in context!")
            return
        img = metadata["result"]
        stack_writer.append(img)
        context.set_object("stack_writer", stack_writer)
        metadata["frame_saved"] = True
        
    def emit_rgb_stack(self, context: WorkflowContext, metadata: Dict[str, Any]):
        stack_writer = context.get_object("stack_writer")
        if stack_writer is None:
            self._logger.debug("No stack writer found in context!")
            return
        rgb_stack = np.stack(stack_writer, axis=-1)
        context.emit_event("rgb_stack", {"rgb_stack": rgb_stack})
        stack_writer = []
        context.set_object("stack_writer", stack_writer)
        
    def set_laser_power(self, power: float, channel: str):
        if channel not in self.allIlluNames:
            self._logger.error(f"Channel {channel} not found in available lasers: {self.allIlluNames}")
            return None
        self._master.lasersManager[channel].setValue(power)
        if self._master.lasersManager[channel].enabled == 0:
            self._master.lasersManager[channel].setEnabled(1)
        self._logger.debug(f"Setting laser power to {power} for channel {channel}")
        return power

    def close_tiff_writer(self, tiff_writers, tiff_index):
        if tiff_writers is not None:
            tiff_writer = tiff_writers[tiff_index]
            tiff_writer.close()
        else:
            raise ValueError("TIFF writer is not initialized.")

    def start_tiff_writer(self, tiff_writers, tiff_index):
        if tiff_writers is not None:
            tiff_writer = tiff_writers[tiff_index]
            tiff_writer.start()
        else:
            raise ValueError("TIFF writer is not initialized.")
    
    def move_stage_xy(self, posX: float = None, posY: float = None, relative: bool = False):
        # {"task":"/motor_act",     "motor":     {         "steppers": [             { "stepperid": 1, "position": -1000, "speed": 30000, "isabs": 0, "isaccel":1, "isen":0, "accel":500000}     ]}}
        self._logger.debug(f"Moving stage to X={posX}, Y={posY}")
        #if posY and posX is None:
            
        self.mStage.move(value=(posX, posY), speed=(self.SPEED_X, self.SPEED_Y), axis="XY", is_absolute=not relative, is_blocking=True, acceleration=self.ACCELERATION)
        #newPosition = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([posX, posY])
        return (posX, posY)

    def move_stage_z(self, posZ: float, relative: bool = False):
        self._logger.debug(f"Moving stage to Z={posZ}")
        self.mStage.move(value=posZ, speed=self.SPEED_Z, axis="Z", is_absolute=not relative, is_blocking=True)
        newPosition = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([newPosition["Z"]])
        return newPosition["Z"]

            
    @APIExport()
    def pauseWorkflow(self):
        status = self.workflow_manager.get_status()["status"]
        if status == "running":
            return self.workflow_manager.pause_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot pause in current state: {status}")

    @APIExport()
    def resumeExperiment(self):
        status = self.workflow_manager.get_status()["status"]
        if status == "paused":
            return self.workflow_manager.resume_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot resume in current state: {status}")

    @APIExport()
    def stopExperiment(self):
        status = self.workflow_manager.get_status()["status"]
        if status in ["running", "paused", "stopping"]:
            return self.workflow_manager.stop_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot stop in current state: {status}")

    @APIExport()
    def getExperimentStatus(self):
        return self.workflow_manager.get_status()
    
    @APIExport()
    def forceStopExperiment(self):
        self.workflow_manager.stop_workflow()
        del self.workflow_manager
        self.workflow_manager = WorkflowsManager()
        
        
    """Couples a 2‑D stage scan with external‑trigger camera acquisition.

    • Puts the connected ``CameraHIK`` into *external* trigger mode
      (one exposure per TTL rising edge on LINE0).
    • Runs ``positioner.start_stage_scanning``.
    • Pops every frame straight from the camera ring‑buffer and writes it to
      disk as ``000123.tif`` (frame‑id used as filename).

    Assumes the micro‑controller (or the positioner itself) raises a TTL pulse
    **after** arriving at each grid co‑ordinate.
    """


    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------
    @APIExport(runOnUIThread=False)
    def startFastStageScanAcquisition(self,
                      xstart:float=0, xstep:float=500, nx:int=10,
                      ystart:float=0, ystep:float=500, ny:int=10,
                      tsettle:float=90, tExposure:float=50,
                      illumination0:int=None, illumination1:int=None,
                      illumination2:int=None, illumination3:int=None, led:float=None):
        """Full workflow: arm camera ➔ launch writer ➔ execute scan."""
        self.fastStageScanIsRunning = True
        self._stop() # ensure all prior runs are stopped
        self.move_stage_xy(posX=xstart, posY=ystart, relative=False)

        # 1. prepare camera ----------------------------------------------------
        self.mDetector.stopAcquisition()
        #self.mDetector.NBuffer        = total_frames + 32   # head‑room
        #self.mDetector.frame_buffer   = collections.deque(maxlen=self.mDetector.NBuffer)
        #self.mDetector.frameid_buffer = collections.deque(maxlen=self.mDetector.NBuffer)
        self.mDetector.setTriggerSource("External trigger")
        self.mDetector.flushBuffers()
        self.mDetector.startAcquisition()

        # compute the metadata for the stage scan (e.g. x/y coordinates and illumination channels)
        # stage will start at xstart, ystart and move in steps of xstep, ystep in snake scan logic 
        
        illum_dict = {
            "illumination0": illumination0,
            "illumination1": illumination1,
            "illumination2": illumination2,
            "illumination3": illumination3, 
            "led": led
        }

        # Count how many illumination entries are valid (not None)
        nIlluminations = sum(val is not None and val > 0 for val in illum_dict.values())        
        nScan = max(nIlluminations, 1)
        total_frames = nx * ny * nScan
        self._logger.info(f"Stage-scan: {nx}×{ny} ({total_frames} frames)")
        def addDataPoint(metadataList, x, y, illuminationChannel, illuminationValue, runningNumber):
            """Helper function to add metadata for each position."""
            metadataList.append({
                "x": x,
                "y": y,
                "illuminationChannel": illuminationChannel,
                "illuminationValue": illuminationValue,
                "runningNumber": runningNumber
            })
            return metadataList
        metadataList = []
        runningNumber = 0
        for iy in range(ny):
            for ix in range(nx):
                x = xstart + ix * xstep
                y = ystart + iy * ystep
                # Snake pattern
                if iy % 2 == 1:
                    x = xstart + (nx - 1 - ix) * xstep

                # If there's at least one valid illumination or LED set, take only one image as "default"
                if nIlluminations == 0:
                    runningNumber += 1
                    addDataPoint(metadataList, x, y, "default", -1, runningNumber)
                else:
                    # Otherwise take an image for each illumination channel > 0
                    for channel, value in illum_dict.items():
                        if value is not None and value > 0:
                            runningNumber += 1
                            addDataPoint(metadataList, x, y, channel, value, runningNumber)
        # 2. start writer thread ----------------------------------------------
        self._stop_writer_evt.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(total_frames,metadataList,),
            daemon=True,
        )
        self._writer_thread.start()
        illumination=(illumination0, illumination1, illumination2, illumination3) if nIlluminations > 0 else (0,0,0,0)
        # 3. execute stage scan (blocks until finished) ------------------------
        self.mStage.start_stage_scanning(
            xstart=0, xstep=xstep, nx=nx,
            ystart=0, ystep=ystep, ny=ny,
            tsettle=tsettle, tExposure=tExposure,
            illumination=illumination, led=led,
        )

        
    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------
    def _writer_loop(self, n_expected: int, metadataList: list, minPeriod: float = 0.2):
        """
        Bulk-writer that uses the (frames, ids) tuple returned by camera.getChunk().
        The call is non-blocking; it returns empty arrays until new frames arrive.
        """
        saved = 0
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mFilePath = os.path.join(self.save_dir, timeStamp + "_FastStageScan")
        # create directory if it does not exist
        if not os.path.exists(mFilePath):
            os.makedirs(mFilePath)
        self._logger.info(f"Writer thread started, saving to {mFilePath}...")
        tLastCapture = time.time()  # for throttling
        while saved < n_expected and not self._stop_writer_evt.is_set():

            frames, ids = self.mDetector.getChunk()        # ← empties camera buffer

            if frames.size == 0:                        # nothing new yet
                time.sleep(0.005)
                continue

            for frame, fid in zip(frames, ids):
                currentMetadata = metadataList[fid] if fid < len(metadataList) else {}
                if currentMetadata == {}:
                    self._logger.warning(f"Metadata for frame {fid} is empty. Skipping this frame.")
                    break
                fileName = os.path.join(mFilePath,  f"FastScan_{currentMetadata['runningNumber']}_x_{round(currentMetadata['x'], 1)}_y_{round(currentMetadata['y'], 1)}_illu-{currentMetadata['illuminationChannel']}_{currentMetadata['illuminationValue']}{fid:06d}.tif")
                tif.imwrite(fileName, frame)
                saved += 1
                self._logger.debug(f"saved {saved}/{n_expected} under {fileName}")

        self._logger.info(f"Writer thread finished ({saved} images).")
        # 4. wait for writer to finish then stop camera ------------------------

        self._logger.info("Grid‑scan completed and all images saved.")

        # 5. clean up and bring camera back to normal mode -----------------
        self.mDetector.stopAcquisition()
        self.mDetector.setTriggerSource("Continuous")
        self.mDetector.flushBuffers()
        self.mDetector.startAcquisition()
        self._logger.info("Camera reset to continuous mode.")
        self.fastStageScanIsRunning = False


    def _stop(self):
        """Abort the acquisition gracefully."""
        self._stop_writer_evt.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2)
        self.mDetector.stopAcquisition()

    @APIExport(runOnUIThread=False)
    def stopFastStageScanAcquisition(self):
        """Stop the stage scan acquisition and writer thread."""
        self.mStage.stop_stage_scanning()
        self.fastStageScanIsRunning = False
        self._logger.info("Stopping stage scan acquisition...")
        self._stop()
        self._logger.info("Stage scan acquisition stopped.")

# Copyright (C) 2025 Benedict Diederich
