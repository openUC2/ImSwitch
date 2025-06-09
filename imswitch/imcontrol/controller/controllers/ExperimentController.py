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
import zarr, numcodecs
from fastapi.responses import FileResponse

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
    IS_OMEZARR_AVAILABLE = True # TODO: True
except Exception as e:
    IS_OMEZARR_AVAILABLE = False

from imswitch.imcontrol.controller.controllers.experiment_controller.OmeTiffStitcher import OmeTiffStitcher
from imswitch.imcontrol.controller.controllers.experiment_controller.ome_writer import OMEWriter, OMEWriterConfig

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict
import uuid

# -----------------------------------------------------------
# Reuse the existing sub-models:
# -----------------------------------------------------------

class OMEFileStorePaths:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tiff_dir  = os.path.join(base_dir, "tiles")
        self.zarr_dir  = os.path.join(base_dir+".ome.zarr")      # ‹scan›_FastStageScan.ome.zarr
        os.makedirs(self.tiff_dir) if not os.path.exists(self.tiff_dir) else None        

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
    ome_write_tiff: bool = Field(False, description="Whether to write OME-TIFF files")
    ome_write_zarr: bool = Field(True, description="Whether to write OME-Zarr files")
    ome_write_stitched_tiff: bool = Field(True, description="Whether to write stitched OME-TIFF files")
        
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
    sigUpdateOMEZarrStore = Signal(dict)  

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # initialize variables
        self.tWait = 0.1
        self.workflow_manager = WorkflowsManager()

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
        self.ExperimentParams.illuSourceMinIntensities = []
        self.ExperimentParams.illuSourceMaxIntensities = []
        self.ExperimentParams.illuIntensities = [0]*len(self.allIlluNames)
        self.ExperimentParams.exposureTimes = [0]*len(self.allIlluNames) 
        self.ExperimentParams.gains = [0]*len(self.allIlluNames) 
        self.ExperimentParams.isDPCpossible = False
        self.ExperimentParams.isDarkfieldpossible = False
        self.ExperimentParams.performanceMode = False
        for laserN in self.availableIlliminations:
            self.ExperimentParams.illuSourceMinIntensities.append(laserN.valueRangeMin)
            self.ExperimentParams.illuSourceMaxIntensities.append(laserN.valueRangeMax)
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
        self._writer_thread_ome = None
        self._current_ome_writer = None  # For normal mode OME writing
        self._stop_writer_evt = threading.Event()

        # fast stage scanning parameters ----------------------------------------
        self.fastStageScanIsRunning = False

        # OME writer configuration -----------------------------------------------
        self._ome_write_tiff = False
        self._ome_write_zarr = True
        self._ome_write_stitched_tiff = True

    @APIExport(requestType="GET")
    def getHardwareParameters(self):
        return self.ExperimentParams

    @APIExport(requestType="GET")
    def getOMEWriterConfig(self):
        """Get current OME writer configuration."""
        return {
            "write_tiff": getattr(self, '_ome_write_tiff', False),
            "write_zarr": getattr(self, '_ome_write_zarr', True),
            "write_stitched_tiff": getattr(self, '_ome_write_stitched_tiff', True)
        }


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
    def getLastScanAsOMEZARR(self): 
        """ Returns the last OME-Zarr folder as a zipped file for download. """ 
        try:
            return self.getOmeZarrUrl()
        except Exception as e:
            self._logger.error(f"Error while getting last scan as OME-Zarr: {e}")
            raise HTTPException(status_code=500, detail="Error while getting last scan as OME-Zarr.")

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
        illuminationIntensities = p.illuIntensities
        if type(illuminationIntensities) is not List  and type(illuminationIntensities) is not list: illuminationIntensities = [p.illuIntensities]
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

        # Prepare directory and filename for saving
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        drivePath = dirtools.UserFileDirs.Data
        dirPath = os.path.join(drivePath, 'ExperimentController', timeStamp)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        mFileName = f"{timeStamp}_{exp_name}"

        workflowSteps = []
        step_id = 0
        file_writers = []  # Initialize outside the loop for context storage

        def performanceMode():
            self._logger.debug("Performance mode is enabled. Executing on hardware directly.")
            for snake_tile in snake_tiles:
                # we need to wait if there is another fast stage scan running
                while self.fastStageScanIsRunning:
                    self._logger.debug("Waiting for fast stage scan to finish...")
                    time.sleep(0.1) # TODO: Probably we want to do that in a thread since we need to return http response here?
                # need to compute the pos/net dx and dy and center pos as well as number of images in X / Y
                xStart, xEnd, yStart, yEnd, xStep, yStep = self.computeScanRanges([snake_tile])
                if xStep == 0:
                    nx = 1
                else:
                    nx = int((xEnd-xStart)//xStep)+1
                if yStep == 0:
                    ny = 1
                else:
                    ny = int((yEnd-yStart)//yStep)+1
                if len(illuminationIntensities) == 1: illumination0 = illuminationIntensities[0]
                else: illumination0 = illuminationIntensities[0] if len(illuminationIntensities) > 0 else None
                if len(illuminationIntensities) == 2: illumination1 = illuminationIntensities[1]
                else: illumination1 = illuminationIntensities[1] if len(illuminationIntensities) > 1 else None
                if len(illuminationIntensities) == 3: illumination2 = illuminationIntensities[2]
                else: illumination2 = illuminationIntensities[2] if len(illuminationIntensities) > 2 else None
                if len(illuminationIntensities) == 4: illumination3 = illuminationIntensities[3]
                else: illumination3 = illuminationIntensities[3] if len(illuminationIntensities) > 3 else None
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
                                                    illumination2=illumination2, illumination3=illumination3, led=led,
                                                    tPeriod=tPeriod, nTimes=nTimes)
                # we need to wait until the acquisition is done so that we can start the next one
            return
        def normalMode():                
                ''' THIS IS THE NORMAL MODE:
                We will move the stage to each point, set the illumination, acquire the frame and save it 
                using the unified OME writer (TIFF stitching + OME-Zarr).
                '''

                # Set up all OME writers at once (similar to original file_writers approach)
                for position_center_index, tiles in enumerate(snake_tiles):
                    experimentName = f"{t}_{exp_name}_{position_center_index}"
                    mFilePath = os.path.join(dirPath, mFileName + str(position_center_index) + "_" + experimentName + "_" + ".ome.tif")
                    self._logger.debug(f"OME-TIFF path: {mFilePath}")
                    
                    # Create a new OMEWriter instance for this tile
                    file_paths = OMEFileStorePaths(mFilePath.replace(".ome.tif", ""))
                    tile_shape = (self.mDetector._shape[-1], self.mDetector._shape[-2])  # (height, width)
                    
                    # Calculate grid parameters from tile
                    all_points = []
                    for point in tiles:
                        if point is not None:
                            all_points.append([point["x"], point["y"]])
                    
                    if all_points:
                        x_coords = [p[0] for p in all_points]
                        y_coords = [p[1] for p in all_points]
                        x_start, x_end = min(x_coords), max(x_coords)
                        y_start, y_end = min(y_coords), max(y_coords)
                        unique_x = sorted(set(x_coords))
                        unique_y = sorted(set(y_coords))
                        x_step = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 100.0
                        y_step = unique_y[1] - unique_y[0] if len(unique_y) > 1 else 100.0
                        nx, ny = len(unique_x), len(unique_y)
                        grid_shape = (nx, ny)
                        grid_geometry = (x_start, y_start, x_step, y_step)
                    else:
                        grid_shape = (1, 1)
                        grid_geometry = (0, 0, 100, 100)
                    
                    writer_config = OMEWriterConfig(
                        write_tiff=self._ome_write_tiff,
                        write_zarr=self._ome_write_zarr,
                        write_stitched_tiff=self._ome_write_stitched_tiff,
                        min_period=0.1,  # Faster for normal mode
                        pixel_size=self.detectorPixelSize[-1] if hasattr(self, 'detectorPixelSize') else 1.0,
                        n_time_points=1,
                        n_z_planes=len(z_positions),
                        n_channels = sum(np.array(illuminationIntensities) > 0) # number of illumination sources with intensities >0 
                    )
                    
                    ome_writer = OMEWriter(
                        file_paths=file_paths,
                        tile_shape=tile_shape,
                        grid_shape=grid_shape,
                        grid_geometry=grid_geometry,
                        config=writer_config,
                        logger=self._logger
                    )
                    file_writers.append(ome_writer)

                # Loop over each tile (each central point and its neighbors)
                for position_center_index, tiles in enumerate(snake_tiles):
                    
                    # iterate over positions
                    for mIndex, mPoint in enumerate(tiles):
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
                                illuIntensity = illuminationIntensities[illuIndex-1]
                                if illuIntensity <= 0: continue

                                # Turn on illumination - if we have only one source, we can skip this step after the first stop of mIndex
                                if sum(np.array(illuminationIntensities)>0) > 1 or  ( mIndex == 0):
                                    workflowSteps.append(WorkflowStep(
                                        name="Turn on illumination",
                                        step_id=step_id,
                                        main_func=self.set_laser_power,
                                        main_params={"power": illuIntensity, "channel": illuSource},
                                        post_funcs=[self.wait_time],
                                        post_params={"seconds": 0.05},
                                    ))
                                    step_id += 1
                                else:
                                    self._logger.debug(f"Skipping illumination {illuSource} as it is already on.")
                                    continue

                                # Acquire frame
                                workflowSteps.append(WorkflowStep(
                                    name="Acquire frame",
                                    step_id=step_id,
                                    main_func=self.acquire_frame,
                                    main_params={"channel": "Mono"},
                                    post_funcs=[self.save_frame_ome], #, self.add_image_to_canvas],
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
                                        "runningNumber": step_id,  # Add running number for compatibility
                                        "illuminationChannel": illuSource,
                                        "illuminationValue": illuminationIntensities[illuIndex],
                                    },
                                ))
                                step_id += 1

                                # Turn off illumination
                                if len(illuminationIntensities) > 1 and sum(np.array(illuminationIntensities)>0)>1: # TODO: Is htis the right approach?
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

                    # Finalize OME writer for this tile
                    workflowSteps.append(WorkflowStep(
                        name=f"Finalize OME writer for tile {position_center_index}",
                        step_id=step_id,
                        main_func=self.dummy_main_func,  # Placeholder for any pre-finalization logic
                        main_params={},
                        post_funcs=[self.finalize_tile_ome_writer],
                        post_params={"tile_index": position_center_index},
                    ))
                    step_id += 1

                # Add a wait period between each loop
                workflowSteps.append(WorkflowStep(
                    name="Wait for next frame",
                    step_id=step_id,
                    main_func=self.dummy_main_func,
                    main_params={},
                    pre_funcs=[self.wait_time],
                    pre_params={"seconds": 0.01}
                ))

                # Finalize OME writer at the end
                workflowSteps.append(WorkflowStep(
                    name="Finalize OME writer",
                    step_id=step_id,
                    main_func=self.finalize_current_ome_writer,
                    main_params={},
                ))
                step_id += 1

                # turn off all illuminations
                for illuIndex, illuSource in enumerate(illuSources):
                    illuIntensity = illuminationIntensities[illuIndex-1]
                    if illuIntensity <= 0:
                        continue
                    # Turn off illumination
                    workflowSteps.append(WorkflowStep(
                        name="Turn off illumination",
                        step_id=step_id,
                        main_func=self.set_laser_power,
                        main_params={"power": 0, "channel": illuSource},
                    ))

                # Final step: mark done
                workflowSteps.append(WorkflowStep(
                    name="Done",
                    step_id=step_id,
                    main_func=self.dummy_main_func,
                    main_params={},
                    pre_funcs=[self.wait_time],
                    pre_params={"seconds": tPeriod}  # Wait for the time period before next iteration
                ))
            
        
        if performanceMode and hasattr(self.mStage, "start_stage_scanning") and hasattr(self.mDetector, "setTriggerSource"):
            performanceMode() # TODO: We should return immediately 
        else:
            for t in range(nTimes):
                '''
                THIS IS THE PERFORMANCE MODE:
                The microcontroller will move the stage in a grid and triggers the camera, 
                ImSwitch listens to the camera and stores the images in a OME-Zarr format.
                The microcontroller will also handle the illumination and the Z-positioning.
                The microcontroller will not handle the autofocus if enabled.
                
                Prepare TIFF writer - reuse timeStamp, dirPath from above
                if performanceMode is True, we will execute on the Hardware directly
                '''
                normalMode()
            # If we are in performance mode, we will not use the OME writer
            def sendProgress(payload):
                self.sigExperimentWorkflowUpdate.emit(payload)

            # Create workflow and context
            wf = Workflow(workflowSteps, self.workflow_manager)
            context = WorkflowContext()
            # Set metadata (optional – store whatever data you want)
            context.set_metadata("experimentName", exp_name)
            context.set_metadata("nTimes", nTimes)
            context.set_metadata("tPeriod", tPeriod)

            # Store file_writers in context if they were created (non-performance mode)
            if len(file_writers) > 0:
                context.set_object("file_writers", file_writers)
            context.on("progress", sendProgress)
            context.on("rgb_stack", sendProgress)
            context.on("rgb_stack", sendProgress)

            # Start the workflow
            self.workflow_manager.start_workflow(wf, context)

        return {"status": "running"}

    def computeScanRanges(self, snake_tiles):
        # Flatten all point dictionaries from all tiles to compute scan range
        all_points = [pt for tile in snake_tiles for pt in tile]
        minX = min(pt["x"] for pt in all_points)
        maxX = max(pt["x"] for pt in all_points)
        minY = min(pt["y"] for pt in all_points)
        maxY = max(pt["y"] for pt in all_points)
        # compute step between two adjacent points in X/Y
        uniqueX = np.unique([pt["x"] for pt in all_points])
        uniqueY = np.unique([pt["y"] for pt in all_points])
        if len(uniqueX) == 1:
            diffX = 0
        else:
            diffX = np.diff(uniqueX).min()
        if len(uniqueY) == 1:
            diffY = 0
        else:
            diffY = np.diff(uniqueY).min()
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

    def save_frame_ome(self, context: WorkflowContext, metadata: Dict[str, Any], **kwargs):
        """
        Saves a single frame using the unified OME writer (both stitched TIFF and OME-Zarr).
        
        Args:
            context: WorkflowContext containing relevant data.
            metadata: A dictionary containing the image data and other metadata.
            **kwargs: Additional keyword arguments, including tile position, channel, etc.
        """
        # Get the latest image from the camera
        img = metadata.get("result")
        if img is None:
            self._logger.debug("No image found in metadata!")
            return
        
        # Get tile index to identify the correct OME writer
        position_center_index = kwargs.get("position_center_index")
        if position_center_index is None:
            self._logger.error("No position_center_index provided for OME writer lookup")
            metadata["frame_saved"] = False
            return
        
        # Prepare metadata for OME writer
        ome_metadata = {
            "x": kwargs.get("posX", 0),
            "y": kwargs.get("posY", 0),
            "z": kwargs.get("posZ", 0),
            "runningNumber": kwargs.get("runningNumber", 0),
            "illuminationChannel": kwargs.get("illuminationChannel", "unknown"),
            "illuminationValue": kwargs.get("illuminationValue", 0),
            "tile_index": kwargs.get("tile_index", 0),
            "time_index": kwargs.get("time_index", 0),
            "z_index": kwargs.get("z_index", 0),
            "channel_index": kwargs.get("channel_index", 0),
        }
        
        try:
            # Get file_writers list from context
            file_writers = context.get_object("file_writers")
            if file_writers is None or position_center_index >= len(file_writers):
                self._logger.error(f"No OME writer found for tile index {position_center_index}")
                metadata["frame_saved"] = False
                return
            
            # Write frame using the specific OME writer from the list
            ome_writer = file_writers[position_center_index]
            chunk_info = ome_writer.write_frame(img, ome_metadata)
            self.setOmeZarrUrl(ome_writer.store.split(dirtools.UserFileDirs.Data)[-1])  # Update OME-Zarr URL in context
            # Emit signal for frontend updates if Zarr chunk was written
            if chunk_info and "rel_chunk" in chunk_info:
                sigZarrDict = {
                    "event": "zarr_chunk",
                    "path": chunk_info["rel_chunk"],
                    "zarr": str(self.getOmeZarrUrl())
                }
                self.sigUpdateOMEZarrStore.emit(sigZarrDict)
            
            metadata["frame_saved"] = True
        except Exception as e:
            self._logger.error(f"Error saving OME frame: {e}")
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
            # TODO: This is not working as the posY and posX are in microns, but the OME-Zarr store expects pixel coordinates.
            # Convert to pixel coordinates
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

    def setOmeZarrUrl(self, url):
        """Set the OME-Zarr URL for the experiment."""
        self._omeZarrUrl = url
        self._logger.info(f"OME-Zarr URL set to: {self._omeZarrUrl}")
        
    def getOmeZarrUrl(self):
        """Get the OME-Zarr URL for the experiment."""
        if self._omeZarrUrl is None:
            raise ValueError("OME-Zarr URL is not set.")
        return self._omeZarrUrl

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------
    @APIExport(runOnUIThread=False)
    def startFastStageScanAcquisition(self,
                      xstart:float=0, xstep:float=500, nx:int=10,
                      ystart:float=0, ystep:float=500, ny:int=10,
                      tsettle:float=90, tExposure:float=50,
                      illumination0:int=None, illumination1:int=None,
                      illumination2:int=None, illumination3:int=None, led:float=None,
                      tPeriod:int=1, nTimes:int=1):
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
        nLastTime = time.time() 
        for iTime in range(nTimes):
            saveOMEZarr = True; 
            nTimePoints = 1  # For now, we assume a single time point
            nZPlanes = 1  # For now, we assume a single Z plane    
            if saveOMEZarr:
                # ------------------------------------------------------------------+
                # 2. open OME-Zarr canvas                                           |
                # ──────────────────────────────────────────────────────────────────+
                timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.mFilePath = os.path.join(self.save_dir,  f"{timeStamp}_FastStageScan")
                # create directory if it does not exist and file paths
                omezarr_store = OMEFileStorePaths(self.mFilePath)
                self.setOmeZarrUrl(self.mFilePath.split(dirtools.UserFileDirs.Data)[-1]+".ome.zarr")
                self._writer_thread_ome = threading.Thread(
                    target=self._writer_loop_ome, args=(omezarr_store, total_frames, metadataList, xstart, ystart, xstep, ystep, nx, ny, 0, nTimePoints, nZPlanes, nIlluminations),
                    daemon=True)
                self._stop_writer_evt.clear()            
                self._writer_thread_ome.start()
            else:    
                # Non-performance mode: also use _writer_loop_ome but configure for TIFF stitching
                self._stop_writer_evt.clear()
                self._writer_thread_ome = threading.Thread(
                    target=self._writer_loop_ome, 
                    args=(omezarr_store, total_frames, metadataList, xstart, ystart, xstep, ystep, nx, ny, 0.2, True, True, False, nTimePoints, nZPlanes, nIlluminations),  # is_tiff=True, write_stitched_tiff=True, is_performance_mode=False
                    daemon=True
                )
                self._writer_thread_ome.start()
            illumination=(illumination0, illumination1, illumination2, illumination3) if nIlluminations > 0 else (0,0,0,0)
            # 3. execute stage scan (blocks until finished) ------------------------
            self.mStage.start_stage_scanning(
                xstart=0, xstep=xstep, nx=nx,
                ystart=0, ystep=ystep, ny=ny,
                tsettle=tsettle, tExposure=tExposure,
                illumination=illumination, led=led,
            )
            #TODO: Make path more uniform - e.g. basetype 
            while nLastTime + tPeriod > time.time() and self.fastStageScanIsRunning:
                time.sleep(0.1)
        return self.getOmeZarrUrl()  # return relative path to the data directory

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------
    def _writer_loop_ome(
        self,
        mFilePath: OMEFileStorePaths,
        n_expected: int,
        metadata_list: list[dict],
        x_start: float,
        y_start: float,
        x_step: float,
        y_step: float,
        nx: int,
        ny: int,
        min_period: float = 0.2,
        is_tiff: bool = False,
        write_stitched_tiff: bool = True,  # Enable stitched TIFF by default
        is_performance_mode: bool = True,  # New parameter to distinguish modes
        nTimePoints: int = 1,
        nZ_planes: int = 1,
        nIlluminations: int = 1
    ):
        """
        Bulk-writer for both fast stage scan (performance mode) and normal stage scan.

        Stores every frame in multiple formats:
        • individual OME-TIFF tile (debug/backup) - optional
        • single chunked OME-Zarr mosaic (browser streaming) 
        • stitched OME-TIFF file (Fiji compatible) - optional

        Parameters
        ----------
        n_expected      total number of frames that will arrive
        metadata_list   list with x, y, illuminationChannel, … for each frame-id
        x_start … ny    grid geometry (needed to locate each tile in the canvas)
        is_performance_mode  whether using hardware triggering or workflow-based acquisition
        """
        # Set up unified OME writer
        tile_shape = (self.mDetector._shape[-1], self.mDetector._shape[-2])  # (height, width)        
        grid_shape = (nx, ny)
        grid_geometry = (x_start, y_start, x_step, y_step)
        writer_config = OMEWriterConfig(
            write_tiff=is_tiff,
            write_zarr=self._ome_write_zarr,
            write_stitched_tiff=write_stitched_tiff,
            min_period=min_period,
            pixel_size=self.detectorPixelSize[-1] if hasattr(self, 'detectorPixelSize') else 1.0,
            n_time_points=nTimePoints,
            n_z_planes= nZ_planes,
            n_channels = nIlluminations
        )
        
        ome_writer = OMEWriter(
            file_paths=mFilePath,
            tile_shape=tile_shape,
            grid_shape=grid_shape,
            grid_geometry=grid_geometry,
            config=writer_config,
            logger=self._logger
        )

        # ------------------------------------------------------------- main loop
        saved = 0
        self._logger.info(f"Writer thread started → {mFilePath.base_dir}")

        if is_performance_mode:
            # Performance mode: get frames from camera buffer
            while saved < n_expected and not self._stop_writer_evt.is_set():
                frames, ids = self.mDetector.getChunk()  # empties camera buffer

                if frames.size == 0:
                    time.sleep(0.005)
                    continue

                for frame, fid in zip(frames, ids):
                    meta = metadata_list[fid] if fid < len(metadata_list) else None
                    if not meta:
                        self._logger.warning(f"missing metadata for frame-id {fid}")
                        continue

                    # Write frame using unified writer
                    chunk_info = ome_writer.write_frame(frame, meta)
                    saved += 1
                    
                    # emit signal to tell frontend about the new chunk
                    if chunk_info and "rel_chunk" in chunk_info:
                        sigZarrDict = {
                            "event": "zarr_chunk",
                            "path": chunk_info["rel_chunk"],
                            "zarr": str(self.getOmeZarrUrl())  # e.g. /files/…/FastStageScan.ome.zarr
                        }
                        self.sigUpdateOMEZarrStore.emit(sigZarrDict)
        else:
            # Normal mode: frames are provided via external queue or workflow
            # This is a placeholder - actual implementation depends on how frames are provided
            self._logger.info("Normal mode writer started - waiting for frames via workflow")
            # In normal mode, frames will be written via separate calls to write_frame

        self._logger.info(f"Writer thread finished ({saved}/{n_expected}) tiles under : {mFilePath.base_dir}")
        
        # Finalize writing (build pyramids, etc.)
        ome_writer.finalize()

        # Store writer reference for normal mode
        if not is_performance_mode:
            self._current_ome_writer = ome_writer
            return  # Don't reset camera in normal mode

        # bring camera back to continuous mode (performance mode only)
        self.mDetector.stopAcquisition()
        self.mDetector.setTriggerSource("Continuous")
        self.mDetector.flushBuffers()
        self.mDetector.startAcquisition()
        self.fastStageScanIsRunning = False

    def write_frame_to_ome_writer(self, frame, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Write a frame to the current OME writer (for normal mode scanning).
        
        Args:
            frame: Image data as numpy array
            metadata: Dictionary containing position and other metadata
            
        Returns:
            Dictionary with information about the written chunk (for Zarr)
        """
        if self._current_ome_writer is not None:
            return self._current_ome_writer.write_frame(frame, metadata)
        else:
            self._logger.warning("No OME writer available for frame writing")
            return None
    
    def finalize_tile_ome_writer(self, context: WorkflowContext, metadata: Dict[str, Any], tile_index: int):
        """Finalize the OME writer for a specific tile."""
        file_writers = context.get_object("file_writers")
        
        if file_writers is not None and tile_index < len(file_writers):
            ome_writer = file_writers[tile_index]
            try:
                self._logger.info(f"Finalizing OME writer for tile: {tile_index}")
                ome_writer.finalize()
                self._logger.info(f"OME writer finalized for tile {tile_index}")
            except Exception as e:
                self._logger.error(f"Error finalizing OME writer for tile {tile_index}: {e}")
        else:
            self._logger.warning(f"No OME writer found for tile index: {tile_index}")

    def finalize_current_ome_writer(self, context: WorkflowContext = None, metadata: Dict[str, Any] = None):
        """Finalize all OME writers and clean up."""
        # Finalize OME writers from context (normal mode)
        if context is not None:
            # Get file_writers list and finalize all
            file_writers = context.get_object("file_writers")
            if file_writers is not None:
                for i, ome_writer in enumerate(file_writers):
                    try:
                        self._logger.info(f"Finalizing OME writer for tile {i}")
                        ome_writer.finalize()
                    except Exception as e:
                        self._logger.error(f"Error finalizing OME writer for tile {i}: {e}")
                # Clear the list from context
                context.remove_object("file_writers")
        
        # Also finalize the instance variable if it exists (performance mode)
        if self._current_ome_writer is not None:
            try:
                self._current_ome_writer.finalize()
                self._current_ome_writer = None
            except Exception as e:
                self._logger.error(f"Error finalizing current OME writer: {e}")
    
    def _stop(self):
        """Abort the acquisition gracefully."""
        self._stop_writer_evt.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2)
        if self._writer_thread_ome is not None:
            self._writer_thread_ome.join(timeout=2)
        self.mDetector.stopAcquisition()

    @APIExport(runOnUIThread=False)
    def stopFastStageScanAcquisition(self):
        """Stop the stage scan acquisition and writer thread."""
        self.mStage.stop_stage_scanning()
        self.fastStageScanIsRunning = False
        self._logger.info("Stopping stage scan acquisition...")
        self._stop()
        self._logger.info("Stage scan acquisition stopped.")

    @APIExport(runOnUIThread=False)
    def startFastStageScanAcquisitionFilePath(self) -> str:
        """Returns the file path of the last saved fast stage scan."""
        if hasattr(self, 'fastStageScanFilePath') and self.fastStageScanFilePath is not None:
            return self.fastStageScanFilePath
        else:
            return "No fast stage scan available yet"
    
    
# Copyright (C) 2025 Benedict Diederich
