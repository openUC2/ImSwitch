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
from imswitch.imcontrol.controller.controllers.experiment_controller import (
    ExperimentPerformanceMode,
    ExperimentNormalMode,
    OMEFileStorePaths
)

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
    ome_write_tiff: bool = Field(False, description="Whether to write OME-TIFF files")
    ome_write_zarr: bool = Field(True, description="Whether to write OME-Zarr files")
    ome_write_stitched_tiff: bool = Field(False, description="Whether to write stitched OME-TIFF files")

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
        self.SPEED_Z_default = 10000
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
        # ensure all subfolders are generated:
        os.makedirs(self.save_dir) if not os.path.exists(self.save_dir) else None

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
        self._ome_write_stitched_tiff = False
        self._ome_write_single_tiff = False

        # Initialize experiment execution modes
        self.performance_mode = ExperimentPerformanceMode(self)
        self.normal_mode = ExperimentNormalMode(self)

    @APIExport(requestType="GET")
    def getHardwareParameters(self):
        return self.ExperimentParams

    @APIExport(requestType="GET")
    def getOMEWriterConfig(self):
        """Get current OME writer configuration."""
        return {
            "write_tiff": getattr(self, '_ome_write_tiff', False),
            "write_zarr": getattr(self, '_ome_write_zarr', True),
            "write_stitched_tiff": getattr(self, '_ome_write_stitched_tiff', False),
            "write_single_tiff": getattr(self, '_ome_write_single_tiff', False)
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
        
        # Handle case where no XY coordinates are provided but z-stack is enabled
        # In this case, we want to scan at the current position
        if len(mExperiment.pointList) == 0 and (mExperiment.parameterValue.zStack or (mExperiment.parameterValue.zStackStepSize > 0 and mExperiment.parameterValue.zStackMax > mExperiment.parameterValue.zStackMin)):
            self._logger.info("No XY coordinates provided but z-stack enabled. Creating fallback point at current position.")
            
            # Get current stage position
            current_position = self.mStage.getPosition()
            current_x = current_position.get("X", 0)
            current_y = current_position.get("Y", 0)
            
            # Create a fallback point at current position
            fallback_tile = [{
                "iterator": 0,
                "centerIndex": 0,
                "iX": 0,
                "iY": 0,
                "x": current_x,
                "y": current_y,
            }]
            tiles.append(fallback_tile)
            return tiles
        
        # Original logic for when pointList is provided
        for iCenter, centerPoint in enumerate(mExperiment.pointList):
            # Collect central and neighbour points (without duplicating the center)
            allPoints = [(n.x, n.y) for n in centerPoint.neighborPointList]
            
            # Handle case where neighborPointList is empty but centerPoint is provided
            # This means scan at the center point position only (useful for z-stack-only)
            if len(allPoints) == 0:
                self._logger.info(f"Empty neighborPointList for center point {iCenter}. Using center point position for z-stack scanning.")
                fallback_tile = [{
                    "iterator": 0,
                    "centerIndex": iCenter,
                    "iX": 0,
                    "iY": 0,
                    "x": centerPoint.x,
                    "y": centerPoint.y,
                }]
                tiles.append(fallback_tile)
                continue
            
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

    @APIExport(requestType="GET")
    def getExperimentStatus(self):
        """Get the current status of running experiments."""
        # Check workflow manager status (normal mode)
        if self.ExperimentParams.performanceMode:
            # Check performance mode status
            workflow_status = self.performance_mode.get_scan_status()
        else:
            # Check normal mode status
            workflow_status = self.workflow_manager.get_status()

        return workflow_status

    @APIExport(requestType="POST")
    def startWellplateExperiment(self, mExperiment: Experiment):
        # Extract key parameters
        exp_name = mExperiment.name
        p = mExperiment.parameterValue

        # Timelapse-related
        nTimes = p.numberOfImages
        tPeriod = p.timeLapsePeriod

        # Z-steps -related
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
        
        # check if any of the illumination sources is turned on, if not, return error
        if not any(illuminationIntensities):
            return HTTPException(status_code=400, detail="No illumination sources are turned on. Please set at least one illumination source intensity.")

        # check if we want to use performance mode
        self.ExperimentParams.performanceMode = p.performanceMode
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
        isZStack = p.zStack or (p.zStackStepSize > 0 and p.zStackMax > p.zStackMin)
        if isZStack:
            z_positions = np.arange(zStackMin, zStackMax + zStackStepSize, zStackStepSize) + currentZ
        else:
            z_positions = [currentZ]  # Get current Z position

        # Prepare directory and filename for saving
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        drivePath = dirtools.UserFileDirs.Data
        dirPath = os.path.join(drivePath, 'ExperimentController', timeStamp)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        mFileName = f"{timeStamp}_{exp_name}"

        workflowSteps = []
        file_writers = []  # Initialize outside the loop for context storage
        
        # OME writer-related
        self._ome_write_tiff = p.ome_write_tiff
        self._ome_write_zarr = p.ome_write_zarr
        self._ome_write_stitched_tiff = p.ome_write_stitched_tiff
        self._ome_write_single_tiff = getattr(p, 'ome_write_single_tiff', False)  # Default to False if not specified

        # determine if each sub scan in snake_tiles is a single tile or a multi-tile scan - if single image we should squah them in a single TIF (e.g. by appending )
        is_single_tile_scan = all(len(tile) == 1 for tile in snake_tiles)
        if is_single_tile_scan:
            self._ome_write_stitched_tiff = False  # Disable stitched TIFF for single tile scans
            self._ome_write_single_tiff = True   # Enable single TIFF writing
        else:
            self._ome_write_single_tiff = False
            
            
        # Decide which execution mode to use
        if performanceMode and self.performance_mode.is_hardware_capable():
            # Execute in performance mode
            experiment_params = {
                'mExperiment': mExperiment,
                'tPeriod': tPeriod,
                'nTimes': nTimes
            }
            result = self.performance_mode.execute_experiment(
                snake_tiles=snake_tiles,
                illumination_intensities=illuminationIntensities,
                experiment_params=experiment_params
            )
            return {"status": "running", "mode": "performance"}
        else:
            # Execute in normal mode using workflow
            all_workflow_steps = []
            all_file_writers = []

            for t in range(nTimes):
                experiment_params = {
                    'mExperiment': mExperiment,
                    'tPeriod': tPeriod,
                    'nTimes': nTimes
                }

                result = self.normal_mode.execute_experiment(
                    snake_tiles=snake_tiles,
                    illumination_intensities=illuminationIntensities,
                    illumination_sources=illuSources,
                    z_positions=z_positions,
                    exposures=exposures,
                    gains=gains,
                    exp_name=exp_name,
                    dir_path=dirPath,
                    m_file_name=mFileName,
                    t=t,
                    n_times=nTimes,  # Pass total number of time points
                    is_auto_focus=isAutoFocus,
                    autofocus_min=autofocusMin,
                    autofocus_max=autofocusMax,
                    autofocus_step_size=autofocusStepSize,
                    t_period=tPeriod
                )

                # Append workflow steps and file writers to the accumulated lists
                all_workflow_steps.extend(result["workflow_steps"])
                all_file_writers.extend(result["file_writers"])

            # Use the accumulated workflow steps and file writers
            workflowSteps = all_workflow_steps
            file_writers = all_file_writers
            # Create workflow progress handler
            def sendProgress(payload):
                self.sigExperimentWorkflowUpdate.emit(payload)

            # Create workflow and context
            from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext
            wf = Workflow(workflowSteps, self.workflow_manager)
            context = WorkflowContext()

            # Set metadata
            context.set_metadata("experimentName", exp_name)
            context.set_metadata("nTimes", nTimes)
            context.set_metadata("tPeriod", tPeriod)
            # Add timing information for proper period calculation
            import time
            context.set_metadata("experiment_start_time", time.time())
            context.set_metadata("timepoint_times", {})  # Track timing for each timepoint

            # Store file_writers in context
            if len(file_writers) > 0:
                context.set_object("file_writers", file_writers)
            context.on("progress", sendProgress)
            context.on("rgb_stack", sendProgress)

            # Start the workflow
            self.workflow_manager.start_workflow(wf, context)

        return {"status": "running"}

    def computeScanRanges(self, snake_tiles):
        """Compute scan ranges - delegated to base class method."""
        return self.performance_mode.compute_scan_ranges(snake_tiles)




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

    def wait_for_next_timepoint(self, timepoint: int, t_period: float, context: WorkflowContext, metadata: Dict[str, Any]):
        """
        Wait for the proper time interval between timepoints, accounting for measurement time.
        
        Args:
            timepoint: Current timepoint index
            t_period: Target period between timepoints in seconds
            context: WorkflowContext containing timing information
            metadata: Metadata dictionary
        """
        import time
        
        current_time = time.time()
        experiment_start_time = context.get_metadata("experiment_start_time", current_time)
        timepoint_times = context.get_metadata("timepoint_times", {})
        
        # Calculate expected time for this timepoint
        expected_time = experiment_start_time + (timepoint + 1) * t_period
        
        # Store timing information for this timepoint
        timepoint_times[str(timepoint)] = current_time
        context.set_metadata("timepoint_times", timepoint_times)
        
        # Calculate how long to wait
        wait_time = max(0, expected_time - current_time)
        
        if wait_time > 0:
            self._logger.info(f"Waiting {wait_time:.2f}s for next timepoint (timepoint {timepoint})")
            time.sleep(wait_time)
        else:
            self._logger.warning(f"Timepoint {timepoint} is running {abs(wait_time):.2f}s behind schedule")
            # Small delay to prevent issues
            time.sleep(0.01)

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
            if ome_writer.store: self.setOmeZarrUrl(ome_writer.store.split(dirtools.UserFileDirs.Data)[-1])  # Update OME-Zarr URL in context
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
        self.mStage.move(value=(posX, posY), speed=(self.SPEED_X_default, self.SPEED_Y_default), axis="XY", is_absolute=not relative, is_blocking=True, acceleration=self.ACCELERATION)
        #newPosition = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([posX, posY])
        return (posX, posY)

    def move_stage_z(self, posZ: float, relative: bool = False, maxSpeedZ=5000):
        self._logger.debug(f"Moving stage to Z={posZ}")
        self.mStage.move(value=posZ, speed=np.min((self.SPEED_Z, maxSpeedZ)), axis="Z", is_absolute=not relative, is_blocking=True)
        newPosition = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([newPosition["Z"]])
        return newPosition["Z"]


    @APIExport()
    def pauseWorkflow(self):
        """Pause the workflow. Only works in normal mode."""
        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check if performance mode is running
        performance_status = self.performance_mode.get_scan_status()
        if performance_status["running"]:
            return {"status": "error", "message": "Cannot pause experiment in performance mode"}

        if workflow_status == "running":
            return self.workflow_manager.pause_workflow()
        else:
            return {"status": "error", "message": f"Cannot pause in current state: {workflow_status}"}

    @APIExport()
    def resumeExperiment(self):
        """Resume the experiment. Only works in normal mode."""
        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check if performance mode is running
        performance_status = self.performance_mode.get_scan_status()
        if performance_status["running"]:
            return {"status": "error", "message": "Cannot resume experiment in performance mode"}

        if workflow_status == "paused":
            return self.workflow_manager.resume_workflow()
        else:
            return {"status": "error", "message": f"Cannot resume in current state: {workflow_status}"}

    @APIExport()
    def stopExperiment(self):
        """Stop the experiment. Works for both normal and performance modes."""
        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check performance mode status
        performance_status = self.performance_mode.get_scan_status()

        results = {}

        # Stop workflow if running
        if workflow_status in ["running", "paused", "stopping"]:
            results["workflow"] = self.workflow_manager.stop_workflow()

        # Stop performance mode if running
        if performance_status["running"]:
            results["performance"] = self.performance_mode.stop_scan()

        # If nothing was running, return appropriate message
        if not results:
            return "No experiments are currently running"

        return results


    @APIExport()
    def forceStopExperiment(self):
        """Force stop the experiment. Works for both normal and performance modes."""
        results = {}

        # Force stop workflow
        try:
            self.workflow_manager.stop_workflow()
            del self.workflow_manager
            self.workflow_manager = WorkflowsManager()
            results["workflow"] = {"status": "force_stopped", "message": "Workflow force stopped"}
        except Exception as e:
            results["workflow"] = {"status": "error", "message": f"Error force stopping workflow: {e}"}

        # Force stop performance mode
        try:
            results["performance"] = self.performance_mode.force_stop_scan()
        except Exception as e:
            results["performance"] = {"status": "error", "message": f"Error force stopping performance mode: {e}"}

        return results


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
            return -1
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

            # 1. prepare camera ----------------------------------------------------
            self.mDetector.stopAcquisition()
            #self.mDetector.NBuffer        = total_frames + 32   # head‑room
            #self.mDetector.frame_buffer   = collections.deque(maxlen=self.mDetector.NBuffer)
            #self.mDetector.frameid_buffer = collections.deque(maxlen=self.mDetector.NBuffer)
            self.mDetector.setTriggerSource("External trigger")
            self.mDetector.flushBuffers()
            self.mDetector.startAcquisition()

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
            self.fastStageScanIsRunning = True  # Set flag to indicate scan is running
            self.mStage.start_stage_scanning(
                xstart=0, xstep=xstep, nx=nx, # we choose xstart/ystart = 0 since this means we start from here in the positive direction with nsteps
                ystart=0, ystep=ystep, ny=ny,
                tsettle=tsettle, tExposure=tExposure,
                illumination=illumination, led=led,
            )
            #TODO: Make path more uniform - e.g. basetype
            while nLastTime + tPeriod < time.time() and self.fastStageScanIsRunning:
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
            write_tiff_single=self._ome_write_single_tiff,
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
            self._logger.warning(f"No OME writer found for tile index when finalizing writer: {tile_index}")

    def finalize_current_ome_writer(self, context: WorkflowContext = None, metadata: Dict[str, Any] = None, *args, **kwargs):
        """Finalize all OME writers and clean up."""
        # TODO: This is misleading as the method closes all filewriters, not just the current one.
        # Finalize OME writers from context (normal mode)
        # optional: extract time_point from args/kwargs if needed
        time_index = kwargs.get("time_index", None)

        if context is not None:
            # Get file_writers list and finalize all
            file_writers = context.get_object("file_writers")
            if file_writers is not None:
                if time_index is not None:
                    self._logger.info(f"Finalizing OME writers for time point {time_index}")
                    try:
                        ome_writer = file_writers[time_index]
                        ome_writer.finalize()
                        self._logger.info(f"OME writer finalized for time point {time_index}")
                    except Exception as e:
                        self._logger.error(f"Error finalizing OME writer for time point {time_index}: {e}")
                else:
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
