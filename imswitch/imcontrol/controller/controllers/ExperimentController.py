from datetime import datetime
import time
from fastapi import HTTPException
import numpy as np
import tifffile as tif
from pydantic import BaseModel
from typing import Any, List, Optional, Dict, Union
import os
import uuid
import threading

from imswitch.imcommon.framework import Signal
from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext, WorkflowStep, WorkflowsManager
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from pydantic import BaseModel, Field

# Import the new component classes
from .experiment_controller import (
    ProtocolManager, HardwareInterface, PerformanceModeExecutor, FileIOManager
)

try:
    from ashlarUC2 import utils
    IS_ASHLAR_AVAILABLE = True
except ImportError:
    IS_ASHLAR_AVAILABLE = False

# Attempt to use OME-Zarr
try:
    from imswitch.imcontrol.controller.controllers.experiment_controller.zarr_data_source import MinimalZarrDataSource
    from imswitch.imcontrol.controller.controllers.experiment_controller.single_multiscale_zarr_data_source import SingleMultiscaleZarrWriter
    IS_OMEZARR_AVAILABLE = False  # TODO: True
except ImportError:
    IS_OMEZARR_AVAILABLE = False

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

        # Initialize variables
        self.tWait = 0.1
        self.workflow_manager = WorkflowsManager()
        self.mFilePaths = []
        self.isPreview = False

        # Initialize component managers
        self.hardware = HardwareInterface(self._master, self._commChannel)
        self.protocol_manager = ProtocolManager()
        self.file_io_manager = FileIOManager()
        self.performance_executor = PerformanceModeExecutor(
            self.hardware, self.file_io_manager.base_save_dir
        )

        # Set default speeds (now handled by HardwareInterface)
        self.SPEED_Y_default = 20000
        self.SPEED_X_default = 20000
        self.SPEED_Z_default = 20000
        self.ACCELERATION = 500000

        # Legacy properties for backwards compatibility
        self.mDetector = self.hardware.mDetector
        self.isRGB = self.hardware.is_rgb
        self.detectorPixelSize = self.hardware.detector_pixel_size
        self.allIlluNames = self.hardware.all_illumination_names
        self.availableIlliminations = self.hardware.available_illuminations
        self.mStage = self.hardware.mStage

        # Stop if external signal (e.g. memory full) is triggered
        self._commChannel.sigExperimentStop.connect(self.stopExperiment)

        # Define changeable Experiment parameters
        self.ExperimentParams = ExperimentWorkflowParams()
        self.ExperimentParams.illuSources = self.allIlluNames
        self.ExperimentParams.illuSourceMinIntensities = [0] * len(self.allIlluNames)
        self.ExperimentParams.illuSourceMaxIntensities = [1023] * len(self.allIlluNames)
        self.ExperimentParams.illuIntensities = [0] * len(self.allIlluNames)
        self.ExperimentParams.exposureTimes = 1000000  # TODO: FIXME
        self.ExperimentParams.gains = [23]  # TODO: FIXME
        self.ExperimentParams.isDPCpossible = False
        self.ExperimentParams.isDarkfieldpossible = False
        self.ExperimentParams.performanceMode = False

        # Legacy performance mode properties
        self.fastStageScanIsRunning = self.performance_executor.is_scan_running()

        
    @APIExport(requestType="GET")
    def getHardwareParameters(self):
        return self.ExperimentParams

    def get_num_xy_steps(self, pointList):
        """Delegate to ProtocolManager."""
        return self.protocol_manager.get_num_xy_steps(pointList)

    def generate_snake_tiles(self, mExperiment):
        """Delegate to ProtocolManager."""
        return self.protocol_manager.generate_snake_tiles(mExperiment)

    def computeScanRanges(self, snake_tiles):
        """Delegate to ProtocolManager."""
        return self.protocol_manager.compute_scan_ranges(snake_tiles)

    @APIExport()
    def getLastFilePathsList(self) -> List[str]:
        """
        Returns the last file paths list.
        """
        return self.mFilePaths
        
        

    @APIExport(requestType="POST")
    def startWellplateExperiment(self, mExperiment: Experiment):
        # Extract and validate parameters using ProtocolManager
        validated_params = self.protocol_manager.validate_experiment_parameters(mExperiment)
        
        exp_name = mExperiment.name
        
        # Extract validated parameters
        illumination_sources = validated_params['illumination_sources']
        illumination_intensities = validated_params['illumination_intensities']
        gains = validated_params['gains']
        exposures = validated_params['exposures']
        performance_mode = validated_params['performance_mode']
        is_z_stack = validated_params['is_z_stack']
        is_autofocus = validated_params['is_autofocus']
        n_times = validated_params['n_times']
        t_period = validated_params['t_period']
        z_stack_min = validated_params['z_stack_min']
        z_stack_max = validated_params['z_stack_max']
        z_stack_step = validated_params['z_stack_step']
        autofocus_min = validated_params['autofocus_min']
        autofocus_max = validated_params['autofocus_max']
        autofocus_step = validated_params['autofocus_step']
        
        # Set movement speeds
        speed = validated_params['speed']
        if speed:
            self.SPEED_X = speed
            self.SPEED_Y = speed
            self.SPEED_Z = speed
        else:
            self.SPEED_X = self.SPEED_X_default
            self.SPEED_Y = self.SPEED_Y_default
            self.SPEED_Z = self.SPEED_Z_default

        # Check if another workflow is running
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        # Start the detector if not already running
        self.hardware.start_detector_acquisition()

        # Generate the list of points to scan based on snake scan
        if mExperiment.parameterValue.resortPointListToSnakeCoordinates:
            pass  # TODO: we need an alternative case
        snake_tiles = self.generate_snake_tiles(mExperiment)
        # remove none values from all_points list
        snake_tiles = [[pt for pt in tile if pt is not None] for tile in snake_tiles]

        # Generate Z-positions
        current_position = self.hardware.get_current_position()
        current_z = current_position["Z"]
        if is_z_stack:
            z_positions = np.arange(z_stack_min, z_stack_max + z_stack_step, z_stack_step) + current_z
        else:
            z_positions = [current_z]  # Get current Z position
        minX, maxX, minY, maxY, diffX, diffY = self.computeScanRanges(snake_tiles)
        mPixelSize = self.detectorPixelSize[-1]  # Pixel size in µm

        # Setup file I/O 
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirPath = self.file_io_manager.create_experiment_directory(exp_name)

        # Prepare OME-Zarr writer setup variables
        z_steps = validated_params['z_steps']
        
        # if performance_mode is True, we will execute on the Hardware directly
        if performance_mode:
            self._logger.debug("Performance mode is enabled. Executing on hardware directly.")
            for snake_tile in snake_tiles:
                # wait if there is another fast stage scan running
                self.performance_executor.wait_for_scan_completion()
                
                # need to compute the position ranges for this tile
                xStart, xEnd, yStart, yEnd, xStep, yStep = self.computeScanRanges([snake_tile])
                nx, ny = int((xEnd - xStart) // xStep) + 1, int((yEnd - yStart) // yStep) + 1
                
                # Extract illumination values
                illumination0 = illumination_intensities[0] if len(illumination_intensities) > 0 else None
                illumination1 = illumination_intensities[1] if len(illumination_intensities) > 1 else None
                illumination2 = illumination_intensities[2] if len(illumination_intensities) > 2 else None
                illumination3 = illumination_intensities[3] if len(illumination_intensities) > 3 else None
                
                # Find LED intensity
                led_index = next((i for i, item in enumerate(illumination_sources) 
                                if "led" in item.lower()), None)
                led = min(illumination_intensities[led_index], 255) if led_index is not None else 0
                
                if nx > 100 or ny > 100:
                    self._logger.error("Too many points in X/Y direction. Please reduce the number of points.")
                    raise HTTPException(status_code=400, 
                                      detail="Too many points in X/Y direction. Please reduce the number of points.")
                
                # Execute fast stage scan using performance executor
                self.performance_executor.start_fast_stage_scan_acquisition(
                    xstart=xStart, xstep=xStep, nx=nx,
                    ystart=yStart, ystep=yStep, ny=ny,
                    tsettle=90, tExposure=50,  # TODO: make these parameters adjustable
                    illumination0=illumination0, illumination1=illumination1,
                    illumination2=illumination2, illumination3=illumination3, 
                    led=led
                )
            return {"status": "running", "store_path": []} 
        # Setup for regular workflow mode (non-performance)
        # Prepare OME-Zarr writer if available
        ome_store = None
        if IS_OMEZARR_AVAILABLE:
            zarr_path = os.path.join(dirPath, f"{timeStamp}_{exp_name}.ome.zarr")
            # compute max coordinates for x/y
            fovX = int(maxX - minX + diffX)
            fovY = int(maxY - minY + diffY)
            ome_store = self.file_io_manager.setup_omezarr_store(
                zarr_path, n_times, z_steps, fovX, fovY
            )
        else:
            self._logger.warning("OME-ZARR not available or not installed.")

        # Setup TIFF writers using FileIOManager
        tiff_writers = self.file_io_manager.setup_tiff_writers(
            mExperiment.pointList, dirPath, exp_name
        )

        workflowSteps = []
        step_id = 0

        for t in range(n_times):


            # Loop over each tile (each central point and its neighbors)
            for position_center_index, tile in enumerate(snake_tiles):
                
                # start TIFF writer using FileIOManager
                self.file_io_manager.start_tiff_writer(position_center_index)
            
                # iterate over positions     
                for mIndex, mPoint in enumerate(tile):
                    try:
                        name = f"Move to point {mPoint['iterator']}"
                    except Exception:
                        name = f"Move to point {mPoint['x']}, {mPoint['y']}"
                    
                    workflowSteps.append(WorkflowStep(
                        name=name,
                        step_id=step_id,
                        main_func=self.hardware.move_stage_xy,
                        main_params={"posX": mPoint["x"], "posY": mPoint["y"], "relative": False},
                    ))
                    
                    # iterate over z-steps
                    for indexZ, iZ in enumerate(z_positions):
                    
                        #move to Z position - but only if we have more than one z position
                        if len(z_positions) > 1 or (len(z_positions) == 1 and mIndex == 0):
                            workflowSteps.append(WorkflowStep(
                                name="Move to Z position",
                                step_id=step_id,
                                main_func=self.hardware.move_stage_z,
                                main_params={"posZ": iZ, "relative": False},
                                pre_funcs=[self.wait_time],
                                pre_params={"seconds": 0.1},
                            ))
                            
                        step_id += 1
                        for illuIndex, illuSource in enumerate(illumination_sources):
                            illuIntensity = illumination_intensities[illuIndex]
                            if illuIntensity <= 0: 
                                continue
                            # Turn on illumination - if we have only one source, we can skip this step after the first iteration
                            if sum(np.array(illumination_intensities) > 0) > 1 or (mIndex == 0):
                                workflowSteps.append(WorkflowStep(
                                    name="Turn on illumination",
                                    step_id=step_id,
                                    main_func=self.hardware.set_laser_power,
                                    main_params={"power": illuIntensity, "channel": illuSource},
                                    post_funcs=[self.wait_time],
                                    post_params={"seconds": 0.05},
                                ))
                                step_id += 1

                        # Acquire frame
                        isPreview = self.isPreview

                        workflowSteps.append(WorkflowStep(
                            name="Acquire frame",
                            step_id=step_id,
                            main_func=self.hardware.acquire_frame,
                            main_params={"channel": "Mono"},
                            post_funcs=[self.save_frame_tiff, self.save_frame_ome_zarr, self.add_image_to_canvas],
                            pre_funcs=[self.set_exposure_time_gain],
                            pre_params={"exposure_time": exposures[illuIndex], "gain": gains[illuIndex]},
                            post_params={
                                "posX": mPoint["x"],
                                "posY": mPoint["y"],
                                "posZ": 0,  # TODO: Add Z position if needed
                                "iX": mPoint["iX"],
                                "iY": mPoint["iY"], 
                                "pixel_size": mPixelSize,
                                "minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY,
                                "channel": illuSource,
                                "time_index": t,
                                "tile_index": mIndex,
                                "position_center_index": position_center_index,
                                "isPreview": isPreview,
                            },
                        ))
                        step_id += 1

                        # Turn off illumination
                        if len(illumination_intensities) > 1 and sum(np.array(illumination_intensities) > 0) > 1:
                            workflowSteps.append(WorkflowStep(
                                name="Turn off illumination",
                                step_id=step_id,
                                main_func=self.hardware.set_laser_power,
                                main_params={"power": 0, "channel": illuSource},
                            ))
                            step_id += 1

                # (Optional) Perform autofocus once per loop, if enabled
                if is_autofocus:
                    workflowSteps.append(WorkflowStep(
                        name="Autofocus",
                        step_id=step_id,
                        main_func=self.hardware.perform_autofocus,
                        main_params={"min_z": autofocus_min, "max_z": autofocus_max, "step_size": autofocus_step},
                    ))
                    step_id += 1

                step_id += 1

                # Close TIFF writer at the end
                workflowSteps.append(WorkflowStep(
                    name="Close TIFF writer",
                    step_id=step_id,
                    main_func=self.file_io_manager.close_tiff_writer,
                    main_params={"writer_index": position_center_index},
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

        if ome_store:
            # Close OME-Zarr store at the end
            workflowSteps.append(WorkflowStep(
                name="Close OME-Zarr store",
                step_id=step_id,
                main_func=self.file_io_manager.close_omezarr_store,
                main_params={},
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
        self.hardware.turn_off_all_illumination(illumination_sources)

        def sendProgress(payload):
            self.sigExperimentWorkflowUpdate.emit(payload)

        # Create workflow and context
        wf = Workflow(workflowSteps, self.workflow_manager)
        context = WorkflowContext()
        # Set metadata
        context.set_metadata("experimentName", exp_name)
        context.set_metadata("nTimes", n_times)
        context.set_metadata("tPeriod", t_period)

        context.set_object("tiff_writers", tiff_writers)
        if ome_store:
            context.set_object("omezarr_store", ome_store)

        # Setup canvas for preview mode
        canvas_width, canvas_height = self.file_io_manager.compute_canvas_dimensions(
            minX, maxX, minY, maxY, diffX, diffY, mPixelSize
        )

        if self.isPreview:
            canvas = self.file_io_manager.create_canvas(
                canvas_width, canvas_height, self.hardware.is_rgb
            )
            context.set_object("canvas", canvas)
            
        context.on("progress", sendProgress)
        context.on("rgb_stack", sendProgress)

        # Start the workflow
        self.workflow_manager.start_workflow(wf, context)

        # Store file paths for retrieval
        self.mFilePaths = self.file_io_manager.get_file_paths()
        return {"status": "running", "store_path": self.mFilePaths}

    ########################################
    # Legacy function wrappers for backwards compatibility
    ########################################
    # Legacy wrapper functions for workflow compatibility
    def acquire_frame(self, channel: str):
        """Legacy wrapper - delegate to HardwareInterface."""
        return self.hardware.acquire_frame(channel)
    
    def set_exposure_time_gain(self, exposure_time: float, gain: float, context: WorkflowContext, metadata: Dict[str, Any]):
        """Legacy wrapper - delegate to HardwareInterface."""
        self.hardware.set_exposure_time_gain(exposure_time, gain)
        
    def dummy_main_func(self):
        """Utility function for workflow steps."""
        self._logger.debug("Dummy main function called")
        return True
 
    def autofocus(self, min_z: float = 0, max_z: float = 0, step_size: float = 0):
        """Legacy wrapper - delegate to HardwareInterface."""
        return self.hardware.perform_autofocus(min_z, max_z, step_size)
        
    def wait_time(self, seconds: int, context: WorkflowContext, metadata: Dict[str, Any]):
        """Utility function for workflow steps."""
        import time
        time.sleep(seconds)

    def save_frame_tiff(self, context: WorkflowContext, metadata: Dict[str, Any], **kwargs):
        """Save frame to TIFF using FileIOManager."""
        # get latest image from the camera 
        img = metadata["result"]

        # get metadata
        posX = kwargs.get("posX", 0)
        posY = kwargs.get("posY", 0)
        posZ = kwargs.get("posZ", 0)
        position_center_index = kwargs.get("position_center_index", 0)
        iX = kwargs.get("iX", 0)
        iY = kwargs.get("iY", 0)
        pixel_size = kwargs.get("pixel_size", 1.0)
        
        # Use FileIOManager to save
        success = self.file_io_manager.save_frame_tiff(
            img, position_center_index, posX, posY, posZ, iX, iY, pixel_size
        )
        metadata["frame_saved"] = success

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
    # Performance mode API delegation
    @APIExport(runOnUIThread=False)
    def startFastStageScanAcquisition(self,
                      xstart: float = 0, xstep: float = 500, nx: int = 10,
                      ystart: float = 0, ystep: float = 500, ny: int = 10,
                      tsettle: float = 90, tExposure: float = 50,
                      illumination0: int = None, illumination1: int = None,
                      illumination2: int = None, illumination3: int = None, 
                      led: float = None):
        """Delegate to PerformanceModeExecutor."""
        return self.performance_executor.start_fast_stage_scan_acquisition(
            xstart, xstep, nx, ystart, ystep, ny, tsettle, tExposure,
            illumination0, illumination1, illumination2, illumination3, led
        )

    @APIExport(runOnUIThread=False)
    def stopFastStageScanAcquisition(self):
        """Delegate to PerformanceModeExecutor."""
        return self.performance_executor.stop_fast_stage_scan_acquisition()

    # Update legacy property for backwards compatibility
    @property
    def fastStageScanIsRunning(self):
        """Legacy property - delegate to PerformanceModeExecutor."""
        return self.performance_executor.is_scan_running()

# Copyright (C) 2025 Benedict Diederich
