from datetime import datetime
import time
from fastapi import HTTPException
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as signal
import skimage.transform as transform
import tifffile as tif
from pydantic import BaseModel
from typing import Any, List, Optional, Dict, Any
import os 
import uuid

from imswitch.imcommon.framework import Signal
from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext, WorkflowStep, WorkflowsManager
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from imswitch import IS_HEADLESS

from pydantic import BaseModel
from typing import Optional, Tuple, List

import h5py
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
    IS_OMEZARR_AVAILABLE = True
except Exception as e:
    IS_OMEZARR_AVAILABLE = False

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict
import uuid

# -----------------------------------------------------------
# Reuse your existing sub-models:
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
    illumination: str
    brightfield: bool
    darkfield: bool
    laserWaveLength: int
    differentialPhaseContrast: bool
    timeLapsePeriod: float
    numberOfImages: int
    autoFocus: bool
    autoFocusMin: float
    autoFocusMax: float
    autoFocusStepSize: float
    zStack: bool
    zStackMin: float
    zStackMax: float
    zStackStepSize: float
    exposureTime: float = None
    gain: float = None

class Experiment(BaseModel):
    # From your old "Experiment" BaseModel:
    name: str
    parameterValue: ParameterValue
    pointList: List[Point]

    # From your old "ExperimentModel":
    number_z_steps: int = Field(0, description="Number of Z slices")
    timepoints: int = Field(1, description="Number of timepoints for time-lapse")
    x_pixels: int = Field(0, description="Image width in pixels")
    y_pixels: int = Field(0, description="Image height in pixels")
    microscope_name: str = Field("FRAME", description="Name of the microscope")
    is_multiposition: bool = Field(False, description="Whether multiple positions are used")
    channels: Dict[str, Dict[str, float]] = Field(
        description="Channel definitions, typically keys like 'Ch0', 'Ch1' etc.",
        default_factory=lambda: {
        "Ch0": {"is_selected": True, "camera_exposure_time": 0}
    })
    multi_positions: Dict[str, Dict[str, float]] = Field(
        description="Multi-position definitions if is_multiposition=True",
        default_factory=lambda: {
            '''
            "pos1": {"x": 0, "y": 0, "z": 0},
            "pos2": {"x": 10000, "y": 20000, "z": 5000},
            '''
        }
    )
    
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
                    "microscope_name": self.microscope_name,
                    "number_z_steps": self.number_z_steps,
                    "is_multiposition": self.is_multiposition,
                    "timepoints": self.timepoints,
                    "channels": self.channels,
                },
                "CameraParameters": {
                    self.microscope_name: {
                        "x_pixels": self.x_pixels,
                        "y_pixels": self.y_pixels,
                    }
                },
            },
            "multi_positions": self.multi_positions,
        }
        return config
    
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

        self.SPEED_Y = 10000
        self.SPEED_X = 10000
        self.SPEED_Z = 10000
        
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
            
        '''
        # define Experiment parameters as ExperimentWorkflowParams
        self.ExperimentParams = ExperimentWorkflowParams()
        
        
        # TODO: Adjust parameters 
        self.ExperimentParams.illuSources = self.allIlluNames
        self.ExperimentParams.illuSourceMinIntensities = [0]*len(self.ExperimentParams.illuSourcesSelected)
        self.ExperimentParams.illuSourceMaxIntensities = [100]*len(self.ExperimentParams.illuSourcesSelected)
        self.ExperimentParams.illuIntensities = [0]*len(self.allIlluNames)
        self.ExperimentParams.exposureTimes = [0]*len(self.allIlluNames)
        self.ExperimentParams.gain = [0]*len(self.allIlluNames)
        
    @APIExport(requestType="GET")
    def getCurrentExperimentParameters(self):
        return self.ExperimentParams
    
    @APIExport(requestType="POST")
    def setCurrentExperimentParameters(self, params: ExperimentWorkflowParams):
        self.ExperimentParams = params
        return self.ExperimentParams

        '''

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

    '''
    def generate_snake_tiles(self, mExperiment):
        """
        Generates Snake Scan coordinates based on a list of central and neighbouring points coming from the GUI
        """
        tiles = []
        for iCenter, centerPoint in enumerate(mExperiment.pointList):
            # 1) Collect all relevant points (central, neighbours) in a list
            allPoints = [(centerPoint.x, centerPoint.y)] + [
                (n.x, n.y) for n in centerPoint.neighborPointList
            ]
            
            # also add center point to the list
            allPoints.append((centerPoint.x, centerPoint.y))
            
            # Sort by y, then by x (i.e. raster)
            allPoints.sort(key=lambda coords: (coords[1], coords[0]))

            # 2) Calculate the number of steps in x and y direction
            num_x_steps, num_y_steps = self.get_num_xy_steps(centerPoint.neighborPointList)
            allPointsSnake = [0]*num_x_steps*num_y_steps
            iTile = 0
            for iY in range(num_y_steps):
                for iX in range(num_x_steps):
                    if iY % 2 == 1 and num_x_steps!=1:
                        # odd
                        mIdex = iY*num_x_steps + num_x_steps - 1 - iX
                    else: 
                        #even
                        mIdex = iTile 
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
        def flatten(xss):
            return [x for xs in xss for x in xs]            
        return flatten(tiles)
    '''
    def generate_snake_tiles(self, mExperiment):
        tiles = []
        for iCenter, centerPoint in enumerate(mExperiment.pointList):
            # Collect central and neighbour points (without duplicating the center)
            allPoints = [(centerPoint.x, centerPoint.y)] + [
                (n.x, n.y) for n in centerPoint.neighborPointList
            ]
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


    @APIExport(requestType="POST")
    def startWellplateExperiment(self, mExperiment: Experiment):
        # Extract key parameters
 
        exp_name = mExperiment.name
        p = mExperiment.parameterValue
        nTimes = p.numberOfImages
        tPeriod = p.timeLapsePeriod
        isAutoFocus = p.autoFocus
        nZSteps = int((mExperiment.parameterValue.zStackMax-mExperiment.parameterValue.zStackMin)//mExperiment.parameterValue.zStackStepSize)+1
        isZStack = p.zStack
        
        # Example usage of a single illumination source
        illuSource = p.illumination
        laserWaveLength = p.laserWaveLength

        # Check if another workflow is running
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        # Start the detector if not already running
        if not self.mDetector._running:
            self.mDetector.startAcquisition()

        # Prepare TIFF writer
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        drivePath = dirtools.UserFileDirs.Data
        dirPath = os.path.join(drivePath, 'recordings', timeStamp)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        mFileName = f"{timeStamp}_{exp_name}"
        mFilePath = os.path.join(dirPath, mFileName + ".tif")
        tiff_writer = tif.TiffWriter(mFilePath)
        
        # Prepare OME-Zarr writer (new)
        # fill ome model
        if IS_OMEZARR_AVAILABLE:
            ome_store = None
            zarr_path = os.path.join(dirPath, mFileName + ".ome.zarr")
            self._logger.debug(f"OME-Zarr path: {zarr_path}")
            ome_store = MinimalZarrDataSource(file_name=zarr_path, mode="w")
            # Configure it from your model
            mExperiment.x_pixels, mExperiment.y_pixels=self.mDetector._shape[-2],self.mDetector._shape[-1]
            mExperiment.number_z_steps = nZSteps
            mExperiment.timepoints = nTimes
            exp_config = mExperiment.to_configuration()
            ome_store.set_metadata_from_configuration_experiment(exp_config)
        else:
            self._logger.error("OME-ZARR not available or not installed.")


        workflowSteps = []
        step_id = 0
        # Generate the list of points to scan based on snake scan
        snake_tiles = self.generate_snake_tiles(mExperiment)
        # Flatten all point dictionaries from all tiles to compute scan range
        all_points = [pt for tile in snake_tiles for pt in tile]

        minX = min(pt["x"] for pt in all_points)
        maxX = max(pt["x"] for pt in all_points)
        minY = min(pt["y"] for pt in all_points)
        maxY = max(pt["y"] for pt in all_points)
        # compute step between two adjacent points in X/Y
        diffX = np.diff(np.unique([pt["x"] for pt in all_points])).min()
        diffY = np.diff(np.unique([pt["y"] for pt in all_points])).min()
        mPixelSize = self.detectorPixelSize[-1]  # Pixel size in µm

        for t in range(nTimes):
            # Loop over each tile (each central point and its neighbors)
            for tile in snake_tiles:
                for mIndex, mPoint in enumerate(tile):
                    try:
                        name = f"Move to point {mPoint['iterator']}"
                    except Exception:
                        name = f"Move to point {mPoint['x']}, {mPoint['y']}"
                    
                    workflowSteps.append(WorkflowStep(
                        name=name,
                        step_id=str(step_id),
                        main_func=self.move_stage_xy,
                        main_params={"posX": mPoint["x"], "posY": mPoint["y"], "relative": False},
                    ))
                    step_id += 1
                    # Turn on illumination (example with "illumination" parameter)
                    workflowSteps.append(WorkflowStep(
                        name="Turn on illumination",
                        step_id=str(step_id),
                        main_func=self.set_laser_power,
                        main_params={"power": laserWaveLength, "channel": illuSource},
                        post_funcs=[self.wait_time],
                        post_params={"seconds": 0.05},
                    ))
                    step_id += 1

                    # Acquire frame
                    try:exposure = p.exposureTime
                    except: exposure = 0.1
                    try:gain = p.gain   
                    except: gain = 1.0
                    isPreview = False
                    '''
                    workflowSteps.append(WorkflowStep(
                        name="Acquire frame",
                        step_id=str(step_id),
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
                        step_id=str(step_id),
                        main_func=self.acquire_frame,
                        main_params={"channel": "Mono"},
                        post_funcs=[self.save_frame_tiff, self.save_frame_ome_zarr, self.add_image_to_canvas],
                        pre_funcs=[self.set_exposure_time_gain],
                        pre_params={"exposure_time": exposure, "gain": gain},
                        post_params={
                            "posX": mPoint["x"],
                            "posY": mPoint["y"],
                            "minX": minX, "minY": minY, "maxX": maxX, "maxY": maxY,
                            "channel": illuSource,
                            "time_index": t,       # or whatever loop index
                            "tile_index": mIndex   # or snake-tile index
                        },
                    ))
                    step_id += 1

                    # Turn off illumination
                    workflowSteps.append(WorkflowStep(
                        name="Turn off illumination",
                        step_id=str(step_id),
                        main_func=self.set_laser_power,
                        main_params={"power": 0, "channel": illuSource},
                    ))
                    step_id += 1

            # (Optional) Perform autofocus once per loop, if enabled
            if isAutoFocus:
                workflowSteps.append(WorkflowStep(
                    name="Autofocus",
                    step_id=str(step_id),
                    main_func=self.autofocus,
                    main_params={},
                ))
                step_id += 1

            # (Optional) Add a wait period between each loop
            workflowSteps.append(WorkflowStep(
                name="Wait for next frame",
                step_id=str(step_id),
                main_func=self.dummy_main_func,
                main_params={},
                pre_funcs=[self.wait_time],
                pre_params={"seconds": 0.1}
            ))
            step_id += 1

        # Close TIFF writer at the end
        workflowSteps.append(WorkflowStep(
            name="Close TIFF writer",
            step_id=str(step_id) + "_done",
            main_func=self.close_tiff_writer,
            main_params={"tiff_writer": tiff_writer},
        ))

        if IS_OMEZARR_AVAILABLE:
            # Close OME-Zarr store at the end (new)
            workflowSteps.append(WorkflowStep(
                name="Close OME-Zarr store",
                step_id=str(step_id) + "_close_omezarr",
                main_func=self.close_ome_zarr_store,
                main_params={"omezarr_store": ome_store},
            ))
            step_id += 1
        
        # Emit final canvas
        workflowSteps.append(WorkflowStep(
            name="Emit Final Canvas",
            step_id=str(step_id),
            main_func=self.dummy_main_func,
            main_params={},
            pre_funcs=[self.emit_final_canvas],
            pre_params={}
        ))
        step_id += 1

        # Final step: mark done
        workflowSteps.append(WorkflowStep(
            name="Done",
            step_id=str(step_id) + "_done2",
            main_func=self.dummy_main_func,
            main_params={},
            pre_funcs=[self.wait_time],
            pre_params={"seconds": 0.1}
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

        context.set_object("tiff_writer", tiff_writer)
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

        if self.mDetector._isRGB:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint16) # numpy assigns y,x,z
        context.set_object("canvas", canvas)
        context.on("progress", sendProgress)
        context.on("rgb_stack", sendProgress)
        
        context.on("progress", sendProgress)
        context.on("rgb_stack", sendProgress)

        # Start the workflow
        self.workflow_manager.start_workflow(wf, context)

        return {"status": "running", "store_path": mFilePath}




    ########################################
    # Hardware-related functions
    ########################################
    def acquire_frame(self, channel: str):
        self._logger.debug(f"Acquiring frame on channel {channel}")
        mFrame = self.mDetector.getLatestFrame()
        return mFrame
    
    def set_exposure_time_gain(self, exposure_time: float, gain: float, context: WorkflowContext, metadata: Dict[str, Any]):
        if gain and gain >=0:
            self._logger.error(f"Setting gain to {gain}")
        if exposure_time and exposure_time >=0:
            self._logger.error(f"Setting exposure time to {exposure_time}")
        
    def dummy_main_func(self):
        self._logger.debug("Dummy main function called")
        return True
 
    def autofocus(self):
        self._logger.error("Performing autofocus...NOT IMPLEMENTED")
   
    def wait_time(self, seconds: int, context: WorkflowContext, metadata: Dict[str, Any]):
        import time
        time.sleep(seconds)

    def save_frame_tiff(self, context: WorkflowContext, metadata: Dict[str, Any], **kwargs):
        # Retrieve the TIFF writer and write the tile
        tiff_writer = context.get_object("tiff_writer")
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
        omeZarrStore.write(img, x=posX, y=posY, z=posZ)
        time.sleep(0.01)

    def add_image_to_canvas(self, context: WorkflowContext, metadata: Dict[str, Any], emitCurrentProgress=True, **kwargs):
        # Retrieve the canvas and add the image
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
            tif.imsave("final_canvas.tif", final_canvas)  
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

    def close_tiff_writer(self, tiff_writer: tif.TiffWriter):
        if tiff_writer is not None:
            tiff_writer.close()
        else:
            raise ValueError("TIFF writer is not initialized.")

        
    def move_stage_xy(self, posX: float, posY: float, relative: bool = False):
        # {"task":"/motor_act",     "motor":     {         "steppers": [             { "stepperid": 1, "position": -1000, "speed": 30000, "isabs": 0, "isaccel":1, "isen":0, "accel":500000}     ]}}
        self._logger.debug(f"Moving stage to X={posX}, Y={posY}")
        self.mStage.move(value=(posX, posY), speed=(self.SPEED_X, self.SPEED_Y), axis="XY", is_absolute=not relative, is_blocking=True)
        newPosition = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([newPosition["X"], newPosition["Y"]])
        return (newPosition["X"], newPosition["Y"])

    def move_stage_z(self, posZ: float, relative: bool = False):
        self._logger.debug(f"Moving stage to Z={posZ}")
        self.mStage.move(value=posZ, speed=self.SPEED_Z, axis="Z", is_absolute=not relative, is_blocking=True)
        newPosition = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([newPosition["Z"]])
        return newPosition["Z"]

    def autofocus(self, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug("Performing autofocus...")
        metadata["autofocus_done"] = True


            
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

# Copyright (C) 2025 Benedict Diederich
