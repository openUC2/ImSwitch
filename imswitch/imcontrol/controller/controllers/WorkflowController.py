import json
import os
import base64
from fastapi import FastAPI, Response, HTTPException
from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext, WorkflowStep, WorkflowsManager
from imswitch import IS_HEADLESS
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
from pydantic import BaseModel
from typing import List, Optional, Union, Tuple

from typing import List, Dict, Any, Optional
import threading
import json
import numpy as np
from pydantic import BaseModel
from fastapi import Query
from collections import deque
from tempfile import TemporaryDirectory
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid



isZARR=True

try:
    from ashlarUC2 import utils
    IS_ASHLAR_AVAILABLE = True
except Exception as e:
    IS_ASHLAR_AVAILABLE = False

from pydantic import BaseModel
from typing import List, Tuple, Dict

class XYZScanRequest(BaseModel):
    coords: List[Tuple[float, float, float]]
    illuminations: List[Dict[str, float]]  # e.g. [{"LED": 100}, {"Laser": 50}]
    file_name: str
    autofocus_on: bool = False


class ScanParameters(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_step: float = 100.0
    y_step: float = 100.0
    autofocus: bool = False
    channel: str = "Mono"
    tile_shape: List[int] = [512, 512]
    dtype: str = "uint16"


class HistoStatus(BaseModel):
    currentPosition: Optional[Tuple[float, float, int, int]] = None  # X, Y, nX, nY
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

    # GUI-Specific Parameters
    illuminationSources: List[str] = ["Brightfield", "Darkfield", "Laser", "DPC"]
    selectedIllumination: Optional[str] = "Brightfield"
    laserWavelengths: List[float] = [405, 488, 532, 635, 785, 10]  # in nm
    selectedLaserWavelength: float = 488.0

    # Time-lapse parameters
    timeLapsePeriod: float = 330.4  # s
    timeLapsePeriodMin: float = 1.0
    timeLapsePeriodMax: float = 1000.0
    numberOfImages: int = 652
    numberOfImagesMin: int = 1
    numberOfImagesMax: int = 1000

    # Autofocus parameters
    autofocusMinFocusPosition: float = 0.0
    autofocusMaxFocusPosition: float = 0.0
    autofocusStepSize: float = 0.1
    autofocusStepSizeMin: float = 0.01
    autofocusStepSizeMax: float = 10.0

    # Z-Stack parameters
    zStackMinFocusPosition: float = 0.0
    zStackMaxFocusPosition: float = 0.0
    zStackStepSize: float = 0.1
    zStackStepSizeMin: float = 0.01
    zStackStepSizeMax: float = 10.0

    @staticmethod
    def from_dict(status_dict: dict) -> "HistoStatus":
        return HistoStatus(**status_dict)

    def to_dict(self) -> dict:
        return self.dict()


class WorkflowStepDefinition(BaseModel):
    id: str
    stepName: str
    mainFuncName: str
    mainParams: Dict[str, Any] = Field(default_factory=dict)
    preFuncs: List[str] = Field(default_factory=list)
    preParams: Dict[str, Any] = Field(default_factory=dict)
    postFuncs: List[str] = Field(default_factory=list)
    postParams: Dict[str, Any] = Field(default_factory=dict)

class WorkflowDefinition(BaseModel):
    steps: List[WorkflowStepDefinition] = Field(default_factory=list)
    # optionally add other metadata fields if needed

class StartWorkflowRequest(BaseModel):
    workflow_id: str

class StitchedImageResponse(BaseModel):
    imageList: List[List[float]]
    image: str

class WorkflowController(LiveUpdatedController):
    """Linked to WorkflowWidget."""

    sigImageReceived = Signal()
    sigUpdatePartialImage = Signal()
    sigUpdateWorkflowState = Signal(str)
    sigStartHistoWorkflow = Signal(float, float, float, float, float, float, bool, list)
    sigStartMultiColorTimelapseWorkflow = Signal(int, int, list, list, list, list, bool)


    WORKFLOW_STORAGE: Dict[str, WorkflowDefinition] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # read offset between cam and microscope from config file in µm
        self.offsetCamMicroscopeX = -2500 #  self._master.WorkflowManager.offsetCamMicroscopeX
        self.offsetCamMicroscopeY = 2500 #  self._master.WorkflowManager.offsetCamMicroscopeY
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
        self.initialOverlap = 0.85

        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.mDetector = self._master.detectorsManager[allDetectorNames[0]] # FIXME: This is hardcoded, need to be changed through the GUI

        # some locking mechanisms
        self.isWorkflowRunning = False
        self.isWorkflowRunning = False

        # Lasers
        allLaserNames = self._master.lasersManager.getAllDeviceNames()
        if "LED" in allLaserNames:
            self.led = self._master.lasersManager["LED"]
        else:
            self.led = None

        # Stages
        self.mStage = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]

        # connect signals
        self.sigStartHistoWorkflow.connect(self.start_xyz_histo_workflow)
        self.workflow_manager = WorkflowsManager()

        # define scan parameter per sample and populate into GUI later
        self.allScanParameters = []
        mFWD = os.path.dirname(os.path.realpath(__file__)).split("imswitch")[0]

        # compute optimal scan step size based on camera resolution and pixel size
        self.bestScanSizeX, self.bestScanSizeY = self.computeOptimalScanStepSize2(overlap = self.initialOverlap)

        # define function registry for workflow steps in API
        self.function_registry = {
            "move_stage_xy": self.move_stage_xy,
            "acquire_frame": self.acquire_frame,
            "set_laser_power": self.set_laser_power,
            "process_data": self.process_data,
            "save_frame_zarr": self.save_frame_zarr,
            "wait_time": self.wait_time,
            "autofocus": self.autofocus,
            "close_zarr": self.close_zarr,
            "save_frame": self.save_frame,
            "set_exposure_time_gain": self.set_exposure_time_gain,
            "dummy_main_func": self.dummy_main_func,
            "addFrametoFile": self.addFrametoFile,
            "append_data": self.append_data,
            "save_data": self.save_data,
            "save_frame_tiff": self.save_frame_tiff,
            "close_tiff_writer": self.close_tiff_writer
        }

    ########################################
    # Helper Functions
    ########################################

    def compute_scan_positions(self, x_min, x_max, y_min, y_max, x_step, y_step):
        # Compute a grid of (x,y) positions
        xs = [x_min + i * x_step for i in range(int((x_max - x_min) / x_step) + 1)]
        ys = [y_min + j * y_step for j in range(int((y_max - y_min) / y_step) + 1)]
        return xs, ys


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

    @APIExport()
    def computeOptimalScanStepSize2(self, overlap: float = 0.75):
        mFrameSize = (self.mDetector._camera.SensorHeight, self.mDetector._camera.SensorWidth)
        bestScanSizeX = mFrameSize[1]*self.mDetector.pixelSizeUm[-1]*overlap
        bestScanSizeY = mFrameSize[0]*self.mDetector.pixelSizeUm[-1]*overlap
        return (bestScanSizeX, bestScanSizeY)

    ########################################
    # Workflow Functions
    ########################################

    def move_stage_xy(self, posX: float, posY: float, relative: bool = False):
        # {"task":"/motor_act",     "motor":     {         "steppers": [             { "stepperid": 1, "position": -1000, "speed": 30000, "isabs": 0, "isaccel":1, "isen":0, "accel":500000}     ]}}
        self._logger.debug(f"Moving stage to X={posX}, Y={posY}")
        self.mStage.move(value=(posX, posY), axis="XY", is_absolute=not relative, is_blocking=True)
        newPosition = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([newPosition["X"], newPosition["Y"]])
        return (newPosition["X"], newPosition["Y"])

    def move_stage_z(self, posZ: float, relative: bool = False):
        self._logger.debug(f"Moving stage to Z={posZ}")
        self.mStage.move(value=posZ, axis="Z", is_absolute=not relative, is_blocking=True)
        newPosition = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([newPosition["Z"]])
        return newPosition["Z"]

    def autofocus(self, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug("Performing autofocus...")
        metadata["autofocus_done"] = True

    def save_data(self, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Saving data for step {metadata['step_id']}")
        context.update_metadata(metadata["step_id"], "saved", True)

    def set_laser_power(self, power: float, channel: str):
        mLaserNames = self._master.lasersManager.getAllDeviceNames()
        if channel not in mLaserNames:
            self._logger.error(f"Channel {channel} not found in available lasers: {mLaserNames}")
            return None
        self._master.lasersManager[channel].setValue(power)
        if self._master.lasersManager[channel].enabled == 0:
            self._master.lasersManager[channel].setEnabled(1)
        self._logger.debug(f"Setting laser power to {power} for channel {channel}")
        return power

    def dummy_main_func(self):
        self._logger.debug("Dummy main function called")
        return True

    def acquire_frame(self, channel: str):
        self._logger.debug(f"Acquiring frame on channel {channel}")
        mFrame = self.mDetector.getLatestFrame()
        return mFrame

    def set_exposure_time_gain(self, exposure_time: float, gain: float, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Setting exposure time to {exposure_time}")
        return exposure_time


    def process_data(self, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Processing data for step {metadata['step_id']}...")
        metadata["processed"] = True

    def save_frame(self, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Saving frame for step {metadata['step_id']}...")
        metadata["frame_saved"] = True

    def save_frame_tiff(self, context: WorkflowContext, metadata: Dict[str, Any]):
        # Retrieve the TIFF writer and write the tile
        tiff_writer = context.get_object("tiff_writer")
        if tiff_writer is None:
            self._logger.debug("No TIFF writer found in context!")
            return
        img = metadata["result"]
        # append the image to the tiff file
        tiff_writer.save(img)
        metadata["frame_saved"] = True

    def close_tiff_writer(self):
        if self.tiff_writer is not None:
            self.tiff_writer.close()
        else:
            raise ValueError("TIFF writer is not initialized.")

    def save_frame_zarr(self, context: "WorkflowContext", metadata: Dict[str, Any]):
        # Retrieve the Zarr writer and write the tile
        zarr_writer = context.get_object("zarr_writer")
        if zarr_writer is None:
            self._logger.debug("No Zarr writer found in context!")
            return
        img = metadata["result"]
        # Compute tile indices (row/col) from metadata
        # This depends on how we map x,y coordinates to grid indices
        col = context.data.get("global", {}).get("last_col")
        row = context.data.get("global", {}).get("last_row")
        if col is not None and row is not None:
            metadata["IndexX"] = col
            metadata["IndexY"] = row
        self._logger.debug(f"Saving frame tile at row={row}, column={col}")
        try:zarr_writer["tiles"].write_tile(img, row, col)
        except:zarr_writer["tiles"].write_tile(img.T, row, col)
        metadata["frame_saved"] = True

    def wait_time(self, seconds: int, context: WorkflowContext, metadata: Dict[str, Any]):
        import time
        time.sleep(seconds)

    def addFrametoFile(self, frame:np.ndarray, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Adding frame to file for step {metadata['step_id']}...")
        metadata["frame_added"] = True

    def append_data(self, context: WorkflowContext, metadata: Dict[str, Any]):
        obj = context.get_object("data_buffer")
        if obj is not None:
            obj.append(metadata["result"])

    def close_zarr(self, context: WorkflowContext, metadata: Dict[str, Any]):
        zarr_writer = context.get_object("zarr_writer")
        if zarr_writer is not None:
            zarr_writer.close()
            context.remove_object("zarr_writer")
            metadata["zarr_closed"] = True


    ########################################
    # Control Flow Functions
    ########################################

    @APIExport()
    def stopWorkflow(self):
        self.isWorkflowRunning = False

    @APIExport()
    def startWorkflowTileBasedByParameters(self, numberTilesX:int=2, numberTilesY:int=2, stepSizeX:int=100, stepSizeY:int=100,
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
            stepSizeX, _ = self.computeOptimalScanStepSize2()
        if stepSizeY<=0 or stepSizeY is None:
            _, stepSizeY = self.computeOptimalScanStepSize2()
        if initPosX is None or type(initPosX)==str :
            initPosX = self.mStage.getPosition()["X"]
        if initPosY is None or type(initPosY)==str:
            initPosY = self.mStage.getPosition()["Y"]

    @APIExport()
    def pause_workflow(self):
        status = self.workflow_manager.get_status()["status"]
        if status == "running":
            return self.workflow_manager.pause_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot pause in current state: {status}")

    @APIExport()
    def resume_workflow(self):
        status = self.workflow_manager.get_status()["status"]
        if status == "paused":
            return self.workflow_manager.resume_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot resume in current state: {status}")

    @APIExport()
    def stop_workflow(self):
        status = self.workflow_manager.get_status()["status"]
        if status in ["running", "paused"]:
            return self.workflow_manager.stop_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot stop in current state: {status}")

    @APIExport()
    def workflow_status(self):
        return self.workflow_manager.get_status()

    @APIExport()
    def force_stop_workflow(self):
        self.workflow_manager.stop_workflow()
        del self.workflow_manager
        self.workflow_manager = WorkflowsManager()

    @APIExport()
    def setWorkflowStatus(self, status: HistoStatus) -> bool:
        return True

    @APIExport()
    def getWorkflowStatus(self) -> HistoStatus:
        return HistoStatus.from_dict(self.status.to_dict())

    ########################################
    # Transmit Worflow Definition via API
    ########################################

    @APIExport(requestType="POST")
    def uploadWorkflow(self, definition: WorkflowDefinition):
        workflow_id = str(uuid.uuid4())
        self.WORKFLOW_STORAGE[workflow_id] = definition
        return {"workflow_id": workflow_id, "status": "stored"}

    @APIExport(requestType="POST")
    def start_workflow_api(self, request: StartWorkflowRequest):
        workflow_id = request.workflow_id
        definition = self.WORKFLOW_STORAGE.get(workflow_id)
        if not definition:
            raise HTTPException(status_code=404, detail="Workflow definition not found")

        # Convert to actual WorkflowSteps
        steps = []
        for step_def in definition.steps:
            main_func = self.function_registry.get(step_def.mainFuncName)
            if not main_func:
                raise HTTPException(status_code=400, detail=f"Unknown function {step_def.mainFuncName}")

            pre_funcs = [self.function_registry[f] for f in step_def.preFuncs if f in self.function_registry]
            post_funcs = [self.function_registry[f] for f in step_def.postFuncs if f in self.function_registry]

            step = WorkflowStep(
                name=step_def.stepName,
                main_func=main_func,
                main_params=step_def.mainParams,
                step_id=step_def.id,
                pre_funcs=pre_funcs,
                pre_params=step_def.preParams,
                post_funcs=post_funcs,
                post_params=step_def.postParams
            )
            steps.append(step)

        # (Edges might define ordering or concurrency—this is up to you
        #  how to interpret or linearize them. For a simple linear workflow,
        #  you might just rely on the array order or define a topological sort.)
        def sendProgress(payload):
            self.sigUpdateWorkflowState.emit(payload)

        wf = Workflow(steps, self.workflow_manager)
        context = WorkflowContext()

        # Insert the tiff writer object into context so `save_frame` can use it
        tiff_writer = tifffile.TiffWriter("timelapse.tif")
        context.set_object("tiff_writer", tiff_writer)
        context.on("progress", sendProgress)
        # Run the workflow
        # context = wf.run_in_background(context)
        self.workflow_manager.start_workflow(wf, context)

        return {"status": "started", "workflow_id": workflow_id}


    ########################################
    # Example Workflow for Histo-Slide Scanner Interface
    ########################################

    @APIExport()
    def start_xyz_histo_workflow(self,
        x_min: float = Query(...),
        x_max: float = Query(...),
        y_min: float = Query(...),
        y_max: float = Query(...),
        x_step: float = Query(100.0),
        y_step: float = Query(100.0),
        autofocus_on: bool = Query(False),
        channel: str = Query("Mono")
    ):

        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        # Start the detector if not already running
        if not self.mDetector._running: self.mDetector.startAcquisition()

        # Compute the scan positions
        xs, ys = self.compute_scan_positions(x_min, x_max, y_min, y_max, x_step, y_step)

        # Setup Zarr store
        tmp_dir = TemporaryDirectory()
        store_path = os.path.join(tmp_dir.name, "tiled.zarr")
        self._logger.debug("Zarr store path: "+store_path)

        # Let's assume single channel "Mono" for simplicity, but can adapt for more.
        def open_ome_zarr(store_path, layout="tiled", mode="a", channel_names=None):
            return None # TODO: Implement this function to open or create a Zarr dataset with the specified layout and channel names.
        dataset = open_ome_zarr(store_path, layout="tiled", mode="a", channel_names=[channel])
        # Calculate grid shape based on the number of xy positions
        grid_shape = (len(ys), len(xs))
        tile_shape = (self.mDetector._camera.SensorHeight, self.mDetector._camera.SensorWidth)
        if self.mDetector._camera.SensorHeight>self.mDetector._camera.SensorWidth: tile_shape = (self.mDetector._camera.SensorWidth, self.mDetector._camera.SensorHeight)
        dtype = "uint16"
        tiles = dataset.make_tiles("tiled_raw", grid_shape=grid_shape, tile_shape=tile_shape, dtype=dtype)
        # Create workflow steps
        # Autofocus mode:
        # if autofocus_on == True: run autofocus before every XY move
        # else no autofocus pre-func
        pre_for_xy = [self.autofocus] if autofocus_on else []

        illuSources = ["LED", "Laser"]

        workflowSteps = []
        step_id = 0
        # We'll add a small function that updates metadata with tile indices for saving
        def update_tile_indices(context: WorkflowContext, metadata: Dict[str, Any]):
            # Based on metadata["x"] and metadata["y"], find their indices in xs, ys
            x_val = metadata["posX"]
            y_val = metadata["posY"]
            col = xs.index(x_val)
            row = ys.index(y_val)
            # Store indices so save_frame can use them
            metadata["IndexX"] = col
            metadata["IndexY"] = row
            context.update_metadata("global", "last_col", col)
            context.update_metadata("global", "last_row", row)

        # In this simplified example, we only do a single Z position (z=0)
        # and a single frame per position. You can easily extend this.
        z_pos = 0
        frames = [0]  # single frame index for simplicity


        for y_i, y_pos in enumerate(ys):
            # we want to have snake scan pattern
            if y_i % 2 == 0:
                xs = xs
            else:
                xs = xs[::-1]
            for x_i, x_pos in enumerate(xs):
                # Move XY
                workflowSteps.append(WorkflowStep(
                    name=f"Move XY to ({x_pos}, {y_pos})",
                    main_func=self.move_stage_xy,
                    main_params={"posX": x_pos, "posY": y_pos, "relative": False},
                    step_id=str(step_id),
                    pre_funcs=pre_for_xy,
                    post_funcs=[update_tile_indices]
                ))
                step_id += 1

                # Move Z (we keep fixed z=0 here for simplicity)
                workflowSteps.append(WorkflowStep(
                    name=f"Move Z to {z_pos}",
                    step_id=str(step_id),
                    main_func=self.move_stage_z,
                    main_params={"posZ": z_pos, "relative": False},
                    pre_funcs=[],
                    post_funcs=[]
                ))
                step_id += 1

                for illu in illuSources:
                    # Set laser power (arbitrary, could be parameterized)
                    workflowSteps.append(WorkflowStep(
                        name=f"Set laser power",
                        step_id=str(step_id),
                        main_func=self.set_laser_power,
                        main_params={"power": 10, "channel": illu},
                        pre_funcs=[],
                        post_funcs=[]
                    ))
                    step_id += 1

                    for fr in frames:
                        # Acquire frame with a short wait, process data, and save frame
                        workflowSteps.append(WorkflowStep(
                            name=f"Acquire frame {channel}",
                            step_id=str(step_id),
                            main_func=self.acquire_frame,
                            main_params={"channel": channel},
                            pre_funcs=[self.wait_time],
                            pre_params={"seconds": .1},
                            post_funcs=[self.process_data, self.save_frame_zarr],
                        ))
                        step_id += 1

        # Close Zarr dataset at the end
        workflowSteps.append(WorkflowStep(
            name="Close Zarr dataset",
            step_id=str(step_id),
            main_func=self.close_zarr,
            main_params={},
        ))

        def sendProgress(payload):
            self.sigUpdateWorkflowState.emit(payload)

        # Create a workflow and context
        wf = Workflow(steps=workflowSteps, workflow=self.workflow_manager)
        context = WorkflowContext()
        # Insert the zarr writer object into context so `save_frame` can use it
        context.set_object("zarr_writer", {"tiles": tiles})
        context.set_object("data_buffer", deque())  # example if needed
        context.on("progress", sendProgress)
        # Run the workflow
        # context = wf.run_in_background(context)
        self.workflow_manager.start_workflow(wf, context)

        #context = wf.run(context)
        # questions
        # How can I pause a running thread?
        # we would need a handle on the running thread to pause it
        # We should not run yet another workflow and wait for the first one to finish


        # Return the store path to the client so they know where data is stored
        return {"status": "completed", "zarr_store_path": store_path}#, "results": context.data}






    @APIExport(requestType="POST")
    def start_xyz_histo_workflow_by_list(self,
        req: XYZScanRequest,  # The pydantic model above
    ):
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        if not self.mDetector._running:
            self.mDetector.startAcquisition()

        coords = req.coords
        illuminations = req.illuminations
        file_name = req.file_name
        autofocus_on = req.autofocus_on

        # Either use `file_name` directly, or create a temporary directory
        # For example, if file_name is just a base, store a .zarr there:
        store_path = file_name + str(np.random.randint(0,100000))
        self._logger.debug("Zarr store path: " + store_path)

        def open_ome_zarr(store_path, layout="tiled", mode="a", channel_names=None):
            """
            Open or create a Zarr dataset with the specified layout and channel names.
            This is a placeholder function, you need to implement it based on your Zarr library.
            """
            # todo: this is a placeholder, you need to implement the actual opening/creation logic
            import zarr
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            return zarr.open(store_path, mode=mode, shape=(0, 0), dtype="uint16", chunks=(1, 1), overwrite=True)
        dataset = open_ome_zarr(store_path, layout="tiled", mode="a", channel_names=["Mono"])
        # If the shape is unknown beforehand, you can guess or wait until images come in.
        # For demonstration, let's define a 512 x 512 tile shape:
        tile_shape = (512, 512)
        dtype = "uint16"
        # If you know how many positions exist, you could do:
        # grid_shape = (len(coords), len(illuminations)) # or something more advanced
        # but if it's purely 2D, you can do 1D scanning in "tiles"
        tiles = dataset.make_tiles("tiled_raw", grid_shape=(len(coords), len(illuminations)), tile_shape=tile_shape, dtype=dtype)

        # Build workflow steps
        workflowSteps = []
        step_id = 0

        # If we want to run autofocus before every XY move:
        pre_for_xy = [self.autofocus] if autofocus_on else []

        def update_tile_indices(context: WorkflowContext, metadata: Dict[str, Any]):
            """
            Example: If you want to store tile indices in metadata, do so here.
            This depends on how you want to map coords + illuminations to tile row/col.
            For simplicity, let row = index of coords, col = index of illumination
            stored in metadata.
            """
            row = metadata.get("coord_index", 0)
            col = metadata.get("illum_index", 0)
            metadata["IndexX"] = col
            metadata["IndexY"] = row
            context.update_metadata("global", "last_col", col)
            context.update_metadata("global", "last_row", row)

        for c_i, (x, y, z) in enumerate(coords):
            # Move XY
            workflowSteps.append(WorkflowStep(
                name=f"Move XY to ({x}, {y})",
                step_id=str(step_id),
                main_func=self.move_stage_xy,
                main_params={"posX": x, "posY": y, "relative": False},
                pre_funcs=pre_for_xy,
                post_funcs=[]
            ))
            step_id += 1

            # Move Z
            workflowSteps.append(WorkflowStep(
                name=f"Move Z to {z}",
                step_id=str(step_id),
                main_func=self.move_stage_z,
                main_params={"posZ": z, "relative": False},
                pre_funcs=[],
                post_funcs=[]
            ))
            step_id += 1

            for ill_i, illum_dict in enumerate(illuminations):
                # e.g. illum_dict = {"LED": 100} or {"Laser": 50, "AnotherChannel": 80}
                for channel_name, power in illum_dict.items():
                    # set laser power
                    workflowSteps.append(WorkflowStep(
                        name=f"Set {channel_name} power to {power}",
                        step_id=str(step_id),
                        main_func=self.set_laser_power,
                        main_params={"power": power, "channel": channel_name},
                        pre_funcs=[],
                        post_funcs=[]
                    ))
                    step_id += 1

                    # Acquire frame
                    # Possibly define a function that sets 'coord_index' and 'illum_index' in the metadata
                    # so you can compute tile positions in update_tile_indices

                    # have a seperate step to udpate indicees
                    def set_indices(context: WorkflowContext, metadata: Dict[str, Any]):
                        metadata["coord_index"] = c_i
                        metadata["illum_index"] = ill_i

                    workflowSteps.append(WorkflowStep(
                        name=f"Update indices",
                        main_func=self.dummy_main_func,
                        main_params={},
                        step_id=str(step_id),
                        pre_funcs=[set_indices],
                    ))

                    workflowSteps.append(WorkflowStep(
                        name=f"Acquire frame {channel_name}",
                        step_id=str(step_id),
                        main_func=self.acquire_frame,
                        main_params={"channel": "Mono"},  # or channel_name if your camera channel naming matches
                        pre_funcs=[self.wait_time],
                        pre_params={"seconds": 0.1},
                        post_funcs=[self.process_data, self.save_frame_zarr, update_tile_indices],
                    ))
                    step_id += 1

                    # Optionally turn off the illumination after acquiring
                    workflowSteps.append(WorkflowStep(
                        name=f"Turn off {channel_name}",
                        step_id=str(step_id),
                        main_func=self.set_laser_power,
                        main_params={"power": 0, "channel": channel_name},
                    ))
                    step_id += 1

        # Close Zarr dataset
        workflowSteps.append(WorkflowStep(
            name="Close Zarr dataset",
            step_id=str(step_id),
            main_func=self.close_zarr,
            main_params={},
        ))

        # Create Workflow & Context
        wf = Workflow(workflowSteps)
        context = WorkflowContext()
        context.set_object("zarr_writer", {"tiles": tiles})
        context.set_object("data_buffer", deque())
        # Start the workflow
        self.workflow_manager.start_workflow(wf, context)

        return {"status": "started", "store_path": store_path}
