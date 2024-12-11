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
import zarr
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url
from ome_zarr.format import CurrentFormat
from typing import Callable, List, Dict, Any, Optional
import threading
import json
import traceback
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, Query
from collections import deque
from tempfile import TemporaryDirectory
from iohub.ngff import open_ome_zarr
import uvicorn
import os 

isZARR=True

try:
    from ashlarUC2 import utils
    from ashlar.scripts.ashlar import process_images
    IS_ASHLAR_AVAILABLE = True
except ImportError:
    IS_ASHLAR_AVAILABLE = False


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
    currentPosition: Optional[Tuple[float, float]] = None
    isWorkflowRunning: bool = False
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
  
class WorkflowController(LiveUpdatedController):
    """Linked to WorkflowWidget."""

    sigImageReceived = Signal()
    sigUpdatePartialImage = Signal()
    sigUpdateWorkflowState = Signal(str)

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
        '''
        self.sigImageReceived.connect(self.displayImage)
        self.sigUpdatePartialImage.connect(self.updatePartialImage)
        self._commChannel.sigUpdateMotorPosition.connect(self.updateAllPositionGUI)
        self._commChannel.sigStartTileBasedTileScanning.connect(self.startWorkflowTileBasedByParameters)
        self._commChannel.sigStopTileBasedTileScanning.connect(self.stopWorkflowTilebased)
        '''
        
        self.workflow_manager = WorkflowsManager()

        
        # define scan parameter per sample and populate into GUI later
        self.allScanParameters = []
        mFWD = os.path.dirname(os.path.realpath(__file__)).split("imswitch")[0]
        '''
        self.allScanParameters.append(ScanParameters("6 Wellplate", 126, 86, 0, 0, mFWD+"imswitch/_data/images/Wellplate6.png"))
        self.allScanParameters.append(ScanParameters("24 Wellplate", 126, 86, 0, 0, mFWD+"imswitch/_data/images/Wellplate24.png"))
        self.allScanParameters.append(ScanParameters("3-Slide Wellplateadapter", 164, 109, 0, 0, mFWD+"imswitch/_data/images/WellplateAdapter3Slides.png"))
        '''
        # compute optimal scan step size based on camera resolution and pixel size
        self.bestScanSizeX, self.bestScanSizeY = self.computeOptimalScanStepSize2(overlap = self.initialOverlap)

    @APIExport()
    def computeOptimalScanStepSize2(self, overlap: float = 0.75):
        mFrameSize = (self.mDetector._camera.SensorHeight, self.mDetector._camera.SensorWidth)
        bestScanSizeX = mFrameSize[1]*self.mDetector.pixelSizeUm[-1]*overlap
        bestScanSizeY = mFrameSize[0]*self.mDetector.pixelSizeUm[-1]*overlap     
        return (bestScanSizeX, bestScanSizeY)

    def turnOnLED(self):
        if self.led is not None:
            self.led.setEnabled(1)
            self.led.setValue(255)
    
    def turnOffLED(self):
        if self.led is not None:
            self.led.setEnabled(0)

    
    def updateAllPositionGUI(self):
        allPositions = self.mStage.position
        if not IS_HEADLESS: self._widget.updateBoxPosition(allPositions["X"], allPositions["Y"])

    def valueIlluChanged(self):
        illuSource = self._widget.getIlluminationSource()
        illuValue = self._widget.illuminationSlider.value()
        self._master.lasersManager
        if not self._master.lasersManager[illuSource].enabled:
            self._master.lasersManager[illuSource].setEnabled(1)
        
        illuValue = illuValue/100*self._master.lasersManager[illuSource].valueRangeMax
        self._master.lasersManager[illuSource].setValue(illuValue)

        
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
    def getWorkflowStatus(self):
        return True

    @APIExport()
    def setWorkflowStatus(self, status: HistoStatus) -> bool:
        return True


    '''
    Device Functions
    '''
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
        self._logger.debug(f"Setting laser power to {power} for channel {channel}")
        return power

    def acquire_frame(self, channel: str):
        self._logger.debug(f"Acquiring frame on channel {channel}")
        mFrame = self.mDetector.getLatestFrame()
        return mFrame

    def process_data(self, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Processing data for step {metadata['step_id']}...")
        metadata["processed"] = True

    def save_frame(self, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Saving frame for step {metadata['step_id']}...")
        metadata["frame_saved"] = True
        
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
        zarr_writer["tiles"].write_tile(img, row, col)
        metadata["frame_saved"] = True
        
    def wait_time(self, seconds: int, context: WorkflowContext, metadata: Dict[str, Any]):
        import time
        time.sleep(seconds)
        metadata["waited"] = seconds
        
    def addFrametoFile(self, frame:np.ndarray, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Adding frame to file for step {metadata['step_id']}...")
        metadata["frame_added"] = True
        
    def append_data(self, context: WorkflowContext, metadata: Dict[str, Any]):
        obj = context.get_object("data_buffer")
        if obj is not None:
            obj.append(metadata["result"])

    def compute_scan_positions(self, x_min, x_max, y_min, y_max, x_step, y_step):
        # Compute a grid of (x,y) positions
        xs = [x_min + i * x_step for i in range(int((x_max - x_min) / x_step) + 1)]
        ys = [y_min + j * y_step for j in range(int((y_max - y_min) / y_step) + 1)]
        return xs, ys

    def close_zarr(self, context: WorkflowContext, metadata: Dict[str, Any]):
        zarr_writer = context.get_object("zarr_writer")
        if zarr_writer is not None:
            zarr_writer.close()
            context.remove_object("zarr_writer")
            metadata["zarr_closed"] = True

    ########################################
    # Histo-Slide Scanner Interface
    ########################################



    @APIExport()
    def start_workflow(self,
        x_min: float = Query(...),
        x_max: float = Query(...),
        y_min: float = Query(...),
        y_max: float = Query(...),
        x_step: float = Query(100.0),
        y_step: float = Query(100.0),
        autofocus_on: bool = Query(False),
        channel: str = Query("Mono")
    ):
        
        if self.workflow_manager.get_status()["status"] in ["running", "paused", "stopping"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")
        
        # Compute the scan positions
        xs, ys = self.compute_scan_positions(x_min, x_max, y_min, y_max, x_step, y_step)

        # Setup Zarr store
        tmp_dir = TemporaryDirectory()
        store_path = os.path.join(tmp_dir.name, "tiled.zarr")
        self._logger.debug("Zarr store path: "+store_path)
        
        # Let's assume single channel "Mono" for simplicity, but can adapt for more.
        dataset = open_ome_zarr(store_path, layout="tiled", mode="a", channel_names=[channel])
        # Calculate grid shape based on the number of xy positions
        grid_shape = (len(ys), len(xs))
        tile_shape = (self.mDetector._camera.SensorHeight, self.mDetector._camera.SensorWidth)
        dtype = "uint16"
        tiles = dataset.make_tiles("tiled_raw", grid_shape=grid_shape, tile_shape=tile_shape, dtype=dtype)

        # Create workflow steps
        # Autofocus mode:
        # if autofocus_on == True: run autofocus before every XY move
        # else no autofocus pre-func
        pre_for_xy = [self.autofocus] if autofocus_on else []

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

                # Set laser power (arbitrary, could be parameterized)
                workflowSteps.append(WorkflowStep(
                    name=f"Set laser power",
                    step_id=str(step_id),
                    main_func=self.set_laser_power,
                    main_params={"power": 10, "channel": channel},
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
        wf = Workflow(workflowSteps)
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

# With a properly structured JSON config, you can load various workflows dynamically. 
# This addresses automating workflow generation for different imaging applications.
