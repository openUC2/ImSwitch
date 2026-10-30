from fastapi import HTTPException
from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext, WorkflowStep, WorkflowsManager
from imswitch.imcommon.model import initLogger, dirtools
from imswitch.imcontrol.model.aufgaben.storage import save_image
import os
import time
import numpy as np
from imswitch.imcommon.model import APIExport
from imswitch.imcommon.framework import Signal
from ..basecontrollers import LiveUpdatedController
from pydantic import BaseModel
from typing import List, Optional
from typing import Tuple

from typing import Dict, Any
from pydantic import Field
import uuid



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

class WorkflowController(LiveUpdatedController):
    """Linked to WorkflowWidget."""

    sigImageReceived = Signal()
    sigUpdatePartialImage = Signal()
    sigUpdateWorkflowState = Signal(str)


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
        self.workflow_manager = WorkflowsManager()

        # define scan parameter per sample and populate into GUI later
        self.allScanParameters = []

        # compute optimal scan step size based on camera resolution and pixel size
        self.bestScanSizeX, self.bestScanSizeY = self.computeOptimalScanStepSize2(overlap = self.initialOverlap)

        # define function registry for workflow steps in API
        self.function_registry = {
            "move_stage_xy": self.move_stage_xy,
            "acquire_frame": self.acquire_frame,
            "set_laser_power": self.set_laser_power,
            "process_data": self.process_data,
            "wait_time": self.wait_time,
            "autofocus": self.autofocus,
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
        #self._commChannel.sigUpdateMotorPosition.emit([newPosition["X"], newPosition["Y"]])
        return (newPosition["X"], newPosition["Y"])

    def move_stage_z(self, posZ: float, relative: bool = False):
        self._logger.debug(f"Moving stage to Z={posZ}")
        self.mStage.move(value=posZ, axis="Z", is_absolute=not relative, is_blocking=True)
        newPosition = self.mStage.getPosition()
        #self._commChannel.sigUpdateMotorPosition.emit([newPosition["Z"]])
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
        """Keep the step's frame; close_tiff_writer writes them all as one OME-TIFF."""
        context.data.setdefault("tiff_frames", []).append(np.asarray(metadata["result"]))
        metadata["frame_saved"] = True

    def close_tiff_writer(self, context: WorkflowContext, metadata: Dict[str, Any]):
        """The frames kept by save_frame_tiff as a time series in <data>/Workflow/."""
        frames = context.data.pop("tiff_frames", [])
        if not frames:
            return
        path = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), "Workflow",
                            time.strftime("%Y%m%d_%H%M%S") + ".ome.tif")
        save_image(path, np.stack(frames), "TYXS" if frames[0].ndim == 3 else "TYX")
        metadata["tiff_path"] = path


    def wait_time(self, seconds: float, context: WorkflowContext, metadata: Dict[str, Any]):
        context.sleep(seconds)  # returns early on stop

    def addFrametoFile(self, frame:np.ndarray, context: WorkflowContext, metadata: Dict[str, Any]):
        self._logger.debug(f"Adding frame to file for step {metadata['step_id']}...")
        metadata["frame_added"] = True

    def append_data(self, context: WorkflowContext, metadata: Dict[str, Any]):
        obj = context.get_object("data_buffer")
        if obj is not None:
            obj.append(metadata["result"])



    ########################################
    # Control Flow Functions
    ########################################

    @APIExport()
    def stopWorkflow(self):
        self.isWorkflowRunning = False


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

        context.on("progress", sendProgress)
        # Run the workflow
        # context = wf.run_in_background(context)
        self.workflow_manager.start_workflow(wf, context)

        return {"status": "started", "workflow_id": workflow_id}


    ########################################
    # Example Workflow for Histo-Slide Scanner Interface
    ########################################







