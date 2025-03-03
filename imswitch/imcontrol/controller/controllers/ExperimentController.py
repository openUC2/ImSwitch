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

class ExperimentWorkflowParams(BaseModel):
    currentPosition: Optional[Tuple[float, float]] = None  # X, Y, nX, nY
    currentIndexPosition: Optional[Tuple[int, int]] = None  # iX, iY
    isExpiermentRunning: bool = False
    stitchResultAvailable: bool = False
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

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        return self.dict()


class NeighborPoint(BaseModel):
    x: float
    y: float

class Point(BaseModel):
    id: uuid.UUID
    name: str
    x: float
    y: float
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

class Experiment(BaseModel):
    name: str
    parameterValue: ParameterValue
    pointList: List[Point]


class ExperimentController(ImConWidgetController):
    """Linked to ExperimentWidget."""

    sigExperimentWorkflowUpdate = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # initialize variables
        self.tWait = 0.1
        self.workflow_manager = WorkflowsManager()

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
            
        # define Experiment parameters as ExperimentWorkflowParams
        self.ExperimentParams = ExperimentWorkflowParams()
        
        
        # TODO: Adjust parameters 
        '''
        self.ExperimentParams.illuSources = self.allIlluNames
        self.ExperimentParams.illuSourceMinIntensities = [0]*len(self.ExperimentParams.illuSourcesSelected)
        self.ExperimentParams.illuSourceMaxIntensities = [100]*len(self.ExperimentParams.illuSourcesSelected)
        self.ExperimentParams.illuIntensities = [0]*len(self.allIlluNames)
        self.ExperimentParams.exposureTimes = [0]*len(self.allIlluNames)
        self.ExperimentParams.gain = [0]*len(self.allIlluNames)
        '''
        
    @APIExport(requestType="GET")
    def getCurrentExperimentParameters(self):
        return self.ExperimentParams
    
    @APIExport(requestType="POST")
    def setCurrentExperimentParameters(self, params: ExperimentWorkflowParams):
        self.ExperimentParams = params
        return self.ExperimentParams

    @APIExport(requestType="POST")
    def startWellplateExperiment(self, mExperiment: Experiment):
        # Extract key parameters
        exp_name = mExperiment.name
        p = mExperiment.parameterValue
        nTimes = p.numberOfImages
        tPeriod = p.timeLapsePeriod
        isAutoFocus = p.autoFocus
        zStackOn = p.zStack

        # Example usage of a single illumination source
        illuSource = p.illumination
        laserWaveLength = p.laserWaveLength

        # Check if another workflow is running
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        # Start the detector if not already running
        if not self.mDetector._running:
            self.mDetector.startAcquisition()

        workflowSteps = []
        step_id = 0

        # Example: Move to each point, take images, possibly do autofocus or z-stack
        for t in range(nTimes):
            # Loop over the list of points
            for mPointList in mExperiment.pointList:
                # for every point create a single-point or neighboring-points list to loop over
                if len(mPointList.neighborPointList)>0:
                    pointList = mPointList.neighborPointList
                else:
                    pointList = [mPointList]
                
                for point in pointList:
                    
                    try:
                        name = f"Move to point {point.id}"
                    except:
                        name = f"Move to point {point.x}, {point.y}"
                    
                    # Move stage to each point.x, point.y
                    workflowSteps.append(WorkflowStep(
                        name=name,
                        step_id=str(step_id),
                        main_func=self.move_stage_xy,
                        main_params={"posX": point.x, "posY": point.y, "relative": False},
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
                    workflowSteps.append(WorkflowStep(
                        name="Acquire frame",
                        step_id=str(step_id),
                        main_func=self.acquire_frame,
                        main_params={"channel": "Mono"},
                        post_funcs=[self.save_frame_tiff], # or self.append_frame_to_stack if preview
                        pre_funcs=[self.set_exposure_time_gain],
                        pre_params={"exposure_time": 0.1, "gain": 1.0}
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
                main_func=self.wait_time,
                main_params={"seconds": tPeriod},
            ))
            step_id += 1

        # Close TIFF writer at the end
        workflowSteps.append(WorkflowStep(
            name="Close TIFF writer",
            step_id=str(step_id) + "_done",
            main_func=self.close_tiff_writer,
            main_params={"tiff_writer": tif.TiffWriter("Experiment.tif")},
        ))

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

        # Prepare TIFF writer
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        drivePath = dirtools.UserFileDirs.Data
        dirPath = os.path.join(drivePath, 'recordings', timeStamp)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        mFileName = f"{timeStamp}_{exp_name}"
        mFilePath = os.path.join(dirPath, mFileName + ".tif")
        tiff_writer = tif.TiffWriter(mFilePath)
        context.set_object("tiff_writer", tiff_writer)

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
        if gain >=0:
            self._logger.error(f"Setting gain to {gain}")
        if exposure_time >=0:
            self._logger.error(f"Setting exposure time to {exposure_time}")
        
    def dummy_main_func(self):
        self._logger.debug("Dummy main function called")
        return True
 
    def autofocus(self):
        self._logger.error("Performing autofocus...NOT IMPLEMENTED")
   
    def wait_time(self, seconds: int, context: WorkflowContext, metadata: Dict[str, Any]):
        import time
        time.sleep(seconds)

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


    ########################################
    # Example Workflow for Multicolour Experiment
    ########################################
    
    @APIExport(requestType="POST")
    def start_multicolour_Experiment_workflow(
            self,
            params: ExperimentWorkflowParams
        ):
        '''
        nTimes: int = 1
        tPeriod: int = 1
        illuSources: List[str] = ["LED", "Laser"]
        illuIntensities: List[int] = [10, 10]
        exposureTimes: List[float] = [0.1, 0.1]
        gain: List[float] = [1.0, 1.0]
        autofocus_on: bool = False
        isPreview: bool = False -> if we use preview mode, we do not save the data, we only show it via Socket Updates
        '''
        
        # assign parameters to global variables
        self.ExperimentParams = params
        nTimes = params.nTimes
        tPeriod = params.tPeriod
        illuSources = params.illuSourcesSelected
        exposureTimes = params.exposureTimes
        gains = params.gain
        intensities = params.illuIntensities
        autofocus_every_nth_frame = params.autofocus_every_n_frames
        isPreview = params.isPreview
        
        # ensure parameters are lists 
        if not isinstance(illuSources, list): illuSources = [illuSources]
        if not isinstance(intensities, list): intensities = [intensities]
        if not isinstance(exposureTimes, list): exposureTimes = [exposureTimes]
        if not isinstance(gains, list): gains = [gains]
        
        # Check if another workflow is running
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")
        # Start the detector if not already running
        if not self.mDetector._running: 
            self.mDetector.startAcquisition()
        
        # construct the workflow steps
        workflowSteps = []
        step_id = 0

        # In this simplified example, we only do a single Z position (z=0)
        # and a single frame per position. You can easily extend this.
        z_pos = 0
        frames = [0]  # single frame index for simplicity
        for t in range(nTimes):
            for illu, intensity, exposure, gain in zip(illuSources, intensities, exposureTimes, gains):
                # Set laser power (arbitrary, could be parameterized)
                workflowSteps.append(WorkflowStep(
                    name=f"Set laser power",
                    step_id=str(step_id),
                    main_func=self.set_laser_power,
                    main_params={"power": intensity, "channel": illu},
                    pre_funcs=[],
                    post_funcs=[self.wait_time], # add a short wait
                    post_params={"seconds": 0.05}
                ))
                
                # Acquire frame with a short wait, process data, and save frame
                workflowSteps.append(WorkflowStep(
                    name=f"Acquire frame",
                    step_id=str(step_id),
                    main_func=self.acquire_frame,
                    main_params={"channel": "Mono"},
                    post_funcs=[self.save_frame_tiff] if not isPreview else [self.append_frame_to_stack],
                    pre_funcs=[self.set_exposure_time_gain],
                    pre_params={"exposure_time": exposure, "gain": gain}
                ))
                
                # Set laser power off
                workflowSteps.append(WorkflowStep(
                    name=f"Set laser power",
                    step_id=str(step_id),
                    main_func=self.set_laser_power,
                    main_params={"power": 0, "channel": illu},
                    pre_funcs=[],
                    post_funcs=[self.wait_time], # add a short wait
                    post_params={"seconds": 0}
                ))


            # Add autofocus step every nth frame
            if autofocus_every_nth_frame > 0 and t % autofocus_every_nth_frame == 0:
                workflowSteps.append(WorkflowStep(
                    name=f"Autofocus",
                    step_id=str(step_id),
                    main_func=self.autofocus,
                    main_params={},
                ))
                                        
            # add delay between frames 
            if isPreview:
                workflowSteps.append(WorkflowStep(
                    name=f"Display frame",
                    step_id=str(step_id),
                    main_func=self.dummy_main_func,
                    main_params={}, 
                    pre_funcs=[self.emit_rgb_stack],
                    pre_params={}  
                ))
            else:
                workflowSteps.append(WorkflowStep(
                    name=f"Wait for next frame",
                    step_id=str(step_id),
                    main_func=self.dummy_main_func,
                    main_params={}, 
                    pre_funcs=[self.wait_time],
                    pre_params={"seconds": tPeriod}  
                ))

                step_id += 1
        
        # Close TIFF writer at the end
        workflowSteps.append(WorkflowStep(
            name="Close TIFF writer",
            step_id=str(step_id)+"_done",
            main_func=self.close_tiff_writer,
            main_params={"tiff_writer": tif.TiffWriter("Experiment.tif")},
        ))
        
        # add final step to notify completion
        workflowSteps.append(WorkflowStep(
                name=f"Done",
                step_id=str(step_id),
                main_func=self.dummy_main_func,
                main_params={}, 
                pre_funcs=[self.wait_time],
                pre_params={"seconds": 0.1}  
            ))
        
        def sendProgress(payload):
            self.sigExperimentWorkflowUpdate.emit(payload)
            
        # Create a workflow and context
        wf = Workflow(workflowSteps, self.workflow_manager)
        context = WorkflowContext()
        # set meta data of the workflow
        context.set_metadata("nTimes", nTimes)
        context.set_metadata("tPeriod", tPeriod)
        context.set_metadata("illuSources", illuSources)
        context.set_metadata("illuIntensities", intensities)
        context.set_metadata("exposureTimes", exposureTimes)
        context.set_metadata("gains", gains)
        context.set_metadata("autofocus_every_nth_frame", autofocus_every_nth_frame)
        context.set_metadata("isPreview", isPreview)
        
        if isPreview:
            # we assume that we are in preview mode and do not save the data
            # but we want to show it via Socket Updates
            # therefore, we accumulate n frames in a list and display them as an RGB image
            stack_writer = []
            context.set_object("stack_writer", stack_writer)
        else:
            # Insert the tiff writer object into context so `save_frame` can use it
            timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drivePath = dirtools.UserFileDirs.Data
            dirPath = os.path.join(drivePath, 'recordings', timeStamp)
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)        
            self._logger.debug("Save TIF at: " + dirPath)
            experimentName = "Experiment"
            mFileName = f'{timeStamp}_{experimentName}'
            mFilePath = os.path.join(dirPath, mFileName+ ".tif")
            tiff_writer = tif.TiffWriter(mFilePath)
            context.set_object("tiff_writer", tiff_writer)
        context.on("progress", sendProgress)
        context.on("rgb_stack", sendProgress)
        # Run the workflow
        # context = wf.run_in_background(context)
        self.workflow_manager.start_workflow(wf, context)

        # return the store path to the client so they know where data is stored
        return {"status": "completed", "store_path": "Experiment.tif"}
            
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
