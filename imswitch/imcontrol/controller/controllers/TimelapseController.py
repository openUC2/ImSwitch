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

from imswitch.imcommon.framework import Signal
from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext, WorkflowStep, WorkflowsManager
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from imswitch import IS_HEADLESS

import h5py
import numpy as np

class TimelapseWorkflowParams(BaseModel):
    nTimes: int = 1
    tPeriod: int = 1
    illuSources: List[str] = []
    illuSourcesSelected: List[str] = []
    illuSourceMinIntensities: List[int] = []
    illuSourceMaxIntensities: List[int] = []
    illuIntensities: List[int] = []
    exposureTimes: List[float] = []
    gain: List[float] = []
    autofocus_every_n_frames: int = 0 # -1 means never

    
class TimelapseController(ImConWidgetController):
    """Linked to TimelapseWidget."""

    sigTimelapseWorkflowUpdate = Signal()

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
            self.positioner = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]
        except:
            self.positioner = None
            
        # define timelapse parameters as TimelapseWorkflowParams
        self.timelapseParams = TimelapseWorkflowParams()
        self.timelapseParams.illuSources = self.allIlluNames
        self.timelapseParams.illuSourceMinIntensities = [0]*len(self.timelapseParams.illuSourcesSelected)
        self.timelapseParams.illuSourceMaxIntensities = [100]*len(self.timelapseParams.illuSourcesSelected)
        self.timelapseParams.illuIntensities = [0]*len(self.allIlluNames)
        self.timelapseParams.exposureTimes = [0]*len(self.allIlluNames)
        self.timelapseParams.gain = [0]*len(self.allIlluNames)
        
    @APIExport(requestType="GET")
    def getCurrentTimelapseParameters(self):
        return self.timelapseParams
    
    @APIExport(requestType="POST")
    def setCurrentTimelapseParameters(self, params: TimelapseWorkflowParams):
        self.timelapseParams = params
        return self.timelapseParams

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
    
        
    ########################################
    # Example Workflow for Multicolour Timelapse
    ########################################
    
    @APIExport(requestType="POST")
    def start_multicolour_timelapse_workflow(
            self,
            params: TimelapseWorkflowParams
        ):
        '''
        nTimes: int = 1
        tPeriod: int = 1
        illuSources: List[str] = ["LED", "Laser"]
        illuIntensities: List[int] = [10, 10]
        exposureTimes: List[float] = [0.1, 0.1]
        gain: List[float] = [1.0, 1.0]
        autofocus_on: bool = False
        '''
        
        # assign parameters to global variables
        self.timelapseParams = params
        nTimes = params.nTimes
        tPeriod = params.tPeriod
        illuSources = params.illuSourcesSelected
        exposureTimes = params.exposureTimes
        gains = params.gain
        intensities = params.illuIntensities
        autofocus_every_nth_frame = params.autofocus_every_n_frames
        
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
                    post_params={"seconds": 0.1}
                ))
                
                # Acquire frame with a short wait, process data, and save frame
                workflowSteps.append(WorkflowStep(
                    name=f"Acquire frame",
                    step_id=str(step_id),
                    main_func=self.acquire_frame,
                    main_params={"channel": "Mono"},
                    post_funcs=[self.save_frame_tiff],
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
            main_params={"tiff_writer": tif.TiffWriter("timelapse.tif")},
        ))
        
        def sendProgress(payload):
            self.sigTimelapseWorkflowUpdate.emit(payload)
            
        # Create a workflow and context
        wf = Workflow(workflowSteps, self.workflow_manager)
        context = WorkflowContext()
        # Insert the tiff writer object into context so `save_frame` can use it
        tiff_writer = tif.TiffWriter("timelapse.tif")
        context.set_object("tiff_writer", tiff_writer)
        context.on("progress", sendProgress)
        # Run the workflow
        # context = wf.run_in_background(context)
        self.workflow_manager.start_workflow(wf, context)

        # return the store path to the client so they know where data is stored
        return {"status": "completed", "store_path": "timelapse.tif"}
            
    @APIExport()
    def pause_workflow_tl(self):
        status = self.workflow_manager.get_status()["status"]
        if status == "running":
            return self.workflow_manager.pause_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot pause in current state: {status}")

    @APIExport()
    def resume_workflow_tl(self):
        status = self.workflow_manager.get_status()["status"]
        if status == "paused":
            return self.workflow_manager.resume_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot resume in current state: {status}")

    @APIExport()
    def stop_workflow_tl(self):
        status = self.workflow_manager.get_status()["status"]
        if status in ["running", "paused"]:
            return self.workflow_manager.stop_workflow()
        else:
            raise HTTPException(status_code=400, detail=f"Cannot stop in current state: {status}")

    @APIExport()
    def workflow_status_tl(self):
        return self.workflow_manager.get_status()
    
    @APIExport()
    def force_stop_workflow_tl(self):
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
