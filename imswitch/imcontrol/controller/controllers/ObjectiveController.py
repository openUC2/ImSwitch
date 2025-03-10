
from imswitch.imcommon.model import dirtools, modulesconfigtools, ostools, APIExport
from imswitch.imcommon.framework import Signal, Worker, Mutex, Timer
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.controller.basecontrollers import LiveUpdatedController

from imswitch.imcommon.model import APIExport, initLogger
from imswitch import IS_HEADLESS


class ObjectiveController(LiveUpdatedController):
    sigObjectiveChanged = Signal(float, float, float, str, float, float) # pixelsize, NA, magnification, objectiveName, FOVx, FOVy
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # filter detector
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector= self._master.detectorsManager[allDetectorNames[0]]

        # Assign configuration values from the dataclass
        self.pixelsizes = self._master.objectiveManager.pixelsizes
        self.NAs = self._master.objectiveManager.NAs
        self.magnifications = self._master.objectiveManager.magnifications
        self.objectiveNames = self._master.objectiveManager.objectiveNames
        self.objectivePositions = self._master.objectiveManager.objectivePositions
        self.homeDirection = self._master.objectiveManager.homeDirection
        self.homePolarity = self._master.objectiveManager.homePolarity
        self.homeSpeed = self._master.objectiveManager.homeSpeed
        self.homeAcceleration = self._master.objectiveManager.homeAcceleration
        self.calibrateOnStart = self._master.objectiveManager.calibrateOnStart
        
        self.detectorWidth, self.detectorHeight = self.detector._camera.SensorWidth, self.detector._camera.SensorHeight
        self.currentObjective = None  # Will be set after calibration

        # Create new objective instance (expects a parent with post_json)
        self._objective = self._master.rs232sManager["ESP32"]._esp32.objective

        if self.calibrateOnStart:
            self.calibrateObjective()
            # After calibration, move to the first objective position (X1)
            self._objective.move(slot=1, isBlocking=True)
            self.currentObjective = 1
            self._updatePixelSize()


    @APIExport(runOnUIThread=True)
    def calibrateObjective(self, homeDirection:int=None, homePolarity:int=None):
        if homeDirection is not None:
            self.homeDirection = homeDirection
        if homePolarity is not None:
            self.homePolarity = homePolarity
        # Calibrate (home) the objective on the microcontroller side (blocking)
        # self._objective.calibrate(isBlocking=True)
        self._objective.home(direction=self.homeDirection, endstoppolarity=self.homePolarity, isBlocking=True)
        status = self._objective.getstatus()
        # Assume status is structured as: {"objective": {"state": 1, ...}}
        try:state = status.get("objective", {}).get("state", 1)
        except:state = 0 # Assume calibration failed
        # state has to be within [0, 1]
        state = 1 if state > 1 else state
        self.currentObjective = state
        self._updatePixelSize()


    @APIExport(runOnUIThread=True)
    def moveToObjective(self, slot: int):
        # slot should be 1 or 2
        if slot not in [1, 2]:
            self._logger.error("Invalid objective slot: %s", slot)
            return
        self._objective.move(slot=slot, isBlocking=True)
        self.currentObjective = slot
        self._updatePixelSize()
        if not IS_HEADLESS:
            self._widget.setCurrentObjectiveInfo(self.currentObjective)

    @APIExport(runOnUIThread=True)
    def getCurrentObjective(self):
        # Return a tuple: (current objective slot, objective name)
        name = self.objectiveNames[0] if self.currentObjective == 1 else self.objectiveNames[1]
        return self.currentObjective, name

    def _updatePixelSize(self):
        # Update the detector's effective pixel size based on the current objective
        if self.currentObjective is None or self.currentObjective not in [1, 2]:
            return
        self.sigObjectiveChanged.emit(self.pixelsizes[self.currentObjective - 1],
                                      self.NAs[self.currentObjective - 1],
                                      self.magnifications[self.currentObjective - 1],
                                      self.objectiveNames[self.currentObjective - 1], 
                                      self.pixelsizes[self.currentObjective - 1]*self.detectorHeight,
                                      self.pixelsizes[self.currentObjective - 1]*self.detectorWidth)
        if self.currentObjective == 1:
            self.detector.setPixelSizeUm(self.pixelsizes[0])
        elif self.currentObjective == 2:
            self.detector.setPixelSizeUm(self.pixelsizes[1])
            
        #objective_params["objective"]["FOV"] = self.pixelsizes[0] * (self.detectorWidth, self.detectorHeight)
        #objective_params["objective"]["pixelsize"] = self.detector.pixelSizeUm[-1]

    def onObj1Clicked(self):
        if self.currentObjective != 1:
            self.moveToObjective(1)

    def onObj2Clicked(self):
        if self.currentObjective != 2:
            self.moveToObjective(2)

    def onCalibrateClicked(self):
        self.calibrateObjective()
        
    @APIExport(runOnUIThread=True)
    def setPositions(self, x1:float=None, x2:float=None, isBlocking:bool=False):
        '''
        overwrite the positions for objective 1 and 2 in the EEPROMof the ESP32
        '''
        return self._objective.setPositions(x1, x2, isBlocking)

    @APIExport(runOnUIThread=True)
    def getstatus(self):
        '''
        get the positions for objective 1 and 2 from the EEPROMof the ESP32
        '''
        # retreive parameters from objective
        objective_params = self._objective.getstatus()
        # compute the FOV and pixelsize and return it
        objective_params["FOV"] = (self.pixelsizes[self.currentObjective - 1]*self.detectorWidth, self.pixelsizes[self.currentObjective - 1]*self.detectorHeight)
        objective_params["pixelsize"] = self.detector.pixelSizeUm[-1]
        return objective_params


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
