
from imswitch.imcommon.model import dirtools, modulesconfigtools, ostools, APIExport
from imswitch.imcommon.framework import Signal, Worker, Mutex, Timer
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.controller.basecontrollers import LiveUpdatedController

from imswitch.imcommon.model import APIExport, initLogger
from imswitch import IS_HEADLESS


from pydantic import BaseModel
from typing import Tuple

class ObjectiveStatusModel(BaseModel):
    x1: float
    x2: float
    z1: float
    z2: float
    pos: float
    isHomed: bool
    state: int
    isRunning: bool
    FOV: Tuple[float, float]
    pixelsize: float
    objectiveName: str
    NA: float
    magnification: int

class ObjectiveController(LiveUpdatedController):
    sigObjectiveChanged = Signal(dict) # pixelsize, NA, magnification, objectiveName, FOVx, FOVy

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
        self.isActive = self._master.objectiveManager.isActive
        self.detectorWidth, self.detectorHeight = self.detector._camera.SensorWidth, self.detector._camera.SensorHeight
        self.currentObjective = None  # Will be set after calibration

        # Create new objective instance (expects a parent with post_json)
        try:
            if not self.isActive:
                raise Exception("Objective is not active in the setup, skipping initialization.")
            self._objective = self._master.rs232sManager["ESP32"]._esp32.objective
        except:
            class dummyObjective:
                def __init__(self, pixelsizes, NAs, magnifications, objectiveNames):
                    self.move = lambda slot, isBlocking: None
                    self.home = lambda direction, endstoppolarity, isBlocking: None
                    self.x1 = 0
                    self.x2 = 0
                    self.slot = 0
                    self.isHomed = 0
                    self.z1 = 0
                    self.z2 = 0
                    self.pixelsizes = pixelsizes
                    self.NAs = NAs
                    self.magnifications = magnifications
                    self.objectiveNames = objectiveNames

                def home(self, direction, endstoppolarity, isBlocking):
                    if direction is not None:
                        self.homeDirection = direction
                    if endstoppolarity is not None:
                        self.homePolarity = endstoppolarity
                    # Simulate homing process
                    self.x1 = 0
                    self.x2 = 0
                    self.slot = 0
                    self.isHomed = 1
                    # Simulate a delay for homing
                    import time
                    time.sleep(1)

                def move(self, slot, isBlocking):
                    self.slot = slot

                def getstatus(self):
                    return {
                        "x1": self.x1,
                        "x2": self.x2,
                        "z1": self.z1,
                        "z2": self.z2,
                        "pos": 0,
                        "isHomed": self.isHomed,
                        "state": self.slot,
                        "isRunning": 0,
                        "FOV": (100,100),
                        "pixelsize": self.pixelsizes[self.slot - 1],
                        "objectiveName": self.objectiveNames[self.slot - 1],
                        "NA": self.NAs[self.slot - 1],
                        "magnification": self.magnifications[self.slot - 1]
                    }

                def setPositions(self, x1, x2, z1, z2, isBlocking):
                    if x1 is not None:
                        self.x1 = x1
                    if x2 is not None:
                        self.x2 = x2
                    if z1 is not None:
                        self.z1 = z1
                    if z2 is not None:
                        self.z2 = z2

            self._objective = dummyObjective(pixelsizes=self.pixelsizes, 
                                             NAs=self.NAs, 
                                             magnifications=self.magnifications, 
                                             objectiveNames=self.objectiveNames)

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
        if not self._master.positionersManager.getAllDeviceNames()[0] == "ESP32Stage":
            self._logger.error("ESP32Stage is not available in the positioners manager, cannot home objective.")
            return
        self._master.positionersManager["ESP32Stage"].home_a() # will home the objective - only the 
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
        if self.currentObjective is None or self.currentObjective not in [1, 2]:
            return

        # update information based on the current objective
        mStatus = self.getstatus()

        # Über das Signal als dict senden
        self.sigObjectiveChanged.emit(mStatus)

        # Setzen der Pixelgröße in der Detector-Objekt
        self.detector.setPixelSizeUm(mStatus["pixelsize"])

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
    def setPositions(self, x1:float=None, x2:float=None, z1:float=None, z2:float=None, isBlocking:bool=False):
        '''
        overwrite the positions for objective 1 and 2 in the EEPROMof the ESP32
        '''
        return self._objective.setPositions(x1, x2, z1, z2, isBlocking)


    @APIExport(runOnUIThread=True)
    def getstatus(self):
        """
        Get the current status of the objective.

        """
        # default status
        status = {
            "x1": 0,
            "x2": 0,
            "z1": 0,
            "z2": 0,
            "pos": 0,
            "isHomed": 0,
            "state": 0,
            "isRunning": 0,
            "FOV": (100,100),
            "pixelsize": 1,
            "objectiveName": "TEST",
            "NA": 1.0,
            "magnification": 1,
            "availableObjectives": [1, 2],
            "availableObjectivesNames": self.objectiveNames,
            "availableObjectivesPositions": self.objectivePositions,
            "homeDirection": self.homeDirection,
            "homePolarity": self.homePolarity,
            "homeSpeed": self.homeSpeed,
            "homeAcceleration": self.homeAcceleration,
            "availableObjectiveMagnifications": self.magnifications,
            "availableObjectiveNAs": self.NAs,
            "availableObjectivePixelSizes": self.pixelsizes,
            "detectorWidth": self.detectorWidth,
            "detectorHeight": self.detectorHeight
        }

        # get the status from the objective
        objective_raw = self._objective.getstatus()
        status.update(objective_raw)

        # calculate the field of view and pixel size
        fov_x = self.pixelsizes[self.currentObjective - 1] * self.detectorWidth
        fov_y = self.pixelsizes[self.currentObjective - 1] * self.detectorHeight
        current_pixelsize = self.pixelsizes[self.currentObjective - 1]
        current_NA = self.NAs[self.currentObjective - 1]
        current_magnification = self.magnifications[self.currentObjective - 1]
        current_objective_name = self.objectiveNames[self.currentObjective-1]

        status["FOV"] = (fov_x, fov_y)
        status["pixelsize"] = current_pixelsize
        status["NA"] = current_NA
        status["magnification"] = current_magnification
        status["objectiveName"] = current_objective_name

        # return the data as a dictionary
        return status


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
