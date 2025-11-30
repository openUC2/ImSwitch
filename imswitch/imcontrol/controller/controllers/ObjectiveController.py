
from imswitch.imcommon.model import dirtools, modulesconfigtools, ostools, APIExport
from imswitch.imcommon.framework import Signal, Worker, Mutex, Timer
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.controller.basecontrollers import LiveUpdatedController

from imswitch.imcommon.model import APIExport, initLogger
from imswitch import IS_HEADLESS


from pydantic import BaseModel
from typing import Tuple
# TODO: we have to take into account that the ps4 controller can trigger a switch of the lenses, hence we should have a callback on that 
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
    """
    Controller for objective operations following the Model-View-Presenter pattern.
    This controller manipulates state through ObjectiveManager and responds to its signals.
    All state is stored in the manager, anot in the controller.
    """
    
    # Signal for backwards compatibility and UI updates
    sigObjectiveChanged = Signal(dict) # pixelsize, NA, magnification, objectiveName, FOVx, FOVy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # Reference to the manager (holds all state)
        self._manager = self._master.objectiveManager
        
        # Connect to manager signals
        self._manager.sigObjectiveStateChanged.connect(self._onObjectiveStateChanged)
        self._manager.sigObjectiveParametersChanged.connect(self._onObjectiveParametersChanged)
        
        # Connect to communication channel signals
        self._commChannel.sigSetObjectiveByName.connect(self._onSetObjectiveByName)
        self._commChannel.sigSetObjectiveByID.connect(self._onSetObjectiveByID)
        # Get detector reference and set dimensions in manager
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]
        detectorWidth = self.detector._camera.SensorWidth
        detectorHeight = self.detector._camera.SensorHeight
        self._manager.setDetectorDimensions(detectorWidth, detectorHeight)

        # Create new objective instance (expects a parent with post_json)
        try:
            if not self._manager.isActive:
                raise Exception("Objective is not active in the setup, skipping initialization.")
            self._objective = self._master.rs232sManager["ESP32"]._esp32.objective
        except:
            class dummyObjective:
                def __init__(self, manager):
                    self._manager = manager
                    self.move = lambda slot, isBlocking: None
                    self.home = lambda direction, endstoppolarity, isBlocking: None
                    self.x1 = 0
                    self.x2 = 0
                    self.slot = 0
                    self.isHomed = 0
                    self.z1 = 0
                    self.z2 = 0

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
                        "isRunning": 0
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

            self._objective = dummyObjective(manager=self._manager)

        # Initialize objective state
        # INDEXING CONVENTION:
        # - Hardware returns 1-based slot (1 or 2)
        # - Manager stores 0-based index (0 or 1)
        if self._manager.calibrateOnStart:
            self.calibrateObjective()
            # After calibration, move to the first objective position (hardware slot 1)
            self._objective.move(slot=1, isBlocking=True)
            self._manager.setCurrentObjective(0)  # Store as 0-based index
        else:
            status = self._objective.getstatus()
            # Hardware returns 1-based slot, convert to 0-based for manager
            hardwareSlot = status.get("state", 1)  # Default to slot 1 if not available
            internalIndex = hardwareSlot - 1 if hardwareSlot >= 1 else 0  # Convert to 0-based
            self._manager.setCurrentObjective(internalIndex)
            isHomed = status.get("isHomed", 0) == 1
            self._manager.setHomedState(isHomed)
        
        # Update detector with current pixel size
        self._updatePixelSize()


    @APIExport(runOnUIThread=True)
    def calibrateObjective(self, homeDirection:int=None, homePolarity:int=None):
        """
        Calibrate (home) the objective turret.
        
        Args:
            homeDirection: Override home direction
            homePolarity: Override home polarity
        """
        # Note: We don't update manager state here as these are temporary overrides
        # The actual configuration should be updated through setObjectiveParameters if needed
        if homeDirection is None:
            homeDirection = self._manager.homeDirection
        if homePolarity is None:
            homePolarity = self._manager.homePolarity
            
        # Calibrate (home) the objective on the microcontroller side (blocking)
        if not self._master.positionersManager.getAllDeviceNames()[0] == "ESP32Stage":
            self._logger.error("ESP32Stage is not available in the positioners manager, cannot home objective.")
            return
        
        self._master.positionersManager["ESP32Stage"].home_a() # will home the objective
        
        # Update homing state in manager
        self._manager.setHomedState(True)
        
        # Get current state from hardware
        # INDEXING: Hardware returns 1-based slot, convert to 0-based for manager
        status = self._objective.getstatus()
        # Assume status is structured as: {"objective": {"state": 1, ...}}
        try:
            hardware_slot = status.get("objective", {}).get("state", 1)  # 1-based from hardware
        except:
            hardware_slot = 1  # Default to slot 1 if status unavailable
        
        # Convert hardware's 1-based slot to 0-based index for internal state
        internal_index = hardware_slot - 1 if hardware_slot >= 1 else 0
        # Clamp to valid range [0, 1]
        internal_index = max(0, min(1, internal_index))
        
        # Update manager state with 0-based index (will emit signal)
        self._manager.setCurrentObjective(internal_index)
        
        # Update detector pixel size
        self._updatePixelSize()


    @APIExport(runOnUIThread=True)
    def moveToObjective(self, slot: int):
        """
        Move to a specific objective slot.
        
        INDEXING CONVENTION:
        - API/external: 1-based slot numbers (1 or 2) for user-facing interfaces
        - Internal state (_manager.setCurrentObjective): 0-based (0 or 1)
        - Hardware (ESP32): 1-based (slot 1, 2)
        
        Args:
            slot: Objective slot number (1 or 2) - 1-based for API consistency
        """
        # Validate 1-based slot input
        if slot not in [1, 2]:
            self._logger.error("Invalid objective slot: %s (must be 1 or 2)", slot)
            return
        
        # Convert to 0-based index for internal state
        internal_slot = slot - 1  # 1 -> 0, 2 -> 1
        
        # Move hardware (hardware uses 1-based indexing, so pass slot directly)
        self._objective.move(slot=slot, isBlocking=True)
        
        # Update manager state with 0-based index (will emit signal)
        self._manager.setCurrentObjective(internal_slot)
        
        # Update detector pixel size
        self._updatePixelSize()
        
        # Update UI if not headless
        if not IS_HEADLESS:
            self._widget.setCurrentObjectiveInfo(self._manager.getCurrentObjective())

    @APIExport(runOnUIThread=True)
    def getCurrentObjective(self):
        """
        Get current objective information.
        
        Returns:
            Tuple of (slot, name)
        """
        # Read from manager state
        slot = self._manager.getCurrentObjective()
        name = self._manager.getCurrentObjectiveName()
        return slot, name

    def _updatePixelSize(self):
        """
        Update pixel size in detector based on current objective.
        Internal method called after objective changes.
        """
        currentObjective = self._manager.getCurrentObjective()
        if currentObjective is None or currentObjective not in [0, 1, 2]:
            return

        # Get status from manager
        mStatus = self.getstatus()

        # Emit signal for backwards compatibility
        self.sigObjectiveChanged.emit(mStatus)

        # Update detector pixel size
        pixelsize = self._manager.getCurrentPixelSize()
        if pixelsize is not None:
            self.detector.setPixelSizeUm(pixelsize)
    
    def _onObjectiveStateChanged(self, status: dict):
        """
        Handle objective state changed signal from manager.
        
        Args:
            status: Full status dictionary from manager
        """
        self._logger.debug(f"Objective state changed: {status.get('currentObjective')}")
        # Could trigger UI updates or other reactions here
        # For now, the signal is mainly for other controllers to observe
    
    def _onObjectiveParametersChanged(self, slot: int, params: dict):
        """
        Handle objective parameters changed signal from manager.
        
        Args:
            slot: Objective slot that was updated
            params: Updated parameters
        """
        self._logger.debug(f"Objective {slot} parameters changed: {params}")
        # If this is the current objective, update detector
        if slot == self._manager.getCurrentObjective():
            self._updatePixelSize()
    
    def _onSetObjectiveByID(self, objective_id):
        """
        Handle request to set objective by ID from communication channel.
        
        This allows other controllers to request objective changes without
        direct coupling.
        
        INDEXING CONVENTION:
        - objective_id uses 1-based numbering (1 or 2) for API consistency
        - Signal type is str but we accept both str and int for flexibility
        
        Args:
            objective_id: ID of objective to switch to (1 or 2, as int or str)
        """
        try:
            # Convert to int if string
            if isinstance(objective_id, str):
                objective_id = int(objective_id)
            
            if objective_id in [1, 2]:
                # Get current slot (0-based from manager) and convert to 1-based for comparison
                current_slot_0based = self._manager.getCurrentObjective()
                current_slot_1based = current_slot_0based + 1 if current_slot_0based is not None else None
                
                if current_slot_1based != objective_id:
                    self._logger.info(f"Received request to move to objective ID {objective_id}")
                    self.moveToObjective(objective_id)  # moveToObjective expects 1-based
                else:
                    self._logger.debug(f"Already on objective ID {objective_id}, no movement needed")
            else:
                self._logger.warning(f"Invalid objective ID {objective_id} received (must be 1 or 2)")
        except ValueError:
            self._logger.error(f"Invalid objective ID format: {objective_id} (must be convertible to int)")
        except Exception as e:
            self._logger.error(f"Failed to set objective by ID {objective_id}: {e}", exc_info=True)
    
    def _onSetObjectiveByName(self, objective_name: str):
        """
        Handle request to set objective by name from communication channel.
        
        This allows other controllers to request objective changes without
        direct coupling.
        
        INDEXING CONVENTION:
        - objective_names array is 0-indexed
        - API/external uses 1-based slot numbers
        - moveToObjective expects 1-based slot
        
        Args:
            objective_name: Name of objective to switch to (e.g., "10x", "20x")
        """
        try:
            # Find slot number for the objective name
            objective_names = self._manager.objectiveNames
            if objective_name in objective_names:
                idx = objective_names.index(objective_name)  # 0-based index in array
                slot_1based = idx + 1  # Convert to 1-based slot for API
                
                # Get current slot (0-based from manager) and convert to 1-based for comparison
                current_slot_0based = self._manager.getCurrentObjective()
                current_slot_1based = current_slot_0based + 1 if current_slot_0based is not None else None
                
                if current_slot_1based != slot_1based:
                    self._logger.info(f"Received request to move to objective '{objective_name}' (slot {slot_1based})")
                    self.moveToObjective(slot_1based)  # moveToObjective expects 1-based
                else:
                    self._logger.debug(f"Already on objective '{objective_name}' (slot {slot_1based}), no movement needed")
            else:
                self._logger.warning(f"Objective '{objective_name}' not found in ObjectiveManager. Available: {objective_names}")
        except Exception as e:
            self._logger.error(f"Failed to set objective by name '{objective_name}': {e}", exc_info=True)

    def onObj1Clicked(self):
        """Handle UI button click for objective 1 (slot 1)"""
        # getCurrentObjective returns 0-based, we need to compare with 0 (slot 1 = index 0)
        if self._manager.getCurrentObjective() != 0:
            self.moveToObjective(1)  # moveToObjective expects 1-based

    def onObj2Clicked(self):
        """Handle UI button click for objective 2 (slot 2)"""
        # getCurrentObjective returns 0-based, we need to compare with 1 (slot 2 = index 1)
        if self._manager.getCurrentObjective() != 1:
            self.moveToObjective(2)  # moveToObjective expects 1-based

    def onCalibrateClicked(self):
        """Handle UI button click for calibration"""
        self.calibrateObjective()

    @APIExport(runOnUIThread=True)
    def setPositions(self, x1:float=None, x2:float=None, z1:float=None, z2:float=None, isBlocking:bool=False):
        """
        Overwrite the positions for objective 1 and 2 in the EEPROM of the ESP32.
        
        Args:
            x1: Position for objective 1 in X
            x2: Position for objective 2 in X
            z1: Position for objective 1 in Z
            z2: Position for objective 2 in Z
            isBlocking: Whether to block until complete
        """
        return self._objective.setPositions(x1, x2, z1, z2, isBlocking)

    @APIExport(runOnUIThread=True)
    def setObjectiveParameters(self, objectiveSlot: int, pixelsize: float = None, 
                               objectiveName: str = None, NA: float = None, 
                               magnification: int = None):
        """
        Set objective parameters for a specific objective slot.
        This updates the manager state and propagates changes to the detector.
        
        Args:
            objectiveSlot: Objective slot number (1 or 2)
            pixelsize: Pixel size in micrometers
            objectiveName: Name of the objective
            NA: Numerical aperture
            magnification: Magnification value
            
        Returns:
            Dictionary with updated parameters
        """
        if objectiveSlot not in [1, 2]:
            raise ValueError("Objective slot must be 1 or 2")
        
        # Update manager state (will emit signal)
        self._manager.setObjectiveParameters(
            slot=objectiveSlot,
            pixelsize=pixelsize,
            NA=NA,
            magnification=magnification,
            objectiveName=objectiveName,
            emitSignal=True
        )
        
        # Return updated parameters from manager
        return self._manager.getObjectiveParameters(objectiveSlot)


    @APIExport(runOnUIThread=True)
    def getstatus(self):
        """
        Get the current status of the objective.
        Combines hardware status with manager state.
        
        INDEXING CONVENTION in returned status:
        - currentObjective: 0-based index from manager (0 or 1)
        - state: 1-based slot from hardware (1 or 2)
        
        Returns:
            Dictionary with complete objective status
        """
        # Get base status from manager (includes all configuration and current state)
        # Note: manager's currentObjective is 0-based
        status = self._manager.getFullStatus()

        # Get hardware-specific status from objective device
        # Note: hardware's "state" field is already 1-based (1 or 2)
        try:
            objective_raw = self._objective.getstatus()
            # Hardware state is already 1-based, no conversion needed
            status.update(objective_raw)
        except Exception as e:
            self._logger.warning(f"Failed to get hardware status: {e}")

        # Return combined status
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
