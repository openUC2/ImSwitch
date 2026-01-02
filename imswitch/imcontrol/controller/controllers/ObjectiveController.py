
from imswitch.imcommon.model import APIExport
from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.controller.basecontrollers import LiveUpdatedController

from imswitch import IS_HEADLESS


from typing import Tuple, Optional


class ObjectiveController(LiveUpdatedController):
    """
    Controller for objective operations - Single Source of Truth for all objective state.
    
    This controller manages:
    - Current objective slot (0 or 1)
    - Homing state
    - Pixel size updates to detector
    - Motor movements via ESP32StageManager axis "A"
    
    The ObjectiveManager is only used for reading/writing configuration parameters
    to the setup JSON file. All runtime state is managed here.
    
    Motor Control:
    - Uses ESP32StageManager axis "A" for all movements
    - Objective positions (x0, x1) are stored in config and moved to via absolute positioning
    - Homing uses ESP32StageManager.home_a()
    """

    # Signal for backwards compatibility and UI updates
    sigObjectiveChanged = Signal(dict)  # pixelsize, NA, magnification, objectiveName, FOVx, FOVy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # Reference to the config manager (only for reading/writing config parameters)
        self._configManager = self._master.objectiveManager

        # === State variables (Single Source of Truth) ===
        self._currentObjective: Optional[int] = None  # 0 or 1, or None if not initialized
        self._isHomed: bool = False
        self._detectorWidth: Optional[int] = None
        self._detectorHeight: Optional[int] = None

        # Cache configuration values for quick access
        self._objectivePositions = list(self._configManager.objectivePositions)  # [x0, x1] in µm
        self._zPositions = list(self._configManager.zPositions)  # [z0, z1] in µm - Z focus offsets
        self._pixelsizes = list(self._configManager.pixelsizes)
        self._NAs = list(self._configManager.NAs)
        self._magnifications = list(self._configManager.magnifications)
        self._objectiveNames = list(self._configManager.objectiveNames)

        # Connect to config manager signals for external parameter updates
        self._configManager.sigObjectiveParametersChanged.connect(self._onConfigParametersChanged)

        # Connect to communication channel signals
        self._commChannel.sigSetObjectiveByName.connect(self._onSetObjectiveByName)
        self._commChannel.sigSetObjectiveByID.connect(self._onSetObjectiveByID)

        # Get detector reference and store dimensions
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]
        self._detectorWidth = self.detector._camera.SensorWidth
        self._detectorHeight = self.detector._camera.SensorHeight
        self._logger.debug(f"Detector dimensions: {self._detectorWidth}x{self._detectorHeight}")

        # Get ESP32Stage positioner for axis "A" motor control
        self._positioner = None
        self._hasMotor = False
        try:
            if not self._configManager.isActive:
                raise Exception("Objective is not active in the setup, skipping motor initialization.")

            positionerNames = self._master.positionersManager.getAllDeviceNames()
            if "ESP32Stage" in positionerNames:
                self._positioner = self._master.positionersManager["ESP32Stage"]
                self._hasMotor = True
                self._logger.info("ESP32Stage positioner found for objective motor control on axis A")
            else:
                self._logger.warning("ESP32Stage not found in positioners. Objective motor control disabled.")
        except Exception as e:
            self._logger.warning(f"Could not initialize objective motor control: {e}")
            self._hasMotor = False

        # Initialize objective state
        if self._configManager.isActive and self._hasMotor:
            calibrateOnStart = self._configManager.calibrateOnStart

            if calibrateOnStart:
                self._logger.info("Calibrating objective on startup...")
                self.calibrateObjective()
                # After calibration, move to the first objective position (slot 0)
                self._moveMotorToSlot(0)
                self._currentObjective = 0
            else:
                # Assume we're at slot 0 if not calibrating
                self._currentObjective = 0
                self._logger.info("Skipping calibration, assuming objective slot 0")
        else:
            # No motor, just set default slot
            self._currentObjective = 0
            self._logger.info("No motor control available, defaulting to objective slot 0")

        # Update detector with current pixel size
        self._updatePixelSize()

    # === State Property Accessors ===

    def _getCurrentPixelSize(self) -> Optional[float]:
        """Get the pixel size for the current objective."""
        if self._currentObjective is not None and 0 <= self._currentObjective < len(self._pixelsizes):
            return self._pixelsizes[self._currentObjective]
        return None

    def _getCurrentNA(self) -> Optional[float]:
        """Get the NA for the current objective."""
        if self._currentObjective is not None and 0 <= self._currentObjective < len(self._NAs):
            return self._NAs[self._currentObjective]
        return None

    def _getCurrentMagnification(self) -> Optional[int]:
        """Get the magnification for the current objective."""
        if self._currentObjective is not None and 0 <= self._currentObjective < len(self._magnifications):
            return self._magnifications[self._currentObjective]
        return None

    def _getCurrentObjectiveName(self) -> str:
        """Get the name of the current objective."""
        if self._currentObjective is not None and 0 <= self._currentObjective < len(self._objectiveNames):
            return self._objectiveNames[self._currentObjective]
        return "default"

    def _getCurrentFOV(self) -> Optional[Tuple[float, float]]:
        """Get the field of view (FOV_x, FOV_y) in µm for the current objective."""
        if self._detectorWidth is None or self._detectorHeight is None:
            return None
        pixelsize = self._getCurrentPixelSize()
        if pixelsize is None:
            return None
        return (pixelsize * self._detectorWidth, pixelsize * self._detectorHeight)

    # === Motor Control Methods (using ESP32StageManager axis A) ===

    def _moveMotorToSlot(self, slot: int) -> bool:
        """
        Move the objective motor to the position for the given slot using axis A.
        
        Args:
            slot: Objective slot (0 or 1)
            
        Returns:
            True if movement successful, False otherwise
        """
        if not self._hasMotor or self._positioner is None:
            self._logger.warning("No motor available for objective movement")
            return False

        if slot not in [0, 1]:
            self._logger.error(f"Invalid slot {slot}, must be 0 or 1")
            return False

        if slot >= len(self._objectivePositions):
            self._logger.error(f"No position configured for slot {slot}")
            return False

        target_position = self._objectivePositions[slot]
        self._logger.info(f"Moving objective motor (axis A) to position {target_position} µm for slot {slot}")

        try:
            # Use ESP32StageManager to move axis A to absolute position
            self._positioner.move(
                value=target_position,
                axis="A",
                is_absolute=True,
                is_blocking=True,
                speed=self._configManager.homeSpeed
            )
            return True
        except Exception as e:
            self._logger.error(f"Failed to move objective motor: {e}", exc_info=True)
            return False

    @APIExport(runOnUIThread=True)
    def calibrateObjective(self, homeDirection: int = None, homePolarity: int = None):
        """
        Calibrate (home) the objective turret using ESP32StageManager axis A.
        
        Args:
            homeDirection: Override home direction (optional)
            homePolarity: Override home polarity (optional)
        """
        if not self._hasMotor or self._positioner is None:
            self._logger.error("No motor available for objective calibration")
            return

        self._logger.info("Starting objective calibration (homing axis A)...")

        try:
            # Home axis A using ESP32StageManager
            self._positioner.home_a(isBlocking=True)
            self._isHomed = True
            self._logger.info("Objective homing completed successfully")
        except Exception as e:
            self._logger.error(f"Objective homing failed: {e}", exc_info=True)
            self._isHomed = False
        # need to state the current status after homing (e.g. 2 )
        self._currentObjective = 2

    @APIExport(runOnUIThread=True)
    def moveToObjective(self, slot: int, skipZ: bool = False):
        """
        Move to a specific objective slot.
        
        Args:
            slot: Objective slot number (0 or 1)
            skipZ: If True, only move axis A (objective turret) without Z focus adjustment.
                   If False (default), also apply Z delta based on z0/z1 config.
        """
        import threading
        mThread = threading.Thread(target=self.moveToObjectiveThread, args=(slot, skipZ))
        mThread.start()

    def moveToObjectiveThread(self, slot: int, skipZ: bool = False):
        if slot not in [0, 1]:
            self._logger.error(f"Invalid objective slot: {slot} (must be 0 or 1)")
            return

        if False and slot == self._currentObjective:
            self._logger.debug(f"Already at objective slot {slot}, no movement needed")
            return

        self._logger.info(f"Moving to objective slot {slot} ({self._objectiveNames[slot]}), skipZ={skipZ}")

        # Calculate Z delta if needed (before changing slot)
        z_delta = 0
        if not skipZ and self._currentObjective is not None and self._currentObjective in [0, 1]:
            # Calculate Z delta: difference between target z position and current z position
            current_z = self._zPositions[self._currentObjective] if self._currentObjective < len(self._zPositions) else 0
            target_z = self._zPositions[slot] if slot < len(self._zPositions) else 0
            z_delta = target_z - current_z
            self._logger.debug(f"Z delta: {z_delta} µm (from z{self._currentObjective}={current_z} to z{slot}={target_z})")
            # Apply Z focus adjustment if needed
            if  z_delta != 0 and self._positioner is not None:
                try:
                    self._logger.info(f"Applying Z focus adjustment: {z_delta} µm")
                    self._positioner.move(
                        value=z_delta,
                        axis="Z",
                        is_absolute=False,  # Relative movement for Z
                        is_blocking=False
                    )
                except Exception as e:
                    self._logger.error(f"Failed to apply Z focus adjustment: {e}", exc_info=True)

        # Move the axis A motor (objective turret)
        success = self._moveMotorToSlot(slot)

        if success or not self._hasMotor:
            # Update state
            self._currentObjective = slot


            # Update detector pixel size
            self._updatePixelSize()

            # Update UI if not headless
            if not IS_HEADLESS:
                self._widget.setCurrentObjectiveInfo(self._currentObjective)

    @APIExport(runOnUIThread=True)
    def getCurrentObjective(self):
        """
        Get current objective information.
        
        Returns:
            Tuple of (slot, name)
        """
        return self._currentObjective, self._getCurrentObjectiveName()

    def _updatePixelSize(self):
        """
        Update pixel size in detector based on current objective.
        Internal method called after objective changes.
        """
        if self._currentObjective is None or self._currentObjective not in [0, 1]:
            return

        # Get status and emit signal
        mStatus = self.getstatus()
        self.sigObjectiveChanged.emit(mStatus)

        # Update detector pixel size
        pixelsize = self._getCurrentPixelSize()
        if pixelsize is not None:
            self.detector.setPixelSizeUm(pixelsize)
            self._logger.debug(f"Updated detector pixel size to {pixelsize} µm/px")

    def _onConfigParametersChanged(self, slot: int, params: dict):
        """
        Handle objective parameters changed signal from config manager.
        Updates local cache and detector if current objective changed.
        
        Args:
            slot: Objective slot that was updated
            params: Updated parameters
        """
        self._logger.debug(f"Objective {slot} config parameters changed: {params}")

        # Update local cache
        if "pixelsize" in params and slot < len(self._pixelsizes):
            self._pixelsizes[slot] = params["pixelsize"]
        if "NA" in params and slot < len(self._NAs):
            self._NAs[slot] = params["NA"]
        if "magnification" in params and slot < len(self._magnifications):
            self._magnifications[slot] = params["magnification"]
        if "objectiveName" in params and slot < len(self._objectiveNames):
            self._objectiveNames[slot] = params["objectiveName"]
        if "position" in params and slot < len(self._objectivePositions):
            self._objectivePositions[slot] = params["position"]
        if "zPosition" in params and slot < len(self._zPositions):
            self._zPositions[slot] = params["zPosition"]

        # If this is the current objective, update detector
        if slot == self._currentObjective:
            self._updatePixelSize()

    def _onSetObjectiveByID(self, objective_id):
        """
        Handle request to set objective by ID from communication channel.
        
        Args:
            objective_id: ID of objective to switch to (0 or 1, as int or str)
        """
        try:
            if isinstance(objective_id, str):
                objective_id = int(objective_id)

            if objective_id in [0, 1]:
                if self._currentObjective != objective_id:
                    self._logger.info(f"Received request to move to objective ID {objective_id}")
                    self.moveToObjective(objective_id)
                else:
                    self._logger.debug(f"Already on objective ID {objective_id}, no movement needed")
            else:
                self._logger.warning(f"Invalid objective ID {objective_id} received (must be 0 or 1)")
        except ValueError:
            self._logger.error(f"Invalid objective ID format: {objective_id} (must be convertible to int)")
        except Exception as e:
            self._logger.error(f"Failed to set objective by ID {objective_id}: {e}", exc_info=True)

    def _onSetObjectiveByName(self, objective_name: str):
        """
        Handle request to set objective by name from communication channel.
        
        Args:
            objective_name: Name of objective to switch to (e.g., "10x", "20x")
        """
        try:
            if objective_name in self._objectiveNames:
                slot = self._objectiveNames.index(objective_name)

                if self._currentObjective != slot:
                    self._logger.info(f"Received request to move to objective '{objective_name}' (slot {slot})")
                    self.moveToObjective(slot)
                else:
                    self._logger.debug(f"Already on objective '{objective_name}' (slot {slot}), no movement needed")
            else:
                self._logger.warning(f"Objective '{objective_name}' not found. Available: {self._objectiveNames}")
        except Exception as e:
            self._logger.error(f"Failed to set objective by name '{objective_name}': {e}", exc_info=True)

    def onObj1Clicked(self):
        """Handle UI button click for objective 1 (slot 0)"""
        if self._currentObjective != 0:
            self.moveToObjective(0)

    def onObj2Clicked(self):
        """Handle UI button click for objective 2 (slot 1)"""
        if self._currentObjective != 1:
            self.moveToObjective(1)

    def onCalibrateClicked(self):
        """Handle UI button click for calibration"""
        self.calibrateObjective()

    # Legacy API method for frontend compatibility
    @APIExport(runOnUIThread=True)
    def setPositions(self, x0: float = None, x1: float = None, z0: float = None, z1: float = None, isBlocking: bool = False):
        """
        Set the motor positions for both objective slots (legacy API).
        This updates the configuration and saves to file.
        
        Args:
            x0: Motor position in µm for objective slot 0 (axis A)
            x1: Motor position in µm for objective slot 1 (axis A)
            z0: Z focus position in µm for objective slot 0
            z1: Z focus position in µm for objective slot 1
            isBlocking: Not used, kept for API compatibility
        """
        # Update local cache and config for axis A positions
        if x0 is not None:
            self._objectivePositions[0] = x0
            self._configManager.setObjectiveParameters(slot=0, position=x0, emitSignal=False)

        if x1 is not None:
            self._objectivePositions[1] = x1
            self._configManager.setObjectiveParameters(slot=1, position=x1, emitSignal=False)

        # Update local cache and config for Z focus positions
        if z0 is not None:
            self._zPositions[0] = z0
            self._configManager.setObjectiveParameters(slot=0, zPosition=z0, emitSignal=False)

        if z1 is not None:
            self._zPositions[1] = z1
            self._configManager.setObjectiveParameters(slot=1, zPosition=z1, emitSignal=False)

        self._logger.info(f"Set objective positions: x0={x0}, x1={x1}, z0={z0}, z1={z1} µm")

    @APIExport(runOnUIThread=True)
    def saveCurrentZPosition(self, slot: int = None):
        """
        Save the current Z axis position as the z0/z1 focus position for the given slot.
        If slot is None, uses the current objective slot.
        
        This is useful for calibrating the focus offset between objectives.
        
        Args:
            slot: Objective slot (0 or 1). If None, uses current objective.
            
        Returns:
            Dictionary with the saved Z position info
        """
        if slot is None:
            slot = self._currentObjective

        if slot not in [0, 1]:
            self._logger.error(f"Invalid slot {slot}, must be 0 or 1")
            return {"success": False, "error": "Invalid slot"}

        if self._positioner is None:
            self._logger.error("No positioner available to read Z position")
            return {"success": False, "error": "No positioner"}

        try:
            positions = self._positioner.getPosition()
            if "Z" not in positions:
                self._logger.error("Z axis not available")
                return {"success": False, "error": "Z axis not available"}

            current_z = positions["Z"]

            # Update local cache and config
            self._zPositions[slot] = current_z
            self._configManager.setObjectiveParameters(slot=slot, zPosition=current_z, emitSignal=True)

            self._logger.info(f"Saved Z position {current_z} µm for objective slot {slot}")

            return {
                "success": True,
                "slot": slot,
                "zPosition": current_z,
                "objectiveName": self._objectiveNames[slot]
            }
        except Exception as e:
            self._logger.error(f"Failed to save Z position: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @APIExport(runOnUIThread=True)
    def setObjectivePosition(self, slot: int, position: float):
        """
        Set the motor position for a specific objective slot.
        This updates the configuration and saves to file.
        
        Args:
            slot: Objective slot number (0 or 1)
            position: Motor position in µm for axis A
        """
        if slot not in [0, 1]:
            raise ValueError("Objective slot must be 0 or 1")

        # Update local cache
        if slot < len(self._objectivePositions):
            self._objectivePositions[slot] = position

        # Update config manager (will save to file)
        self._configManager.setObjectiveParameters(slot=slot, position=position)

        self._logger.info(f"Set objective {slot} position to {position} µm")

    @APIExport(runOnUIThread=True)
    def setObjectiveParameters(self, objectiveSlot: int, pixelsize: float = None,
                               objectiveName: str = None, NA: float = None,
                               magnification: int = None, position: float = None):
        """
        Set objective parameters for a specific objective slot.
        This updates both the controller state and the config file.
        
        Args:
            objectiveSlot: Objective slot number (0 or 1)
            pixelsize: Pixel size in micrometers
            objectiveName: Name of the objective
            NA: Numerical aperture
            magnification: Magnification value
            position: Motor position in µm for axis A
            
        Returns:
            Dictionary with updated parameters
        """
        if objectiveSlot not in [0, 1]:
            raise ValueError("Objective slot must be 0 or 1")

        # Update config manager (will save to file and emit signal)
        self._configManager.setObjectiveParameters(
            slot=objectiveSlot,
            pixelsize=pixelsize,
            NA=NA,
            magnification=magnification,
            objectiveName=objectiveName,
            position=position,
            emitSignal=True
        )

        # Return updated parameters
        return self._configManager.getObjectiveParameters(objectiveSlot)

    @APIExport(runOnUIThread=True)
    def getstatus(self):
        """
        Get the current status of the objective.
        
        Returns:
            Dictionary with complete objective status
        """
        # Build status from controller state (single source of truth)
        status = {
            # Current state
            "currentObjective": self._currentObjective,
            "isHomed": self._isHomed,

            # Current objective parameters
            "objectiveName": self._getCurrentObjectiveName(),
            "pixelsize": self._getCurrentPixelSize(),
            "NA": self._getCurrentNA(),
            "magnification": self._getCurrentMagnification(),
            "FOV": self._getCurrentFOV(),

            # Available objectives configuration
            "availableObjectives": [0, 1],
            "availableObjectivesNames": list(self._objectiveNames),
            "availableObjectivesPositions": list(self._objectivePositions),
            "availableObjectiveZPositions": list(self._zPositions),  # z0, z1 focus offsets
            "availableObjectiveMagnifications": list(self._magnifications),
            "availableObjectiveNAs": list(self._NAs),
            "availableObjectivePixelSizes": list(self._pixelsizes),

            # Legacy API aliases for x0, x1, z0, z1 (for frontend compatibility)
            "x0": self._objectivePositions[0] if len(self._objectivePositions) > 0 else None,
            "x1": self._objectivePositions[1] if len(self._objectivePositions) > 1 else None,
            "z0": self._zPositions[0] if len(self._zPositions) > 0 else None,
            "z1": self._zPositions[1] if len(self._zPositions) > 1 else None,

            # Motor configuration
            "homeDirection": self._configManager.homeDirection,
            "homePolarity": self._configManager.homePolarity,
            "homeSpeed": self._configManager.homeSpeed,
            "homeAcceleration": self._configManager.homeAcceleration,

            # Detector info
            "detectorWidth": self._detectorWidth,
            "detectorHeight": self._detectorHeight,

            # Motor availability
            "hasMotor": self._hasMotor,
            "isActive": self._configManager.isActive,

            # Current motor position (if available)
            "motorPosition": None,
            "zPosition": None
        }

        # Get current motor positions from positioner if available
        if self._hasMotor and self._positioner is not None:
            try:
                positions = self._positioner.getPosition()
                if "A" in positions:
                    status["motorPosition"] = positions["A"]
                if "Z" in positions:
                    status["zPosition"] = positions["Z"]
            except Exception as e:
                self._logger.debug(f"Could not get motor position: {e}")

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
