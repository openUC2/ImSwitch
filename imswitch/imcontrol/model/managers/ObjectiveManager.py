import enum
import glob
import cv2
import os

import numpy as np
from PIL import Image
import os
from imswitch.imcommon.model import dirtools

import json

from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger


class ObjectiveManager(SignalInterface):
    """
    Manager for objective state following the Model-View-Presenter pattern.
    This class holds ALL state related to objectives and provides methods to query/update it.
    Controllers should interact with state ONLY through this manager.
    """
    
    # Signal emitted when objective state changes (objective switched, parameters updated, etc.)
    sigObjectiveStateChanged = Signal(dict)
    
    # Signal emitted when objective parameters are updated
    sigObjectiveParametersChanged = Signal(int, dict)  # slot, parameters

    def __init__(self, ObjectiveInfo, setupInfo=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        
        # Store reference to setupInfo for saving configuration
        self._setupInfo = setupInfo

        if ObjectiveInfo is None:
            # define default
            self.__ObjectiveInfo = {
                "pixelsizes": [0.1, 0.2],
                "NAs": [0.2, 0.1],
                "magnifications": [20, 10],
                "objectiveNames": ["20x", "10x"],
                "objectivePositions": [0,10000],
                "homeDirection": -1,
                "homePolarity": 1,
                "homeSpeed": 20000,
                "homeAcceleration": 20000,
                "calibrateOnStart": True
            }


        self.__ObjectiveInfo = ObjectiveInfo
        
        # Configuration parameters (can be updated dynamically)
        self._pixelsizes = list(self.__ObjectiveInfo.pixelsizes)
        self._NAs = list(self.__ObjectiveInfo.NAs)
        self._magnifications = list(self.__ObjectiveInfo.magnifications)
        self._objectiveNames = list(self.__ObjectiveInfo.objectiveNames)
        self._objectivePositions = list(self.__ObjectiveInfo.objectivePositions)
        self._homeDirection = self.__ObjectiveInfo.homeDirection
        self._homePolarity = self.__ObjectiveInfo.homePolarity
        self._homeSpeed = self.__ObjectiveInfo.homeSpeed
        self._homeAcceleration = self.__ObjectiveInfo.homeAcceleration
        self._calibrateOnStart = self.__ObjectiveInfo.calibrateOnStart
        self._isActive = self.__ObjectiveInfo.active
        
        # Track current objective slot (1 or 2, or None if not initialized)
        # This provides a centralized state that other controllers can query
        self._currentObjective = None
        
        # Track homing state
        self._isHomed = False
        
        # Detector dimensions (set externally after initialization)
        self._detectorWidth = None
        self._detectorHeight = None

    # === Property accessors (read-only from outside) ===
    
    @property
    def pixelsizes(self):
        """Get pixel sizes for all objectives (read-only)"""
        return list(self._pixelsizes)
    
    @property
    def NAs(self):
        """Get numerical apertures for all objectives (read-only)"""
        return list(self._NAs)
    
    @property
    def magnifications(self):
        """Get magnifications for all objectives (read-only)"""
        return list(self._magnifications)
    
    @property
    def objectiveNames(self):
        """Get objective names (read-only)"""
        return list(self._objectiveNames)
    
    @property
    def objectivePositions(self):
        """Get objective positions (read-only)"""
        return list(self._objectivePositions)
    
    @property
    def homeDirection(self):
        """Get home direction"""
        return self._homeDirection
    
    @property
    def homePolarity(self):
        """Get home polarity"""
        return self._homePolarity
    
    @property
    def homeSpeed(self):
        """Get home speed"""
        return self._homeSpeed
    
    @property
    def homeAcceleration(self):
        """Get home acceleration"""
        return self._homeAcceleration
    
    @property
    def calibrateOnStart(self):
        """Get calibrate on start flag"""
        return self._calibrateOnStart
    
    @property
    def isActive(self):
        """Get active flag"""
        return self._isActive
    
    @property
    def isHomed(self):
        """Get homing state"""
        return self._isHomed
    
    # === State management methods ===
    
    def setDetectorDimensions(self, width: int, height: int):
        """
        Set detector dimensions for FOV calculations.
        
        Args:
            width: Detector width in pixels
            height: Detector height in pixels
        """
        self._detectorWidth = width
        self._detectorHeight = height
        self.__logger.debug(f"Detector dimensions set: {width}x{height}")
    
    def getDetectorDimensions(self):
        """
        Get detector dimensions.
        
        Returns:
            Tuple of (width, height) or (None, None) if not set
        """
        return (self._detectorWidth, self._detectorHeight)
    
    def getCurrentObjective(self) -> int:
        """
        Get the current objective slot.
        
        Returns:
            Current objective slot (1 or 2) or None if not set
        """
        return self._currentObjective
    
    def setCurrentObjective(self, slot: int, emitSignal: bool = True):
        """
        Set the current objective slot (0-based index).
        
        This should be called by ObjectiveController when the objective changes.
        Provides a centralized state that other components can query.
        
        INDEXING CONVENTION:
        - This method uses 0-based indexing (0 or 1) for internal state
        - External APIs (getCurrentObjectiveID) return 1-based (1 or 2)
        - Hardware uses 1-based (slot 1, 2)
        
        Args:
            slot: Objective index (0 or 1 for the two objectives, or None if not set)
            emitSignal: Whether to emit the state changed signal
            
        Raises:
            ValueError: If slot is not 0, 1, or None
        """
        if slot not in [0, 1, None]:
            raise ValueError(f"Objective slot must be 0, 1, or None (0-based index), got {slot}")
        
        old_slot = self._currentObjective
        self._currentObjective = slot
        
        if old_slot != slot:
            self.__logger.info(f"Current objective changed: {old_slot} → {slot}")
            if slot is not None and 0 <= slot < len(self._objectiveNames):
                self.__logger.info(f"Active objective: {self._objectiveNames[slot]}")
            
            if emitSignal:
                self.sigObjectiveStateChanged.emit(self.getFullStatus())
    
    def setHomedState(self, isHomed: bool, emitSignal: bool = True):
        """
        Set the homing state.
        
        Args:
            isHomed: Whether the objective is homed
            emitSignal: Whether to emit the state changed signal
        """
        old_state = self._isHomed
        self._isHomed = isHomed
        
        if old_state != isHomed:
            self.__logger.info(f"Objective homed state changed: {old_state} → {isHomed}")
            
            if emitSignal:
                self.sigObjectiveStateChanged.emit(self.getFullStatus())
    
    def getCurrentObjectiveName(self) -> str:
        """
        Get the name of the current objective.
        
        Returns:
            Objective name (e.g., "10x", "20x") or "default" if not set
        """
        if self._currentObjective is not None and 0 <= self._currentObjective < len(self._objectiveNames):
            return self._objectiveNames[self._currentObjective]
        elif len(self._objectiveNames) > 0:
            return self._objectiveNames[0]  # Return first as default
        return "default"

    def getCurrentObjectiveID(self) -> int:
        """
        Get the current objective slot as 1-based ID for API consistency.
        
        INDEXING CONVENTION:
        - Internal _currentObjective: 0-based (0 or 1)
        - This method returns: 1-based (1 or 2) for API/external use
        
        Returns:
            Current objective slot (1 or 2) or None if not set
        """
        if self._currentObjective is not None:
            return self._currentObjective + 1  # Convert 0-based to 1-based
        return None
    
    def getCurrentPixelSize(self) -> float:
        """
        Get the pixel size of the current objective.
        
        Returns:
            Pixel size in micrometers or None if no objective is set
        """
        if self._currentObjective is not None and 0 <= self._currentObjective  < len(self._pixelsizes):
            return self._pixelsizes[self._currentObjective]
        return None
    
    def getCurrentNA(self) -> float:
        """
        Get the NA of the current objective.
        
        Returns:
            Numerical aperture or None if no objective is set
        """
        if self._currentObjective is not None and 0 <= self._currentObjective < len(self._NAs):
            return self._NAs[self._currentObjective ]
        return None
    
    def getCurrentMagnification(self) -> int:
        """
        Get the magnification of the current objective.
        
        Returns:
            Magnification or None if no objective is set
        """
        if self._currentObjective is not None and 0 <= self._currentObjective  < len(self._magnifications):
            return self._magnifications[self._currentObjective ]
        return None
    
    def getCurrentFOV(self):
        """
        Get the field of view for the current objective.
        
        Returns:
            Tuple of (fov_x, fov_y) in micrometers or None if dimensions not set
        """
        if self._detectorWidth is None or self._detectorHeight is None:
            return None
        
        pixelsize = self.getCurrentPixelSize()
        if pixelsize is None:
            return None
        
        fov_x = pixelsize * self._detectorWidth
        fov_y = pixelsize * self._detectorHeight
        return (fov_x, fov_y)

    def setObjectiveParameters(self, slot: int, *, pixelsize: float = None,
                               NA: float = None, magnification: float = None,
                               objectiveName: str = None, emitSignal: bool = True):
        """
        Update objective parameters for a specific slot.
        
        INDEXING CONVENTION:
        - API/external: 1-based slot numbers (1 or 2) for user-facing interfaces
        - Internal: 0-based array indexing
        - Hardware: 1-based (ESP32 uses slot 1, 2)
        
        Args:
            slot: Objective slot number (1 or 2) - 1-based for API consistency
            pixelsize: Pixel size in micrometers
            NA: Numerical aperture
            magnification: Magnification value
            objectiveName: Name of the objective
            emitSignal: Whether to emit the parameters changed signal
            
        Raises:
            ValueError: If slot is not 1 or 2
        """
        if slot not in [1, 2]:
            raise ValueError(f"Objective slot must be 1 or 2, got {slot}")
        
        idx = slot - 1  # Convert 1-based slot to 0-based array index
        changes = {}
        
        if pixelsize is not None:
            self._pixelsizes[idx] = pixelsize
            changes["pixelsize"] = pixelsize
            self.__logger.info(f"Updated pixelsize for objective {slot}: {pixelsize} µm/px")
        
        if NA is not None:
            self._NAs[idx] = NA
            changes["NA"] = NA
            self.__logger.info(f"Updated NA for objective {slot}: {NA}")
        
        if magnification is not None:
            self._magnifications[idx] = magnification
            changes["magnification"] = magnification
            self.__logger.info(f"Updated magnification for objective {slot}: {magnification}x")
        
        if objectiveName is not None:
            self._objectiveNames[idx] = objectiveName
            changes["objectiveName"] = objectiveName
            self.__logger.info(f"Updated name for objective {slot}: {objectiveName}")
        
        if changes:
            # Save to configuration file
            self._saveObjectiveConfigToFile()
            
            if emitSignal:
                params = self.getObjectiveParameters(slot)
                self.sigObjectiveParametersChanged.emit(slot, params)
                
                # If this is the current objective, also emit state changed
                if self._currentObjective == slot:
                    self.sigObjectiveStateChanged.emit(self.getFullStatus())
    def getObjectiveParameters(self, slot: int) -> dict:
        """
        Get parameters for a specific objective slot.
        
        INDEXING CONVENTION:
        - API/external: 1-based slot numbers (1 or 2)
        - Internal: 0-based array indexing
        
        Args:
            slot: Objective slot number (1 or 2) - 1-based for API consistency
            
        Returns:
            Dictionary with objective parameters
            
        Raises:
            ValueError: If slot is not 1 or 2
        """
        if slot not in [1, 2]:
            raise ValueError(f"Objective slot must be 1 or 2, got {slot}")
        
        idx = slot - 1  # Convert 1-based slot to 0-based array index
        return {
            "objectiveSlot": slot,
            "pixelsize": self._pixelsizes[idx],
            "objectiveName": self._objectiveNames[idx],
            "NA": self._NAs[idx],
            "magnification": self._magnifications[idx],
            "position": self._objectivePositions[idx] if idx < len(self._objectivePositions) else None
        }
    
    def getFullStatus(self) -> dict:
        """
        Get complete status information about objectives.
        
        Returns:
            Dictionary with all objective status information
        """
        status = {
            "currentObjective": self._currentObjective,
            "isHomed": self._isHomed,
            "availableObjectives": [1, 2],
            "availableObjectivesNames": list(self._objectiveNames),
            "availableObjectivesPositions": list(self._objectivePositions),
            "availableObjectiveMagnifications": list(self._magnifications),
            "availableObjectiveNAs": list(self._NAs),
            "availableObjectivePixelSizes": list(self._pixelsizes),
            "homeDirection": self._homeDirection,
            "homePolarity": self._homePolarity,
            "homeSpeed": self._homeSpeed,
            "homeAcceleration": self._homeAcceleration,
            "detectorWidth": self._detectorWidth,
            "detectorHeight": self._detectorHeight
        }
        
        # Add current objective specific info
        if self._currentObjective is not None:
            status["objectiveName"] = self.getCurrentObjectiveName()
            status["pixelsize"] = self.getCurrentPixelSize()
            status["NA"] = self.getCurrentNA()
            status["magnification"] = self.getCurrentMagnification()
            status["FOV"] = self.getCurrentFOV()
        
        return status
    
    def _saveObjectiveConfigToFile(self):
        """
        Save current objective configuration to setup configuration file.
        
        This persists the in-memory objective parameters (pixelsizes, NAs, magnifications, etc.)
        to the configuration file on disk. Similar to how PixelCalibration data is saved.
        """
        try:
            # Update the ObjectiveInfo with current values
            self.__ObjectiveInfo.pixelsizes = list(self._pixelsizes)
            self.__ObjectiveInfo.NAs = list(self._NAs)
            self.__ObjectiveInfo.magnifications = list(self._magnifications)
            self.__ObjectiveInfo.objectiveNames = list(self._objectiveNames)
            self.__ObjectiveInfo.objectivePositions = list(self._objectivePositions)
            self.__ObjectiveInfo.homeDirection = self._homeDirection
            self.__ObjectiveInfo.homePolarity = self._homePolarity
            self.__ObjectiveInfo.homeSpeed = self._homeSpeed
            self.__ObjectiveInfo.homeAcceleration = self._homeAcceleration
            self.__ObjectiveInfo.calibrateOnStart = self._calibrateOnStart
            self.__ObjectiveInfo.active = self._isActive
            
            # Import configfiletools to save the configuration
            from imswitch.imcontrol.model import configfiletools
            
            # Get the setup info reference (should be available through a parent reference)
            # Note: This assumes ObjectiveManager has access to setupInfo
            # If not available directly, this should be passed during initialization
            if hasattr(self, '_setupInfo'):
                options, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(options, self._setupInfo)
                self.__logger.info("Objective configuration saved to setup file")
            else:
                self.__logger.warning("Cannot save objective config: no setupInfo reference available")
                
        except Exception as e:
            self.__logger.error(f"Failed to save objective configuration to file: {e}", exc_info=True)
        
    def update(self):
        """Legacy update method - kept for compatibility"""
        return None


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
