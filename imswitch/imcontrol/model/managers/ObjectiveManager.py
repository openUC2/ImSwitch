import os
import json

from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger


class ObjectiveManager(SignalInterface):
    """
    Manager for objective configuration - provides read/write access to objective
    parameters from the setup JSON file.
    
    This is a simplified manager that only handles configuration persistence.
    All state management (current objective, homing state, etc.) is handled
    by ObjectiveController which is the single source of truth.
    
    The ObjectiveController uses the ESP32StageManager axis "A" for motor control
    instead of the legacy objective_get/objective_act hardware interface.
    """
    
    # Signal emitted when objective parameters are updated in config
    sigObjectiveParametersChanged = Signal(int, dict)  # slot, parameters

    def __init__(self, ObjectiveInfo, setupInfo=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        
        # Store reference to setupInfo for saving configuration
        self._setupInfo = setupInfo

        if ObjectiveInfo is None:
            # Define default configuration
            class DefaultObjectiveInfo:
                pixelsizes = [0.1, 0.2]
                NAs = [0.2, 0.1]
                magnifications = [20, 10]
                objectiveNames = ["20x", "10x"]
                objectivePositions = [-1000, 29000]
                zPositions = [0, 0]  # z0, z1 - relative Z focus offsets
                homeDirection = -1
                homePolarity = 1
                homeSpeed = 20000
                homeAcceleration = 20000
                calibrateOnStart = True
                active = False
            self.__ObjectiveInfo = DefaultObjectiveInfo()
        else:
            self.__ObjectiveInfo = ObjectiveInfo
        
        # Configuration parameters loaded from JSON (can be updated dynamically)
        self._pixelsizes = list(self.__ObjectiveInfo.pixelsizes)
        self._NAs = list(self.__ObjectiveInfo.NAs)
        self._magnifications = list(self.__ObjectiveInfo.magnifications)
        self._objectiveNames = list(self.__ObjectiveInfo.objectiveNames)
        self._objectivePositions = list(self.__ObjectiveInfo.objectivePositions)
        # Z positions (focus offsets) for each objective - default to [0, 0] if not present
        self._zPositions = list(getattr(self.__ObjectiveInfo, 'zPositions', [0, 0]))
        self._homeDirection = self.__ObjectiveInfo.homeDirection
        self._homePolarity = self.__ObjectiveInfo.homePolarity
        self._homeSpeed = self.__ObjectiveInfo.homeSpeed
        self._homeAcceleration = self.__ObjectiveInfo.homeAcceleration
        self._calibrateOnStart = self.__ObjectiveInfo.calibrateOnStart
        self._isActive = self.__ObjectiveInfo.active

    # === Configuration Property accessors (read-only from outside) ===
    
    @property
    def pixelsizes(self):
        """Get pixel sizes for all objectives from config"""
        return list(self._pixelsizes)
    
    @property
    def NAs(self):
        """Get numerical apertures for all objectives from config"""
        return list(self._NAs)
    
    @property
    def magnifications(self):
        """Get magnifications for all objectives from config"""
        return list(self._magnifications)
    
    @property
    def objectiveNames(self):
        """Get objective names from config"""
        return list(self._objectiveNames)
    
    @property
    def objectivePositions(self):
        """Get objective positions (in µm) from config for axis A motor movement"""
        return list(self._objectivePositions)
    
    @property
    def zPositions(self):
        """Get Z focus positions (z0, z1) in µm from config for Z-axis focus adjustment"""
        return list(self._zPositions)
    
    @property
    def homeDirection(self):
        """Get home direction from config"""
        return self._homeDirection
    
    @property
    def homePolarity(self):
        """Get home polarity from config"""
        return self._homePolarity
    
    @property
    def homeSpeed(self):
        """Get home speed from config"""
        return self._homeSpeed
    
    @property
    def homeAcceleration(self):
        """Get home acceleration from config"""
        return self._homeAcceleration
    
    @property
    def calibrateOnStart(self):
        """Get calibrate on start flag from config"""
        return self._calibrateOnStart
    
    @property
    def isActive(self):
        """Get active flag from config"""
        return self._isActive

    # === Configuration update methods ===
    
    def setObjectiveParameters(self, slot: int, *, pixelsize: float = None,
                               NA: float = None, magnification: float = None,
                               objectiveName: str = None, position: float = None,
                               zPosition: float = None, emitSignal: bool = True):
        """
        Update objective parameters for a specific slot and save to config file.
        
        Args:
            slot: Objective slot number (0 or 1)
            pixelsize: Pixel size in micrometers
            NA: Numerical aperture
            magnification: Magnification value
            objectiveName: Name of the objective
            position: Motor position for this objective on axis A (in µm)
            zPosition: Z focus position for this objective (in µm)
            emitSignal: Whether to emit the parameters changed signal
            
        Raises:
            ValueError: If slot is not 0 or 1
        """
        if slot not in [0, 1]:
            raise ValueError(f"Objective slot must be 0 or 1, got {slot}")
        
        changes = {}
        
        if pixelsize is not None:
            self._pixelsizes[slot] = pixelsize
            changes["pixelsize"] = pixelsize
            self.__logger.info(f"Updated pixelsize for objective {slot}: {pixelsize} µm/px")
        
        if NA is not None:
            self._NAs[slot] = NA
            changes["NA"] = NA
            self.__logger.info(f"Updated NA for objective {slot}: {NA}")
        
        if magnification is not None:
            self._magnifications[slot] = magnification
            changes["magnification"] = magnification
            self.__logger.info(f"Updated magnification for objective {slot}: {magnification}x")
        
        if objectiveName is not None:
            self._objectiveNames[slot] = objectiveName
            changes["objectiveName"] = objectiveName
            self.__logger.info(f"Updated name for objective {slot}: {objectiveName}")
        
        if position is not None:
            self._objectivePositions[slot] = position
            changes["position"] = position
            self.__logger.info(f"Updated position for objective {slot}: {position} µm")
        
        if zPosition is not None:
            self._zPositions[slot] = zPosition
            changes["zPosition"] = zPosition
            self.__logger.info(f"Updated Z position for objective {slot}: {zPosition} µm")
        
        if changes:
            # Save to configuration file
            self._saveObjectiveConfigToFile()
            
            if emitSignal:
                params = self.getObjectiveParameters(slot)
                self.sigObjectiveParametersChanged.emit(slot, params)
                    
    def getObjectiveParameters(self, slot: int) -> dict:
        """
        Get parameters for a specific objective slot from config.
        
        Args:
            slot: Objective slot number (0 or 1)
            
        Returns:
            Dictionary with objective parameters
            
        Raises:
            ValueError: If slot is not 0 or 1
        """
        if slot not in [0, 1]:
            raise ValueError(f"Objective slot must be 0 or 1, got {slot}")
        
        return {
            "objectiveSlot": slot,
            "pixelsize": self._pixelsizes[slot],
            "objectiveName": self._objectiveNames[slot],
            "NA": self._NAs[slot],
            "magnification": self._magnifications[slot],
            "position": self._objectivePositions[slot] if slot < len(self._objectivePositions) else None,
            "zPosition": self._zPositions[slot] if slot < len(self._zPositions) else None
        }
    
    def getAllConfig(self) -> dict:
        """
        Get all configuration parameters.
        
        Returns:
            Dictionary with all objective configuration
        """
        return {
            "availableObjectives": [0, 1],
            "availableObjectivesNames": list(self._objectiveNames),
            "availableObjectivesPositions": list(self._objectivePositions),
            "availableObjectiveZPositions": list(self._zPositions),
            "availableObjectiveMagnifications": list(self._magnifications),
            "availableObjectiveNAs": list(self._NAs),
            "availableObjectivePixelSizes": list(self._pixelsizes),
            "homeDirection": self._homeDirection,
            "homePolarity": self._homePolarity,
            "homeSpeed": self._homeSpeed,
            "homeAcceleration": self._homeAcceleration,
            "calibrateOnStart": self._calibrateOnStart,
            "isActive": self._isActive
        }
    
    def _saveObjectiveConfigToFile(self):
        """
        Save current objective configuration to setup configuration file.
        
        This persists the in-memory objective parameters (pixelsizes, NAs, magnifications, etc.)
        to the configuration file on disk.
        """
        try:
            # Update the ObjectiveInfo with current values
            self.__ObjectiveInfo.pixelsizes = list(self._pixelsizes)
            self.__ObjectiveInfo.NAs = list(self._NAs)
            self.__ObjectiveInfo.magnifications = list(self._magnifications)
            self.__ObjectiveInfo.objectiveNames = list(self._objectiveNames)
            self.__ObjectiveInfo.objectivePositions = list(self._objectivePositions)
            self.__ObjectiveInfo.zPositions = list(self._zPositions)
            self.__ObjectiveInfo.homeDirection = self._homeDirection
            self.__ObjectiveInfo.homePolarity = self._homePolarity
            self.__ObjectiveInfo.homeSpeed = self._homeSpeed
            self.__ObjectiveInfo.homeAcceleration = self._homeAcceleration
            self.__ObjectiveInfo.calibrateOnStart = self._calibrateOnStart
            self.__ObjectiveInfo.active = self._isActive
            
            # Import configfiletools to save the configuration
            from imswitch.imcontrol.model import configfiletools
            
            if self._setupInfo is not None:
                options, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(options, self._setupInfo)
                self.__logger.info("Objective configuration saved to setup file")
            else:
                self.__logger.warning("Cannot save objective config: no setupInfo reference available")
                
        except Exception as e:
            self.__logger.error(f"Failed to save objective configuration to file: {e}", exc_info=True)


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
