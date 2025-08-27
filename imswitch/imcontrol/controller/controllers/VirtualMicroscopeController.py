from imswitch.imcommon.model import APIExport, initLogger
from imswitch.imcontrol.controller.basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS
import numpy as np
from typing import Dict, Optional, Tuple, List
from pydantic import BaseModel


class VirtualMicroscopeConfig(BaseModel):
    """Configuration for virtual microscope simulation parameters"""
    # Stage drift simulation
    drift_enabled: bool = False
    drift_rate_x: float = 0.1  # pixels per second
    drift_rate_y: float = 0.1
    drift_rate_z: float = 0.05
    
    # Objective parameters
    objectives: Dict[str, Dict] = {
        "20x_0.75": {
            "magnification": 20,
            "NA": 0.75,
            "pixel_scale": 0.325,  # um per pixel
            "type": "air"
        },
        "60x_1.42": {
            "magnification": 60,
            "NA": 1.42,
            "pixel_scale": 0.108,  # um per pixel  
            "type": "oil"
        }
    }
    current_objective: str = "20x_0.75"
    
    # Exposure and gain simulation
    exposure_time: float = 100.0  # ms
    gain: float = 1.0
    
    # Photobleaching simulation
    bleaching_enabled: bool = False
    bleaching_rate: float = 0.01  # fraction per exposure
    
    # Multi-channel simulation
    channels: Dict[str, Dict] = {
        "488": {"wavelength": 488, "intensity": 1.0, "color": "cyan"},
        "561": {"wavelength": 561, "intensity": 1.0, "color": "green"},  
        "640": {"wavelength": 640, "intensity": 1.0, "color": "red"}
    }
    active_channels: List[str] = ["488"]
    
    # Noise parameters
    readout_noise: float = 50.0
    shot_noise_enabled: bool = True
    dark_current: float = 0.1
    
    # Sampling and aliasing
    nyquist_sampling: bool = True
    aliasing_enabled: bool = False


class VirtualMicroscopeController(LiveUpdatedController):
    """Controller for enhanced Virtual Microscope simulation with API endpoints"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)
        
        # Initialize simulation configuration
        self._config = VirtualMicroscopeConfig()
        
        # Get virtual microscope manager from RS232 devices
        self._virtualMicroscopeManager = None
        if hasattr(self._master, 'rs232sManager'):
            rs232_managers = self._master.rs232sManager
            for name, manager in rs232_managers:
                if 'VirtualMicroscope' in name:
                    self._virtualMicroscopeManager = manager
                    break
        
        if self._virtualMicroscopeManager is None:
            self._logger.warning("Virtual Microscope Manager not found")
            return
            
        # Initialize simulation state
        self._last_drift_time = None
        self._bleaching_factor = 1.0
        self._frame_count = 0
        
        # Connect to existing controllers for integration
        self._connectToExistingControllers()

    @APIExport(runOnUIThread=True)
    def getConfig(self) -> Dict:
        """Get current virtual microscope configuration"""
        return self._config.dict()

    @APIExport(runOnUIThread=True)
    def updateConfig(self, config: Dict) -> Dict:
        """Update virtual microscope configuration"""
        try:
            # Update configuration with provided values
            for key, value in config.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                    
            self._logger.info(f"Updated virtual microscope config: {config}")
            return {"status": "success", "config": self._config.dict()}
        except Exception as e:
            self._logger.error(f"Failed to update config: {str(e)}")
            return {"status": "error", "message": str(e)}



    @APIExport(runOnUIThread=True)
    def enableStageDrift(self, enabled: bool, drift_rate_x: float = None, 
                        drift_rate_y: float = None, drift_rate_z: float = None) -> Dict:
        """Enable/disable stage drift simulation with relative increments"""
        try:
            self._config.drift_enabled = enabled
            if drift_rate_x is not None:
                self._config.drift_rate_x = drift_rate_x
            if drift_rate_y is not None:
                self._config.drift_rate_y = drift_rate_y  
            if drift_rate_z is not None:
                self._config.drift_rate_z = drift_rate_z
                
            if enabled:
                # Reset drift timing for relative increments
                import time
                self._last_drift_time = time.time()
            else:
                self._last_drift_time = None
                
            self._logger.info(f"Stage drift {'enabled' if enabled else 'disabled'} with rates: X={self._config.drift_rate_x}, Y={self._config.drift_rate_y}, Z={self._config.drift_rate_z}")
            return {"status": "success", "drift_enabled": enabled}
        except Exception as e:
            self._logger.error(f"Failed to set stage drift: {str(e)}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=True)
    def enablePhotobleaching(self, enabled: bool, bleaching_rate: float = None) -> Dict:
        """Enable/disable photobleaching simulation"""
        try:
            self._config.bleaching_enabled = enabled
            if bleaching_rate is not None:
                self._config.bleaching_rate = bleaching_rate
                
            if not enabled:
                self._bleaching_factor = 1.0  # Reset bleaching
                self._frame_count = 0
                
            self._logger.info(f"Photobleaching {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "bleaching_enabled": enabled}
        except Exception as e:
            self._logger.error(f"Failed to set photobleaching: {str(e)}")
            return {"status": "error", "message": str(e)}



    @APIExport(runOnUIThread=True)
    def setActiveChannels(self, channels: List[str]) -> Dict:
        """Set active laser channels for multi-channel imaging"""
        try:
            available_channels = list(self._config.channels.keys())
            invalid_channels = [ch for ch in channels if ch not in available_channels]
            
            if invalid_channels:
                return {
                    "status": "error", 
                    "message": f"Invalid channels: {invalid_channels}. Available: {available_channels}"
                }
                
            self._config.active_channels = channels
            
            # Calculate combined intensity from active channels
            if self._virtualMicroscopeManager:
                total_intensity = sum(
                    self._config.channels[ch]["intensity"] 
                    for ch in channels
                ) * 1000  # Base scaling
                self._virtualMicroscopeManager._illuminator.set_intensity(1, total_intensity)
                
            self._logger.info(f"Set active channels: {channels}")
            return {"status": "success", "active_channels": channels}
        except Exception as e:
            self._logger.error(f"Failed to set channels: {str(e)}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=True)
    def simulateAliasing(self, enabled: bool) -> Dict:
        """Enable/disable aliasing artifacts for educational purposes"""
        try:
            self._config.aliasing_enabled = enabled
            self._config.nyquist_sampling = not enabled  # Inverse relationship
            
            self._logger.info(f"Aliasing simulation {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "aliasing_enabled": enabled}
        except Exception as e:
            self._logger.error(f"Failed to set aliasing: {str(e)}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=True)
    def getStatus(self) -> Dict:
        """Get current virtual microscope status and parameters"""
        try:
            status = {
                "config": self._config.dict(),
                "drift_active": self._last_drift_time is not None,
                "frame_count": self._frame_count,
                "bleaching_factor": self._bleaching_factor
            }
            
            if self._virtualMicroscopeManager:
                position = self._virtualMicroscopeManager._positioner.get_position()
                intensity = self._virtualMicroscopeManager._illuminator.get_intensity(1)
                objective_state = getattr(self._virtualMicroscopeManager, 'currentObjective', 1)
                
                status.update({
                    "position": position,
                    "illuminator_intensity": intensity,
                    "current_objective_slot": objective_state
                })
                
                # Add SLM status if available
                if hasattr(self._virtualMicroscopeManager._virtualMicroscope, 'slm'):
                    slm_status = self._virtualMicroscopeManager._virtualMicroscope.slm.get_status()
                    status["slm"] = slm_status
                
            return status
        except Exception as e:
            self._logger.error(f"Failed to get status: {str(e)}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=True)
    def setSLMPattern(self, pattern_type: str, **kwargs) -> Dict:
        """Set SLM pattern for structured illumination and beam shaping"""
        try:
            if not self._virtualMicroscopeManager:
                return {"status": "error", "message": "Virtual Microscope Manager not available"}
                
            if not hasattr(self._virtualMicroscopeManager._virtualMicroscope, 'slm'):
                return {"status": "error", "message": "Virtual SLM not available"}
            
            slm = self._virtualMicroscopeManager._virtualMicroscope.slm
            success = slm.set_pattern(pattern_type, **kwargs)
            
            if success:
                slm.set_active(True)
                return {
                    "status": "success",
                    "pattern_type": pattern_type,
                    "parameters": kwargs,
                    "slm_status": slm.get_status()
                }
            else:
                return {"status": "error", "message": "Failed to set SLM pattern"}
                
        except Exception as e:
            self._logger.error(f"Failed to set SLM pattern: {str(e)}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=True)
    def applySLMAberrationCorrection(self, **zernike_coeffs) -> Dict:
        """Apply aberration correction using Zernike polynomials on SLM"""
        try:
            if not self._virtualMicroscopeManager:
                return {"status": "error", "message": "Virtual Microscope Manager not available"}
                
            if not hasattr(self._virtualMicroscopeManager._virtualMicroscope, 'slm'):
                return {"status": "error", "message": "Virtual SLM not available"}
            
            slm = self._virtualMicroscopeManager._virtualMicroscope.slm
            success = slm.apply_aberration_correction(**zernike_coeffs)
            
            if success:
                return {
                    "status": "success",
                    "zernike_coefficients": zernike_coeffs,
                    "slm_status": slm.get_status()
                }
            else:
                return {"status": "error", "message": "Failed to apply aberration correction"}
                
        except Exception as e:
            self._logger.error(f"Failed to apply SLM aberration correction: {str(e)}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=True)
    def getSLMPattern(self) -> Dict:
        """Get current SLM pattern as base64 encoded image"""
        try:
            if not self._virtualMicroscopeManager:
                return {"status": "error", "message": "Virtual Microscope Manager not available"}
                
            if not hasattr(self._virtualMicroscopeManager._virtualMicroscope, 'slm'):
                return {"status": "error", "message": "Virtual SLM not available"}
            
            slm = self._virtualMicroscopeManager._virtualMicroscope.slm
            pattern = slm.get_pattern()
            
            # Convert to base64 for web transmission
            import base64
            from io import BytesIO
            try:
                from PIL import Image
                img = Image.fromarray(pattern)
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return {
                    "status": "success",
                    "pattern_base64": img_str,
                    "pattern_shape": pattern.shape,
                    "slm_status": slm.get_status()
                }
            except ImportError:
                # Fallback without PIL
                return {
                    "status": "success", 
                    "pattern_shape": pattern.shape,
                    "pattern_available": True,
                    "slm_status": slm.get_status(),
                    "note": "Pattern data available but PIL not installed for base64 encoding"
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get SLM pattern: {str(e)}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=True)
    def resetSLM(self) -> Dict:
        """Reset SLM to blank state"""
        try:
            if not self._virtualMicroscopeManager:
                return {"status": "error", "message": "Virtual Microscope Manager not available"}
                
            if not hasattr(self._virtualMicroscopeManager._virtualMicroscope, 'slm'):
                return {"status": "error", "message": "Virtual SLM not available"}
            
            slm = self._virtualMicroscopeManager._virtualMicroscope.slm
            slm.reset()
            
            return {
                "status": "success",
                "message": "SLM reset to blank state",
                "slm_status": slm.get_status()
            }
            
        except Exception as e:
            self._logger.error(f"Failed to reset SLM: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _applyDrift(self):
        """Internal method to apply stage drift if enabled using relative increments"""
        if not self._config.drift_enabled or not hasattr(self, '_last_drift_time') or self._last_drift_time is None:
            return
            
        if not self._virtualMicroscopeManager:
            return
            
        try:
            import time
            current_time = time.time()
            time_interval = current_time - self._last_drift_time
            
            # Apply relative drift increments based on time interval
            drift_x = self._config.drift_rate_x * time_interval
            drift_y = self._config.drift_rate_y * time_interval  
            drift_z = self._config.drift_rate_z * time_interval
            
            # Apply relative position changes
            self._virtualMicroscopeManager._positioner.move(
                x=drift_x, y=drift_y, z=drift_z, is_absolute=False
            )
            
            # Update timing for next interval
            self._last_drift_time = current_time
            
        except Exception as e:
            self._logger.error(f"Error applying drift: {str(e)}")

    def _applyPhotobleaching(self):
        """Internal method to apply photobleaching if enabled"""
        if not self._config.bleaching_enabled:
            return
            
        self._frame_count += 1
        
        # Apply exponential decay
        self._bleaching_factor *= (1 - self._config.bleaching_rate)
        
        # Update illuminator intensity to reflect bleaching
        if self._virtualMicroscopeManager:
            # Get current intensity and apply bleaching factor
            current_intensity = self._virtualMicroscopeManager._illuminator.get_intensity(1)
            if current_intensity > 0:
                bleached_intensity = current_intensity * self._bleaching_factor
                self._virtualMicroscopeManager._illuminator.set_intensity(1, bleached_intensity)

    def update(self):
        """Update method called periodically by the framework"""
        super().update()
        
        # Apply ongoing simulations
        self._applyDrift()
        self._applyPhotobleaching()
        
    def _connectToExistingControllers(self):
        """Connect to existing controllers to listen for changes"""
        try:
            # Connect to SettingsController changes via shared attributes
            if hasattr(self._commChannel, 'sharedAttrs'):
                self._commChannel.sharedAttrs.sigAttributeSet.connect(self._onDetectorSettingChanged)
                
            # Connect to ObjectiveController if available
            if hasattr(self._master, 'objectiveController'):
                objective_controller = self._master.objectiveController
                if hasattr(objective_controller, 'sigObjectiveChanged'):
                    objective_controller.sigObjectiveChanged.connect(self._onObjectiveChanged)
                    
        except Exception as e:
            self._logger.warning(f"Could not fully connect to existing controllers: {str(e)}")
            
    def _onDetectorSettingChanged(self, key, value):
        """Handle detector setting changes from SettingsController"""
        try:
            if not isinstance(key, tuple) or len(key) < 3:
                return
                
            category, detector_name, param_category = key[:3]
            if category != 'Detector':
                return
                
            # Handle exposure time changes
            if len(key) == 4 and param_category == 'Param' and key[3] == 'exposure':
                self._config.exposure_time = value
                if self._virtualMicroscopeManager:
                    self._virtualMicroscopeManager.updateExposureGain(
                        exposure_time=value, 
                        gain=self._config.gain
                    )
                self._logger.info(f"Virtual microscope updated exposure: {value}")
                
            # Handle gain changes  
            elif len(key) == 4 and param_category == 'Param' and key[3] == 'gain':
                self._config.gain = value
                if self._virtualMicroscopeManager:
                    self._virtualMicroscopeManager.updateExposureGain(
                        exposure_time=self._config.exposure_time,
                        gain=value
                    )
                self._logger.info(f"Virtual microscope updated gain: {value}")
                
        except Exception as e:
            self._logger.error(f"Error handling detector setting change: {str(e)}")
            
    def _onObjectiveChanged(self, status_dict):
        """Handle objective changes from ObjectiveController"""
        try:
            if 'state' in status_dict:
                objective_slot = status_dict['state']
                if self._virtualMicroscopeManager:
                    self._virtualMicroscopeManager.setObjective(objective_slot)
                    
                # Update our config to reflect the change
                if objective_slot == 1:
                    self._config.current_objective = "20x_0.75"
                elif objective_slot == 2:
                    self._config.current_objective = "60x_1.42"
                    
                self._logger.info(f"Virtual microscope updated objective: slot {objective_slot}")
                
        except Exception as e:
            self._logger.error(f"Error handling objective change: {str(e)}")


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