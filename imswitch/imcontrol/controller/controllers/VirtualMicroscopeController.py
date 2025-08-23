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
        self._drift_start_time = None
        self._initial_position = {"X": 0, "Y": 0, "Z": 0}
        self._bleaching_factor = 1.0
        self._frame_count = 0

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
    def switchObjective(self, objective_name: str) -> Dict:
        """Switch to different virtual objective"""
        try:
            if objective_name not in self._config.objectives:
                available = list(self._config.objectives.keys())
                return {"status": "error", "message": f"Unknown objective. Available: {available}"}
                
            self._config.current_objective = objective_name
            obj_params = self._config.objectives[objective_name]
            
            # Apply objective-specific settings to virtual microscope
            if self._virtualMicroscopeManager:
                # Simulate objective change via binning/scaling
                camera = self._virtualMicroscopeManager._camera
                if objective_name == "60x_1.42":
                    camera.binning = True  # Higher magnification
                else:
                    camera.binning = False  # Lower magnification
                    
            self._logger.info(f"Switched to objective {objective_name}: {obj_params}")
            return {
                "status": "success", 
                "objective": objective_name,
                "parameters": obj_params
            }
        except Exception as e:
            self._logger.error(f"Failed to switch objective: {str(e)}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=True)
    def enableStageDrift(self, enabled: bool, drift_rate_x: float = None, 
                        drift_rate_y: float = None, drift_rate_z: float = None) -> Dict:
        """Enable/disable stage drift simulation"""
        try:
            import time
            
            self._config.drift_enabled = enabled
            if drift_rate_x is not None:
                self._config.drift_rate_x = drift_rate_x
            if drift_rate_y is not None:
                self._config.drift_rate_y = drift_rate_y  
            if drift_rate_z is not None:
                self._config.drift_rate_z = drift_rate_z
                
            if enabled and self._drift_start_time is None:
                self._drift_start_time = time.time()
                if self._virtualMicroscopeManager:
                    pos = self._virtualMicroscopeManager._positioner.get_position()
                    self._initial_position = pos.copy()
                    
            elif not enabled:
                self._drift_start_time = None
                
            self._logger.info(f"Stage drift {'enabled' if enabled else 'disabled'}")
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
    def setExposureAndGain(self, exposure_time: float = None, gain: float = None) -> Dict:
        """Set virtual exposure time and gain"""
        try:
            if exposure_time is not None:
                self._config.exposure_time = max(1.0, exposure_time)
            if gain is not None:
                self._config.gain = max(0.1, gain)
                
            # Apply to virtual illuminator
            if self._virtualMicroscopeManager:
                # Scale illuminator intensity based on exposure and gain
                base_intensity = 1000
                scaled_intensity = base_intensity * (self._config.exposure_time / 100.0) * self._config.gain
                self._virtualMicroscopeManager._illuminator.set_intensity(1, scaled_intensity)
                
            return {
                "status": "success", 
                "exposure_time": self._config.exposure_time,
                "gain": self._config.gain
            }
        except Exception as e:
            self._logger.error(f"Failed to set exposure/gain: {str(e)}")
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
                "drift_active": self._drift_start_time is not None,
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
        """Internal method to apply stage drift if enabled"""
        if not self._config.drift_enabled or self._drift_start_time is None:
            return
            
        if not self._virtualMicroscopeManager:
            return
            
        try:
            import time
            elapsed = time.time() - self._drift_start_time
            
            # Calculate drift offsets
            drift_x = self._config.drift_rate_x * elapsed
            drift_y = self._config.drift_rate_y * elapsed
            drift_z = self._config.drift_rate_z * elapsed
            
            # Apply drift to position
            new_x = self._initial_position["X"] + drift_x
            new_y = self._initial_position["Y"] + drift_y
            new_z = self._initial_position["Z"] + drift_z
            
            self._virtualMicroscopeManager._positioner.move(
                x=new_x, y=new_y, z=new_z, is_absolute=True
            )
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
            base_intensity = 1000 * self._config.gain * (self._config.exposure_time / 100.0)
            bleached_intensity = base_intensity * self._bleaching_factor
            self._virtualMicroscopeManager._illuminator.set_intensity(1, bleached_intensity)

    def update(self):
        """Update method called periodically by the framework"""
        super().update()
        
        # Apply ongoing simulations
        self._applyDrift()
        self._applyPhotobleaching()


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