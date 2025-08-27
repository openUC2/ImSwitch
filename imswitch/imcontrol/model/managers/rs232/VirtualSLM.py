"""
Virtual SLM (Spatial Light Modulator) simulation for ImSwitch Virtual Microscope

This module provides simulation capabilities for SLM-based microscopy techniques
such as structured illumination, beam shaping, and holographic manipulation.
"""

import numpy as np
import threading
from typing import Dict, List, Optional, Tuple, Union
from imswitch.imcommon.model import initLogger


class VirtualSLM:
    """
    Virtual Spatial Light Modulator for advanced microscopy simulation
    
    Supports various SLM patterns and operations for educational and research purposes:
    - Structured illumination patterns (sinusoidal, square wave)
    - Beam shaping (donut, top-hat, Gaussian)
    - Holographic patterns for optical trapping
    - Aberration correction via Zernike polynomials
    - Custom pattern upload and display
    """
    
    def __init__(self, parent, width=1920, height=1152):
        """
        Initialize Virtual SLM
        
        Parameters:
        -----------
        parent : VirtualMicroscopy
            Parent virtual microscopy instance
        width : int
            SLM width in pixels (default: 1920 for typical SLM)
        height : int  
            SLM height in pixels (default: 1152 for typical SLM)
        """
        self._parent = parent
        self._logger = initLogger(self, tryInheritParent=True)
        
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        
        # Current SLM state
        self._current_pattern = np.zeros((height, width), dtype=np.uint8)
        self._is_active = False
        self._pattern_type = "blank"
        
        # Pattern parameters
        self._pattern_params = {
            "frequency": 10,  # lines per mm
            "phase": 0,       # phase offset in radians
            "amplitude": 255, # pattern amplitude (0-255)
            "angle": 0,       # pattern rotation angle
            "center_x": width // 2,
            "center_y": height // 2
        }
        
        # Aberration correction parameters (Zernike coefficients)
        self._zernike_coeffs = {
            "tip": 0.0,           # Z2 - tip
            "tilt": 0.0,          # Z3 - tilt  
            "defocus": 0.0,       # Z4 - defocus
            "astig_0": 0.0,       # Z5 - astigmatism 0°
            "astig_45": 0.0,      # Z6 - astigmatism 45°
            "coma_x": 0.0,        # Z7 - coma x
            "coma_y": 0.0,        # Z8 - coma y
            "spherical": 0.0      # Z9 - spherical aberration
        }
        
        self._logger.info(f"Initialized Virtual SLM ({width}x{height} pixels)")
    
    def set_pattern(self, pattern_type: str, **kwargs) -> bool:
        """
        Set SLM pattern type and parameters
        
        Parameters:
        -----------
        pattern_type : str
            Pattern type: 'blank', 'sinusoidal', 'square', 'donut', 'tophat', 'gaussian', 'custom'
        **kwargs : dict
            Pattern-specific parameters
            
        Returns:
        --------
        bool : Success status
        """
        with self.lock:
            try:
                # Update pattern parameters
                for key, value in kwargs.items():
                    if key in self._pattern_params:
                        self._pattern_params[key] = value
                
                self._pattern_type = pattern_type
                self._current_pattern = self._generate_pattern(pattern_type)
                
                self._logger.info(f"Set SLM pattern: {pattern_type} with params {kwargs}")
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to set SLM pattern: {e}")
                return False
    
    def _generate_pattern(self, pattern_type: str) -> np.ndarray:
        """Generate the specified pattern"""
        
        if pattern_type == "blank":
            return np.zeros((self.height, self.width), dtype=np.uint8)
            
        elif pattern_type == "sinusoidal":
            return self._generate_sinusoidal()
            
        elif pattern_type == "square":
            return self._generate_square_wave()
            
        elif pattern_type == "donut":
            return self._generate_donut()
            
        elif pattern_type == "tophat":
            return self._generate_tophat()
            
        elif pattern_type == "gaussian":
            return self._generate_gaussian()
            
        else:
            self._logger.warning(f"Unknown pattern type: {pattern_type}, using blank")
            return np.zeros((self.height, self.width), dtype=np.uint8)
    
    def _generate_sinusoidal(self) -> np.ndarray:
        """Generate sinusoidal grating pattern for structured illumination"""
        y, x = np.ogrid[:self.height, :self.width]
        
        # Create coordinate system centered at pattern center
        x_centered = x - self._pattern_params["center_x"] 
        y_centered = y - self._pattern_params["center_y"]
        
        # Apply rotation
        angle = np.radians(self._pattern_params["angle"])
        x_rot = x_centered * np.cos(angle) - y_centered * np.sin(angle)
        
        # Generate sinusoidal pattern
        frequency = self._pattern_params["frequency"] * 2 * np.pi / self.width
        phase = self._pattern_params["phase"]
        amplitude = self._pattern_params["amplitude"]
        
        pattern = 127 + (amplitude / 2) * np.sin(frequency * x_rot + phase)
        return np.clip(pattern, 0, 255).astype(np.uint8)
    
    def _generate_square_wave(self) -> np.ndarray:
        """Generate square wave pattern"""
        y, x = np.ogrid[:self.height, :self.width]
        
        x_centered = x - self._pattern_params["center_x"] 
        y_centered = y - self._pattern_params["center_y"]
        
        angle = np.radians(self._pattern_params["angle"])
        x_rot = x_centered * np.cos(angle) - y_centered * np.sin(angle)
        
        frequency = self._pattern_params["frequency"] * 2 * np.pi / self.width
        phase = self._pattern_params["phase"]
        amplitude = self._pattern_params["amplitude"]
        
        pattern = 127 + (amplitude / 2) * np.sign(np.sin(frequency * x_rot + phase))
        return np.clip(pattern, 0, 255).astype(np.uint8)
    
    def _generate_donut(self) -> np.ndarray:
        """Generate donut pattern for STED-like applications"""
        y, x = np.ogrid[:self.height, :self.width]
        
        center_x = self._pattern_params["center_x"]
        center_y = self._pattern_params["center_y"] 
        
        # Calculate distance from center
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Generate donut profile (higher intensity at edges)
        r_norm = r / (min(self.width, self.height) / 4)  # Normalize radius
        donut = np.sin(np.pi * r_norm)**2  # Donut shape
        
        pattern = self._pattern_params["amplitude"] * donut
        return np.clip(pattern, 0, 255).astype(np.uint8)
    
    def _generate_tophat(self) -> np.ndarray:
        """Generate top-hat (flat circular) pattern"""
        y, x = np.ogrid[:self.height, :self.width]
        
        center_x = self._pattern_params["center_x"]
        center_y = self._pattern_params["center_y"]
        
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        radius = min(self.width, self.height) / 6
        
        pattern = np.where(r <= radius, self._pattern_params["amplitude"], 0)
        return pattern.astype(np.uint8)
    
    def _generate_gaussian(self) -> np.ndarray:
        """Generate Gaussian beam pattern"""
        y, x = np.ogrid[:self.height, :self.width]
        
        center_x = self._pattern_params["center_x"]
        center_y = self._pattern_params["center_y"]
        
        sigma = min(self.width, self.height) / 8  # Beam waist
        
        r_sq = (x - center_x)**2 + (y - center_y)**2
        gaussian = np.exp(-r_sq / (2 * sigma**2))
        
        pattern = self._pattern_params["amplitude"] * gaussian
        return np.clip(pattern, 0, 255).astype(np.uint8)
    
    def apply_aberration_correction(self, **zernike_coeffs) -> bool:
        """
        Apply aberration correction using Zernike polynomials
        
        Parameters:
        -----------
        **zernike_coeffs : dict
            Zernike coefficients for aberration correction
            
        Returns:
        --------
        bool : Success status
        """
        with self.lock:
            try:
                # Update Zernike coefficients
                for mode, coeff in zernike_coeffs.items():
                    if mode in self._zernike_coeffs:
                        self._zernike_coeffs[mode] = coeff
                
                # Apply aberration correction to current pattern
                correction = self._generate_zernike_correction()
                self._current_pattern = np.clip(
                    self._current_pattern.astype(np.float32) + correction, 0, 255
                ).astype(np.uint8)
                
                self._logger.info(f"Applied aberration correction: {zernike_coeffs}")
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to apply aberration correction: {e}")
                return False
    
    def _generate_zernike_correction(self) -> np.ndarray:
        """Generate Zernike aberration correction pattern"""
        y, x = np.ogrid[:self.height, :self.width]
        
        # Normalize coordinates to unit circle
        x_norm = (2 * x / self.width) - 1
        y_norm = (2 * y / self.height) - 1
        
        # Calculate polar coordinates
        rho = np.sqrt(x_norm**2 + y_norm**2)
        phi = np.arctan2(y_norm, x_norm)
        
        # Apply unit circle mask
        mask = rho <= 1
        
        # Initialize correction pattern
        correction = np.zeros((self.height, self.width))
        
        # Add Zernike modes (simplified implementation)
        if self._zernike_coeffs["tip"] != 0:
            correction += self._zernike_coeffs["tip"] * rho * np.cos(phi) * mask
            
        if self._zernike_coeffs["tilt"] != 0:
            correction += self._zernike_coeffs["tilt"] * rho * np.sin(phi) * mask
            
        if self._zernike_coeffs["defocus"] != 0:
            correction += self._zernike_coeffs["defocus"] * (2 * rho**2 - 1) * mask
            
        if self._zernike_coeffs["astig_0"] != 0:
            correction += self._zernike_coeffs["astig_0"] * rho**2 * np.cos(2 * phi) * mask
            
        if self._zernike_coeffs["astig_45"] != 0:
            correction += self._zernike_coeffs["astig_45"] * rho**2 * np.sin(2 * phi) * mask
        
        # Scale correction to appropriate range
        return 20 * correction  # Adjust scaling as needed
    
    def upload_custom_pattern(self, pattern: np.ndarray) -> bool:
        """
        Upload custom pattern to SLM
        
        Parameters:
        -----------
        pattern : np.ndarray
            Custom pattern array (will be resized if needed)
            
        Returns:
        --------
        bool : Success status
        """
        with self.lock:
            try:
                # Resize pattern if needed
                if pattern.shape != (self.height, self.width):
                    from scipy.ndimage import zoom
                    scale_y = self.height / pattern.shape[0]
                    scale_x = self.width / pattern.shape[1]
                    pattern = zoom(pattern, (scale_y, scale_x), order=1)
                
                # Ensure proper data type and range
                pattern = np.clip(pattern, 0, 255).astype(np.uint8)
                
                self._current_pattern = pattern
                self._pattern_type = "custom"
                
                self._logger.info(f"Uploaded custom pattern: {pattern.shape}")
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to upload custom pattern: {e}")
                return False
    
    def get_pattern(self) -> np.ndarray:
        """Get current SLM pattern"""
        with self.lock:
            return self._current_pattern.copy()
    
    def set_active(self, active: bool):
        """Set SLM active state"""
        with self.lock:
            self._is_active = active
            self._logger.info(f"SLM {'activated' if active else 'deactivated'}")
    
    def is_active(self) -> bool:
        """Check if SLM is active"""
        with self.lock:
            return self._is_active
    
    def get_status(self) -> Dict:
        """Get comprehensive SLM status"""
        with self.lock:
            return {
                "active": self._is_active,
                "pattern_type": self._pattern_type,
                "pattern_params": self._pattern_params.copy(),
                "zernike_coeffs": self._zernike_coeffs.copy(),
                "dimensions": (self.height, self.width)
            }
    
    def reset(self):
        """Reset SLM to blank state"""
        with self.lock:
            self._current_pattern = np.zeros((self.height, self.width), dtype=np.uint8)
            self._pattern_type = "blank"
            self._is_active = False
            
            # Reset parameters to defaults
            self._pattern_params = {
                "frequency": 10,
                "phase": 0,
                "amplitude": 255,
                "angle": 0,
                "center_x": self.width // 2,
                "center_y": self.height // 2
            }
            
            self._zernike_coeffs = {key: 0.0 for key in self._zernike_coeffs}
            
            self._logger.info("SLM reset to blank state")


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