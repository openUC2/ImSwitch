"""
Base class for galvo scanner managers.

This module provides the abstract base class for all galvo scanner implementations.
Galvo scanners are used for high-speed laser scanning in microscopy applications.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class GalvoScanConfig:
    """Configuration dataclass for galvo scan parameters."""
    nx: int = 256
    ny: int = 256
    x_min: int = 500
    x_max: int = 3500
    y_min: int = 500
    y_max: int = 3500
    sample_period_us: int = 1
    frame_count: int = 0
    bidirectional: bool = False


class GalvoScannerManager(ABC):
    """
    Abstract base class for managers that control galvo scanners.
    
    Each type of galvo scanner corresponds to a manager derived from this class.
    Galvo scanners allow fast 2D positioning of laser beams using mirror galvanometers.
    
    Attributes:
        name: Unique name of the galvo scanner device
        config: Current scan configuration
        is_scanning: Whether the scanner is currently active
    """

    @abstractmethod
    def __init__(self, galvoScannerInfo, name: str):
        """
        Initialize the galvo scanner manager.
        
        Args:
            galvoScannerInfo: Configuration info from setup file
            name: Unique device name from setup file
        """
        self._galvoScannerInfo = galvoScannerInfo
        self._name = name
        
        # Initialize with default config from galvoScannerInfo or defaults
        props = galvoScannerInfo.managerProperties if galvoScannerInfo else {}
        self._config = GalvoScanConfig(
            nx=props.get('nx', 256),
            ny=props.get('ny', 256),
            x_min=props.get('x_min', 500),
            x_max=props.get('x_max', 3500),
            y_min=props.get('y_min', 500),
            y_max=props.get('y_max', 3500),
            sample_period_us=props.get('sample_period_us', 1),
            frame_count=props.get('frame_count', 0),
            bidirectional=props.get('bidirectional', False)
        )
        
        self._is_scanning = False
        self._current_frame = 0
        self._current_line = 0

    @property
    def name(self) -> str:
        """Unique galvo scanner name from setup info."""
        return self._name

    @property
    def config(self) -> GalvoScanConfig:
        """Current scan configuration."""
        return self._config

    @property
    def is_scanning(self) -> bool:
        """Whether the scanner is currently active."""
        return self._is_scanning

    @property
    def current_frame(self) -> int:
        """Current frame number during scanning."""
        return self._current_frame

    @property
    def current_line(self) -> int:
        """Current line number during scanning."""
        return self._current_line

    @abstractmethod
    def start_scan(self, nx: int = None, ny: int = None, 
                   x_min: int = None, x_max: int = None,
                   y_min: int = None, y_max: int = None,
                   sample_period_us: int = None, frame_count: int = None,
                   bidirectional: bool = None, timeout: int = 1) -> Dict[str, Any]:
        """
        Start galvo scanning with the specified parameters.
        
        Args:
            nx: Number of X samples per line (default: from config)
            ny: Number of Y lines (default: from config)
            x_min: Min X position 0-4095 (default: from config)
            x_max: Max X position 0-4095 (default: from config)
            y_min: Min Y position 0-4095 (default: from config)
            y_max: Max Y position 0-4095 (default: from config)
            sample_period_us: Microseconds per sample, 0=max speed (default: from config)
            frame_count: Number of frames, 0=infinite (default: from config)
            bidirectional: Enable bidirectional scanning (default: from config)
            timeout: Request timeout in seconds
            
        Returns:
            dict: Response from the hardware
        """
        pass

    @abstractmethod
    def stop_scan(self, timeout: int = 1) -> Dict[str, Any]:
        """
        Stop the galvo scanner.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            dict: Response from the hardware
        """
        pass

    @abstractmethod
    def get_status(self, timeout: int = 1) -> Dict[str, Any]:
        """
        Get the current galvo scanner status.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            dict: Status including running, current_frame, current_line, config
        """
        pass

    def update_config(self, **kwargs) -> None:
        """
        Update the scan configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key) and value is not None:
                setattr(self._config, key, value)

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get the current configuration as a dictionary.
        
        Returns:
            dict: Current configuration parameters
        """
        return {
            'nx': self._config.nx,
            'ny': self._config.ny,
            'x_min': self._config.x_min,
            'x_max': self._config.x_max,
            'y_min': self._config.y_min,
            'y_max': self._config.y_max,
            'sample_period_us': self._config.sample_period_us,
            'frame_count': self._config.frame_count,
            'bidirectional': self._config.bidirectional
        }


# Copyright (C) 2020-2025 ImSwitch developers
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
