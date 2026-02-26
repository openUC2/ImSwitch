"""
Base class for galvo scanner managers.

This module provides the abstract base class for all galvo scanner implementations.
Galvo scanners are used for high-speed laser scanning in microscopy applications.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import os
import numpy as np


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
    pre_samples: int = 0
    fly_samples: int = 0
    trig_delay_us: int = 0
    trig_width_us: int = 0
    line_settle_samples: int = 0
    enable_trigger: int = 1
    apply_x_lut: int = 0


@dataclass
class AffineTransform:
    """
    2x3 affine transformation matrix for camera-to-galvo coordinate mapping.
    
    The transform maps camera pixel coordinates to galvo DAC coordinates:
        galvo_x = a11 * cam_x + a12 * cam_y + tx
        galvo_y = a21 * cam_x + a22 * cam_y + ty
    """
    a11: float = 1.0
    a12: float = 0.0
    tx: float = 0.0
    a21: float = 0.0
    a22: float = 1.0
    ty: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """Return as 2x3 numpy array."""
        return np.array([
            [self.a11, self.a12, self.tx],
            [self.a21, self.a22, self.ty]
        ])

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'AffineTransform':
        """Create from a 2x3 numpy array."""
        return cls(
            a11=float(matrix[0, 0]),
            a12=float(matrix[0, 1]),
            tx=float(matrix[0, 2]),
            a21=float(matrix[1, 0]),
            a22=float(matrix[1, 1]),
            ty=float(matrix[1, 2])
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            'a11': self.a11, 'a12': self.a12, 'tx': self.tx,
            'a21': self.a21, 'a22': self.a22, 'ty': self.ty
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'AffineTransform':
        """Create from dictionary."""
        return cls(
            a11=d.get('a11', 1.0), a12=d.get('a12', 0.0), tx=d.get('tx', 0.0),
            a21=d.get('a21', 0.0), a22=d.get('a22', 1.0), ty=d.get('ty', 0.0)
        )

    def transform_point(self, cam_x: float, cam_y: float) -> tuple:
        """Transform a single camera coordinate to galvo DAC coordinate."""
        galvo_x = self.a11 * cam_x + self.a12 * cam_y + self.tx
        galvo_y = self.a21 * cam_x + self.a22 * cam_y + self.ty
        return (galvo_x, galvo_y)

    def transform_points(self, points: List[Dict]) -> List[Dict]:
        """
        Transform a list of points from camera coordinates to galvo DAC coordinates.
        
        Args:
            points: List of dicts with 'x', 'y' (camera coords) and other fields
            
        Returns:
            List of dicts with 'x', 'y' replaced by galvo DAC coords (clamped 0-4095)
        """
        result = []
        for pt in points:
            gx, gy = self.transform_point(pt['x'], pt['y'])
            new_pt = dict(pt)
            new_pt['x'] = int(max(0, min(4095, round(gx))))
            new_pt['y'] = int(max(0, min(4095, round(gy))))
            result.append(new_pt)
        return result

    @staticmethod
    def compute_from_point_pairs(
        cam_points: List[tuple],
        galvo_points: List[tuple]
    ) -> 'AffineTransform':
        """
        Compute an affine transform from 3+ corresponding point pairs.
        
        Uses least-squares to solve for the 2x3 affine matrix:
            [gx]   [a11 a12 tx] [cx]
            [gy] = [a21 a22 ty] [cy]
                                [ 1]
        
        Args:
            cam_points: List of (cx, cy) camera pixel coordinates
            galvo_points: List of (gx, gy) galvo DAC coordinates
            
        Returns:
            AffineTransform computed from the point pairs
        """
        n = len(cam_points)
        if n < 3:
            raise ValueError("At least 3 point pairs required for affine calibration")

        # Build the system: A * params = b
        # For each point: gx = a11*cx + a12*cy + tx
        #                 gy = a21*cx + a22*cy + ty
        A = np.zeros((2 * n, 6))
        b = np.zeros(2 * n)
        for i, (cp, gp) in enumerate(zip(cam_points, galvo_points)):
            cx, cy = cp
            gx, gy = gp
            A[2*i, 0] = cx
            A[2*i, 1] = cy
            A[2*i, 2] = 1.0
            b[2*i] = gx
            A[2*i+1, 3] = cx
            A[2*i+1, 4] = cy
            A[2*i+1, 5] = 1.0
            b[2*i+1] = gy

        # Least-squares solution
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        matrix = np.array([
            [result[0], result[1], result[2]],
            [result[3], result[4], result[5]]
        ])
        return AffineTransform.from_matrix(matrix)


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
            bidirectional=props.get('bidirectional', False),
            pre_samples=props.get('pre_samples', 0),
            fly_samples=props.get('fly_samples', 0),
            trig_delay_us=props.get('trig_delay_us', 0),
            trig_width_us=props.get('trig_width_us', 0),
            line_settle_samples=props.get('line_settle_samples', 0),
            enable_trigger=props.get('enable_trigger', 1),
            apply_x_lut=props.get('apply_x_lut', 0)
        )
        
        self._is_scanning = False
        self._current_frame = 0
        self._current_line = 0
        
        # Affine transform for camera-to-galvo mapping
        self._affine_transform = AffineTransform()
        self._calibration_config_path = None
        
        # Arbitrary points state
        self._arbitrary_points = []
        self._arb_scan_paused = False
        self._arb_scan_running = False

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
                   bidirectional: bool = None,
                   pre_samples: int = None, fly_samples: int = None,
                   trig_delay_us: int = None, trig_width_us: int = None,
                   line_settle_samples: int = None, enable_trigger: int = None,
                   apply_x_lut: int = None,
                   timeout: int = 1) -> Dict[str, Any]:
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
            pre_samples: Pre-scan samples (default: from config)
            fly_samples: Fly-back samples (default: from config)
            trig_delay_us: Trigger delay in microseconds (default: from config)
            trig_width_us: Trigger width in microseconds (default: from config)
            line_settle_samples: Line settling samples (default: from config)
            enable_trigger: Enable trigger output 0/1 (default: from config)
            apply_x_lut: Apply X lookup table 0/1 (default: from config)
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
            'bidirectional': self._config.bidirectional,
            'pre_samples': self._config.pre_samples,
            'fly_samples': self._config.fly_samples,
            'trig_delay_us': self._config.trig_delay_us,
            'trig_width_us': self._config.trig_width_us,
            'line_settle_samples': self._config.line_settle_samples,
            'enable_trigger': self._config.enable_trigger,
            'apply_x_lut': self._config.apply_x_lut
        }

    # ========================
    # Affine Transform Methods
    # ========================

    @property
    def affine_transform(self) -> AffineTransform:
        """Current camera-to-galvo affine transform."""
        return self._affine_transform

    def set_affine_transform(self, a11: float = None, a12: float = None,
                              tx: float = None, a21: float = None,
                              a22: float = None, ty: float = None) -> Dict[str, Any]:
        """
        Set the affine transform parameters.
        
        Args:
            a11, a12, tx, a21, a22, ty: Affine matrix elements
            
        Returns:
            dict: Updated affine transform
        """
        if a11 is not None: self._affine_transform.a11 = a11
        if a12 is not None: self._affine_transform.a12 = a12
        if tx is not None: self._affine_transform.tx = tx
        if a21 is not None: self._affine_transform.a21 = a21
        if a22 is not None: self._affine_transform.a22 = a22
        if ty is not None: self._affine_transform.ty = ty
        return self._affine_transform.to_dict()

    def get_affine_transform_dict(self) -> Dict[str, float]:
        """Get the affine transform as a dictionary."""
        return self._affine_transform.to_dict()

    def reset_affine_transform(self):
        """Reset affine transform to identity."""
        self._affine_transform = AffineTransform()

    def compute_affine_from_calibration(
        self,
        cam_points: List[tuple],
        galvo_points: List[tuple]
    ) -> Dict[str, Any]:
        """
        Compute and set affine transform from calibration point pairs.
        
        Args:
            cam_points: List of (cx, cy) camera pixel coordinates
            galvo_points: List of (gx, gy) galvo DAC coordinates
            
        Returns:
            dict: Computed affine transform parameters
        """
        self._affine_transform = AffineTransform.compute_from_point_pairs(
            cam_points, galvo_points
        )
        return self._affine_transform.to_dict()

    def transform_camera_to_galvo(self, cam_x: float, cam_y: float) -> tuple:
        """Transform camera pixel coordinates to galvo DAC coordinates."""
        return self._affine_transform.transform_point(cam_x, cam_y)

    def save_affine_config(self, config_path: str = None) -> str:
        """
        Save affine transform to a JSON configuration file.
        
        Args:
            config_path: Path to save config. If None, uses default.
            
        Returns:
            str: Path where config was saved
        """
        if config_path is None:
            config_path = self._calibration_config_path
        if config_path is None:
            # Default path next to the setup JSON
            config_path = os.path.join(
                os.path.expanduser('~'), '.imswitch',
                f'galvo_affine_{self._name}.json'
            )
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        data = {
            'scanner_name': self._name,
            'affine_transform': self._affine_transform.to_dict()
        }
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._calibration_config_path = config_path
        return config_path

    def load_affine_config(self, config_path: str = None) -> bool:
        """
        Load affine transform from a JSON configuration file.
        
        Args:
            config_path: Path to load config from. If None, uses default.
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if config_path is None:
            config_path = self._calibration_config_path
        if config_path is None:
            config_path = os.path.join(
                os.path.expanduser('~'), '.imswitch',
                f'galvo_affine_{self._name}.json'
            )
        
        if not os.path.exists(config_path):
            return False
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            if 'affine_transform' in data:
                self._affine_transform = AffineTransform.from_dict(data['affine_transform'])
                self._calibration_config_path = config_path
                return True
        except Exception:
            pass
        return False

    # ========================
    # Arbitrary Points Methods
    # ========================

    @abstractmethod
    def set_arbitrary_points(self, points: List[Dict], laser_trigger: str = "AUTO",
                              timeout: int = 1) -> Dict[str, Any]:
        """
        Send arbitrary points to the scanner hardware.
        
        Args:
            points: List of point dicts with 'x', 'y', 'dwell_us', optional 'laser_intensity'
            laser_trigger: Trigger mode
            timeout: Request timeout
            
        Returns:
            dict: Response from hardware
        """
        pass

    @abstractmethod
    def stop_arbitrary_scan(self, timeout: int = 1) -> Dict[str, Any]:
        """Stop arbitrary point scanning."""
        pass

    @abstractmethod
    def pause_arbitrary_scan(self, timeout: int = 1) -> Dict[str, Any]:
        """Pause arbitrary point scanning."""
        pass

    @abstractmethod
    def resume_arbitrary_scan(self, timeout: int = 1) -> Dict[str, Any]:
        """Resume arbitrary point scanning from paused position."""
        pass

    def get_arbitrary_points(self) -> List[Dict]:
        """Get the currently stored arbitrary points."""
        return self._arbitrary_points

    def get_arbitrary_scan_state(self) -> Dict[str, Any]:
        """Get the state of arbitrary point scanning."""
        return {
            'running': self._arb_scan_running,
            'paused': self._arb_scan_paused,
            'num_points': len(self._arbitrary_points),
            'points': self._arbitrary_points
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
