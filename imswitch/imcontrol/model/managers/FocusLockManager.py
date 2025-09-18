import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from imswitch.imcommon.model import initLogger, dirtools
from .FocusLockConfig import FocusLockConfig


class FocusLockManager:
    """
    Manager for focus map functionality and global focus corrections.
    Handles persistence, interpolation, and parameter management for focus maps.
    """
    
    def __init__(self, profile_name: str = "default"):
        """
        Initialize the FocusLockManager.
        
        Args:
            profile_name: The profile name for config storage
        """
        self._logger = initLogger(self)
        self._profile_name = profile_name
        
        # Initialize configuration
        self._config = FocusLockConfig(profile_name)
        
        # Initialize storage paths
        self._config_dir = Path(dirtools.UserFileDirs.Root) / "focus"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._map_file = self._config_dir / f"focusmap_{profile_name}.json"
        self._params_file = self._config_dir / f"focuslock_params_{profile_name}.json"
        
        # Initialize data structures
        self._focus_map = None
        self._focus_params = self._load_focus_params()
        
        # Load existing data
        self.load_map()
        
    def _load_focus_params(self) -> Dict[str, Any]:
        """Load focus lock parameters from config system."""
        # Try to load from legacy params file first
        if self._params_file.exists():
            try:
                with open(self._params_file, 'r') as f:
                    legacy_params = json.load(f)
                self._logger.info("Loaded legacy focus lock parameters")
                return legacy_params
            except Exception as e:
                self._logger.error(f"Failed to load legacy params: {e}")
        
        # Use configuration system
        return self._config.get_focuslock_params()
    
    def load_map(self) -> None:
        """Load focus map from disk."""
        if not self._map_file.exists():
            self._logger.info(f"No focus map found at {self._map_file}")
            return
            
        try:
            with open(self._map_file, 'r') as f:
                self._focus_map = json.load(f)
            self._logger.info(f"Loaded focus map with {len(self._focus_map.get('points', []))} points")
        except Exception as e:
            self._logger.error(f"Failed to load focus map: {e}")
            self._focus_map = None
    
    def save_map(self) -> None:
        """Save focus map to disk."""
        if self._focus_map is None:
            return
            
        try:
            with open(self._map_file, 'w') as f:
                json.dump(self._focus_map, f, indent=2)
            self._logger.info(f"Saved focus map to {self._map_file}")
        except Exception as e:
            self._logger.error(f"Failed to save focus map: {e}")
    
    def load_params(self) -> None:
        """Load focus lock parameters - now uses config system."""
        # Parameters are loaded in __init__ from config system
        pass
    
    def save_params(self) -> None:
        """Save focus lock parameters via config system."""
        try:
            # Update configuration
            self._config.update_section("focuslock", {
                "enabled": self._focus_params.get("lock_enabled", False),
                "settle_band_um": self._focus_params.get("settle_band_um", 1.0),
                "settle_timeout_ms": self._focus_params.get("settle_timeout_ms", 1500),
                "settle_window_ms": self._focus_params.get("settle_window_ms", 200),
                "watchdog": self._focus_params.get("watchdog", {})
            })
            self._config.save_config()
            
            # Also save to legacy file for backward compatibility
            with open(self._params_file, 'w') as f:
                json.dump(self._focus_params, f, indent=2)
            
            self._logger.info("Saved focus lock parameters")
        except Exception as e:
            self._logger.error(f"Failed to save focus lock parameters: {e}")
    
    def clear_map(self) -> None:
        """Clear the current focus map."""
        self._focus_map = None
        if self._map_file.exists():
            self._map_file.unlink()
        self._logger.info("Cleared focus map")
    
    def add_point(self, x_um: float, y_um: float, z_um: float) -> None:
        """
        Add a calibration point to the focus map.
        
        Args:
            x_um: X coordinate in micrometers
            y_um: Y coordinate in micrometers  
            z_um: Z coordinate in micrometers
        """
        if self._focus_map is None:
            self._focus_map = {
                "profile": self._profile_name,
                "method": "plane",
                "points": [],
                "fit": {},
                "channel_z_offsets": {"DAPI": 0.0, "FITC": 0.8, "TRITC": 1.2},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        
        # Add the point
        point = {"x_um": float(x_um), "y_um": float(y_um), "z_um": float(z_um)}
        self._focus_map["points"].append(point)
        self._focus_map["updated_at"] = datetime.now().isoformat()
        
        self._logger.info(f"Added focus point: {point}")
    
    def fit(self, method: str = "plane") -> Dict[str, Any]:
        """
        Fit a surface to the calibration points.
        
        Args:
            method: Fitting method ("plane" is currently supported)
            
        Returns:
            Dictionary with fit coefficients and metrics
        """
        if self._focus_map is None or len(self._focus_map.get("points", [])) < 3:
            raise ValueError("Need at least 3 points for plane fitting")
        
        if method != "plane":
            raise ValueError(f"Unsupported fitting method: {method}")
        
        points = self._focus_map["points"]
        coefficients = self._fit_plane(points)
        
        # Store fit results
        self._focus_map["fit"] = {
            "plane": {
                "a": coefficients[0],
                "b": coefficients[1], 
                "c": coefficients[2]
            }
        }
        self._focus_map["method"] = method
        self._focus_map["updated_at"] = datetime.now().isoformat()
        
        self._logger.info(f"Fitted {method} with coefficients: {coefficients}")
        return self._focus_map["fit"]
    
    def _fit_plane(self, points: List[Dict[str, float]]) -> Tuple[float, float, float]:
        """
        Fit a plane to the given points using least squares.
        Plane equation: z = a*x + b*y + c
        
        Args:
            points: List of point dictionaries with x_um, y_um, z_um keys
            
        Returns:
            Tuple of (a, b, c) coefficients
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points for plane fitting")
        
        # Extract coordinates
        x_coords = np.array([p["x_um"] for p in points])
        y_coords = np.array([p["y_um"] for p in points])
        z_coords = np.array([p["z_um"] for p in points])
        
        # Check for collinearity using cross product
        if len(points) >= 3:
            v1 = np.array([x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]])
            v2 = np.array([x_coords[2] - x_coords[0], y_coords[2] - y_coords[0]])
            cross_product = np.cross(v1, v2)
            if abs(cross_product) < 1e-10:
                raise ValueError("Points are collinear - cannot fit plane")
        
        # Solve using least squares: [x y 1] * [a b c]^T = z
        A = np.column_stack([x_coords, y_coords, np.ones(len(points))])
        coefficients, residuals, rank, s = np.linalg.lstsq(A, z_coords, rcond=None)
        
        return tuple(coefficients)
    
    def get_z_offset(self, x_um: float, y_um: float) -> float:
        """
        Get the Z offset for given XY coordinates.
        
        Args:
            x_um: X coordinate in micrometers
            y_um: Y coordinate in micrometers
            
        Returns:
            Z offset in micrometers
        """
        if self._focus_map is None or "fit" not in self._focus_map:
            return 0.0
        
        if "plane" in self._focus_map["fit"]:
            plane = self._focus_map["fit"]["plane"]
            a, b, c = plane["a"], plane["b"], plane["c"]
            return float(a * x_um + b * y_um + c)
        
        return 0.0
    
    def get_channel_offset(self, channel_name: str) -> float:
        """
        Get the Z offset for a specific channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Channel Z offset in micrometers
        """
        if self._focus_map is None:
            return 0.0
        
        channel_offsets = self._focus_map.get("channel_z_offsets", {})
        return float(channel_offsets.get(channel_name, 0.0))
    
    def set_channel_offset(self, channel_name: str, offset_um: float) -> None:
        """
        Set the Z offset for a specific channel.
        
        Args:
            channel_name: Name of the channel
            offset_um: Z offset in micrometers
        """
        if self._focus_map is None:
            self._focus_map = {
                "profile": self._profile_name,
                "method": "plane",
                "points": [],
                "fit": {},
                "channel_z_offsets": {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        
        if "channel_z_offsets" not in self._focus_map:
            self._focus_map["channel_z_offsets"] = {}
        
        self._focus_map["channel_z_offsets"][channel_name] = float(offset_um)
        self._focus_map["updated_at"] = datetime.now().isoformat()
    
    def get_params(self) -> Dict[str, Any]:
        """Get current focus lock parameters."""
        return self._focus_params.copy()
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set focus lock parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        self._focus_params.update(params)
        self.save_params()
    
    def get_map_data(self) -> Optional[Dict[str, Any]]:
        """Get current focus map data."""
        return self._focus_map.copy() if self._focus_map else None
    
    def is_map_active(self) -> bool:
        """Check if focus map is active and has valid fit data."""
        return (self._focus_map is not None and 
                "fit" in self._focus_map and 
                len(self._focus_map.get("points", [])) >= 3)
    
    def get_map_stats(self) -> Dict[str, Any]:
        """Get statistics about the current focus map."""
        if not self.is_map_active():
            return {"active": False, "point_count": 0}
        
        points = self._focus_map["points"]
        z_values = [p["z_um"] for p in points]
        
        return {
            "active": True,
            "point_count": len(points),
            "method": self._focus_map.get("method", "unknown"),
            "z_range_um": max(z_values) - min(z_values) if z_values else 0,
            "created_at": self._focus_map.get("created_at"),
            "updated_at": self._focus_map.get("updated_at")
        }
    
    def get_config(self) -> 'FocusLockConfig':
        """Get the configuration object for advanced usage."""
        return self._config
    
    def is_focus_map_enabled_in_config(self) -> bool:
        """Check if focus map is enabled in configuration."""
        return self._config.is_focus_map_enabled()
    
    def is_focus_lock_live_enabled_in_config(self) -> bool:
        """Check if live focus lock is enabled in configuration."""
        return self._config.is_focus_lock_live_enabled()
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration section.
        
        Args:
            section: Configuration section to update
            updates: Dictionary of updates to apply
        """
        self._config.update_section(section, updates)
        self._config.save_config()
        
        # Reload parameters if focuslock section was updated
        if section == "focuslock":
            self._focus_params = self._config.get_focuslock_params()
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment-related configuration."""
        return self._config.get_experiment_params()
    
    def get_z_move_order(self) -> str:
        """Get Z move order preference from config."""
        return self._config.get_z_move_order()