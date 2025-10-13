"""
Per-objective calibration data storage and management.

This module handles saving and loading stage-to-camera calibration data
with support for multiple objectives. Each objective can have its own
affine transformation matrix and associated metadata.

Copyright 2024, released under GNU GPL v3
"""
import json
import os
import numpy as np
import logging
from typing import Dict, Optional, Any
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays and types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


class CalibrationStorage:
    """
    Manages per-objective stage-to-camera calibration data.
    
    The calibration file format is:
    {
        "format_version": "2.0",
        "objectives": {
            "objective_1": {
                "affine_matrix": [[...], [...]], 
                "metrics": {...},
                "timestamp": "...",
                "objective_info": {...}
            },
            ...
        },
        "legacy_data": {...}  # For backward compatibility
    }
    """
    
    FORMAT_VERSION = "2.0"
    
    def __init__(self, filepath: str = "camera_stage_calibration.json", logger: Optional[logging.Logger] = None):
        """
        Initialize calibration storage.
        
        Args:
            filepath: Path to calibration file
            logger: Optional logger instance
        """
        self.filepath = filepath
        self.logger = logger or logging.getLogger(__name__)
        self._data = None
        self._load()
    
    def _load(self):
        """Load calibration data from file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self._data = json.load(f)
                
                # Check format version
                version = self._data.get("format_version", "1.0")
                if version != self.FORMAT_VERSION:
                    self.logger.info(f"Migrating calibration file from v{version} to v{self.FORMAT_VERSION}")
                    self._migrate_format(version)
                else:
                    self.logger.info(f"Loaded calibration data from {self.filepath}")
            except Exception as e:
                self.logger.error(f"Failed to load calibration file: {e}")
                self._data = self._create_empty_data()
        else:
            self.logger.info(f"No existing calibration file at {self.filepath}, creating new")
            self._data = self._create_empty_data()
    
    def _create_empty_data(self) -> Dict:
        """Create an empty calibration data structure."""
        return {
            "format_version": self.FORMAT_VERSION,
            "objectives": {},
            "legacy_data": {}
        }
    
    def _migrate_format(self, old_version: str):
        """
        Migrate calibration data from older format.
        
        Args:
            old_version: Old format version string
        """
        if old_version == "1.0" or "camera_stage_mapping_calibration" in self._data:
            # Old format had a single calibration, move it to legacy_data
            legacy = {}
            for key in list(self._data.keys()):
                if key not in ["format_version", "objectives"]:
                    legacy[key] = self._data.pop(key)
            
            new_data = self._create_empty_data()
            new_data["legacy_data"] = legacy
            
            # Try to extract affine matrix from legacy data
            if "camera_stage_mapping_calibration" in legacy:
                old_calib = legacy["camera_stage_mapping_calibration"]
                if "image_to_stage_displacement" in old_calib:
                    # Convert 2x2 matrix to 2x3 by adding zero translation
                    A = np.array(old_calib["image_to_stage_displacement"])
                    if A.shape == (2, 2):
                        affine_matrix = np.column_stack([A, [0, 0]])
                        new_data["objectives"]["default"] = {
                            "affine_matrix": affine_matrix.tolist(),
                            "metrics": {"source": "migrated_from_v1.0"},
                            "timestamp": datetime.now().isoformat(),
                            "objective_info": {"name": "default"}
                        }
                        self.logger.info("Migrated legacy calibration to 'default' objective")
            
            self._data = new_data
        else:
            # Unknown version, keep as-is but add new structure
            if "objectives" not in self._data:
                self._data["objectives"] = {}
            self._data["format_version"] = self.FORMAT_VERSION
    
    def save_calibration(
        self,
        objective_id: str,
        affine_matrix: np.ndarray,
        metrics: Dict[str, Any],
        objective_info: Optional[Dict[str, Any]] = None
    ):
        """
        Save calibration data for a specific objective.
        
        Args:
            objective_id: Unique identifier for the objective (e.g., "10x", "objective_1")
            affine_matrix: 2x3 affine transformation matrix
            metrics: Dictionary of calibration quality metrics
            objective_info: Optional additional information about the objective
        """
        if affine_matrix.shape != (2, 3):
            raise ValueError(f"Affine matrix must be 2x3, got {affine_matrix.shape}")
        
        calibration_entry = {
            "affine_matrix": affine_matrix.tolist(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "objective_info": objective_info or {}
        }
        
        self._data["objectives"][objective_id] = calibration_entry
        self._save()
        
        self.logger.info(f"Saved calibration for objective '{objective_id}'")
    
    def load_calibration(self, objective_id: str) -> Optional[Dict[str, Any]]:
        """
        Load calibration data for a specific objective.
        
        Args:
            objective_id: Unique identifier for the objective
        
        Returns:
            Dictionary with calibration data or None if not found
        """
        if objective_id in self._data["objectives"]:
            calib = self._data["objectives"][objective_id].copy()
            # Convert affine_matrix back to numpy array
            calib["affine_matrix"] = np.array(calib["affine_matrix"])
            return calib
        else:
            self.logger.warning(f"No calibration found for objective '{objective_id}'")
            return None
    
    def list_objectives(self) -> list:
        """
        Get list of all objectives with calibration data.
        
        Returns:
            List of objective identifiers
        """
        return list(self._data["objectives"].keys())
    
    def delete_calibration(self, objective_id: str) -> bool:
        """
        Delete calibration data for a specific objective.
        
        Args:
            objective_id: Unique identifier for the objective
        
        Returns:
            True if deleted, False if not found
        """
        if objective_id in self._data["objectives"]:
            del self._data["objectives"][objective_id]
            self._save()
            self.logger.info(f"Deleted calibration for objective '{objective_id}'")
            return True
        else:
            return False
    
    def _save(self):
        """Save current data to file."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(self.filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            with open(self.filepath, 'w') as f:
                json.dump(self._data, f, indent=4, sort_keys=True, cls=NumpyEncoder)
            
            self.logger.debug(f"Saved calibration data to {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save calibration file: {e}")
            raise
    
    def get_legacy_data(self) -> Dict:
        """
        Get legacy calibration data for backward compatibility.
        
        Returns:
            Dictionary with legacy calibration data
        """
        return self._data.get("legacy_data", {})
    
    def export_to_legacy_format(self, objective_id: str = None) -> Dict:
        """
        Export calibration to legacy format for backward compatibility.
        
        If objective_id is not specified, uses the first available objective.
        
        Args:
            objective_id: Optional objective identifier
        
        Returns:
            Dictionary in legacy format
        """
        if objective_id is None:
            objectives = self.list_objectives()
            if not objectives:
                return {}
            objective_id = objectives[0]
        
        calib = self.load_calibration(objective_id)
        if calib is None:
            return {}
        
        # Extract 2x2 matrix (ignore translation component)
        affine_matrix = calib["affine_matrix"]
        A = affine_matrix[:, :2]
        
        # Create legacy format
        legacy = {
            "camera_stage_mapping_calibration": {
                "image_to_stage_displacement": A.tolist(),
                "backlash_vector": [0, 0, 0],  # Not used in new system
                "backlash": 0  # Not used in new system
            }
        }
        
        return legacy
    
    def get_affine_matrix(self, objective_id: str) -> Optional[np.ndarray]:
        """
        Get just the affine matrix for an objective.
        
        Args:
            objective_id: Objective identifier
        
        Returns:
            2x3 numpy array or None if not found
        """
        calib = self.load_calibration(objective_id)
        if calib:
            return calib["affine_matrix"]
        return None
    
    def get_metrics(self, objective_id: str) -> Optional[Dict]:
        """
        Get calibration metrics for an objective.
        
        Args:
            objective_id: Objective identifier
        
        Returns:
            Dictionary of metrics or None if not found
        """
        calib = self.load_calibration(objective_id)
        if calib:
            return calib["metrics"]
        return None
