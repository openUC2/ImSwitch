"""
Base class for experiment execution modes.

This module provides common functionality shared between performance mode
and normal mode experiment execution.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod

from imswitch.imcommon.model import dirtools
from ..experiment_controller.ome_writer import OMEWriter, OMEWriterConfig


class OMEFileStorePaths:
    """Helper class for managing OME file storage paths."""
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tiff_dir = os.path.join(base_dir, "tiles")
        self.zarr_dir = os.path.join(base_dir + ".ome.zarr")
        os.makedirs(self.tiff_dir) if not os.path.exists(self.tiff_dir) else None


class ExperimentModeBase(ABC):
    """
    Base class for experiment execution modes.
    
    Provides common functionality for both performance mode and normal mode
    experiment execution, including parameter processing, scan range computation,
    and OME writer configuration.
    """
    
    def __init__(self, experiment_controller):
        """Initialize the base mode with reference to the main controller."""
        self.controller = experiment_controller
        self._logger = experiment_controller._logger
        
    def compute_scan_ranges(self, snake_tiles: List[List[Dict]]) -> Tuple[float, float, float, float, float, float]:
        """
        Compute scan ranges from snake tiles.
        
        Args:
            snake_tiles: List of tiles containing point dictionaries
            
        Returns:
            Tuple of (minX, maxX, minY, maxY, diffX, diffY)
        """
        # Flatten all point dictionaries from all tiles to compute scan range
        all_points = [pt for tile in snake_tiles for pt in tile]
        minX = min(pt["x"] for pt in all_points)
        maxX = max(pt["x"] for pt in all_points)
        minY = min(pt["y"] for pt in all_points)
        maxY = max(pt["y"] for pt in all_points)
        
        # compute step between two adjacent points in X/Y
        uniqueX = np.unique([pt["x"] for pt in all_points])
        uniqueY = np.unique([pt["y"] for pt in all_points])
        
        if len(uniqueX) == 1:
            diffX = 0
        else:
            diffX = np.diff(uniqueX).min()
            
        if len(uniqueY) == 1:
            diffY = 0
        else:
            diffY = np.diff(uniqueY).min()
            
        return minX, maxX, minY, maxY, diffX, diffY
    
    def create_ome_file_paths(self, base_path: str) -> 'OMEFileStorePaths':
        """Create OME file storage paths."""
        return OMEFileStorePaths(base_path)
    
    def create_writer_config(self, 
                           write_tiff: bool = False,
                           write_zarr: bool = True, 
                           write_stitched_tiff: bool = True,
                           write_tiff_single: bool = False,
                           min_period: float = 0.2,
                           n_time_points: int = 1,
                           n_z_planes: int = 1,
                           n_channels: int = 1) -> OMEWriterConfig:
        """
        Create OME writer configuration.
        
        Args:
            write_tiff: Whether to write individual TIFF files
            write_zarr: Whether to write OME-Zarr format
            write_stitched_tiff: Whether to write stitched TIFF
            write_tiff_single: Whether to append tiles to a single TIFF file
            min_period: Minimum period between writes
            n_time_points: Number of time points
            n_z_planes: Number of Z planes
            n_channels: Number of channels
            
        Returns:
            OMEWriterConfig instance
        """
        pixel_size = self.controller.detectorPixelSize[-1] if hasattr(self.controller, 'detectorPixelSize') else 1.0
        
        return OMEWriterConfig(
            write_tiff=write_tiff,
            write_zarr=write_zarr,
            write_stitched_tiff=write_stitched_tiff,
            write_tiff_single=write_tiff_single,
            min_period=min_period,
            pixel_size=pixel_size,
            n_time_points=n_time_points,
            n_z_planes=n_z_planes,
            n_channels=n_channels
        )
    
    def prepare_illumination_parameters(self, illumination_intensities: List[float]) -> Dict[str, Optional[float]]:
        """
        Prepare illumination parameters in the format expected by hardware.
        
        Args:
            illumination_intensities: List of illumination intensities
            
        Returns:
            Dictionary with illumination0-3 and led parameters
        """
        illum_dict = {
            "illumination0": illumination_intensities[0] if len(illumination_intensities) > 0 else None,
            "illumination1": illumination_intensities[1] if len(illumination_intensities) > 1 else None,
            "illumination2": illumination_intensities[2] if len(illumination_intensities) > 2 else None,
            "illumination3": illumination_intensities[3] if len(illumination_intensities) > 3 else None,
            "led": 0  # Default LED value
        }
        return illum_dict
    
    def create_experiment_directory(self, exp_name: str) -> Tuple[str, str, str]:
        """
        Create experiment directory and generate file paths.
        
        Args:
            exp_name: Experiment name
            
        Returns:
            Tuple of (timeStamp, dirPath, mFileName)
        """
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        drivePath = dirtools.UserFileDirs.Data
        dirPath = os.path.join(drivePath, 'ExperimentController', timeStamp)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        mFileName = f"{timeStamp}_{exp_name}"
        
        return timeStamp, dirPath, mFileName
    
    def calculate_grid_parameters(self, tiles: List[Dict]) -> Tuple[Tuple[int, int], Tuple[float, float, float, float]]:
        """
        Calculate grid parameters from tile list.
        
        Args:
            tiles: List of point dictionaries
            
        Returns:
            Tuple of (grid_shape, grid_geometry)
        """
        all_points = []
        for point in tiles:
            if point is not None:
                all_points.append([point["x"], point["y"]])
        
        if all_points:
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            x_start, x_end = min(x_coords), max(x_coords)
            y_start, y_end = min(y_coords), max(y_coords)
            unique_x = sorted(set(x_coords))
            unique_y = sorted(set(y_coords))
            x_step = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 100.0
            y_step = unique_y[1] - unique_y[0] if len(unique_y) > 1 else 100.0
            nx, ny = len(unique_x), len(unique_y)
            grid_shape = (nx, ny)
            grid_geometry = (x_start, y_start, x_step, y_step)
        else:
            grid_shape = (1, 1)
            grid_geometry = (0, 0, 100, 100)
            
        return grid_shape, grid_geometry
    
    @abstractmethod
    def execute_experiment(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the experiment. Must be implemented by subclasses.
        
        Args:
            **kwargs: Experiment parameters
            
        Returns:
            Dictionary with execution results
        """
        pass