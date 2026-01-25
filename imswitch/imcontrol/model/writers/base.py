"""
Base writer interface and context dataclasses.

Provides the common interface that all writers must implement,
along with context objects for session, detector, and frame metadata.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import time


class WriterCapabilities(Enum):
    """Writer capability flags."""
    SINGLE_FILE = "single_file"  # All data in one file
    MULTI_FILE = "multi_file"  # Multiple files per session
    STREAMING = "streaming"  # Can write while acquiring
    METADATA_RICH = "metadata_rich"  # Supports full OME metadata
    MULTI_DETECTOR = "multi_detector"  # Supports multiple detectors
    MULTI_CHANNEL = "multi_channel"  # Supports multiple channels
    TIME_SERIES = "time_series"  # Supports time-lapse
    Z_STACK = "z_stack"  # Supports z-stacks
    TILED = "tiled"  # Supports tiled/mosaic images


@dataclass
class SessionContext:
    """
    Session-level metadata for a recording/acquisition session.
    
    Contains metadata that applies to the entire session, such as
    instrument configuration, user info, and global acquisition parameters.
    """
    session_id: str  # Unique session identifier (UUID)
    start_time: float = field(default_factory=time.time)
    base_path: str = ""
    
    # User-provided metadata
    project: Optional[str] = None
    experiment: Optional[str] = None
    sample: Optional[str] = None
    user: Optional[str] = None
    description: Optional[str] = None
    
    # Acquisition parameters
    n_time_points: int = 1
    n_z_planes: int = 1
    n_channels: int = 1
    time_interval_s: Optional[float] = None
    z_step_um: Optional[float] = None
    
    # Instrument configuration (from MetadataHub)
    objectives: Dict[str, Any] = field(default_factory=dict)
    light_sources: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'base_path': self.base_path,
            'project': self.project,
            'experiment': self.experiment,
            'sample': self.sample,
            'user': self.user,
            'description': self.description,
            'n_time_points': self.n_time_points,
            'n_z_planes': self.n_z_planes,
            'n_channels': self.n_channels,
            'time_interval_s': self.time_interval_s,
            'z_step_um': self.z_step_um,
            'objectives': self.objectives,
            'light_sources': self.light_sources,
            'metadata': self.metadata,
        }


@dataclass
class DetectorContext:
    """
    Detector-specific metadata context.
    
    This is a simplified version that can be created from the MetadataHub
    DetectorContext or used standalone by writers.
    """
    name: str
    shape_px: Tuple[int, int]  # (width, height)
    pixel_size_um: float
    dtype: str = 'uint16'
    
    # Optional fields
    fov_um: Optional[Tuple[float, float]] = None
    binning: int = 1
    roi: Optional[Tuple[int, int, int, int]] = None
    channel_name: Optional[str] = None
    channel_color: Optional[str] = None
    wavelength_nm: Optional[float] = None
    exposure_ms: Optional[float] = None
    gain: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'name': self.name,
            'shape_px': self.shape_px,
            'pixel_size_um': self.pixel_size_um,
            'dtype': self.dtype,
            'fov_um': self.fov_um,
            'binning': self.binning,
            'roi': self.roi,
            'channel_name': self.channel_name,
            'channel_color': self.channel_color,
            'wavelength_nm': self.wavelength_nm,
            'exposure_ms': self.exposure_ms,
            'gain': self.gain,
        }


@dataclass
class FrameEvent:
    """
    Per-frame metadata event.
    
    Captures metadata at the time of frame acquisition.
    Can be created from MetadataHub FrameEvent or standalone.
    """
    frame_number: int
    timestamp: float = field(default_factory=time.time)
    detector_name: Optional[str] = None
    
    # Positional metadata
    stage_x_um: Optional[float] = None
    stage_y_um: Optional[float] = None
    stage_z_um: Optional[float] = None
    
    # Acquisition settings
    exposure_ms: Optional[float] = None
    laser_power_mw: Optional[float] = None
    
    # Indices
    t_index: int = 0
    z_index: int = 0
    c_index: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'detector_name': self.detector_name,
            'stage_x_um': self.stage_x_um,
            'stage_y_um': self.stage_y_um,
            'stage_z_um': self.stage_z_um,
            'exposure_ms': self.exposure_ms,
            'laser_power_mw': self.laser_power_mw,
            't_index': self.t_index,
            'z_index': self.z_index,
            'c_index': self.c_index,
            'metadata': self.metadata,
        }


class WriterBase(ABC):
    """
    Base interface for all file format writers.
    
    Writers implement this interface to handle writing acquisition data
    in various formats. They can be used from RecordingManager,
    ExperimentController, or any other acquisition pipeline.
    
    Lifecycle:
        1. __init__(session_ctx) - Initialize with session metadata
        2. open(detectors) - Open files/datasets for detectors
        3. write(detector, frames, events) - Write frames with metadata (called multiple times)
        4. finalize() - Flush buffers, write final metadata
        5. close() - Close files and clean up
    """
    
    def __init__(self, session_ctx: SessionContext):
        """
        Initialize writer with session context.
        
        Args:
            session_ctx: Session-level metadata
        """
        self.session_ctx = session_ctx
        self._is_open = False
        self._is_finalized = False
    
    @abstractmethod
    def open(self, detectors: Dict[str, DetectorContext]) -> None:
        """
        Open files/datasets for writing.
        
        Args:
            detectors: Dictionary mapping detector name to DetectorContext
        """
        pass
    
    @abstractmethod
    def write(self, 
              detector_name: str,
              frames: np.ndarray,
              events: Optional[List[FrameEvent]] = None) -> None:
        """
        Write frames for a detector.
        
        Args:
            detector_name: Name of the detector
            frames: Image data as numpy array (can be 2D, 3D, 4D, etc.)
            events: Optional list of FrameEvent objects (one per frame)
        """
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """
        Finalize writing (flush buffers, write final metadata).
        
        Called after all frames have been written but before close().
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close files and clean up resources.
        
        Should be idempotent (safe to call multiple times).
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> List[WriterCapabilities]:
        """
        Return list of capabilities supported by this writer.
        
        Returns:
            List of WriterCapabilities flags
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_file_extension(cls) -> str:
        """
        Return the primary file extension for this writer.
        
        Returns:
            Extension string (e.g., '.ome.tiff', '.zarr', '.png')
        """
        pass
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        if not self._is_finalized:
            self.finalize()
        self.close()
        return False


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
