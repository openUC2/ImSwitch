"""
DataStore Adapter for RecordingManager.

Provides a Storer-compatible interface that wraps AcquisitionDataStore,
allowing the existing RecordingManager to use the new unified I/O service.
"""

import os
import time
import uuid
import logging
from typing import Dict, Optional, Any
import numpy as np

from ..io import AcquisitionDataStore, SessionInfo
from ..io.writers import WriterDetectorContext

logger = logging.getLogger(__name__)


class DataStoreAdapter:
    """
    Adapter that wraps AcquisitionDataStore for RecordingManager compatibility.
    
    This allows RecordingManager to use the new unified I/O service while
    maintaining backward compatibility with the existing Storer interface.
    
    Usage in RecordingManager:
        # Instead of:
        # store = TiffStorer(filepath, detectorsManager)
        # store.snap(images, attrs)
        
        # Use:
        adapter = DataStoreAdapter(filepath, detectorsManager, metadata_hub=hub)
        adapter.snap(images, attrs)
    """
    
    def __init__(self, 
                 filepath: str,
                 detectors_manager,
                 metadata_hub = None,
                 write_zarr: bool = False,
                 write_tiff: bool = True,
                 project: str = None,
                 sample: str = None):
        """
        Initialize the adapter.
        
        Args:
            filepath: Base file path (similar to Storer)
            detectors_manager: DetectorsManager for detector info
            metadata_hub: Optional MetadataHub for metadata
            write_zarr: Write OME-Zarr format
            write_tiff: Write OME-TIFF format
            project: Optional project name
            sample: Optional sample name
        """
        self.filepath = filepath
        self._detectors_manager = detectors_manager
        self._metadata_hub = metadata_hub
        self.write_zarr = write_zarr
        self.write_tiff = write_tiff
        self.project = project
        self.sample = sample
        
        self._data_store: Optional[AcquisitionDataStore] = None
        self._session_id = str(uuid.uuid4())[:8]
    
    def _get_detector_contexts(self) -> Dict[str, WriterDetectorContext]:
        """Extract detector contexts from DetectorsManager."""
        contexts = {}
        
        for det_name in self._detectors_manager.detectorNames:
            detector = self._detectors_manager[det_name]
            
            # Get shape from detector
            shape = detector.shape
            if hasattr(shape, '__iter__'):
                shape_px = tuple(shape)
            else:
                shape_px = (shape, shape)
            
            # Get pixel size (default to 1.0 if not available)
            pixel_size = getattr(detector, 'pixelSizeUm', 1.0)
            if pixel_size is None:
                pixel_size = 1.0
            
            # Get dtype
            dtype = 'uint16'
            if hasattr(detector, 'dtype'):
                dtype = str(detector.dtype)
            
            contexts[det_name] = WriterDetectorContext(
                name=det_name,
                shape_px=shape_px,
                pixel_size_um=float(pixel_size),
                dtype=dtype,
                channel_name=det_name,
            )
        
        return contexts
    
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, Any] = None):
        """
        Snap images using the DataStore.
        
        Args:
            images: Dict mapping detector names to image arrays
            attrs: Optional metadata attributes
        """
        # Extract base directory and filename
        base_dir = os.path.dirname(self.filepath) or '.'
        
        # Create session info
        session_info = SessionInfo(
            session_id=f"snap_{self._session_id}_{int(time.time())}",
            base_path=base_dir,
            project=self.project,
            sample=self.sample,
            n_time_points=1,
            n_z_planes=1,
            n_channels=len(images),
        )
        
        # Get hub snapshot if available
        hub_snapshot = None
        if self._metadata_hub:
            import json
            try:
                hub_snapshot = json.loads(self._metadata_hub.to_json())
            except:
                pass
        
        # Create data store
        data_store = AcquisitionDataStore(
            session_info=session_info,
            metadata_hub=self._metadata_hub,
            hub_snapshot=hub_snapshot,
            write_zarr=self.write_zarr,
            write_tiff=self.write_tiff,
            background_writes=False,  # Sync for snap
        )
        
        # Open with detector contexts
        detector_contexts = self._get_detector_contexts()
        # Filter to only include detectors in images
        filtered_contexts = {k: v for k, v in detector_contexts.items() if k in images}
        
        try:
            data_store.open(detector_contexts=filtered_contexts)
            
            # Write each image
            for det_name, image in images.items():
                # Create frame event from attrs if available
                frame_event = None
                if attrs:
                    from ..io.writers import WriterFrameEvent
                    frame_event = self._attrs_to_frame_event(attrs, det_name)
                
                data_store.write_frame(det_name, image, frame_event=frame_event)
            
        finally:
            data_store.close()
        
        logger.info(f"Snap saved via DataStore: {session_info.base_path}")
    
    def _attrs_to_frame_event(self, attrs: Dict[str, Any], detector_name: str):
        """Convert legacy attrs dict to FrameEvent."""
        from ..io.writers import WriterFrameEvent
        
        # Extract position from attrs
        stage_x = None
        stage_y = None
        stage_z = None
        
        for key, value in attrs.items():
            key_lower = str(key).lower()
            if 'position' in key_lower and 'x' in key_lower:
                stage_x = float(value) if value else None
            elif 'position' in key_lower and 'y' in key_lower:
                stage_y = float(value) if value else None
            elif 'position' in key_lower and 'z' in key_lower:
                stage_z = float(value) if value else None
        
        # Also try :X:Position style
        for key, value in attrs.items():
            key_str = str(key)
            if ':X:Position' in key_str:
                stage_x = float(value) if value else None
            elif ':Y:Position' in key_str:
                stage_y = float(value) if value else None
            elif ':Z:Position' in key_str:
                stage_z = float(value) if value else None
        
        # Get exposure
        exposure_ms = None
        for key, value in attrs.items():
            if 'exposure' in str(key).lower():
                exposure_ms = float(value) if value else None
                break
        
        return WriterFrameEvent(
            frame_number=0,
            detector_name=detector_name,
            stage_x_um=stage_x,
            stage_y_um=stage_y,
            stage_z_um=stage_z,
            exposure_ms=exposure_ms,
            metadata={'legacy_attrs': attrs}
        )


class StreamingDataStoreAdapter:
    """
    Adapter for streaming recordings (multiple frames).
    
    Maintains an open DataStore session for continuous writing.
    """
    
    def __init__(self,
                 base_path: str,
                 detectors_manager,
                 metadata_hub = None,
                 write_zarr: bool = True,
                 write_tiff: bool = False,
                 n_time_points: int = 100,
                 n_z_planes: int = 1,
                 project: str = None,
                 sample: str = None):
        """
        Initialize streaming adapter.
        
        Args:
            base_path: Base directory for session data
            detectors_manager: DetectorsManager for detector info
            metadata_hub: Optional MetadataHub
            write_zarr: Write OME-Zarr format
            write_tiff: Write OME-TIFF format
            n_time_points: Expected number of time points
            n_z_planes: Number of z planes
            project: Optional project name
            sample: Optional sample name
        """
        self.base_path = base_path
        self._detectors_manager = detectors_manager
        self._metadata_hub = metadata_hub
        self.write_zarr = write_zarr
        self.write_tiff = write_tiff
        self.n_time_points = n_time_points
        self.n_z_planes = n_z_planes
        self.project = project
        self.sample = sample
        
        self._data_store: Optional[AcquisitionDataStore] = None
        self._is_open = False
        self._frame_counters: Dict[str, int] = {}
    
    def open(self, detector_names: list = None):
        """
        Open the streaming session.
        
        Args:
            detector_names: List of detector names to include
        """
        if self._is_open:
            return
        
        # Get detector contexts
        detector_contexts = {}
        for det_name in self._detectors_manager.detectorNames:
            if detector_names and det_name not in detector_names:
                continue
            
            detector = self._detectors_manager[det_name]
            shape = detector.shape
            if hasattr(shape, '__iter__'):
                shape_px = tuple(shape)
            else:
                shape_px = (shape, shape)
            
            pixel_size = getattr(detector, 'pixelSizeUm', 1.0) or 1.0
            dtype = str(getattr(detector, 'dtype', 'uint16'))
            
            detector_contexts[det_name] = WriterDetectorContext(
                name=det_name,
                shape_px=shape_px,
                pixel_size_um=float(pixel_size),
                dtype=dtype,
                channel_name=det_name,
            )
            self._frame_counters[det_name] = 0
        
        # Create session info
        session_info = SessionInfo(
            session_id=str(uuid.uuid4()),
            base_path=self.base_path,
            project=self.project,
            sample=self.sample,
            n_time_points=self.n_time_points,
            n_z_planes=self.n_z_planes,
            n_channels=len(detector_contexts),
        )
        
        # Get hub snapshot
        hub_snapshot = None
        if self._metadata_hub:
            import json
            try:
                hub_snapshot = json.loads(self._metadata_hub.to_json())
            except:
                pass
        
        # Create and open data store
        self._data_store = AcquisitionDataStore(
            session_info=session_info,
            metadata_hub=self._metadata_hub,
            hub_snapshot=hub_snapshot,
            write_zarr=self.write_zarr,
            write_tiff=self.write_tiff,
            background_writes=True,  # Async for streaming
        )
        
        self._data_store.open(detector_contexts=detector_contexts)
        self._is_open = True
        
        logger.info(f"Streaming session opened: {session_info.session_id}")
    
    def write_frame(self, detector_name: str, frame: np.ndarray, 
                   t_index: int = None, z_index: int = 0):
        """
        Write a single frame.
        
        Args:
            detector_name: Detector name
            frame: Image data
            t_index: Time index (auto-incremented if None)
            z_index: Z index
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        if t_index is None:
            t_index = self._frame_counters[detector_name]
            self._frame_counters[detector_name] += 1
        
        self._data_store.write_frame(
            detector_name=detector_name,
            frame=frame,
            t_index=t_index,
            z_index=z_index,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get write statistics."""
        if self._data_store:
            return self._data_store.get_statistics()
        return {}
    
    def close(self):
        """Close the streaming session."""
        if not self._is_open:
            return
        
        if self._data_store:
            self._data_store.close()
            self._data_store = None
        
        self._is_open = False
        logger.info("Streaming session closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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
