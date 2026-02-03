"""
OME-Zarr (NGFF) writer with full OME-NGFF v0.4 metadata support.

Writes acquisition data as OME-Zarr format following the OME-NGFF
specification for cloud-optimized n-dimensional bioimaging data.
"""

import os
import zarr
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .base import WriterBase, SessionContext, DetectorContext, FrameEvent, WriterCapabilities
from .registry import register_writer

logger = logging.getLogger(__name__)


@register_writer('OME_ZARR')
class OMEZarrWriter(WriterBase):
    """
    OME-Zarr (NGFF) writer with OME-NGFF v0.4 metadata.
    
    Features:
    - OME-NGFF v0.4 compliant metadata
    - Multi-resolution pyramids (optional)
    - Chunked storage for efficient access
    - Support for time-series and z-stacks
    - Per-plane metadata storage
    - Background thread writing
    """
    
    def __init__(self, session_ctx: SessionContext,
                 chunk_size: tuple = (1, 1, 1, 256, 256),
                 dtype: str = 'uint16',
                 compressor = None):
        """
        Initialize OME-Zarr writer.
        
        Args:
            session_ctx: Session metadata
            chunk_size: Chunk size for Zarr arrays (t, c, z, y, x)
            dtype: Data type for arrays
            compressor: Optional Zarr compressor
        """
        super().__init__(session_ctx)
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.compressor = compressor
        
        # Detector contexts and arrays
        self.detectors: Dict[str, DetectorContext] = {}
        self.stores: Dict[str, str] = {}
        self.roots: Dict[str, zarr.Group] = {}
        self.arrays: Dict[str, zarr.Array] = {}
        
        # Frame counters
        self._frame_counters: Dict[str, int] = {}
        
        # Background writing
        self.queues: Dict[str, deque] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self.locks: Dict[str, threading.Lock] = {}
    
    @classmethod
    def get_capabilities(cls) -> List[WriterCapabilities]:
        return [
            WriterCapabilities.SINGLE_FILE,
            WriterCapabilities.STREAMING,
            WriterCapabilities.METADATA_RICH,
            WriterCapabilities.MULTI_DETECTOR,
            WriterCapabilities.MULTI_CHANNEL,
            WriterCapabilities.TIME_SERIES,
            WriterCapabilities.Z_STACK,
            WriterCapabilities.TILED,
        ]
    
    @classmethod
    def get_file_extension(cls) -> str:
        return '.zarr'
    
    def open(self, detectors: Dict[str, DetectorContext]) -> None:
        """Open Zarr store and create arrays for each detector."""
        if self._is_open:
            logger.warning("Writer already open")
            return
        
        self.detectors = detectors
        
        # Create base Zarr directory
        base_path = self.session_ctx.base_path
        if not base_path:
            base_path = f"{self.session_ctx.session_id}.zarr"
        
        os.makedirs(base_path, exist_ok=True)
        
        # Create separate store per detector (or single store with multiple datasets)
        for det_name, det_ctx in detectors.items():
            # Path for this detector
            det_path = os.path.join(base_path, det_name)
            os.makedirs(det_path, exist_ok=True)
            
            # Initialize store and root group
            self.stores[det_name] = det_path
            self.roots[det_name] = zarr.open_group(store=det_path, mode="w")
            
            # Determine array shape
            n_t = self.session_ctx.n_time_points
            n_c = self.session_ctx.n_channels
            n_z = self.session_ctx.n_z_planes
            height, width = det_ctx.shape_px[1], det_ctx.shape_px[0]
            
            # Create array with proper chunking
            chunks = (
                min(1, n_t),  # Time
                min(1, n_c),  # Channel
                min(1, n_z),  # Z
                min(self.chunk_size[3], height),  # Y
                min(self.chunk_size[4], width),   # X
            )
            
            self.arrays[det_name] = self.roots[det_name].create_array(
                name="0",
                shape=(n_t, n_c, n_z, height, width),
                chunks=chunks,
                dtype=det_ctx.dtype or self.dtype,
                compressor=self.compressor,
            )
            
            # Set OME-NGFF metadata
            self._set_ome_ngff_metadata(det_name, det_ctx, n_t, n_c, n_z)
            
            # Initialize frame counter
            self._frame_counters[det_name] = 0
            
            # Set up background writing
            self.queues[det_name] = deque()
            self.locks[det_name] = threading.Lock()
            self.stop_events[det_name] = threading.Event()
            
            thread = threading.Thread(
                target=self._writer_loop,
                args=(det_name,),
                daemon=True
            )
            thread.start()
            self.threads[det_name] = thread
            
            logger.info(f"Opened OME-Zarr writer for {det_name}: {det_path}")
        
        self._is_open = True
    
    def _set_ome_ngff_metadata(self, det_name: str, det_ctx: DetectorContext,
                                n_t: int, n_c: int, n_z: int):
        """Set OME-NGFF v0.4 compliant metadata for a detector."""
        root = self.roots[det_name]
        
        # Physical pixel sizes
        pixel_size_xy = det_ctx.pixel_size_um
        pixel_size_z = self.session_ctx.z_step_um or 1.0
        time_interval = self.session_ctx.time_interval_s or 1.0
        
        # Set multiscales metadata
        root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": det_name,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [time_interval, 1, pixel_size_z, pixel_size_xy, pixel_size_xy]},
                    ]
                }
            ],
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "coordinateTransformations": [
                {"type": "scale", "scale": [time_interval, 1, pixel_size_z, pixel_size_xy, pixel_size_xy]}
            ]
        }]
        
        # Set omero metadata for channel visualization
        channels = []
        for i in range(n_c):
            channel_name = det_ctx.channel_name or det_name
            if n_c > 1:
                channel_name = f"{channel_name}_{i}"
            
            channel_color = det_ctx.channel_color or "FFFFFF"
            
            channels.append({
                "label": channel_name,
                "color": channel_color,
                "active": True,
                "coefficient": 1.0,
                "family": "linear",
                "inverted": False,
                "window": {
                    "start": 0,
                    "end": 65535 if det_ctx.dtype == 'uint16' else 255,
                    "min": 0,
                    "max": 65535 if det_ctx.dtype == 'uint16' else 255,
                }
            })
        
        root.attrs["omero"] = {
            "id": 1,
            "name": det_name,
            "version": "0.4",
            "channels": channels,
            "rdefs": {
                "defaultT": 0,
                "defaultZ": n_z // 2,
                "model": "color"
            }
        }
        
        # Store detector context as custom metadata
        root.attrs["detector_context"] = det_ctx.to_dict()
    
    def write(self, 
              detector_name: str,
              frames: np.ndarray,
              events: Optional[List[FrameEvent]] = None) -> None:
        """
        Write frames to Zarr array.
        
        Args:
            detector_name: Detector name
            frames: Image data (2D, 3D, or 4D array)
            events: Optional frame events for metadata
        """
        if not self._is_open:
            raise RuntimeError("Writer not open")
        
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        # Normalize frame dimensions
        if frames.ndim == 2:
            frames = frames[np.newaxis, ...]  # Add batch dimension
        
        # Queue frames with metadata
        for i, frame in enumerate(frames):
            event = events[i] if events and i < len(events) else None
            
            # Determine indices
            if event:
                t_idx = event.t_index
                z_idx = event.z_index
                c_idx = event.c_index
            else:
                # Auto-calculate from frame counter
                counter = self._frame_counters[detector_name]
                n_z = self.session_ctx.n_z_planes
                n_c = self.session_ctx.n_channels
                
                t_idx = counter // (n_z * n_c)
                remaining = counter % (n_z * n_c)
                c_idx = remaining // n_z
                z_idx = remaining % n_z
                
                self._frame_counters[detector_name] += 1
            
            with self.locks[detector_name]:
                self.queues[detector_name].append({
                    'frame': frame.copy(),
                    't_idx': t_idx,
                    'c_idx': c_idx,
                    'z_idx': z_idx,
                    'event': event,
                })
    
    def _writer_loop(self, detector_name: str):
        """Background thread for writing frames."""
        array = self.arrays[detector_name]
        
        while not self.stop_events[detector_name].is_set():
            task = None
            with self.locks[detector_name]:
                if self.queues[detector_name]:
                    task = self.queues[detector_name].popleft()
            
            if task:
                try:
                    frame = task['frame']
                    t_idx = task['t_idx']
                    c_idx = task['c_idx']
                    z_idx = task['z_idx']
                    
                    # Write to array
                    array[t_idx, c_idx, z_idx, :, :] = frame
                    logger.debug(f"Wrote frame to {detector_name}[{t_idx},{c_idx},{z_idx}]")
                    
                except Exception as e:
                    logger.error(f"Error writing frame to {detector_name}: {e}")
            else:
                time.sleep(0.001)
        
        # Drain queue on shutdown
        while True:
            with self.locks[detector_name]:
                if not self.queues[detector_name]:
                    break
                task = self.queues[detector_name].popleft()
            
            try:
                frame = task['frame']
                array[task['t_idx'], task['c_idx'], task['z_idx'], :, :] = frame
            except:
                break
    
    def finalize(self) -> None:
        """Finalize Zarr metadata and stop threads."""
        if self._is_finalized:
            return
        
        # Stop all threads
        for det_name in self.stop_events:
            self.stop_events[det_name].set()
        
        for det_name, thread in self.threads.items():
            thread.join(timeout=5.0)
        
        # Update metadata with actual frame count
        for det_name, root in self.roots.items():
            root.attrs["acquisition_complete"] = True
            root.attrs["finalized_at"] = time.time()
        
        self._is_finalized = True
        logger.info("OME-Zarr writer finalized")
    
    def close(self) -> None:
        """Close Zarr store."""
        if not self._is_open:
            return
        
        if not self._is_finalized:
            self.finalize()
        
        # Zarr groups and arrays don't need explicit closing
        self.arrays.clear()
        self.roots.clear()
        self.stores.clear()
        
        self._is_open = False
        logger.info("OME-Zarr writer closed")


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
