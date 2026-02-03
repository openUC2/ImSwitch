"""
OME-TIFF writer with full OME-XML metadata support.

Writes acquisition data as OME-TIFF files with standards-compliant
OME-XML metadata embedded in the TIFF tags.
"""

import os
import time
import threading
import tifffile
from collections import deque
from typing import Dict, List, Optional
import numpy as np
import logging

from .base import WriterBase, SessionContext, DetectorContext, FrameEvent, WriterCapabilities
from .registry import register_writer
from .uuid_gen import compute_content_id

logger = logging.getLogger(__name__)


@register_writer('OME_TIFF')
class OMETiffWriter(WriterBase):
    """
    OME-TIFF writer with background writing and OME-XML metadata.
    
    Features:
    - Asynchronous writing via background thread
    - Full OME-XML metadata generation
    - Per-frame positional metadata
    - BigTIFF support for large datasets
    - RGB and grayscale support
    """
    
    def __init__(self, session_ctx: SessionContext, 
                 bigtiff: bool = True,
                 append_mode: bool = True):
        """
        Initialize OME-TIFF writer.
        
        Args:
            session_ctx: Session metadata
            bigtiff: Use BigTIFF format for large files
            append_mode: Write all frames to single file (True) or multiple files (False)
        """
        super().__init__(session_ctx)
        self.bigtiff = bigtiff
        self.append_mode = append_mode
        self.detectors: Dict[str, DetectorContext] = {}
        self.file_paths: Dict[str, str] = {}
        self.queues: Dict[str, deque] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self.locks: Dict[str, threading.Lock] = {}
    
    @classmethod
    def get_capabilities(cls) -> List[WriterCapabilities]:
        return [
            WriterCapabilities.STREAMING,
            WriterCapabilities.METADATA_RICH,
            WriterCapabilities.MULTI_DETECTOR,
            WriterCapabilities.TIME_SERIES,
            WriterCapabilities.Z_STACK,
        ]
    
    @classmethod
    def get_file_extension(cls) -> str:
        return '.ome.tiff'
    
    def open(self, detectors: Dict[str, DetectorContext]) -> None:
        """Open OME-TIFF files for each detector."""
        if self._is_open:
            logger.warning("Writer already open")
            return
        
        self.detectors = detectors
        
        # Ensure output directory exists
        if self.session_ctx.base_path:
            os.makedirs(self.session_ctx.base_path, exist_ok=True)
        
        # Set up file paths and threads for each detector
        for det_name, det_ctx in detectors.items():
            # Generate file path
            filename = f"{self.session_ctx.session_id}_{det_name}.ome.tiff"
            if self.session_ctx.base_path:
                filepath = os.path.join(self.session_ctx.base_path, filename)
            else:
                filepath = filename
            
            self.file_paths[det_name] = filepath
            self.queues[det_name] = deque()
            self.locks[det_name] = threading.Lock()
            self.stop_events[det_name] = threading.Event()
            
            # Start background writer thread
            thread = threading.Thread(
                target=self._writer_loop,
                args=(det_name,),
                daemon=True
            )
            thread.start()
            self.threads[det_name] = thread
            
            logger.info(f"Opened OME-TIFF writer for {det_name}: {filepath}")
        
        self._is_open = True
    
    def write(self, 
              detector_name: str,
              frames: np.ndarray,
              events: Optional[List[FrameEvent]] = None) -> None:
        """
        Write frames for a detector.
        
        Args:
            detector_name: Detector name
            frames: Image data (2D, 3D, or 4D array)
            events: Optional frame events for metadata
        """
        if not self._is_open:
            raise RuntimeError("Writer not open")
        
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        # Handle different frame dimensions
        if frames.ndim == 2:
            # Single frame
            frames = frames[np.newaxis, ...]
        elif frames.ndim == 3:
            # Stack of frames (already correct)
            pass
        elif frames.ndim == 4:
            # Multi-channel or 3D stack - flatten to 3D
            frames = frames.reshape(-1, frames.shape[-2], frames.shape[-1])
        
        # Enqueue frames with metadata
        for i, frame in enumerate(frames):
            event = events[i] if events and i < len(events) else None
            metadata = self._build_frame_metadata(detector_name, event)
            
            with self.locks[detector_name]:
                self.queues[detector_name].append((frame, metadata))
    
    def _build_frame_metadata(self, detector_name: str, event: Optional[FrameEvent]) -> Dict:
        """Build OME metadata for a single frame."""
        det_ctx = self.detectors[detector_name]
        
        metadata = {
            "Pixels": {
                "PhysicalSizeX": det_ctx.pixel_size_um,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": det_ctx.pixel_size_um,
                "PhysicalSizeYUnit": "µm",
            },
        }
        
        if event:
            plane_metadata = {}
            
            if event.stage_x_um is not None:
                plane_metadata["PositionX"] = event.stage_x_um
                plane_metadata["PositionXUnit"] = "µm"
            
            if event.stage_y_um is not None:
                plane_metadata["PositionY"] = event.stage_y_um
                plane_metadata["PositionYUnit"] = "µm"
            
            if event.stage_z_um is not None:
                plane_metadata["PositionZ"] = event.stage_z_um
                plane_metadata["PositionZUnit"] = "µm"
            
            if event.exposure_ms is not None:
                plane_metadata["ExposureTime"] = event.exposure_ms
                plane_metadata["ExposureTimeUnit"] = "ms"
            
            if event.timestamp:
                plane_metadata["DeltaT"] = event.timestamp - self.session_ctx.start_time
                plane_metadata["DeltaTUnit"] = "s"
            
            if plane_metadata:
                metadata["Plane"] = plane_metadata
        
        # Add session-level metadata
        if self.session_ctx.project:
            metadata["Project"] = self.session_ctx.project
        if self.session_ctx.sample:
            metadata["Sample"] = self.session_ctx.sample
        if self.session_ctx.user:
            metadata["User"] = self.session_ctx.user
        
        # Add content ID
        content_id = compute_content_id({
            'session_id': self.session_ctx.session_id,
            'detector': detector_name,
            'timestamp': time.time()
        })
        metadata["ContentID"] = content_id
        
        return metadata
    
    def _writer_loop(self, detector_name: str):
        """Background thread that writes frames to disk."""
        filepath = self.file_paths[detector_name]
        det_ctx = self.detectors[detector_name]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        
        # Determine photometric mode
        photometric = "rgb" if det_ctx.dtype in ['rgb', 'RGB'] else None
        
        try:
            with tifffile.TiffWriter(filepath, bigtiff=self.bigtiff, append=self.append_mode) as tif:
                stop_event = self.stop_events[detector_name]
                
                while not stop_event.is_set() or len(self.queues[detector_name]) > 0:
                    # Get frame from queue
                    with self.locks[detector_name]:
                        if self.queues[detector_name]:
                            frame, metadata = self.queues[detector_name].popleft()
                        else:
                            frame = None
                    
                    if frame is not None:
                        try:
                            # Write frame with metadata
                            tif.write(
                                data=frame,
                                metadata=metadata,
                                photometric=photometric
                            )
                        except Exception as e:
                            logger.error(f"Error writing frame for {detector_name}: {e}")
                    else:
                        # Sleep briefly to avoid busy loop
                        time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error in writer loop for {detector_name}: {e}")
    
    def finalize(self) -> None:
        """Finalize writing (signal threads to stop after queue is empty)."""
        if self._is_finalized:
            return
        
        # Signal all threads to stop
        for stop_event in self.stop_events.values():
            stop_event.set()
        
        self._is_finalized = True
        logger.info("OME-TIFF writer finalized")
    
    def close(self) -> None:
        """Close files and wait for threads to finish."""
        if not self._is_open:
            return
        
        # Wait for all threads to finish
        for det_name, thread in self.threads.items():
            if thread.is_alive():
                logger.debug(f"Waiting for writer thread for {det_name}...")
                thread.join(timeout=30.0)
                if thread.is_alive():
                    logger.warning(f"Writer thread for {det_name} did not finish in time")
        
        self._is_open = False
        logger.info("OME-TIFF writer closed")


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
