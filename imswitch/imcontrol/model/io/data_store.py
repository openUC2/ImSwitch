"""
Unified AcquisitionDataStore for ImSwitch.

Provides a single I/O service that can be used by both RecordingManager/Controller
and ExperimentController for consistent data writing with OME-compliant metadata.

This replaces the scattered writer implementations with a centralized,
MetadataHub-integrated service.
"""

import os
import time
import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import numpy as np

from .session import SessionInfo, SessionManager
from .writers import (
    WriterBase,
    SessionContext,
    WriterDetectorContext,
    WriterFrameEvent,
    OMETiffWriter,
    OMEZarrWriter,
    convert_detector_context,
    convert_frame_event,
    get_writer,
)

# Try to import MetadataHub
try:
    from ..metadata import MetadataHub, DetectorContext, FrameEvent
    HAS_METADATA_HUB = True
except ImportError:
    HAS_METADATA_HUB = False
    MetadataHub = None
    DetectorContext = None
    FrameEvent = None

logger = logging.getLogger(__name__)


@dataclass
class WriteStatistics:
    """Statistics for data store operations."""
    frames_written: Dict[str, int]  # per detector
    bytes_written: int
    events_consumed: int
    events_dropped: int
    start_time: float
    last_write_time: float
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frames_written': self.frames_written,
            'bytes_written': self.bytes_written,
            'events_consumed': self.events_consumed,
            'events_dropped': self.events_dropped,
            'start_time': self.start_time,
            'last_write_time': self.last_write_time,
            'duration_s': self.last_write_time - self.start_time if self.last_write_time else 0,
            'errors': self.errors,
        }


class AcquisitionDataStore:
    """
    Unified data store service for acquisition data.
    
    Coordinates writing to multiple formats (OME-Zarr, OME-TIFF) with:
    - Integrated MetadataHub for consistent metadata
    - Frame event alignment (N frames == N events consumed)
    - Session management and multi-instance access
    - Background writing support
    
    Usage:
        # With MetadataHub (recommended)
        store = AcquisitionDataStore(
            session_info=session_info,
            metadata_hub=hub,
            write_zarr=True,
            write_tiff=True
        )
        
        # Open with detector contexts from hub
        store.open()
        
        # Write frames - events are auto-consumed from hub
        store.write_frame(detector_name, frame)
        
        # Close session
        store.close()
    
    Or standalone (legacy mode):
        store = AcquisitionDataStore(
            session_info=session_info,
            write_zarr=True
        )
        store.open(detector_contexts={...})
        store.write_frame(detector_name, frame, frame_event=manual_event)
        store.close()
    """
    
    def __init__(self,
                 session_info: SessionInfo,
                 metadata_hub = None,  # MetadataHub instance (optional)
                 hub_snapshot: Optional[Dict[str, Any]] = None,
                 write_zarr: bool = True,
                 write_tiff: bool = False,
                 write_individual_tiffs: bool = False,
                 zarr_chunk_size: tuple = (1, 256, 256),
                 tiff_bigtiff: bool = True,
                 background_writes: bool = True,
                 on_write_complete: Optional[Callable[[str, int], None]] = None):
        """
        Initialize the data store.
        
        Args:
            session_info: Session metadata
            metadata_hub: Optional MetadataHub for automatic metadata extraction
            hub_snapshot: Optional static hub snapshot (if no live hub)
            write_zarr: Write OME-Zarr format
            write_tiff: Write OME-TIFF format
            write_individual_tiffs: Write individual TIFF files per position
            zarr_chunk_size: Chunk size for Zarr arrays
            tiff_bigtiff: Use BigTIFF format
            background_writes: Enable background thread for writes
            on_write_complete: Callback(detector_name, frame_number) when write completes
        """
        self.session_info = session_info
        self.metadata_hub = metadata_hub
        self.hub_snapshot = hub_snapshot or {}
        
        # Writer configuration
        self.write_zarr = write_zarr
        self.write_tiff = write_tiff
        self.write_individual_tiffs = write_individual_tiffs
        self.zarr_chunk_size = zarr_chunk_size
        self.tiff_bigtiff = tiff_bigtiff
        self.background_writes = background_writes
        self.on_write_complete = on_write_complete
        
        # State
        self._is_open = False
        self._session_manager: Optional[SessionManager] = None
        self._session_path: Optional[Path] = None
        
        # Writers
        self._zarr_writer: Optional[WriterBase] = None
        self._tiff_writer: Optional[WriterBase] = None
        
        # Detector contexts (writer format)
        self._detector_contexts: Dict[str, WriterDetectorContext] = {}
        
        # Background writing
        self._write_queue: deque = deque()
        self._write_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = WriteStatistics(
            frames_written={},
            bytes_written=0,
            events_consumed=0,
            events_dropped=0,
            start_time=time.time(),
            last_write_time=0,
            errors=[]
        )
    
    def open(self, detector_contexts: Optional[Dict[str, Any]] = None) -> Path:
        """
        Open the data store for writing.
        
        Args:
            detector_contexts: Optional detector context dicts (if not using MetadataHub)
            
        Returns:
            Path to session directory
        """
        if self._is_open:
            logger.warning("Data store already open")
            return self._session_path
        
        # Get detector contexts
        if detector_contexts:
            # Use provided contexts
            for name, ctx in detector_contexts.items():
                if isinstance(ctx, dict):
                    self._detector_contexts[name] = WriterDetectorContext(**ctx)
                elif hasattr(ctx, 'name'):
                    self._detector_contexts[name] = convert_detector_context(ctx)
                else:
                    self._detector_contexts[name] = ctx
        elif self.metadata_hub:
            # Get from MetadataHub
            hub_contexts = self.metadata_hub.export_detector_contexts()
            for name, ctx_dict in hub_contexts.items():
                self._detector_contexts[name] = WriterDetectorContext(
                    name=name,
                    shape_px=ctx_dict.get('shape_px', (512, 512)),
                    pixel_size_um=ctx_dict.get('pixel_size_um', 1.0),
                    dtype=ctx_dict.get('dtype', 'uint16'),
                    fov_um=ctx_dict.get('fov_um'),
                    binning=ctx_dict.get('binning', 1),
                    channel_name=ctx_dict.get('channel_name', name),
                    channel_color=ctx_dict.get('channel_color'),
                    wavelength_nm=ctx_dict.get('wavelength_nm'),
                    exposure_ms=ctx_dict.get('exposure_ms'),
                    gain=ctx_dict.get('gain'),
                )
            # Take hub snapshot now
            if not self.hub_snapshot:
                self.hub_snapshot = json.loads(self.metadata_hub.to_json())
        
        if not self._detector_contexts:
            raise ValueError("No detector contexts provided and no MetadataHub available")
        
        # Initialize stats
        for det_name in self._detector_contexts:
            self._stats.frames_written[det_name] = 0
        
        # Create session directory
        base_dir = Path(self.session_info.base_path) if self.session_info.base_path else Path.cwd()
        self._session_manager = SessionManager(base_dir)
        self._session_path = self._session_manager.create_session(
            session_info=self.session_info,
            hub_snapshot=self.hub_snapshot,
            detector_contexts={k: v.to_dict() for k, v in self._detector_contexts.items()}
        )
        
        # Update session info with actual path
        self.session_info.base_path = str(self._session_path)
        
        # Create session context for writers
        session_ctx = SessionContext(
            session_id=self.session_info.session_id,
            start_time=self.session_info.start_time,
            base_path=str(self._session_path),
            project=self.session_info.project,
            experiment=self.session_info.experiment,
            sample=self.session_info.sample,
            user=self.session_info.user,
            description=self.session_info.description,
            n_time_points=self.session_info.n_time_points,
            n_z_planes=self.session_info.n_z_planes,
            n_channels=self.session_info.n_channels,
            time_interval_s=self.session_info.time_interval_s,
            z_step_um=self.session_info.z_step_um,
        )
        
        # Open writers
        if self.write_zarr:
            zarr_path = self._session_path / SessionManager.ZARR_DIR
            zarr_ctx = SessionContext(
                session_id=session_ctx.session_id,
                start_time=session_ctx.start_time,
                base_path=str(zarr_path),
                project=session_ctx.project,
                experiment=session_ctx.experiment,
                sample=session_ctx.sample,
                user=session_ctx.user,
                n_time_points=session_ctx.n_time_points,
                n_z_planes=session_ctx.n_z_planes,
                n_channels=session_ctx.n_channels,
            )
            self._zarr_writer = OMEZarrWriter(zarr_ctx, chunk_size=self.zarr_chunk_size)
            self._zarr_writer.open(self._detector_contexts)
        
        if self.write_tiff:
            self._tiff_writer = OMETiffWriter(session_ctx, bigtiff=self.tiff_bigtiff)
            self._tiff_writer.open(self._detector_contexts)
        
        # Start background thread if enabled
        if self.background_writes:
            self._stop_event.clear()
            self._write_thread = threading.Thread(target=self._background_write_loop, daemon=True)
            self._write_thread.start()
        
        self._is_open = True
        self.session_info.status = "acquiring"
        logger.info(f"AcquisitionDataStore opened: {self._session_path}")
        
        return self._session_path
    
    def write_frame(self,
                    detector_name: str,
                    frame: np.ndarray,
                    frame_event: Optional[Any] = None,
                    t_index: int = 0,
                    z_index: int = 0,
                    c_index: int = 0,
                    position_index: int = 0):
        """
        Write a frame with aligned metadata.
        
        Args:
            detector_name: Detector name
            frame: Image data (2D numpy array)
            frame_event: Optional FrameEvent (if None, consumed from MetadataHub)
            t_index: Time point index
            z_index: Z plane index
            c_index: Channel index
            position_index: Position/tile index
        """
        if not self._is_open:
            raise RuntimeError("Data store not open")
        
        if detector_name not in self._detector_contexts:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        # Get or create frame event
        if frame_event is None and self.metadata_hub:
            # Pop event from hub to ensure alignment
            events = self.metadata_hub.pop_frame_events(detector_name, n=1)
            if events:
                frame_event = events[0]
                self._stats.events_consumed += 1
            else:
                # No event available - create one from current state
                logger.debug(f"No frame event in queue for {detector_name}, creating from snapshot")
                snapshot = self.metadata_hub.create_pre_trigger_snapshot(detector_name)
                frame_event = self.metadata_hub.create_frame_event_from_snapshot(snapshot)
                self._stats.events_dropped += 1
        
        # Convert to writer format
        writer_event = None
        if frame_event:
            writer_event = convert_frame_event(frame_event) if not isinstance(frame_event, WriterFrameEvent) else frame_event
            # Add indices
            writer_event.t_index = t_index
            writer_event.z_index = z_index
            writer_event.c_index = c_index
        
        # Queue or write directly
        write_task = {
            'detector_name': detector_name,
            'frame': frame.copy(),  # Copy for thread safety
            'event': writer_event,
            'indices': (t_index, z_index, c_index, position_index),
        }
        
        if self.background_writes:
            with self._lock:
                self._write_queue.append(write_task)
        else:
            self._execute_write(write_task)
    
    def _execute_write(self, task: Dict[str, Any]):
        """Execute a single write operation."""
        detector_name = task['detector_name']
        frame = task['frame']
        event = task['event']
        
        try:
            # Write to Zarr
            if self._zarr_writer:
                self._zarr_writer.write(detector_name, frame, events=[event] if event else None)
            
            # Write to TIFF
            if self._tiff_writer:
                self._tiff_writer.write(detector_name, frame, events=[event] if event else None)
            
            # Update statistics
            with self._lock:
                self._stats.frames_written[detector_name] = self._stats.frames_written.get(detector_name, 0) + 1
                self._stats.bytes_written += frame.nbytes
                self._stats.last_write_time = time.time()
            
            # Callback
            if self.on_write_complete and event:
                self.on_write_complete(detector_name, event.frame_number)
                
        except Exception as e:
            logger.error(f"Write error for {detector_name}: {e}")
            with self._lock:
                self._stats.errors.append(f"{detector_name}: {str(e)}")
    
    def _background_write_loop(self):
        """Background thread for writing frames."""
        while not self._stop_event.is_set():
            task = None
            with self._lock:
                if self._write_queue:
                    task = self._write_queue.popleft()
            
            if task:
                self._execute_write(task)
            else:
                time.sleep(0.001)  # Small sleep when queue is empty
        
        # Drain remaining tasks on shutdown
        while True:
            with self._lock:
                if not self._write_queue:
                    break
                task = self._write_queue.popleft()
            self._execute_write(task)
    
    def flush(self, timeout: float = 10.0) -> bool:
        """
        Flush all pending writes.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if all writes completed, False if timeout
        """
        if not self.background_writes:
            return True
        
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if not self._write_queue:
                    return True
            time.sleep(0.01)
        
        logger.warning(f"Flush timeout, {len(self._write_queue)} tasks pending")
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current write statistics."""
        with self._lock:
            return self._stats.to_dict()
    
    def get_session_path(self) -> Optional[Path]:
        """Get the session directory path."""
        return self._session_path
    
    def get_zarr_path(self) -> Optional[Path]:
        """Get the OME-Zarr store path."""
        if self._session_path:
            return self._session_path / SessionManager.ZARR_DIR
        return None
    
    def close(self):
        """
        Close the data store and finalize all files.
        """
        if not self._is_open:
            return
        
        logger.info("Closing AcquisitionDataStore...")
        
        # Stop background thread
        if self._write_thread:
            self._stop_event.set()
            self._write_thread.join(timeout=10.0)
        
        # Flush pending writes
        self.flush()
        
        # Finalize writers
        if self._zarr_writer:
            try:
                self._zarr_writer.finalize()
                self._zarr_writer.close()
            except Exception as e:
                logger.error(f"Error closing Zarr writer: {e}")
        
        if self._tiff_writer:
            try:
                self._tiff_writer.finalize()
                self._tiff_writer.close()
            except Exception as e:
                logger.error(f"Error closing TIFF writer: {e}")
        
        # Update session info
        self.session_info.status = "completed"
        self.session_info.frames_written = sum(self._stats.frames_written.values())
        
        # Finalize session
        if self._session_manager:
            self._session_manager.finalize_session(self.session_info)
        
        self._is_open = False
        logger.info(f"AcquisitionDataStore closed. Stats: {self._stats.to_dict()}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for simple use cases
def create_data_store(
    base_path: str,
    project: str = None,
    sample: str = None,
    write_zarr: bool = True,
    write_tiff: bool = False,
    metadata_hub = None  # Optional MetadataHub instance
) -> AcquisitionDataStore:
    """
    Create a data store with minimal configuration.
    
    Args:
        base_path: Directory for session data
        project: Optional project name
        sample: Optional sample name
        write_zarr: Write OME-Zarr format
        write_tiff: Write OME-TIFF format
        metadata_hub: Optional MetadataHub instance
        
    Returns:
        AcquisitionDataStore instance (not yet opened)
    """
    import uuid
    
    session_info = SessionInfo(
        session_id=str(uuid.uuid4()),
        base_path=base_path,
        project=project,
        sample=sample,
    )
    
    return AcquisitionDataStore(
        session_info=session_info,
        metadata_hub=metadata_hub,
        write_zarr=write_zarr,
        write_tiff=write_tiff,
    )


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
