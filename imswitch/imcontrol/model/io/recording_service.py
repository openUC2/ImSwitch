"""
Unified Recording Service for ImSwitch.

This is the central service for all recording operations:
- Single image capture (snap) with metadata: TIFF, PNG, JPG
- Video recording with start/stop: MP4
- Streaming recordings: OME-Zarr, OME-TIFF
- Stitched mosaic output: 2D stitched OME-TIFF

This replaces the legacy RecordingManager functionality.
"""

import enum
import os
import time
import datetime
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

try:
    import tifffile as tiff
except ImportError:
    tiff = None

try:
    import cv2
except ImportError:
    cv2 = None

from imswitch.imcommon.framework import Signal, SignalInterface

logger = logging.getLogger(__name__)


# =============================================================================
# Enums (shared with legacy RecordingManager for compatibility)
# =============================================================================

class SaveMode(enum.Enum):
    """Where to save captured data."""
    Disk = 1
    RAM = 2
    DiskAndRAM = 3
    Numpy = 4


class SaveFormat(enum.Enum):
    """Output file format."""
    TIFF = 1
    ZARR = 3
    MP4 = 4
    PNG = 5
    JPG = 6
    OME_TIFF = 7      # OME-TIFF with full metadata
    OME_ZARR = 8      # OME-Zarr (NGFF)
    STITCHED_TIFF = 9  # 2D stitched mosaic


class RecMode(enum.Enum):
    """Recording mode for continuous capture."""
    SpecFrames = 1   # Record specific number of frames
    SpecTime = 2     # Record for specific time
    UntilStop = 3    # Record until stopped
    ScanOnce = 4     # Single scan
    ScanLapse = 5    # Multiple scans (timelapse)


# =============================================================================
# Result dataclasses
# =============================================================================

@dataclass
class SnapResult:
    """Result of a snap operation."""
    success: bool
    detector_name: str
    filepath: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'detector_name': self.detector_name,
            'filepath': self.filepath,
            'error': self.error,
            'timestamp': self.timestamp,
        }


@dataclass
class RecordingStatus:
    """Current status of a recording."""
    is_recording: bool = False
    format: Optional[SaveFormat] = None
    start_time: Optional[float] = None
    frame_count: int = 0
    filepath: Optional[str] = None
    error: Optional[str] = None
    
    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


# =============================================================================
# Background Storage Worker
# =============================================================================

@dataclass
class StorageTask:
    """A task for background storage operations."""
    task_type: str  # 'snap', 'frame', 'video_frame', 'finalize'
    filepath: str
    data: Any = None
    metadata: Optional[Dict[str, Any]] = None
    callback: Optional[Callable[[bool, str], None]] = None
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)


class BackgroundStorageWorker:
    """
    Background worker for non-blocking file I/O.
    
    Handles writing images/frames to disk without blocking the main thread.
    """
    
    def __init__(self, max_queue_size: int = 100):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._lock = threading.Lock()
        
        # Statistics
        self._tasks_completed = 0
        self._tasks_failed = 0
    
    def start(self):
        """Start the background worker thread."""
        if self._is_running:
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._is_running = True
        self._logger.info("BackgroundStorageWorker started")
    
    def stop(self, wait: bool = True, timeout: float = 5.0):
        """Stop the background worker."""
        if not self._is_running:
            return
        
        self._stop_event.set()
        
        if wait and self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        
        self._is_running = False
        self._logger.info(f"BackgroundStorageWorker stopped. Completed: {self._tasks_completed}, Failed: {self._tasks_failed}")
    
    def submit(self, task: StorageTask) -> bool:
        """Submit a storage task to the queue."""
        if not self._is_running:
            self._logger.warning("Cannot submit task: worker not running")
            return False
        
        try:
            self._task_queue.put_nowait(task)
            return True
        except queue.Full:
            self._logger.warning("Storage queue is full")
            if task.callback:
                task.callback(False, "Queue full")
            return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._task_queue.qsize()
    
    def _worker_loop(self):
        """Main worker loop."""
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.1)
                try:
                    self._execute_task(task)
                    self._tasks_completed += 1
                except Exception as e:
                    self._logger.error(f"Task execution failed: {e}")
                    self._tasks_failed += 1
                    if task.callback:
                        task.callback(False, str(e))
                finally:
                    self._task_queue.task_done()
            except queue.Empty:
                continue
        
        # Drain queue on shutdown
        while not self._task_queue.empty():
            try:
                task = self._task_queue.get_nowait()
                self._execute_task(task)
            except:
                break
    
    def _execute_task(self, task: StorageTask):
        """Execute a single storage task."""
        success = True
        message = "OK"
        
        try:
            if task.task_type == 'snap_tiff':
                self._write_tiff(task.filepath, task.data, task.metadata)
            elif task.task_type == 'snap_png':
                self._write_png(task.filepath, task.data)
            elif task.task_type == 'snap_jpg':
                self._write_jpg(task.filepath, task.data)
            elif task.task_type == 'append_tiff':
                self._append_tiff(task.filepath, task.data)
            else:
                message = f"Unknown task type: {task.task_type}"
                success = False
        except Exception as e:
            success = False
            message = str(e)
            raise
        finally:
            if task.callback:
                task.callback(success, message)
    
    def _write_tiff(self, filepath: str, data: np.ndarray, metadata: Dict = None):
        """Write TIFF file with optional OME-XML metadata."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        if metadata:
            try:
                # Try to build proper OME-XML metadata using ome_types
                from imswitch.imcontrol.model.io.ome_writers.ome_tiff_metadata import (
                    build_ome_metadata_from_dict, OME_TYPES_AVAILABLE
                )
                
                if OME_TYPES_AVAILABLE:
                    # Add image dimensions to metadata
                    metadata_with_dims = metadata.copy()
                    metadata_with_dims["height"] = data.shape[0]
                    metadata_with_dims["width"] = data.shape[1] if data.ndim > 1 else 1
                    metadata_with_dims["dtype"] = str(data.dtype)
                    
                    # Build OME-XML string
                    ome_xml = build_ome_metadata_from_dict(metadata_with_dims)
                    if ome_xml:
                        tiff.imwrite(filepath, data, description=ome_xml.encode("ascii", "xmlcharrefreplace").decode("ascii"))
                    else:
                        # Fallback to simple metadata as ImageJ description
                        import json
                        desc_str = json.dumps(metadata, default=str)
                        tiff.imwrite(filepath, data, description=desc_str)
                else:
                    # ome_types not available, use simple JSON description
                    import json
                    desc_str = json.dumps(metadata, default=str)
                    tiff.imwrite(filepath, data, description=desc_str)
            except Exception as e:
                logger.warning(f"Failed to write metadata to TIFF: {e}")
                tiff.imwrite(filepath, data)
        else:
            tiff.imwrite(filepath, data)
    
    def _write_png(self, filepath: str, data: np.ndarray):
        """Write PNG file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        img = data.copy()
        if img.dtype in (np.float32, np.float64):
            img = cv2.convertScaleAbs(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(filepath, img)
    
    def _write_jpg(self, filepath: str, data: np.ndarray):
        """Write JPEG file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        img = data.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(filepath, img)
    
    def _append_tiff(self, filepath: str, data: np.ndarray):
        """Append to existing TIFF file."""
        tiff.imwrite(filepath, data, append=True)


# =============================================================================
# MP4 Video Writer
# =============================================================================

class MP4Writer:
    """
    MP4 video writer with start/stop recording.
    
    Uses OpenCV VideoWriter for MP4 encoding.
    """
    
    def __init__(self, filepath: str, fps: float = 30.0, 
                 frame_size: Optional[Tuple[int, int]] = None,
                 codec: str = 'mp4v'):
        """
        Initialize MP4 writer.
        
        Args:
            filepath: Output file path
            fps: Frames per second
            frame_size: (width, height) - auto-detected from first frame if None
            codec: FourCC codec code
        """
        self._filepath = filepath
        self._fps = fps
        self._frame_size = frame_size
        self._codec = codec
        self._writer = None
        self._is_recording = False
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def start(self):
        """Start recording."""
        if self._is_recording:
            return
        
        # Writer is created on first frame to auto-detect size
        self._is_recording = True
        self._frame_count = 0
        self._start_time = time.time()
        self._logger.info(f"MP4 recording started: {self._filepath}")
    
    def stop(self):
        """Stop recording and finalize file."""
        if not self._is_recording:
            return
        
        with self._lock:
            self._is_recording = False
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        
        self._logger.info(f"MP4 recording stopped: {self._frame_count} frames written")
    
    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video."""
        if not self._is_recording:
            return
        # apparently, opencv swaps axes, so we do that here too 
        if len(frame.shape) >=2:
            frame = frame.transpose(1,0,2) if frame.ndim ==3 else frame.T
        else:
            frame = frame.T
        with self._lock:
            # Initialize writer on first frame
            if self._writer is None:
                h, w = frame.shape[:2]
                if self._frame_size is None:
                    self._frame_size = (h, w)
                
                os.makedirs(os.path.dirname(self._filepath), exist_ok=True) if os.path.dirname(self._filepath) else None
                fourcc = cv2.VideoWriter_fourcc(*self._codec)
                self._writer = cv2.VideoWriter(
                    self._filepath, fourcc, self._fps, self._frame_size # requieres height, width
                )
            
            # Convert grayscale to BGR
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Resize if needed
            h, w = frame.shape[:2]
            if (w, h) != self._frame_size:
                frame = cv2.resize(frame, self._frame_size)
            
            self._writer.write(frame)
            self._frame_count += 1
    
    @property
    def is_recording(self) -> bool:
        return self._is_recording
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed recording time in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    @property
    def filepath(self) -> str:
        return self._filepath


# =============================================================================
# Recording Service - Main Entry Point
# =============================================================================

class RecordingService(SignalInterface):
    """
    Unified Recording Service for all acquisition I/O operations.
    
    This is the central service that replaces RecordingManager.
    It provides:
    - Single image capture (snap) with metadata
    - Video recording (MP4) with start/stop
    - Streaming recordings (OME-Zarr, OME-TIFF)
    - Integration with MetadataHub for comprehensive metadata
    
    Usage:
        service = RecordingService(detectors_manager)
        service.set_metadata_hub(metadata_hub)
        
        # Snap
        results = service.snap(format=SaveFormat.TIFF)
        
        # Video recording
        service.start_video_recording('/path/to/video.mp4')
        service.add_video_frame(frame)
        service.stop_video_recording()
        
        # Streaming to Zarr
        service.start_streaming('/path/to/data.zarr', format=SaveFormat.OME_ZARR)
        service.write_frame(detector_name, frame, metadata)
        service.stop_streaming()
    """
    
    # Signals for UI compatibility
    sigRecordingStarted = Signal()
    sigRecordingEnded = Signal()
    sigRecordingFrameNumUpdated = Signal(int)
    sigRecordingTimeUpdated = Signal(int)
    sigSnapCompleted = Signal(str, np.ndarray, str, bool)  # (name, image, path, savedToDisk)
    
    def __init__(self, detectors_manager=None, metadata_hub=None):
        """
        Initialize the recording service.
        
        Args:
            detectors_manager: DetectorsManager for detector access
            metadata_hub: Optional MetadataHub for comprehensive metadata
        """
        super().__init__()
        self._detectors_manager = detectors_manager
        self._metadata_hub = metadata_hub
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Background worker for async I/O
        self._storage_worker = BackgroundStorageWorker()
        self._storage_worker.start()
        
        # Video recording state
        self._video_writer: Optional[MP4Writer] = None
        self._video_detector_name: Optional[str] = None  # Detector to capture from
        
        # Streaming recording state
        self._is_streaming = False
        self._streaming_format: Optional[SaveFormat] = None
        self._streaming_writer = None
        self._streaming_frame_count = 0
        self._streaming_start_time: Optional[float] = None
        self._streaming_detector_names: List[str] = []
        self._streaming_adapter = None  # StreamingDataStoreAdapter
        
        # Frame callback connection
        self._frame_callback_connected = False
        self._comm_channel = None  # Will be set externally
        
        # Memory recordings (RAM mode)
        self._mem_recordings: Dict[str, Any] = {}
    
    def __del__(self):
        self.shutdown()
    
    def shutdown(self):
        """Shutdown the service and release resources."""
        self.stop_video_recording()
        self.stop_streaming()
        self._storage_worker.stop(wait=True, timeout=10.0)
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def set_detectors_manager(self, detectors_manager):
        """Set or update the detectors manager."""
        self._detectors_manager = detectors_manager
    
    def set_metadata_hub(self, metadata_hub):
        """Set or update the metadata hub."""
        self._metadata_hub = metadata_hub
    
    def set_comm_channel(self, comm_channel):
        """
        Set the communication channel for frame callbacks.
        
        This enables automatic frame capture during recording by connecting
        to sigNewFrame or sigUpdateImage signals.
        """
        self._comm_channel = comm_channel
    
    def set_streaming_adapter(self, adapter):
        """Set the streaming adapter for continuous recordings."""
        self._streaming_adapter = adapter
    
    # =========================================================================
    # Frame Callback for Continuous Recording
    # =========================================================================
    
    def _on_new_frame(self, detector_name: str, frame: np.ndarray, init: bool, scale: list, is_current_detector: bool):
        """
        (
            str, np.ndarray, bool, list, bool
        )  # (detectorName, image, init, scale, isCurrentDetector)
        
        Callback for new frames - writes to active recording.
        
        This is called by the signal connection when a new frame is available.
        """
        if frame is None:
            return
        
        # Write to video recording if active
        if self._video_writer is not None and self._video_writer.is_recording:
            if self._video_detector_name is None or self._video_detector_name == detector_name:
                self._video_writer.write_frame(frame)
                self.sigRecordingFrameNumUpdated.emit(self._video_writer.frame_count)
        
        # Write to streaming adapter if active
        if self._streaming_adapter is not None and hasattr(self._streaming_adapter, '_is_open'):
            if self._streaming_adapter._is_open:
                if not self._streaming_detector_names or detector_name in self._streaming_detector_names:
                    try:
                        self._streaming_adapter.write_frame(detector_name, frame)
                        self._streaming_frame_count += 1
                        self.sigRecordingFrameNumUpdated.emit(self._streaming_frame_count)
                    except Exception as e:
                        self._logger.error(f"Error writing frame to streaming adapter: {e}")
    
    def _connect_frame_callback(self):
        """Connect to frame signal for continuous recording."""
        if self._frame_callback_connected:
            return
        
        if self._comm_channel is not None and hasattr(self._comm_channel, 'sigUpdateImage'):
            try:
                self._comm_channel.sigUpdateImage.connect(self._on_new_frame)
                self._frame_callback_connected = True
                self._logger.info("Frame callback connected for continuous recording")
            except Exception as e:
                self._logger.error(f"Failed to connect frame callback: {e}")
    
    def _disconnect_frame_callback(self):
        """Disconnect frame signal callback."""
        if not self._frame_callback_connected:
            return
        
        if self._comm_channel is not None and hasattr(self._comm_channel, 'sigUpdateImage'):
            try:
                self._comm_channel.sigUpdateImage.disconnect(self._on_new_frame)
                self._frame_callback_connected = False
                self._logger.info("Frame callback disconnected")
            except Exception as e:
                self._logger.debug(f"Error disconnecting frame callback: {e}")
    
    # =========================================================================
    # Snap Operations (Single Image Capture)
    # =========================================================================
    
    def snap(self,
             detector_names: List[str] = None,
             savepath: str = "",
             save_mode: SaveMode = SaveMode.Disk,
             format: SaveFormat = SaveFormat.TIFF,
             attrs: Dict[str, Any] = None,
             async_write: bool = True,
             callback: Callable[[SnapResult], None] = None) -> Dict[str, SnapResult]:
        """
        Capture and save images from detectors.
        
        Args:
            detector_names: List of detector names (None = all)
            savepath: Base path for saving (without extension)
            save_mode: Where to save (Disk, RAM, DiskAndRAM, Numpy)
            format: Output format (TIFF, PNG, JPG)
            attrs: Metadata attributes (uses MetadataHub if not provided)
            async_write: If True, write in background
            callback: Optional callback for completion
            
        Returns:
            Dict mapping detector names to SnapResult
        """
        if self._detectors_manager is None:
            raise RuntimeError("DetectorsManager not set")
        
        if detector_names is None:
            detector_names = list(self._detectors_manager.detectorNames)
        
        # Get metadata from hub if available and attrs not provided
        if attrs is None and self._metadata_hub is not None:
            attrs = self._metadata_hub.get_snapshot()
        
        results = {}
        images = {}
        
        # Capture images from all detectors
        for det_name in detector_names:
            try:
                frame = self._detectors_manager[det_name].getLatestFrame()
                if frame is not None:
                    images[det_name] = frame.copy()
            except Exception as e:
                self._logger.error(f"Failed to capture from {det_name}: {e}")
                results[det_name] = SnapResult(
                    success=False,
                    detector_name=det_name,
                    error=str(e)
                )
        
        # Process based on save mode
        if save_mode == SaveMode.Numpy:
            # Return numpy arrays directly
            for det_name, image in images.items():
                results[det_name] = SnapResult(
                    success=True,
                    detector_name=det_name,
                    metadata={'shape': image.shape, 'dtype': str(image.dtype)}
                )
            return images  # Return images dict for Numpy mode
        
        # Save to disk
        if save_mode in (SaveMode.Disk, SaveMode.DiskAndRAM):
            for det_name, image in images.items():
                filepath = self._generate_filepath(savepath, det_name, format)
                # TODO: We essentially loose all our metadta: {'WidefieldCamera': {'Detector:WidefieldCamera:Model': 'VirtualCamera', 'Detector:WidefieldCamera:Pixel size': [...], 'Detector:WidefieldCamera:Binning': 1, 'Detector:WidefieldCamera:ROI': [...], 'Detector:WidefieldCamera:Param:exposure': 0, 'Detector:WidefieldCamera:Param:exposure_mode': 'Auto', 'Detector:WidefieldCamera:Param:gain': 0, 'Detector:WidefieldCamera:Param:blacklevel': 100, 'Detector:WidefieldCamera:Param:binning': 1, 'Detector:WidefieldCamera:Param:image_width': 1000, 'Detector:WidefieldCamera:Param:image_height': 1000, 'Detector:WidefieldCamera:Param:frame_rate': -1, 'Detector:WidefieldCamera:Param:mode': True, 'Detector:WidefieldCamera:Param:flat_fielding': True, 'Detector:WidefieldCamera:Param:trigger_source': 'Continous', 'Detector:WidefieldCamera:Param:Camera pixel size': 1.0, 'Detector:FocusLock:Model': 'VirtualCamera', 'Detector:FocusLock:Pixel size': [...], 'Detector:FocusLock:Binning': 1, ...}} to  {'Detector': 'WidefieldCamera', 'Channel': 'Brightfield_WidefieldCamera', 'DateTime': '2026-01-27T14:12:22.757871'}
                metadata = self._build_metadata(det_name, image, attrs)
                if async_write:
                    result = self._snap_async(det_name, image, filepath, format, metadata, callback)
                else:
                    result = self._snap_sync(det_name, image, filepath, format, metadata)
                
                results[det_name] = result
        
        # Save to RAM
        if save_mode in (SaveMode.RAM, SaveMode.DiskAndRAM):
            for det_name, image in images.items():
                name = os.path.basename(f'{savepath}_{det_name}')
                self._mem_recordings[name] = image.copy()
                self.sigSnapCompleted.emit(
                    name, image, savepath, save_mode == SaveMode.DiskAndRAM
                )
        
        return results
    
    def snap_numpy(self, detector_names: List[str] = None,
                   attrs: Dict[str, Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Capture images and return as numpy arrays (no saving).
        
        Args:
            detector_names: List of detector names (None = all)
            attrs: Optional metadata attributes per detector
            
        Returns:
            Dict mapping detector names to image arrays
        """
        return self.snap(
            detector_names=detector_names,
            save_mode=SaveMode.Numpy,
            attrs=attrs
        )
    
    def save_image(self, detector_name: str, image: np.ndarray,
                   savepath: str, format: SaveFormat = SaveFormat.TIFF,
                   attrs: Dict[str, Any] = None, async_write: bool = True) -> SnapResult:
        """
        Save a pre-captured image to disk.
        
        This method saves an already captured image (not from detector)
        using the same infrastructure as snap operations.
        
        Args:
            detector_name: Name of the detector (for metadata)
            image: The image data to save
            savepath: Base path for saving (without extension)
            format: Output format (TIFF, PNG, JPG)
            attrs: Metadata attributes
            async_write: If True, write in background
            
        Returns:
            SnapResult with save status
        """
        filepath = self._generate_filepath(savepath, detector_name, format)
        metadata = self._build_metadata(detector_name, image, attrs)
        
        if async_write:
            return self._snap_async(detector_name, image, filepath, format, metadata)
        else:
            return self._snap_sync(detector_name, image, filepath, format, metadata)
    
    def _snap_async(self, detector_name: str, image: np.ndarray,
                    filepath: str, format: SaveFormat,
                    metadata: Dict = None, callback: Callable = None) -> SnapResult:
        """Queue image for background writing."""
        task_type = self._format_to_task_type(format)
        
        task = StorageTask(
            task_type=task_type,
            filepath=filepath,
            data=image.copy(),
            metadata=metadata if format == SaveFormat.TIFF else None,
            callback=lambda ok, msg: callback(SnapResult(
                success=ok, detector_name=detector_name, filepath=filepath, error=msg if not ok else None
            )) if callback else None
        )
        
        self._storage_worker.submit(task)
        
        return SnapResult(
            success=True,
            detector_name=detector_name,
            filepath=filepath,
            metadata=metadata
        )
    
    def _snap_sync(self, detector_name: str, image: np.ndarray,
                   filepath: str, format: SaveFormat,
                   metadata: Dict = None) -> SnapResult:
        """Write image synchronously."""
        try:
            dirname = os.path.dirname(filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            
            if format in (SaveFormat.TIFF, SaveFormat.OME_TIFF):
                if metadata:
                    tiff.imwrite(filepath, image, metadata=metadata, imagej=False)
                else:
                    tiff.imwrite(filepath, image)
            elif format == SaveFormat.PNG:
                img = image.copy()
                if img.dtype in (np.float32, np.float64):
                    img = cv2.convertScaleAbs(img)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(filepath, img)
            elif format == SaveFormat.JPG:
                img = image.copy()
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(filepath, img)
            else:
                raise ValueError(f"Unsupported format for snap: {format}")
            
            return SnapResult(
                success=True,
                detector_name=detector_name,
                filepath=filepath,
                metadata=metadata
            )
        except Exception as e:
            self._logger.error(f"Snap failed: {e}")
            return SnapResult(
                success=False,
                detector_name=detector_name,
                filepath=filepath,
                error=str(e)
            )
    
    def _format_to_task_type(self, format: SaveFormat) -> str:
        """Convert SaveFormat to task type string."""
        mapping = {
            SaveFormat.TIFF: 'snap_tiff',
            SaveFormat.OME_TIFF: 'snap_tiff',
            SaveFormat.PNG: 'snap_png',
            SaveFormat.JPG: 'snap_jpg',
        }
        return mapping.get(format, 'snap_tiff')
    
    def _generate_filepath(self, basepath: str, detector_name: str, format: SaveFormat) -> str:
        """Generate full filepath with extension."""
        ext_map = {
            SaveFormat.TIFF: '.ome.tiff',
            SaveFormat.OME_TIFF: '.ome.tiff',
            SaveFormat.PNG: '.png',
            SaveFormat.JPG: '.jpg',
            SaveFormat.ZARR: '.zarr',
            SaveFormat.OME_ZARR: '.zarr',
            SaveFormat.MP4: '.mp4',
        }
        ext = ext_map.get(format, '.tiff')
        return f"{basepath}_{detector_name}{ext}"
    
    # =========================================================================
    # Video Recording (MP4 with start/stop)
    # =========================================================================
    
    def start_video_recording(self, filepath: str, fps: float = 30.0,
                               detector_name: str = None,
                               auto_capture: bool = True) -> bool:
        """
        Start MP4 video recording.
        
        Args:
            filepath: Output file path
            fps: Frames per second
            detector_name: Detector to record from (None = first detector or all)
            auto_capture: If True, automatically capture frames via signal callback
            
        Returns:
            True if started successfully
        """
        if self._video_writer is not None and self._video_writer.is_recording:
            self._logger.warning("Video recording already in progress")
            return False
        
        self._video_writer = MP4Writer(filepath, fps=fps)
        self._video_detector_name = detector_name
        self._video_writer.start()
        
        # Connect frame callback for automatic frame capture
        if auto_capture:
            self._connect_frame_callback()
        
        self.sigRecordingStarted.emit()
        self._logger.info(f"Video recording started: {filepath}")
        return True
    
    def add_video_frame(self, frame: np.ndarray):
        """Add a frame to the current video recording."""
        if self._video_writer is None or not self._video_writer.is_recording:
            return
        
        self._video_writer.write_frame(frame)
        self.sigRecordingFrameNumUpdated.emit(self._video_writer.frame_count)
    
    def stop_video_recording(self) -> int:
        """
        Stop video recording.
        
        Returns:
            Number of frames recorded
        """
        if self._video_writer is None:
            return 0
        
        # Disconnect frame callback
        self._disconnect_frame_callback()
        
        frame_count = self._video_writer.frame_count
        self._video_writer.stop()
        self._video_writer = None
        self._video_detector_name = None
        self.sigRecordingEnded.emit()
        return frame_count
    
    @property
    def is_video_recording(self) -> bool:
        """Check if video recording is in progress."""
        return self._video_writer is not None and self._video_writer.is_recording
    
    @property
    def video_recording_duration(self) -> float:
        """Get current video recording duration in seconds."""
        if self._video_writer is None:
            return 0.0
        return self._video_writer.elapsed_time
    
    @property
    def video_recording_frame_count(self) -> int:
        """Get current video recording frame count."""
        if self._video_writer is None:
            return 0
        return self._video_writer.frame_count
    
    # =========================================================================
    # Streaming Recording (OME-Zarr, OME-TIFF)
    # =========================================================================
    
    def start_streaming(self, filepath: str, format: SaveFormat = SaveFormat.OME_ZARR,
                        detector_contexts: Dict = None) -> bool:
        """
        Start streaming recording to OME-Zarr or OME-TIFF.
        
        Args:
            filepath: Output file/directory path
            format: SaveFormat.OME_ZARR or SaveFormat.OME_TIFF
            detector_contexts: Detector configuration for writers
            
        Returns:
            True if started successfully
        """
        if self._is_streaming:
            self._logger.warning("Streaming already in progress")
            return False
        
        try:
            from .writers import OMEZarrWriter, OMETiffWriter
            from .session import SessionInfo
            
            # Create writer based on format
            if format == SaveFormat.OME_ZARR:
                self._streaming_writer = OMEZarrWriter(filepath)
            elif format == SaveFormat.OME_TIFF:
                self._streaming_writer = OMETiffWriter(filepath)
            else:
                raise ValueError(f"Unsupported streaming format: {format}")
            
            # Initialize writer with detector contexts
            if detector_contexts:
                self._streaming_writer.open(detector_contexts)
            
            self._is_streaming = True
            self._streaming_format = format
            self._streaming_frame_count = 0
            self.sigRecordingStarted.emit()
            self._logger.info(f"Streaming started: {filepath} ({format.name})")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to start streaming: {e}")
            return False
    
    def write_streaming_frame(self, detector_name: str, frame: np.ndarray,
                               metadata: Dict = None):
        """Write a frame to the streaming output."""
        if not self._is_streaming or self._streaming_writer is None:
            return
        
        try:
            self._streaming_writer.write_frame(detector_name, frame, metadata)
            self._streaming_frame_count += 1
            self.sigRecordingFrameNumUpdated.emit(self._streaming_frame_count)
        except Exception as e:
            self._logger.error(f"Failed to write streaming frame: {e}")
    
    def stop_streaming(self) -> int:
        """
        Stop streaming recording.
        
        Returns:
            Number of frames recorded
        """
        if not self._is_streaming:
            return 0
        
        frame_count = self._streaming_frame_count
        
        if self._streaming_writer is not None:
            try:
                self._streaming_writer.close()
            except Exception as e:
                self._logger.error(f"Error closing streaming writer: {e}")
            self._streaming_writer = None
        
        self._is_streaming = False
        self._streaming_format = None
        self._streaming_frame_count = 0
        self.sigRecordingEnded.emit()
        
        return frame_count
    
    @property
    def is_streaming(self) -> bool:
        """Check if streaming is in progress."""
        return self._is_streaming
    
    @property
    def streaming_duration(self) -> float:
        """Get current streaming recording duration in seconds."""
        if self._streaming_start_time is None:
            return 0.0
        return time.time() - self._streaming_start_time
    
    @property
    def streaming_frame_count(self) -> int:
        """Get current streaming frame count."""
        return self._streaming_frame_count
    
    # =========================================================================
    # Combined Recording State
    # =========================================================================
    
    @property
    def is_recording(self) -> bool:
        """Check if any recording (video or streaming) is in progress."""
        return self.is_video_recording or self.is_streaming or (
            self._streaming_adapter is not None and 
            hasattr(self._streaming_adapter, '_is_open') and 
            self._streaming_adapter._is_open
        )
    
    @property
    def recording_duration(self) -> float:
        """Get duration of current recording in seconds."""
        if self.is_video_recording:
            return self.video_recording_duration
        elif self._streaming_start_time is not None:
            return self.streaming_duration
        return 0.0
    
    @property
    def recording_frame_count(self) -> int:
        """Get frame count of current recording."""
        if self.is_video_recording:
            return self.video_recording_frame_count
        return self._streaming_frame_count
    
    # =========================================================================
    # Metadata Building
    # =========================================================================
    
    def _build_metadata(self, detector_name: str, image: np.ndarray,
                        attrs: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Build comprehensive metadata from attributes.
        
        Preserves ALL original attributes while also extracting key OME-compatible fields.
        The full attrs dictionary is stored under 'AcquisitionAttributes' to prevent data loss.
        """
        metadata = {}
        
        # Always add basic info
        metadata['Detector'] = detector_name
        metadata['DateTime'] = datetime.datetime.now().isoformat()
        
        # Add image shape info
        if image is not None:
            metadata['ImageShape'] = list(image.shape)
            metadata['ImageDtype'] = str(image.dtype)
        
        if not attrs:
            return metadata
        
        def _get_value(val):
            """Extract value from SharedAttrValue or return raw value."""
            return val.value if hasattr(val, 'value') else val
        
        def _serialize_value(val):
            """Serialize value for JSON-compatible storage."""
            val = _get_value(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (list, tuple)):
                return [_serialize_value(v) for v in val]
            elif isinstance(val, dict):
                return {k: _serialize_value(v) for k, v in val.items()}
            elif isinstance(val, (np.integer, np.floating)):
                return float(val)
            return val
        
        def _search_attr(patterns: list, search_dict: dict):
            """Search for attribute using multiple key patterns."""
            for pattern in patterns:
                if pattern in search_dict:
                    return _get_value(search_dict[pattern])
                for key in search_dict.keys():
                    if pattern in str(key):
                        return _get_value(search_dict[key])
            return None
        
        try:
            # PRESERVE ALL ORIGINAL ATTRIBUTES
            # Store full attrs dictionary to prevent data loss
            # Handle nested dict structure (e.g., {detector_name: {attrs...}})
            if detector_name in attrs and isinstance(attrs[detector_name], dict):
                # Attrs are nested under detector name
                detector_attrs = attrs[detector_name]
                metadata['AcquisitionAttributes'] = {
                    k: _serialize_value(v) for k, v in detector_attrs.items()
                }
                # Use detector-specific attrs for extraction
                flat_attrs = detector_attrs
            else:
                # Flat attrs structure
                metadata['AcquisitionAttributes'] = {
                    k: _serialize_value(v) for k, v in attrs.items()
                }
                flat_attrs = attrs
            
            # Extract key OME-compatible fields for quick access
            # Pixel size
            pixel_size = _search_attr([
                f'Detector:{detector_name}:Pixel size',
                f'Detector:{detector_name}:PixelSizeUm',
                'Detector:PixelSizeUm',
                'PixelSizeUm',
                'Param:Camera pixel size'
            ], flat_attrs)
            if pixel_size:
                if isinstance(pixel_size, (list, tuple)) and len(pixel_size) > 0:
                    pixel_size = pixel_size[0] if len(pixel_size) == 1 else pixel_size[1]
                metadata['PhysicalSizeX'] = float(pixel_size)
                metadata['PhysicalSizeY'] = float(pixel_size)
                metadata['PhysicalSizeXUnit'] = 'µm'
                metadata['PhysicalSizeYUnit'] = 'µm'
            
            # Exposure
            exposure = _search_attr([
                f'Detector:{detector_name}:Param:exposure',
                f'Detector:{detector_name}:ExposureMs',
                'Param:exposure',
                'ExposureMs'
            ], flat_attrs)
            if exposure is not None:
                try:
                    exp_val = float(exposure)
                    if exp_val > 0:
                        metadata['ExposureTime'] = exp_val / 1000.0  # Convert ms to s
                        metadata['ExposureTimeUnit'] = 's'
                except (ValueError, TypeError):
                    pass
            
            # Gain
            gain = _search_attr([
                f'Detector:{detector_name}:Param:gain',
                'Param:gain',
                'Gain'
            ], flat_attrs)
            if gain is not None:
                try:
                    metadata['Gain'] = float(gain)
                except (ValueError, TypeError):
                    pass
            
            # Binning
            binning = _search_attr([
                f'Detector:{detector_name}:Binning',
                f'Detector:{detector_name}:Param:binning',
                'Binning'
            ], flat_attrs)
            if binning is not None:
                try:
                    metadata['Binning'] = int(binning)
                except (ValueError, TypeError):
                    pass
            
            # ROI
            roi = _search_attr([
                f'Detector:{detector_name}:ROI',
                'ROI'
            ], flat_attrs)
            if roi is not None:
                metadata['ROI'] = _serialize_value(roi)
            
            # Model
            model = _search_attr([
                f'Detector:{detector_name}:Model',
                'Model'
            ], flat_attrs)
            if model:
                metadata['DetectorModel'] = str(model)
            
            # Stage positions (search in full attrs, not detector-specific)
            for axis in ['X', 'Y', 'Z']:
                for key, val in attrs.items():
                    key_str = str(key)
                    if 'Positioner:' in key_str and f':{axis}:Position' in key_str:
                        try:
                            metadata[f'Position{axis}'] = float(_get_value(val))
                            metadata[f'Position{axis}Unit'] = 'µm'
                        except (ValueError, TypeError):
                            pass
                        break
            
            # Illumination (lasers/LEDs)
            laser_sources = {}
            for key, val in attrs.items():
                key_str = str(key)
                if key_str.startswith('Laser:'):
                    parts = key_str.split(':')
                    if len(parts) >= 3:
                        laser_name = parts[1]
                        attr_name = parts[2]
                        if laser_name not in laser_sources:
                            laser_sources[laser_name] = {}
                        laser_sources[laser_name][attr_name] = _get_value(val)
            
            active_lasers = []
            for laser_name, laser_data in laser_sources.items():
                is_enabled = laser_data.get('Enabled', False)
                value = laser_data.get('Value', 0)
                wavelength = laser_data.get('WavelengthNm', 0)
                if is_enabled and value:
                    try:
                        if float(value) > 0:
                            active_lasers.append({
                                'Name': laser_name,
                                'WavelengthNm': float(wavelength) if wavelength else None,
                                'Power': float(value),
                            })
                    except (ValueError, TypeError):
                        pass
            
            if active_lasers:
                metadata['ActiveLasers'] = active_lasers
                if active_lasers[0].get('WavelengthNm'):
                    metadata['ExcitationWavelength'] = active_lasers[0]['WavelengthNm']
                    metadata['Channel'] = f"{int(active_lasers[0]['WavelengthNm'])}nm"
                else:
                    metadata['Channel'] = active_lasers[0]['Name']
            else:
                metadata['Channel'] = f"Brightfield_{detector_name}"
            
            # Objective
            objective_name = _search_attr(['Objective:Name', 'ObjectiveName'], attrs)
            if objective_name:
                metadata['Objective'] = str(objective_name)
            
            magnification = _search_attr(['Objective:Magnification'], attrs)
            if magnification:
                metadata['Magnification'] = float(magnification)
            
            na = _search_attr(['Objective:NA'], attrs)
            if na:
                metadata['NumericalAperture'] = float(na)
            
            # Timestamp
            metadata['DateTime'] = datetime.datetime.now().isoformat()
            
            return metadata if metadata else None
            
        except Exception as e:
            self._logger.warning(f"Error building metadata: {e}")
            return None
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_pending_io_count(self) -> int:
        """Get number of pending I/O operations."""
        return self._storage_worker.get_queue_size()
    
    def wait_for_io_complete(self, timeout: float = 30.0) -> bool:
        """Wait for all pending I/O operations to complete."""
        start = time.time()
        while self.get_pending_io_count() > 0:
            if time.time() - start > timeout:
                self._logger.warning(f"Timeout waiting for I/O completion")
                return False
            time.sleep(0.1)
        return True
    
    def get_status(self) -> RecordingStatus:
        """Get current recording status."""
        is_rec = self.is_recording
        
        # Determine format and filepath
        if self.is_video_recording:
            format_type = SaveFormat.MP4
            filepath = self._video_writer.filepath if self._video_writer else None
            start_time = self._video_writer._start_time if self._video_writer else None
        elif self._is_streaming:
            format_type = self._streaming_format
            filepath = None
            start_time = self._streaming_start_time
        elif self._streaming_adapter is not None:
            format_type = SaveFormat.OME_ZARR
            filepath = getattr(self._streaming_adapter, 'base_path', None)
            start_time = self._streaming_start_time
        else:
            format_type = None
            filepath = None
            start_time = None
        
        return RecordingStatus(
            is_recording=is_rec,
            format=format_type,
            start_time=start_time,
            frame_count=self.recording_frame_count,
            filepath=filepath,
        )
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get recording status as a dictionary (for API responses)."""
        status = self.get_status()
        return {
            'is_recording': status.is_recording,
            'format': status.format.name if status.format else None,
            'duration_seconds': status.elapsed_time,
            'frame_count': status.frame_count,
            'filepath': status.filepath,
        }


# =============================================================================
# Global Instance and Factory
# =============================================================================

_recording_service: Optional[RecordingService] = None


def get_recording_service() -> RecordingService:
    """Get or create the global RecordingService instance."""
    global _recording_service
    if _recording_service is None:
        _recording_service = RecordingService()
    return _recording_service


def create_recording_service(detectors_manager=None, metadata_hub=None) -> RecordingService:
    """Create a new RecordingService with the given managers."""
    return RecordingService(detectors_manager, metadata_hub)


def shutdown_recording_service():
    """Shutdown the global RecordingService."""
    global _recording_service
    if _recording_service is not None:
        _recording_service.shutdown()
        _recording_service = None


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
