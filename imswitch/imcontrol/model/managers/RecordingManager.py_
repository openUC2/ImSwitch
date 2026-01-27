"""
Recording Manager for ImSwitch.

DEPRECATION NOTICE:
===================
This module is DEPRECATED and will be removed in a future version.
Please use the new unified io module instead:

    from imswitch.imcontrol.model.io import (
        RecordingService,
        SaveMode,
        SaveFormat,
        get_recording_service
    )
    
    # Create service
    service = get_recording_service()
    service.set_detectors_manager(detectors_manager)
    
    # Snap images
    results = service.snap(format=SaveFormat.TIFF)
    
    # Video recording
    service.start_video_recording('/path/to/video.mp4')
    service.add_video_frame(frame)
    service.stop_video_recording()

The new io module provides:
- Unified snap/recording/streaming operations
- MetadataHub integration for OME-compliant metadata
- OME-Zarr and OME-TIFF writers
- 2D stitched mosaic support
- MP4 video recording

This module remains for backwards compatibility only.
"""

import enum
import os
import time
import queue
import threading
import warnings
from io import BytesIO
from typing import Dict, Optional, Type, List, Callable, Any, Tuple
import h5py
try:
    import zarr
except:
    pass
import numpy as np
import tifffile as tiff
import cv2


from imswitch.imcommon.framework import Signal, SignalInterface, Thread, Worker
from imswitch.imcommon.model import initLogger
import abc
import logging

from imswitch.imcontrol.model.managers.DetectorsManager import DetectorsManager

logger = logging.getLogger(__name__)
# Fallback to ome-zarr if vanilla implementation is not available
try:
    from ome_zarr.writer import write_multiscales_metadata # TODO: This fails with newer numpy versions!
    from ome_zarr.format import format_from_version
    IS_OME_ZARR = True
except ImportError:
    IS_OME_ZARR = False


# =============================================================================
# Background Storage Queue - Asynchronous File I/O
# =============================================================================

class StorageTask:
    """
    A task to be executed by the background storage worker.
    
    Encapsulates all data needed for a file I/O operation.
    """
    def __init__(self, 
                 task_type: str,
                 filepath: str,
                 data: Any = None,
                 attrs: Dict[str, Any] = None,
                 callback: Callable[[bool, str], None] = None,
                 priority: int = 0):
        """
        Args:
            task_type: Type of task ('snap', 'append', 'finalize')
            filepath: Target file path
            data: Image data (numpy array or dict of arrays)
            attrs: Metadata attributes
            callback: Optional callback(success: bool, message: str)
            priority: Task priority (lower = higher priority)
        """
        self.task_type = task_type
        self.filepath = filepath
        self.data = data
        self.attrs = attrs
        self.callback = callback
        self.priority = priority
        self.timestamp = time.time()
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)


class BackgroundStorageWorker:
    """
    Background worker that handles file I/O operations asynchronously.
    
    Uses a priority queue to manage storage tasks without blocking
    the main acquisition thread. This ensures that image acquisition
    continues smoothly while files are being written.
    
    Features:
    - Priority-based task queue
    - Non-blocking snap/append operations
    - Automatic error handling with callbacks
    - Graceful shutdown with queue drain
    """
    
    def __init__(self, max_queue_size: int = 100):
        """
        Args:
            max_queue_size: Maximum number of pending tasks (0 = unlimited)
        """
        self._logger = initLogger(self)
        self._task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._pending_tasks = 0
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
        """
        Stop the background worker.
        
        Args:
            wait: If True, wait for pending tasks to complete
            timeout: Maximum time to wait for shutdown (seconds)
        """
        if not self._is_running:
            return
        
        self._stop_event.set()
        
        if wait and self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        
        self._is_running = False
        self._logger.info(f"BackgroundStorageWorker stopped. Completed: {self._tasks_completed}, Failed: {self._tasks_failed}")
    
    def submit_task(self, task: StorageTask) -> bool:
        """
        Submit a storage task to the queue.
        
        Args:
            task: StorageTask to execute
            
        Returns:
            True if task was queued, False if queue is full
        """
        if not self._is_running:
            self._logger.warning("Cannot submit task: worker not running")
            return False
        
        try:
            self._task_queue.put_nowait(task)
            with self._lock:
                self._pending_tasks += 1
            return True
        except queue.Full:
            self._logger.warning("Storage queue is full, task dropped")
            if task.callback:
                task.callback(False, "Queue full")
            return False
    
    def get_queue_size(self) -> int:
        """Get current number of pending tasks."""
        with self._lock:
            return self._pending_tasks
    
    def _worker_loop(self):
        """Main worker loop - processes tasks from queue."""
        while not self._stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop event
                task = self._task_queue.get(timeout=0.1)
                
                try:
                    self._execute_task(task)
                    with self._lock:
                        self._pending_tasks -= 1
                        self._tasks_completed += 1
                except Exception as e:
                    self._logger.error(f"Task execution failed: {e}")
                    with self._lock:
                        self._pending_tasks -= 1
                        self._tasks_failed += 1
                    if task.callback:
                        task.callback(False, str(e))
                finally:
                    self._task_queue.task_done()
                    
            except queue.Empty:
                continue
        
        # Drain remaining tasks on shutdown
        while not self._task_queue.empty():
            try:
                task = self._task_queue.get_nowait()
                self._execute_task(task)
                self._task_queue.task_done()
            except:
                break
    
    def _execute_task(self, task: StorageTask):
        """Execute a single storage task."""
        success = True
        message = "OK"
        
        try:
            if task.task_type == 'snap_tiff':
                self._snap_tiff(task.filepath, task.data, task.attrs)
            elif task.task_type == 'snap_png':
                self._snap_png(task.filepath, task.data)
            elif task.task_type == 'snap_jpg':
                self._snap_jpg(task.filepath, task.data)
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
    
    def _snap_tiff(self, filepath: str, data: np.ndarray, attrs: Dict[str, Any] = None):
        """Write TIFF file with optional OME metadata."""
        if attrs:
            tiff.imwrite(filepath, data, metadata=attrs, imagej=False)
        else:
            tiff.imwrite(filepath, data)
        self._logger.debug(f"Saved TIFF: {filepath}")
    
    def _snap_png(self, filepath: str, data: np.ndarray):
        """Write PNG file."""
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = cv2.convertScaleAbs(data)
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(filepath, data)
        self._logger.debug(f"Saved PNG: {filepath}")
    
    def _snap_jpg(self, filepath: str, data: np.ndarray):
        """Write JPEG file."""
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(filepath, data)
        self._logger.debug(f"Saved JPG: {filepath}")
    
    def _append_tiff(self, filepath: str, data: np.ndarray):
        """Append to existing TIFF file."""
        tiff.imwrite(filepath, data, append=True) # TODO: Add metadata
        self._logger.debug(f"Appended to TIFF: {filepath}")

    

# Global background storage worker instance
_background_storage_worker: Optional[BackgroundStorageWorker] = None


def get_background_storage_worker() -> BackgroundStorageWorker:
    """Get or create the global background storage worker."""
    global _background_storage_worker
    if _background_storage_worker is None:
        _background_storage_worker = BackgroundStorageWorker()
        _background_storage_worker.start()
    return _background_storage_worker


def shutdown_background_storage():
    """Shutdown the global background storage worker."""
    global _background_storage_worker
    if _background_storage_worker is not None:
        _background_storage_worker.stop(wait=True)
        _background_storage_worker = None


def _create_zarr_store(path):
    """
    Create a Zarr store compatible with both Zarr 2.x and 3.x
    
    Args:
        path: Path to the store
        
    Returns:
        Store object compatible with current Zarr version
    """
    if hasattr(zarr.storage, 'DirectoryStore'):
        # Zarr 2.x compatibility
        return zarr.storage.DirectoryStore(path)
    elif hasattr(zarr.storage, 'LocalStore'):
        # Zarr 3.x with LocalStore
        return zarr.storage.LocalStore(path)
    else:
        # Zarr 3.x with direct path usage
        return path


def _build_snap_metadata(detector_name: str, image: np.ndarray, 
                         attrs: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """
    Build OME-TIFF compatible metadata from attributes.
    
    This is a shared function used by both _snap_background and TiffStorer.
    For new code, prefer using SnapService from imswitch.imcontrol.model.io
    which has the same logic.
    
    Args:
        detector_name: Name of the detector
        image: Image array
        attrs: Dictionary of metadata attributes
        
    Returns:
        Dictionary of OME metadata or None
    """
    if not attrs:
        return None
    
    metadata = {}
    
    def _get_value(val):
        """Extract value from SharedAttrValue or return raw value."""
        return val.value if hasattr(val, 'value') else val
    
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
        import datetime
        
        # Detector info
        metadata['Detector'] = detector_name
        
        # Pixel size
        pixel_size = _search_attr([
            f'Detector:{detector_name}:PixelSizeUm',
            'Detector:PixelSizeUm',
            'PixelSizeUm'
        ], attrs)
        if pixel_size:
            metadata['PhysicalSizeX'] = float(pixel_size)
            metadata['PhysicalSizeY'] = float(pixel_size)
            metadata['PhysicalSizeXUnit'] = 'µm'
            metadata['PhysicalSizeYUnit'] = 'µm'
        
        # Exposure
        exposure = _search_attr([
            f'Detector:{detector_name}:ExposureMs',
            'Detector:ExposureMs',
            'ExposureMs'
        ], attrs)
        if exposure:
            metadata['ExposureTime'] = float(exposure) / 1000.0
            metadata['ExposureTimeUnit'] = 's'
        
        # Stage positions
        for axis in ['X', 'Y', 'Z']:
            for key, val in attrs.items():
                key_str = str(key)
                if 'Positioner:' in key_str and f':{axis}:Position' in key_str:
                    metadata[f'Position{axis}'] = float(_get_value(val))
                    metadata[f'Position{axis}Unit'] = 'µm'
                    break
        
        # Active lasers
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
            if is_enabled and value and float(value) > 0:
                active_lasers.append({
                    'Name': laser_name,
                    'WavelengthNm': float(wavelength) if wavelength else None,
                    'Power': float(value),
                })
        
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
        logger.warning(f"Error building snap metadata: {e}")
        return None


# =============================================================================
# DEPRECATED: Legacy Storer classes - Kept for backwards compatibility only
# These are replaced by the unified io/writers and io/snap_service modules.
# =============================================================================

class Storer(abc.ABC):
    """
    DEPRECATED: Base class for storing data.
    Use imswitch.imcontrol.model.io.SnapService for snap operations.
    """
    def __init__(self, filepath, detectorManager):
        import warnings
        warnings.warn(
            "Storer classes are deprecated. Use model.io.SnapService instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.filepath = filepath
        self.detectorManager: DetectorsManager = detectorManager

    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        raise NotImplementedError

    def stream(self, data=None, **kwargs):
        raise NotImplementedError




class TiffStorer(Storer):
    """
    DEPRECATED: A storer that stores the images in TIFF files with OME metadata.
    Use imswitch.imcontrol.model.io.SnapService for new code.
    """
    
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        for channel, image in images.items():
            path = f'{self.filepath}_{channel}.ome.tiff'
            if not hasattr(image, "shape"):
                logger.error(f"Could not save image to tiff file {path}")
                continue
            
            try:
                # Use the shared metadata building function
                ome_metadata = _build_snap_metadata(channel, image, attrs)
                
                if ome_metadata:
                    tiff.imwrite(path, image, metadata=ome_metadata, imagej=False)
                else:
                    tiff.imwrite(path, image)
                
                logger.info(f"Saved image to tiff file {path}")
            except Exception as e:
                logger.error(f"Error saving tiff file {path}: {e}")
                tiff.imwrite(path, image)


class PNGStorer(Storer):
    """
    DEPRECATED: A storer that stores images in PNG format.
    Use imswitch.imcontrol.model.io.SnapService for new code.
    """
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        for channel, image in images.items():
            path = f'{self.filepath}_{channel}.png'
            # if image is BW only, we have to convert it to RGB
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = cv2.convertScaleAbs(image)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(path, image)
            del image
            logger.info(f"Saved image to png file {path}")


class JPGStorer(Storer):
    """
    DEPRECATED: A storer that stores images in JPG format.
    Use imswitch.imcontrol.model.io.SnapService for new code.
    """
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        for channel, image in images.items():
            path = f'{self.filepath}_{channel}.jpg'
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(path, image)
            logger.info(f"Saved image to jpg file {path}")


class MP4Storer(Storer):
    """
    DEPRECATED: A storer that writes frames to MP4 format.
    Video recording should use a dedicated video writer module.
    """
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        # not yet implemented
        pass


# ZarrStorer is deprecated - use io/writers/OMEZarrWriter instead
class ZarrStorer(Storer):
    """
    DEPRECATED: A storer that stores images in Zarr format.
    Use imswitch.imcontrol.model.io.OMEZarrWriter for new code.
    """
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        if not IS_OME_ZARR:
            logger.error("OME Zarr is not installed. Please install ome-zarr.")
            return
        
        path = f'{self.filepath}.zarr'
        try:
            datasets: List[dict] = []
            store = _create_zarr_store(path)
            root = zarr.group(store=store)

            for channel, image in images.items():
                shape = self.detectorManager[channel].shape
                root.create_dataset(channel, data=image, shape=tuple(reversed(shape)),
                                        chunks=(512, 512), dtype='i2')
                datasets.append({"path": channel, "transformation": None})
            
            metadata_kwargs = attrs if attrs else {}
            write_multiscales_metadata(root, datasets, format_from_version("0.2"), shape, **metadata_kwargs)
            logger.info(f"Saved image to zarr file {path}")
        except Exception as e:
            logger.error(f"Error saving zarr file {path}: {e}")


class SaveMode(enum.Enum):
    """
    DEPRECATED: Use imswitch.imcontrol.model.io.SaveMode instead.
    """
    Disk = 1
    RAM = 2
    DiskAndRAM = 3
    Numpy = 4


class SaveFormat(enum.Enum):
    """
    DEPRECATED: Use imswitch.imcontrol.model.io.SaveFormat instead.
    """
    TIFF = 1
    ZARR = 3
    MP4 = 4
    PNG = 5
    JPG = 6


# DEPRECATED: Use io/writers directly
DEFAULT_STORER_MAP: Dict[str, Type[Storer]] = {
    SaveFormat.ZARR: ZarrStorer,
    SaveFormat.TIFF: TiffStorer,
    SaveFormat.MP4: MP4Storer,
    SaveFormat.PNG: PNGStorer,
    SaveFormat.JPG: JPGStorer
}


class RecordingManager(SignalInterface):
    """
    DEPRECATED: Use RecordingService from imswitch.imcontrol.model.io instead.
    
    This class is kept for backwards compatibility only.
    For new code, use:
    
        from imswitch.imcontrol.model.io import RecordingService
        service = RecordingService(detectors_manager)
        service.snap(format=SaveFormat.TIFF)
    """

    sigRecordingStarted = Signal()
    sigRecordingEnded = Signal()
    sigRecordingFrameNumUpdated = Signal(int)  # (frameNumber)
    sigRecordingTimeUpdated = Signal(int)  # (recTime)
    sigMemorySnapAvailable = Signal(
        str, np.ndarray, object, bool
    )  # (name, image, filePath, savedToDisk)
    sigMemoryRecordingAvailable = Signal(
        str, object, object, bool
    )  # (name, file, filePath, savedToDisk)

    def __init__(self, detectorsManager, storerMap: Optional[Dict[str, Type[Storer]]] = None):
        super().__init__()
        self.__logger = initLogger(self)
        self.__storerMap = storerMap or DEFAULT_STORER_MAP
        self._memRecordings = {}  # { filePath: bytesIO }
        self.__detectorsManager = detectorsManager
        self.__record = False

        if 1: #not IS_HEADLESS: # TODO: Merge the two RecordingWorkers
            self._thread = Thread()
            self.__recordingWorker = RecordingWorker(self)
            self.__recordingWorker.moveToThread(self._thread)
            self._thread.started.connect(self.__recordingWorker.run)
        else:
            self.__recordingWorker = RecordingWorkerNoQt(self)
            self._thread = Thread(target=self.__recordingWorker.run)

    def __del__(self):
        self.endRecording(emitSignal=False, wait=True)
        # Wait for any pending background I/O to complete
        self.wait_for_io_complete(timeout=10.0)
        if hasattr(super(), '__del__'):
            super().__del__()

    @property
    def record(self):
        """ Whether a recording is currently being recorded. """
        return self.__record

    @property
    def detectorsManager(self):
        return self.__detectorsManager

    def startRecording(self, detectorNames, recMode, savename, saveMode, attrs,
                       saveFormat=SaveFormat.TIFF, singleMultiDetectorFile=False, singleLapseFile=False,
                       recFrames=None, recTime=None):
        """ Starts a recording with the specified detectors, recording mode,
        file name prefix and attributes to save to the recording per detector.
        In SpecFrames mode, recFrames (the number of frames) must be specified,
        and in SpecTime mode, recTime (the recording time in seconds) must be
        specified. """
        # TODO: This is not used in most cases other than recording MP4, so I guess it would be wise to entirely remove this part and create a new way for saving videos by copying the existing implementation into a new place; Also we want to merge it with the 
        self.__logger.info('Starting recording')
        self.__record = True
        self.__recordingWorker.detectorNames = detectorNames
        self.__recordingWorker.recMode = recMode
        self.__recordingWorker.savename = savename
        self.__recordingWorker.saveMode = saveMode
        self.__recordingWorker.saveFormat = saveFormat
        self.__recordingWorker.attrs = attrs
        self.__recordingWorker.recFrames = recFrames
        self.__recordingWorker.recTime = recTime
        self.__recordingWorker.singleMultiDetectorFile = singleMultiDetectorFile
        self.__recordingWorker.singleLapseFile = singleLapseFile
        self.__detectorsManager.execOnAll(lambda c: c.flushBuffers(),
                                          condition=lambda c: c.forAcquisition)
        if 0: #IS_HEADLESS:
            self._thread = Thread(target=self.__recordingWorker.run) # TODO: Merge the two RecordingWorkers
        self._thread.start()

    def endRecording(self, emitSignal=True, wait=True):
        """ Ends the current recording. Unless emitSignal is false, the
        sigRecordingEnded signal will be emitted. Unless wait is False, this
        method will wait until the recording is complete before returning. """

        self.__detectorsManager.execOnAll(lambda c: c.flushBuffers(),
                                          condition=lambda c: c.forAcquisition)

        if self.__record:
            self.__logger.info('Stopping recording')
        self.__record = False
        self._thread.quit()
        if emitSignal:
            self.sigRecordingEnded.emit()
        if wait:
            self._thread.wait()

    def snap(self, detectorNames=None, savename="", saveMode=SaveMode.Disk, saveFormat=SaveFormat.TIFF, attrs=None,
             use_background_io: bool = True, io_callback: Callable[[bool, str], None] = None):
        """ 
        DEPRECATED: Use RecordingService.snap() from imswitch.imcontrol.model.io instead.
        
        Saves an image with the specified detectors to a file
        with the specified name prefix, save mode, file format and attributes
        to save to the capture per detector.
        
        Args:
            detectorNames: List of detector names to capture. If None, all detectors.
            savename: File path prefix for saving.
            saveMode: SaveMode.Disk, SaveMode.RAM, SaveMode.DiskAndRAM, or SaveMode.Numpy
            saveFormat: SaveFormat.TIFF, SaveFormat.PNG, SaveFormat.JPG, etc.
            attrs: Dictionary of metadata attributes to save.
            use_background_io: If True (default), use background queue for non-blocking I/O.
                              Set to False for synchronous writes (blocks until complete).
            io_callback: Optional callback(success: bool, message: str) called when
                        background I/O completes. Only used when use_background_io=True.
        """
        import warnings
        warnings.warn(
            "RecordingManager.snap() is deprecated. Use RecordingService.snap() from "
            "imswitch.imcontrol.model.io instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # TODO: Move to central io module using RecordingService.snap()
        if detectorNames is None:
            detectorNames = self.__detectorsManager.detectorNames

        try:
            images = {}

            # Acquire data
            for detectorName in detectorNames:
                images[detectorName] = self.__detectorsManager[detectorName].getLatestFrame()
                image = images[detectorName]

            if saveFormat:
                if saveMode == SaveMode.Disk or saveMode == SaveMode.DiskAndRAM:
                    if use_background_io:
                        # Use background queue for non-blocking I/O
                        self._snap_background(images, savename, saveFormat, attrs, io_callback)
                    else:
                        # Synchronous write (original behavior)
                        storer = self.__storerMap[saveFormat]
                        store = storer(savename, self.__detectorsManager)
                        store.snap(images, attrs)

                if saveMode == SaveMode.RAM or saveMode == SaveMode.DiskAndRAM:
                    for channel, image in images.items():
                        name = os.path.basename(f'{savename}_{channel}')
                        self.sigMemorySnapAvailable.emit(name, image, savename, saveMode == SaveMode.DiskAndRAM)

        except Exception as e:
            self.__logger.error(f'Failed to snap image: {e}')

        finally:
            # self.__detectorsManager.stopAcquisition(acqHandle)
            if saveMode == SaveMode.Numpy:
                return images

    def _snap_background(self, images: Dict[str, np.ndarray], savename: str, 
                         saveFormat: SaveFormat, attrs: Dict[str, str] = None,
                         callback: Callable[[bool, str], None] = None):
        """
        Queue images for background saving using BackgroundStorageWorker.
        
        This method is non-blocking - it submits tasks to the background
        storage worker and returns immediately.
        
        Note: For new code, prefer using SnapService from imswitch.imcontrol.model.io
        """
        worker = get_background_storage_worker()
        
        for channel, image in images.items():
            # Build OME metadata using the shared metadata building function
            ome_attrs = _build_snap_metadata(channel, image, attrs)
            
            # Determine task type and filepath based on format
            if saveFormat == SaveFormat.TIFF:
                task_type = 'snap_tiff'
                filepath = f'{savename}_{channel}.tiff'
                task_attrs = ome_attrs
            elif saveFormat == SaveFormat.PNG:
                task_type = 'snap_png'
                filepath = f'{savename}_{channel}.png'
                task_attrs = None  # PNG doesn't support metadata
            elif saveFormat == SaveFormat.JPG:
                task_type = 'snap_jpg'
                filepath = f'{savename}_{channel}.jpg'
                task_attrs = None  # JPG doesn't support metadata
            else:
                # Unsupported format for background I/O, fall back to sync
                self.__logger.warning(f"Format {saveFormat} not supported for background I/O, using sync write")
                storer = self.__storerMap[saveFormat]
                store = storer(savename, self.__detectorsManager)
                store.snap({channel: image}, attrs)
                continue
            
            # Make a copy of the image data for thread safety
            image_copy = image.copy()
            
            task = StorageTask(
                task_type=task_type,
                filepath=filepath,
                data=image_copy,
                attrs=task_attrs,
                callback=callback,
                priority=0  # Normal priority
            )
            
            if not worker.submit_task(task):
                self.__logger.warning(f"Failed to queue storage task for {filepath}")
    
    def get_pending_io_count(self) -> int:
        """
        Get the number of pending background I/O operations.
        
        Useful for checking if all writes have completed before
        ending an experiment or closing the application.
        """
        try:
            worker = get_background_storage_worker()
            return worker.get_queue_size()
        except:
            return 0
    
    def wait_for_io_complete(self, timeout: float = 30.0) -> bool:
        """
        Wait for all pending background I/O operations to complete.
        
        Args:
            timeout: Maximum time to wait in seconds.
            
        Returns:
            True if all I/O completed, False if timeout reached.
        """
        start = time.time()
        while self.get_pending_io_count() > 0:
            if time.time() - start > timeout:
                self.__logger.warning(f"Timeout waiting for I/O completion, {self.get_pending_io_count()} tasks pending")
                return False
            time.sleep(0.1)
        return True


    def snapImagePrev(self, detectorName, savename, saveFormat, image, attrs):
        """ Saves a previously taken image to a file with the specified name prefix,
        file format and attributes to save to the capture per detector. """
        fileExtension = str(saveFormat.name).lower()
        filePath = self.getSaveFilePath(f'{savename}_{detectorName}.{fileExtension}')

        if saveFormat == SaveFormat.TIFF:
            tiff.imwrite(filePath, image)
        elif saveFormat == SaveFormat.PNG:
            cv2.imwrite(filePath, image)
        elif saveFormat == SaveFormat.JPG:
            cv2.imwrite(filePath, image)
        elif saveFormat == SaveFormat.ZARR:
            if not IS_OME_ZARR:
                logger.error("OME Zarr is not installed. Please install ome-zarr.")
                return
            path = self.getSaveFilePath(f'{savename}.{fileExtension}')
            store = _create_zarr_store(path)
            root = zarr.group(store=store)
            shape = self.__detectorsManager[detectorName].shape
            d = root.create_dataset(detectorName, data=image, shape=tuple(reversed(shape)), chunks=(512, 512),
                                    dtype='i2')
            datasets = {"path": detectorName, "transformation": None}
            write_multiscales_metadata(root, datasets, format_from_version("0.2"), shape, **attrs)
            store.close()
        else:
            raise ValueError(f'Unsupported save format "{saveFormat}"')

    def getSaveFilePath(self, path, allowOverwriteDisk=False, allowOverwriteMem=False):
        newPath = path
        numExisting = 0

        def existsFunc(pathToCheck):
            if not allowOverwriteDisk and os.path.exists(pathToCheck):
                return True
            if not allowOverwriteMem and pathToCheck in self._memRecordings:
                return True
            return False

        while existsFunc(newPath):
            numExisting += 1
            pathWithoutExt, pathExt = os.path.splitext(path)
            newPath = f'{pathWithoutExt}_{numExisting}{pathExt}'
        return newPath




class RecordingWorker(Worker):
    def __init__(self, recordingManager):
        super().__init__()
        self.__logger = initLogger(self)
        self.__recordingManager = recordingManager
        self.__logger = initLogger(self)

    def run(self):
        acqHandle = self.__recordingManager.detectorsManager.startAcquisition()
        try:
            self._record()

        finally:
            self.__recordingManager.detectorsManager.stopAcquisition(acqHandle)

    def _record(self):
        if self.saveFormat == SaveFormat.ZARR:
            files, fileDests, filePaths = self._getFiles()

        shapes = {detectorName: self.__recordingManager.detectorsManager[detectorName].shape
                  for detectorName in self.detectorNames}

        currentFrame = {}
        datasets = {}
        filenames = {}

        for detectorName in self.detectorNames:
            currentFrame[detectorName] = 0

            datasetName = detectorName
            if self.recMode == RecMode.ScanLapse and self.singleLapseFile:
                # Add scan number to dataset name
                scanNum = 0
                datasetNameWithScan = f'{datasetName}_scan{scanNum}'
                while datasetNameWithScan in files[detectorName]:
                    scanNum += 1
                    datasetNameWithScan = f'{datasetName}_scan{scanNum}'
                datasetName = datasetNameWithScan

            # Initial number of frames must not be 0; otherwise, too much disk space may get
            # allocated. We remove this default frame later on if no frames are captured.
            shape = shapes[detectorName]
            if len(shape) > 2:
                shape = shape[-2:]

            if self.saveFormat == SaveFormat.TIFF:
                fileExtension = str(self.saveFormat.name).lower()
                filenames[detectorName] = self.__recordingManager.getSaveFilePath(
                    f'{self.savename}_{detectorName}.{fileExtension}', False, False)

            elif self.saveFormat == SaveFormat.ZARR:
                if not IS_OME_ZARR:
                    logger.error("OME Zarr is not installed. Please install ome-zarr.")
                    return
                datasets[detectorName] = files[detectorName].create_dataset(datasetName, shape=(1, *reversed(shape)),
                                                                            dtype='i2', chunks=(1, 512, 512)
                                                                            )
                datasets[detectorName].attrs['detector_name'] = detectorName
                # For ImageJ compatibility
                datasets[detectorName].attrs['element_size_um'] \
                    = self.__recordingManager.detectorsManager[detectorName].pixelSizeUm
                datasets[detectorName].attrs['writing'] = True
                info: List[dict] = [{"path": datasetName, "transformation": None}]
                write_multiscales_metadata(files[detectorName], info, format_from_version("0.2"), shape, **self.attrs[detectorName])

        self.__recordingManager.sigRecordingStarted.emit()
        try:
            if len(self.detectorNames) < 1:
                raise ValueError('No detectors to record specified')

            if self.recMode in [RecMode.SpecFrames, RecMode.ScanOnce, RecMode.ScanLapse]:
                recFrames = self.recFrames
                if recFrames is None:
                    raise ValueError('recFrames must be specified in SpecFrames, ScanOnce or'
                                     ' ScanLapse mode')

                while (self.__recordingManager.record and
                       any([currentFrame[detectorName] < recFrames
                            for detectorName in self.detectorNames])):
                    for detectorName in self.detectorNames:
                        if currentFrame[detectorName] >= recFrames:
                            continue  # Reached requested number of frames with this detector, skip

                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)

                        if n > 0:
                            it = currentFrame[detectorName]
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue
                            elif self.saveFormat == SaveFormat.ZARR:
                                dataset = datasets[detectorName]
                                if it == 0:
                                    dataset[0, :, :] = newFrames[0, :, :]
                                    if n > 0:
                                        dataset.append(newFrames[1:n, :, :])
                                else:
                                    dataset.append(newFrames)
                                currentFrame[detectorName] += n

                            # Things get a bit weird if we have multiple detectors when we report
                            # the current frame number, since the detectors may not be synchronized.
                            # For now, we will report the lowest number.
                            self.__recordingManager.sigRecordingFrameNumUpdated.emit(
                                min(list(currentFrame.values()))
                            )
                    time.sleep(0.0001)  # Prevents freezing for some reason

                self.__recordingManager.sigRecordingFrameNumUpdated.emit(0)
            elif self.recMode == RecMode.SpecTime:
                recTime = self.recTime
                if recTime is None:
                    raise ValueError('recTime must be specified in SpecTime mode')

                start = time.time()
                currentRecTime = 0
                shouldStop = False
                while True:
                    for detectorName in self.detectorNames:
                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)
                        if n > 0:
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue
                            elif  self.saveFormat == SaveFormat.ZARR:
                                it = currentFrame[detectorName]
                                dataset = datasets[detectorName]
                                dataset.resize(n + it, axis=0)
                                dataset[it:it + n, :, :] = newFrames
                            currentFrame[detectorName] += n
                            self.__recordingManager.sigRecordingTimeUpdated.emit(
                                np.around(currentRecTime, decimals=2)
                            )
                            currentRecTime = time.time() - start

                    if shouldStop:
                        break  # Enter loop one final time, then stop

                    if not self.__recordingManager.record or currentRecTime >= recTime:
                        shouldStop = True

                    time.sleep(0.0001)  # Prevents freezing for some reason

                self.__recordingManager.sigRecordingTimeUpdated.emit(0)
            elif self.recMode == RecMode.UntilStop:
                shouldStop = False
                while True:
                    for detectorName in self.detectorNames:
                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)
                        if n > 0:
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue

                            elif self.saveFormat == SaveFormat.ZARR:
                                it = currentFrame[detectorName]
                                dataset = datasets[detectorName]
                                if it == 0:
                                    dataset[0, :, :] = newFrames[0, :, :]
                                    if n > 0:
                                        dataset.append(newFrames[1:n, :, :])
                                else:
                                    dataset.append(newFrames)

                            currentFrame[detectorName] += n

                    if shouldStop:
                        break

                    if not self.__recordingManager.record:
                        shouldStop = True  # Enter loop one final time, then stop

                    time.sleep(0.0001)  # Prevents freezing for some reason
            else:
                raise ValueError('Unsupported recording mode specified')
        finally:

            if self.saveFormat == SaveFormat.ZARR:
                for detectorName, file in files.items():
                    # Remove default frame if no frames have been captured
                    if self.saveMode == SaveMode.RAM or self.saveMode == SaveMode.DiskAndRAM:
                        filePath = filePaths[detectorName]
                        name = os.path.basename(filePath)
                        if self.saveMode == SaveMode.RAM:
                            file.close()
                            self.__recordingManager.sigMemoryRecordingAvailable.emit(
                                name, fileDests[detectorName], filePath, False
                            )
                        else:
                            file.flush()
                            self.__recordingManager.sigMemoryRecordingAvailable.emit(
                                name, file, filePath, True
                            )
                    else:
                        datasets[detectorName].attrs['writing'] = False
                        self.store.close()
            emitSignal = True
            if self.recMode in [RecMode.SpecFrames, RecMode.ScanOnce, RecMode.ScanLapse]:
                emitSignal = False
            self.__recordingManager.endRecording(emitSignal=emitSignal, wait=False)

    def _getFiles(self):
        singleMultiDetectorFile = self.singleMultiDetectorFile
        singleLapseFile = self.recMode == RecMode.ScanLapse and self.singleLapseFile

        files = {}
        fileDests = {}
        filePaths = {}
        extension = 'zarr'

        for detectorName in self.detectorNames:
            if singleMultiDetectorFile:
                baseFilePath = f'{self.savename}.{extension}'
            else:
                baseFilePath = f'{self.savename}_{detectorName}.{extension}'

            filePaths[detectorName] = self.__recordingManager.getSaveFilePath(
                baseFilePath,
                allowOverwriteDisk=singleLapseFile and self.saveMode != SaveMode.RAM,
                allowOverwriteMem=singleLapseFile and self.saveMode == SaveMode.RAM
            )

        for detectorName in self.detectorNames:
            if self.saveMode == SaveMode.RAM:
                memRecordings = self.__recordingManager._memRecordings
                if (filePaths[detectorName] not in memRecordings or
                        memRecordings[filePaths[detectorName]].closed):
                    memRecordings[filePaths[detectorName]] = BytesIO()
                fileDests[detectorName] = memRecordings[filePaths[detectorName]]
            else:
                fileDests[detectorName] = filePaths[detectorName]

            if singleMultiDetectorFile and len(files) > 0:
                files[detectorName] = list(files.values())[0]
            else:
                if  self.saveFormat == SaveFormat.ZARR:
                    self.store = _create_zarr_store(fileDests[detectorName])
                    files[detectorName] = zarr.group(store=self.store, overwrite=True)

        return files, fileDests, filePaths

    def _getNewFrames(self, detectorName):
        newFrames, frameIndices = self.__recordingManager.detectorsManager[detectorName].getChunk()
        newFrames = np.array(newFrames)
        return newFrames

class RecordingWorkerNoQt(Worker):
    def __init__(self, recordingManager):
        super().__init__()
        self.__logger = initLogger(self)
        self.__recordingManager = recordingManager
        self.__logger = initLogger(self)

    def run(self):
        self.__logger.info('Recording worker NoQT started')
        acqHandle = self.__recordingManager.detectorsManager.startAcquisition()
        try:
            self._record()

        finally:
            self.__recordingManager.detectorsManager.stopAcquisition(acqHandle)

    def moveToThread(self, thread) -> None:
        return super().moveToThread(thread)

    def _record(self):
        self.__logger.info('Recording started in mode: ' + str(self.recMode))
        if self.saveFormat == SaveFormat.ZARR:
            files, fileDests, filePaths = self._getFiles()

        shapes = {detectorName: self.__recordingManager.detectorsManager[detectorName].shape
                  for detectorName in self.detectorNames}

        currentFrame = {}
        datasets = {}
        filenames = {}

        for detectorName in self.detectorNames:
            currentFrame[detectorName] = 0

            datasetName = detectorName
            if self.recMode == RecMode.ScanLapse and self.singleLapseFile:
                # Add scan number to dataset name
                scanNum = 0
                datasetNameWithScan = f'{datasetName}_scan{scanNum}'
                while datasetNameWithScan in files[detectorName]:
                    scanNum += 1
                    datasetNameWithScan = f'{datasetName}_scan{scanNum}'
                datasetName = datasetNameWithScan

            # Initial number of frames must not be 0; otherwise, too much disk space may get
            # allocated. We remove this default frame later on if no frames are captured.
            shape = shapes[detectorName]
            if len(shape) > 2:
                shape = shape[-2:]

            if self.saveFormat == SaveFormat.MP4:
                # Need to initiliaze videowriter for each detector
                self.__logger.debug("Initialize MP4 recorder")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fileExtension = str(self.saveFormat.name).lower()
                filePath = self.__recordingManager.getSaveFilePath(f'{self.savename}_{detectorName}.{fileExtension}')
                self.__logger.debug("Saving Video to file: " + filePath)
                filenames[detectorName] = filePath
                datasets[detectorName] = cv2.VideoWriter(filePath, fourcc, 20.0, shapes[detectorName])
                #datasets[detectorName] = cv2.VideoWriter(filePath, cv2.VideoWriter_fourcc(*'MJPG'), 10, shapes[detectorName])


            elif self.saveFormat == SaveFormat.TIFF:
                fileExtension = str(self.saveFormat.name).lower()
                filenames[detectorName] = self.__recordingManager.getSaveFilePath(
                    f'{self.savename}_{detectorName}.{fileExtension}', False, False)

            elif self.saveFormat == SaveFormat.PNG:
                fileExtension = str(self.saveFormat.name).lower()
                filenames[detectorName] = self.__recordingManager.getSaveFilePath(
                    f'{self.savename}_{detectorName}.{fileExtension}', False, False)

            elif self.saveFormat == SaveFormat.JPG:
                fileExtension = str(self.saveFormat.name).lower()
                filenames[detectorName] = self.__recordingManager.getSaveFilePath(
                    f'{self.savename}_{detectorName}.{fileExtension}', False, False)

            elif self.saveFormat == SaveFormat.ZARR:
                if not IS_OME_ZARR:
                    logger.error("OME Zarr is not installed. Please install ome-zarr.")
                    return
                datasets[detectorName] = files[detectorName].create_dataset(datasetName, shape=(1, *reversed(shape)),
                                                                            dtype='i2', chunks=(1, 512, 512)
                                                                            )
                datasets[detectorName].attrs['detector_name'] = detectorName
                # For ImageJ compatibility
                datasets[detectorName].attrs['element_size_um'] \
                    = self.__recordingManager.detectorsManager[detectorName].pixelSizeUm
                datasets[detectorName].attrs['writing'] = True
                info: List[dict] = [{"path": datasetName, "transformation": None}]
                write_multiscales_metadata(files[detectorName], info, format_from_version("0.2"), shape, **self.attrs[detectorName])


        self.__recordingManager.sigRecordingStarted.emit()
        try:
            if len(self.detectorNames) < 1:
                raise ValueError('No detectors to record specified')

            if self.recMode in [RecMode.SpecFrames, RecMode.ScanOnce, RecMode.ScanLapse]:
                recFrames = self.recFrames
                if recFrames is None:
                    raise ValueError('recFrames must be specified in SpecFrames, ScanOnce or'
                                     ' ScanLapse mode')

                while (self.__recordingManager.record and
                       any([currentFrame[detectorName] < recFrames
                            for detectorName in self.detectorNames])):
                    for detectorName in self.detectorNames:
                        if currentFrame[detectorName] >= recFrames:
                            continue  # Reached requested number of frames with this detector, skip

                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)

                        if n > 0:
                            it = currentFrame[detectorName]
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue

                            elif self.saveFormat == SaveFormat.ZARR:
                                dataset = datasets[detectorName]
                                if it == 0:
                                    dataset[0, :, :] = newFrames[0, :, :]
                                    if n > 0:
                                        dataset.append(newFrames[1:n, :, :])
                                else:
                                    dataset.append(newFrames)
                                currentFrame[detectorName] += n
                            elif self.saveFormat == SaveFormat.MP4:
                                for iframe in range(n):
                                    frame = newFrames[iframe,:,:]
                                    #https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
                                    frame = cv2.cvtColor(cv2.convertScaleAbs(frame), cv2.COLOR_GRAY2BGR)
                                    self.__logger.debug(type(frame))

                                    datasets[detectorName].write(frame)


                            # Things get a bit weird if we have multiple detectors when we report
                            # the current frame number, since the detectors may not be synchronized.
                            # For now, we will report the lowest number.
                            self.__recordingManager.sigRecordingFrameNumUpdated.emit(
                                min(list(currentFrame.values()))
                            )
                    time.sleep(0.0001)  # Prevents freezing for some reason

                self.__recordingManager.sigRecordingFrameNumUpdated.emit(0)
            elif self.recMode == RecMode.SpecTime:
                recTime = self.recTime
                if recTime is None:
                    raise ValueError('recTime must be specified in SpecTime mode')

                start = time.time()
                currentRecTime = 0
                shouldStop = False
                while True:
                    for detectorName in self.detectorNames:
                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)
                        if n > 0:
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue
                            elif self.saveFormat == SaveFormat.ZARR:
                                it = currentFrame[detectorName]
                                dataset = datasets[detectorName]
                                dataset.resize(n + it, axis=0)
                                dataset[it:it + n, :, :] = newFrames
                            elif self.saveFormat == SaveFormat.MP4:
                                for iframe in range(n):
                                    frame = newFrames[iframe,:,:]
                                    #https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
                                    frame = cv2.cvtColor(cv2.convertScaleAbs(frame), cv2.COLOR_GRAY2BGR)
                                    self.__logger.debug(type(frame))

                                    datasets[detectorName].write(frame)

                            currentFrame[detectorName] += n
                            self.__recordingManager.sigRecordingTimeUpdated.emit(
                                np.around(currentRecTime, decimals=2)
                            )
                            currentRecTime = time.time() - start

                    if shouldStop:
                        break  # Enter loop one final time, then stop

                    if not self.__recordingManager.record or currentRecTime >= recTime:
                        shouldStop = True

                    time.sleep(0.0001)  # Prevents freezing for some reason

                self.__recordingManager.sigRecordingTimeUpdated.emit(0)
            elif self.recMode == RecMode.UntilStop:
                shouldStop = False
                while True:
                    for detectorName in self.detectorNames:
                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)
                        if n > 0:
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue

                            elif self.saveFormat == SaveFormat.ZARR:
                                it = currentFrame[detectorName]
                                dataset = datasets[detectorName]
                                if it == 0:
                                    dataset[0, :, :] = newFrames[0, :, :]
                                    if n > 0:
                                        dataset.append(newFrames[1:n, :, :])
                                else:
                                    dataset.append(newFrames)
                            elif self.saveFormat == SaveFormat.MP4:
                                for iframe in range(n):
                                    frame = newFrames[iframe,:,:]
                                    #https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
                                    frame = cv2.cvtColor(cv2.convertScaleAbs(frame), cv2.COLOR_GRAY2BGR)

                                    datasets[detectorName].write(frame)


                            currentFrame[detectorName] += n

                    if shouldStop:
                        break

                    if not self.__recordingManager.record:
                        shouldStop = True  # Enter loop one final time, then stop

                    time.sleep(0.0001)  # Prevents freezing for some reason
            else:
                raise ValueError('Unsupported recording mode specified')
        finally:

            if self.saveFormat == SaveFormat.ZARR:
                for detectorName, file in files.items():
                    # Remove default frame if no frames have been captured

                    # Handle memory recordings
                    if self.saveMode == SaveMode.RAM or self.saveMode == SaveMode.DiskAndRAM:
                        filePath = filePaths[detectorName]
                        name = os.path.basename(filePath)
                        if self.saveMode == SaveMode.RAM:
                            file.close()
                            self.__recordingManager.sigMemoryRecordingAvailable.emit(
                                name, fileDests[detectorName], filePath, False
                            )
                        else:
                            file.flush()
                            self.__recordingManager.sigMemoryRecordingAvailable.emit(
                                name, file, filePath, True
                            )
                    else:
                        datasets[detectorName].attrs['writing'] = False
                        if self.saveFormat == SaveFormat.MP4:
                            for detectorName, file in files.items():
                                datasets[detectorName].release()
                        else:
                            self.store.close()
            emitSignal = True
            if self.recMode in [RecMode.SpecFrames, RecMode.ScanOnce, RecMode.ScanLapse]:
                emitSignal = False
            self.__recordingManager.endRecording(emitSignal=emitSignal, wait=False)

    def _getFiles(self):
        singleMultiDetectorFile = self.singleMultiDetectorFile
        singleLapseFile = self.recMode == RecMode.ScanLapse and self.singleLapseFile

        files = {}
        fileDests = {}
        filePaths = {}
        extension = 'zarr'

        for detectorName in self.detectorNames:
            if singleMultiDetectorFile:
                baseFilePath = f'{self.savename}.{extension}'
            else:
                baseFilePath = f'{self.savename}_{detectorName}.{extension}'

            filePaths[detectorName] = self.__recordingManager.getSaveFilePath(
                baseFilePath,
                allowOverwriteDisk=singleLapseFile and self.saveMode != SaveMode.RAM,
                allowOverwriteMem=singleLapseFile and self.saveMode == SaveMode.RAM
            )

        for detectorName in self.detectorNames:
            if self.saveMode == SaveMode.RAM:
                memRecordings = self.__recordingManager._memRecordings
                if (filePaths[detectorName] not in memRecordings or
                        memRecordings[filePaths[detectorName]].closed):
                    memRecordings[filePaths[detectorName]] = BytesIO()
                fileDests[detectorName] = memRecordings[filePaths[detectorName]]
            else:
                fileDests[detectorName] = filePaths[detectorName]

            if singleMultiDetectorFile and len(files) > 0:
                files[detectorName] = list(files.values())[0]
            else:
                if self.saveFormat == SaveFormat.ZARR:
                    self.store = _create_zarr_store(fileDests[detectorName])
                    files[detectorName] = zarr.group(store=self.store, overwrite=True)

        return files, fileDests, filePaths

    def _getNewFrames(self, detectorName):
        newFrames = self.__recordingManager.detectorsManager[detectorName].getChunk()
        newFrames = np.array(newFrames)
        return newFrames


class RecMode(enum.Enum):
    SpecFrames = 1
    SpecTime = 2
    ScanOnce = 3
    ScanLapse = 4
    UntilStop = 5


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
