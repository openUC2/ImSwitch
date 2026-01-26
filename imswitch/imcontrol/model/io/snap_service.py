"""
Unified snap service for ImSwitch.

Provides centralized snap (single image capture) functionality that can be used
by RecordingController, API endpoints, and other components.

This replaces the scattered snap implementations in RecordingManager, TiffStorer, etc.
"""

import os
import time
import datetime
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import tifffile as tiff
except ImportError:
    tiff = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


@dataclass
class SnapResult:
    """Result of a snap operation."""
    success: bool
    detector_name: str
    filepath: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'detector_name': self.detector_name,
            'filepath': self.filepath,
            'error': self.error,
            'timestamp': self.timestamp,
        }


class SnapService:
    """
    Centralized snap service for single image capture.
    
    Features:
    - Synchronous and asynchronous snap operations
    - Multiple format support (TIFF, PNG, JPG)
    - OME-TIFF metadata generation
    - Background writing via queue
    """
    
    def __init__(self, detectors_manager=None, metadata_hub=None):
        """
        Initialize snap service.
        
        Args:
            detectors_manager: DetectorsManager for detector access
            metadata_hub: Optional MetadataHub for metadata
        """
        self._detectors_manager = detectors_manager
        self._metadata_hub = metadata_hub
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Background writing
        self._write_queue = queue.Queue(maxsize=100)
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._is_running = False
    
    def set_detectors_manager(self, detectors_manager):
        """Set or update the detectors manager."""
        self._detectors_manager = detectors_manager
    
    def set_metadata_hub(self, metadata_hub):
        """Set or update the metadata hub."""
        self._metadata_hub = metadata_hub
    
    def start_background_worker(self):
        """Start the background writing thread."""
        if self._is_running:
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._worker_thread.start()
        self._is_running = True
        self._logger.info("SnapService background worker started")
    
    def stop_background_worker(self, wait: bool = True, timeout: float = 5.0):
        """Stop the background writing thread."""
        if not self._is_running:
            return
        
        self._stop_event.set()
        if wait and self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        
        self._is_running = False
        self._logger.info("SnapService background worker stopped")
    
    def snap(self,
             detector_names: List[str] = None,
             savepath: str = "",
             format: str = "tiff",
             attrs: Dict[str, Any] = None,
             async_write: bool = True,
             callback: Callable[[SnapResult], None] = None) -> Dict[str, SnapResult]:
        """
        Capture and save images from detectors.
        
        Args:
            detector_names: List of detector names (None = all)
            savepath: Base path for saving (without extension)
            format: Output format ('tiff', 'png', 'jpg')
            attrs: Metadata attributes
            async_write: If True, write in background
            callback: Optional callback for async completion
            
        Returns:
            Dict mapping detector names to SnapResult
        """
        if self._detectors_manager is None:
            raise RuntimeError("DetectorsManager not set")
        
        if detector_names is None:
            detector_names = list(self._detectors_manager.detectorNames)
        
        results = {}
        
        # Capture images
        images = {}
        for det_name in detector_names:
            try:
                image = self._detectors_manager[det_name].getLatestFrame()
                if image is not None:
                    images[det_name] = image.copy()
            except Exception as e:
                self._logger.error(f"Failed to capture from {det_name}: {e}")
                results[det_name] = SnapResult(
                    success=False,
                    detector_name=det_name,
                    error=str(e)
                )
        
        # Save images
        for det_name, image in images.items():
            filepath = self._generate_filepath(savepath, det_name, format)
            metadata = self._build_metadata(det_name, image, attrs)
            
            if async_write and self._is_running:
                # Queue for background writing
                task = {
                    'detector_name': det_name,
                    'image': image,
                    'filepath': filepath,
                    'format': format,
                    'metadata': metadata,
                    'callback': callback,
                }
                try:
                    self._write_queue.put_nowait(task)
                    results[det_name] = SnapResult(
                        success=True,
                        detector_name=det_name,
                        filepath=filepath,
                    )
                except queue.Full:
                    self._logger.warning(f"Write queue full, writing synchronously")
                    result = self._write_image(det_name, image, filepath, format, metadata)
                    results[det_name] = result
            else:
                # Synchronous write
                result = self._write_image(det_name, image, filepath, format, metadata)
                results[det_name] = result
                if callback:
                    callback(result)
        
        return results
    
    def snap_numpy(self, detector_names: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Capture images and return as numpy arrays (no saving).
        
        Args:
            detector_names: List of detector names (None = all)
            
        Returns:
            Dict mapping detector names to image arrays
        """
        if self._detectors_manager is None:
            raise RuntimeError("DetectorsManager not set")
        
        if detector_names is None:
            detector_names = list(self._detectors_manager.detectorNames)
        
        images = {}
        for det_name in detector_names:
            try:
                image = self._detectors_manager[det_name].getLatestFrame()
                if image is not None:
                    images[det_name] = image.copy()
            except Exception as e:
                self._logger.error(f"Failed to capture from {det_name}: {e}")
        
        return images
    
    def _generate_filepath(self, basepath: str, detector_name: str, format: str) -> str:
        """Generate full filepath with extension."""
        ext_map = {
            'tiff': '.ome.tiff',
            'tif': '.ome.tiff',
            'png': '.png',
            'jpg': '.jpg',
            'jpeg': '.jpg',
        }
        ext = ext_map.get(format.lower(), '.tiff')
        return f"{basepath}_{detector_name}{ext}"
    
    def _build_metadata(self, detector_name: str, image: np.ndarray, 
                        attrs: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Build OME-TIFF compatible metadata from attributes.
        
        This consolidates the metadata building logic from TiffStorer._build_ome_metadata.
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
            self._logger.warning(f"Error building metadata: {e}")
            return None
    
    def _write_image(self, detector_name: str, image: np.ndarray,
                     filepath: str, format: str, 
                     metadata: Dict[str, Any] = None) -> SnapResult:
        """Write a single image to disk."""
        try:
            # Ensure directory exists
            dirname = os.path.dirname(filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            
            format_lower = format.lower()
            
            if format_lower in ('tiff', 'tif'):
                if tiff is None:
                    raise ImportError("tifffile not installed")
                if metadata:
                    tiff.imwrite(filepath, image, metadata=metadata, imagej=False)
                else:
                    tiff.imwrite(filepath, image)
                    
            elif format_lower == 'png':
                if cv2 is None:
                    raise ImportError("cv2 not installed")
                img = image.copy()
                if img.dtype in (np.float32, np.float64):
                    img = cv2.convertScaleAbs(img)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(filepath, img)
                
            elif format_lower in ('jpg', 'jpeg'):
                if cv2 is None:
                    raise ImportError("cv2 not installed")
                img = image.copy()
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(filepath, img)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self._logger.debug(f"Saved {detector_name} to {filepath}")
            return SnapResult(
                success=True,
                detector_name=detector_name,
                filepath=filepath,
            )
            
        except Exception as e:
            self._logger.error(f"Failed to write {filepath}: {e}")
            return SnapResult(
                success=False,
                detector_name=detector_name,
                filepath=filepath,
                error=str(e),
            )
    
    def _background_worker(self):
        """Background thread for writing images."""
        while not self._stop_event.is_set():
            try:
                task = self._write_queue.get(timeout=0.1)
                result = self._write_image(
                    task['detector_name'],
                    task['image'],
                    task['filepath'],
                    task['format'],
                    task['metadata'],
                )
                if task.get('callback'):
                    task['callback'](result)
                self._write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self._logger.error(f"Background worker error: {e}")
        
        # Drain queue on shutdown
        while not self._write_queue.empty():
            try:
                task = self._write_queue.get_nowait()
                self._write_image(
                    task['detector_name'],
                    task['image'],
                    task['filepath'],
                    task['format'],
                    task['metadata'],
                )
            except:
                break
    
    def get_pending_count(self) -> int:
        """Get number of pending write operations."""
        return self._write_queue.qsize()
    
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """Wait for all pending writes to complete."""
        start = time.time()
        while self._write_queue.qsize() > 0:
            if time.time() - start > timeout:
                return False
            time.sleep(0.05)
        return True


# Global snap service instance
_snap_service: Optional[SnapService] = None


def get_snap_service() -> SnapService:
    """Get or create the global snap service."""
    global _snap_service
    if _snap_service is None:
        _snap_service = SnapService()
        _snap_service.start_background_worker()
    return _snap_service


def shutdown_snap_service():
    """Shutdown the global snap service."""
    global _snap_service
    if _snap_service is not None:
        _snap_service.stop_background_worker()
        _snap_service = None


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
