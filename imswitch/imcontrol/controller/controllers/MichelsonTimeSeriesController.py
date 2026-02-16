"""
MichelsonTimeSeriesController - Camera-based ROI intensity time-series acquisition

This controller captures mean intensity from a small camera ROI over time,
useful for demonstrating Michelson interferometer arm-length changes without
requiring an external photodetector.

Features:
- Camera-based ROI mean intensity extraction
- Ring buffer storage for time-series data
- Configurable ROI size (5x5, 10x10, etc.)
- Rate-limited frame processing
- REST API endpoints for start/stop/data retrieval
- CSV export capability
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import time
import threading
from collections import deque

from imswitch.imcommon.model import initLogger, APIExport
from imswitch.imcommon.framework import Signal
from ..basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS


@dataclass
class MichelsonParams:
    """Michelson time-series acquisition parameters"""
    roi_center: Optional[List[int]] = None  # [x, y] in pixels
    roi_size: int = 10  # square ROI size (5, 10, 20, etc.)
    update_freq: float = 30.0  # acquisition rate in Hz
    buffer_duration: float = 60.0  # buffer length in seconds
    decimation: int = 1  # process every Nth frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi_center": self.roi_center,
            "roi_size": self.roi_size,
            "update_freq": self.update_freq,
            "buffer_duration": self.buffer_duration,
            "decimation": self.decimation,
        }


@dataclass
class MichelsonState:
    """Michelson time-series acquisition state"""
    is_capturing: bool = False
    frame_count: int = 0
    sample_count: int = 0
    actual_fps: float = 0.0
    start_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_capturing": self.is_capturing,
            "frame_count": self.frame_count,
            "sample_count": self.sample_count,
            "actual_fps": self.actual_fps,
            "start_time": self.start_time,
        }


class MichelsonTimeSeriesController(LiveUpdatedController):
    """
    Controller for camera-based Michelson interferometer time-series analysis.
    
    This controller extracts mean intensity from a small ROI in the camera image
    over time, storing the results in a ring buffer for visualization and export.
    
    Features:
    - Small ROI extraction (5x5, 10x10, etc.)
    - Ring buffer with configurable duration
    - Rate-limited processing via decimation
    - REST API for control and data retrieval
    - CSV export capability
    
    Architecture:
    - update() is called for every frame via sigUpdateImage
    - Decimation controls how many frames are skipped
    - Data is stored in a ring buffer (deque)
    """

    sigDataUpdated = Signal(object)  # emits dict with timestamp, mean, std
    sigStateChanged = Signal(object)  # emits state dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # Get camera from setup or use first available detector
        if hasattr(self._setupInfo, 'michelson') and self._setupInfo.michelson is not None:
            self.camera = getattr(self._setupInfo.michelson, 'camera', None)
        else:
            self.camera = None

        # If no camera specified, use first available detector
        if self.camera is None:
            try:
                all_detectors = self._master.detectorsManager.getAllDeviceNames()
                if all_detectors:
                    self.camera = all_detectors[0]
                    self._logger.info(f"Using first available detector: {self.camera}")
                else:
                    self._logger.error("No detectors available")
                    return
            except Exception as e:
                self._logger.error(f"Failed to get detector list: {e}")
                return

        # Initialize parameters
        self._params = MichelsonParams()
        self._state = MichelsonState()
        self._processing_lock = threading.Lock()

        # Frame counter for decimation
        self._frame_counter = 0

        # Ring buffer for time-series data
        # Each entry: (timestamp, mean_intensity, std_intensity)
        self._buffer_maxlen = int(self._params.buffer_duration * self._params.update_freq)
        self._data_buffer = deque(maxlen=self._buffer_maxlen)
        
        # For FPS calculation
        self._fps_timestamps = deque(maxlen=30)

        # Connect to CommunicationChannel signal for frame updates
        self._commChannel.sigUpdateImage.connect(self.update)

        self._logger.info("MichelsonTimeSeriesController initialized successfully")

    def update(self, detectorName, image, init, scale, isCurrentDetector):
        """
        Periodic update called for every frame via sigUpdateImage.
        
        Extracts mean intensity from ROI and stores in ring buffer.
        """
        # Only process frames from our target camera
        if detectorName != self.camera:
            return

        # Skip if not capturing
        if not self._state.is_capturing:
            return

        # Skip if image is None
        if image is None:
            return

        self._state.frame_count += 1

        # Decimation: process every Nth frame
        if self._frame_counter >= self._params.decimation - 1:
            self._frame_counter = 0
            self._process_frame(image)
        else:
            self._frame_counter += 1

    def _process_frame(self, image):
        """Process a single frame: extract ROI and compute statistics"""
        try:
            current_time = time.monotonic()
            
            # Update FPS calculation
            self._fps_timestamps.append(current_time)
            if len(self._fps_timestamps) >= 2:
                dt = self._fps_timestamps[-1] - self._fps_timestamps[0]
                if dt > 0:
                    self._state.actual_fps = (len(self._fps_timestamps) - 1) / dt

            # Extract ROI
            h, w = image.shape[:2]
            
            # Determine ROI center
            if self._params.roi_center is not None and self._params.roi_center[0] is not None:
                cx, cy = self._params.roi_center
            else:
                cx, cy = w // 2, h // 2

            # Calculate ROI bounds
            half_size = self._params.roi_size // 2
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w, cx + half_size)
            y2 = min(h, cy + half_size)

            # Extract ROI
            roi = image[y1:y2, x1:x2]
            
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                roi = np.mean(roi, axis=2)

            # Compute statistics
            mean_intensity = float(np.mean(roi))
            std_intensity = float(np.std(roi))
            
            # Relative timestamp from start
            relative_time = current_time - self._state.start_time

            # Add to buffer
            data_point = {
                'timestamp': relative_time,
                'mean': mean_intensity,
                'std': std_intensity,
            }
            self._data_buffer.append(data_point)
            self._state.sample_count = len(self._data_buffer)

            # Emit signal
            self.sigDataUpdated.emit(data_point)

        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")

    def _update_buffer_size(self):
        """Update buffer size based on current parameters"""
        new_maxlen = int(self._params.buffer_duration * self._params.update_freq / self._params.decimation)
        new_maxlen = max(100, new_maxlen)  # At least 100 samples
        
        if new_maxlen != self._buffer_maxlen:
            self._buffer_maxlen = new_maxlen
            # Create new buffer with updated size, preserving data
            old_data = list(self._data_buffer)
            self._data_buffer = deque(old_data[-new_maxlen:], maxlen=new_maxlen)

    # =========================
    # API: Parameter Control
    # =========================
    @APIExport(runOnUIThread=True)
    def get_parameters_michelson(self) -> Dict[str, Any]:
        """Get current Michelson time-series parameters"""
        return self._params.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def set_parameters_michelson(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update Michelson time-series parameters"""
        with self._processing_lock:
            for key, value in params.items():
                if hasattr(self._params, key):
                    setattr(self._params, key, value)
            
            # Update buffer size if relevant params changed
            if any(k in params for k in ['buffer_duration', 'update_freq', 'decimation']):
                self._update_buffer_size()

        return self._params.to_dict()

    @APIExport(runOnUIThread=True)
    def set_roi_michelson(self, center_x: int = None, center_y: int = None, size: int = 10) -> Dict[str, Any]:
        """
        Set ROI center and size for intensity extraction
        
        Args:
            center_x: X coordinate of ROI center (None = image center)
            center_y: Y coordinate of ROI center (None = image center)
            size: Square ROI size in pixels (5, 10, 20, etc.)
        """
        center = [center_x, center_y] if center_x is not None and center_y is not None else None
        return self.set_parameters_michelson({"roi_center": center, "roi_size": size})

    @APIExport(runOnUIThread=True)
    def set_buffer_duration_michelson(self, duration: float) -> Dict[str, Any]:
        """Set buffer duration in seconds"""
        return self.set_parameters_michelson({"buffer_duration": duration})

    @APIExport(runOnUIThread=True)
    def set_decimation_michelson(self, decimation: int) -> Dict[str, Any]:
        """Set decimation factor (process every Nth frame)"""
        return self.set_parameters_michelson({"decimation": max(1, decimation)})

    # =========================
    # API: Acquisition Control
    # =========================
    @APIExport(runOnUIThread=True)
    def get_state_michelson(self) -> Dict[str, Any]:
        """Get current acquisition state"""
        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def start_capture_michelson(self) -> Dict[str, Any]:
        """
        Start time-series capture
        
        Returns:
            Current state dictionary
        """
        with self._processing_lock:
            self._state.is_capturing = True
            self._state.frame_count = 0
            self._state.sample_count = 0
            self._state.start_time = time.monotonic()
            self._frame_counter = 0
            self._fps_timestamps.clear()

        # Ensure camera is running
        self._ensure_camera_running()

        self._logger.info(f"Started Michelson time-series capture (ROI: {self._params.roi_size}x{self._params.roi_size})")
        self._emit_state_changed()

        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def stop_capture_michelson(self) -> Dict[str, Any]:
        """
        Stop time-series capture
        
        Returns:
            Current state dictionary
        """
        with self._processing_lock:
            self._state.is_capturing = False

        self._logger.info(f"Stopped Michelson time-series capture ({self._state.sample_count} samples)")
        self._emit_state_changed()

        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def clear_buffer_michelson(self) -> Dict[str, Any]:
        """
        Clear the time-series buffer
        
        Returns:
            Current state dictionary
        """
        with self._processing_lock:
            self._data_buffer.clear()
            self._state.sample_count = 0

        self._logger.info("Cleared Michelson time-series buffer")
        return self._state.to_dict()

    # =========================
    # API: Data Retrieval
    # =========================
    @APIExport(runOnUIThread=True)
    def get_timeseries_michelson(self, last_n: int = None, last_seconds: float = None) -> Dict[str, Any]:
        """
        Get time-series data from buffer
        
        Args:
            last_n: Return last N samples (if specified)
            last_seconds: Return samples from last N seconds (if specified)
            
        Returns:
            Dictionary with:
            - timestamps: list of relative timestamps
            - means: list of mean intensities
            - stds: list of standard deviations
            - sample_count: total number of samples
            - actual_fps: actual acquisition rate
        """
        with self._processing_lock:
            data = list(self._data_buffer)

        if len(data) == 0:
            return {
                'timestamps': [],
                'means': [],
                'stds': [],
                'sample_count': 0,
                'actual_fps': self._state.actual_fps,
            }

        # Filter by last_n or last_seconds
        if last_n is not None:
            data = data[-last_n:]
        elif last_seconds is not None:
            current_time = time.monotonic() - self._state.start_time
            min_time = current_time - last_seconds
            data = [d for d in data if d['timestamp'] >= min_time]

        return {
            'timestamps': [d['timestamp'] for d in data],
            'means': [d['mean'] for d in data],
            'stds': [d['std'] for d in data],
            'sample_count': len(data),
            'actual_fps': self._state.actual_fps,
        }

    @APIExport(runOnUIThread=True)
    def get_latest_sample_michelson(self) -> Dict[str, Any]:
        """
        Get the most recent sample
        
        Returns:
            Dictionary with timestamp, mean, std, or empty if no data
        """
        with self._processing_lock:
            if len(self._data_buffer) > 0:
                return dict(self._data_buffer[-1])
            else:
                return {}

    @APIExport(runOnUIThread=True)
    def export_csv_michelson(self) -> str:
        """
        Export time-series data as CSV string
        
        Returns:
            CSV formatted string with headers: timestamp,mean,std
        """
        with self._processing_lock:
            data = list(self._data_buffer)

        lines = ['timestamp,mean,std']
        for d in data:
            lines.append(f"{d['timestamp']:.6f},{d['mean']:.6f},{d['std']:.6f}")

        return '\n'.join(lines)

    @APIExport(runOnUIThread=True)
    def get_statistics_michelson(self) -> Dict[str, Any]:
        """
        Get statistics on the buffered data
        
        Returns:
            Dictionary with min, max, mean, std of the mean intensities
        """
        with self._processing_lock:
            data = list(self._data_buffer)

        if len(data) == 0:
            return {
                'data_min': None,
                'data_max': None,
                'data_mean': None,
                'data_std': None,
                'sample_count': 0,
            }

        means = [d['mean'] for d in data]
        return {
            'data_min': float(np.min(means)),
            'data_max': float(np.max(means)),
            'data_mean': float(np.mean(means)),
            'data_std': float(np.std(means)),
            'sample_count': len(means),
        }

    # =========================
    # Helper methods
    # =========================
    def _ensure_camera_running(self):
        """Ensure camera is running, start if necessary"""
        try:
            detector = self._master.detectorsManager[self.camera]
            if hasattr(detector, '_running') and not detector._running:
                self._logger.info(f"Starting camera {self.camera}")
                detector.startAcquisition()
        except Exception as e:
            self._logger.error(f"Failed to start camera: {e}")

    def _emit_state_changed(self):
        """Emit state changed signal"""
        self.sigStateChanged.emit(self._state.to_dict())


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
