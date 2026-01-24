"""
Central Metadata Hub for ImSwitch with OME-types integration.

Aggregates hardware state and detector-specific metadata,
providing a clean interface for recording and OME writers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict
import time
import threading
import numpy as np

try:
    from ome_types import OME
    from ome_types.model import Image, Pixels, Channel, Plane
    from ome_types.model import Instrument, Objective as OMEObjective
    from ome_types.model.simple_types import UnitsLength, UnitsTime
    from ome_types.model import PixelsType
    HAS_OME_TYPES = True
except ImportError:
    HAS_OME_TYPES = False
    # Set all to None for consistency
    OME = None
    Image = None
    Pixels = None
    Channel = None
    Plane = None
    Instrument = None
    OMEObjective = None
    UnitsLength = None
    UnitsTime = None
    PixelsType = None

from .schema import SharedAttrValue, MetadataSchema


@dataclass
class DetectorContext:
    """
    Detector-specific metadata context.
    
    Tracks all metadata needed to generate OME-compliant detector/image metadata.
    Similar to Micro-Manager's ImageMetadata but adapted for ImSwitch.
    """
    name: str
    
    # Required physical properties
    shape_px: Tuple[int, int]  # (width, height) in pixels
    pixel_size_um: float  # Physical pixel size in micrometers
    dtype: str = 'uint16'  # Numpy dtype string
    
    # Optional derived/explicit properties
    fov_um: Optional[Tuple[float, float]] = None  # Field of view (width, height) in um
    binning: int = 1
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    
    # Channel info
    channel_name: Optional[str] = None
    channel_color: Optional[str] = None  # Hex color like "00FF00"
    wavelength_nm: Optional[float] = None
    
    # Camera settings
    exposure_ms: Optional[float] = None
    gain: Optional[float] = None
    temperature_c: Optional[float] = None
    bit_depth: Optional[int] = None
    
    # Frame info from camera hardware
    frame_number: Optional[int] = None  # Hardware frame number
    frame_timestamp: Optional[float] = None  # Hardware frame timestamp
    is_rgb: bool = False  # Whether detector outputs RGB images
    
    # Transforms and calibration
    affine_transform: Optional[np.ndarray] = None  # 3x3 affine matrix
    objective_name: Optional[str] = None
    objective_magnification: Optional[float] = None
    objective_na: Optional[float] = None
    calibration_hash: Optional[str] = None
    
    # Metadata
    last_update: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.fov_um is None:
            # Calculate FOV from shape and pixel size
            self.fov_um = (
                self.shape_px[0] * self.pixel_size_um,
                self.shape_px[1] * self.pixel_size_um
            )
        
        if self.channel_name is None:
            self.channel_name = self.name
    
    def update(self, **kwargs):
        """Update context fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            'name': self.name,
            'shape_px': self.shape_px,
            'pixel_size_um': self.pixel_size_um,
            'fov_um': self.fov_um,
            'dtype': self.dtype,
            'binning': self.binning,
            'roi': self.roi,
            'channel_name': self.channel_name,
            'channel_color': self.channel_color,
            'wavelength_nm': self.wavelength_nm,
            'exposure_ms': self.exposure_ms,
            'gain': self.gain,
            'temperature_c': self.temperature_c,
            'bit_depth': self.bit_depth,
            'frame_number': self.frame_number,
            'frame_timestamp': self.frame_timestamp,
            'is_rgb': self.is_rgb,
            'objective_name': self.objective_name,
            'objective_magnification': self.objective_magnification,
            'objective_na': self.objective_na,
            'last_update': self.last_update,
        }
    
    def to_ome_pixels(self, size_z: int = 1, size_t: int = 1, size_c: int = 1) -> 'Pixels':
        """
        Generate OME Pixels object for this detector.
        
        Args:
            size_z: Number of Z planes
            size_t: Number of time points
            size_c: Number of channels
        
        Returns:
            OME Pixels object
        """
        if not HAS_OME_TYPES:
            raise ImportError("ome-types is required for OME metadata generation")
        
        # Map numpy dtype to OME PixelsType
        dtype_map = {
            'uint8': PixelsType.UINT8,
            'uint16': PixelsType.UINT16,
            'uint32': PixelsType.UINT32,
            'int8': PixelsType.INT8,
            'int16': PixelsType.INT16,
            'int32': PixelsType.INT32,
            'float32': PixelsType.FLOAT,
            'float64': PixelsType.DOUBLE,
        }
        pixel_type = dtype_map.get(self.dtype, PixelsType.UINT16)
        
        # Create Pixels with physical dimensions
        pixels = Pixels(
            id=f"Pixels:{self.name}",
            dimension_order="XYZCT",  # Standard order
            type=pixel_type,
            size_x=self.shape_px[0],
            size_y=self.shape_px[1],
            size_z=size_z,
            size_c=size_c,
            size_t=size_t,
            physical_size_x=self.pixel_size_um,
            physical_size_x_unit=UnitsLength.MICROMETER,
            physical_size_y=self.pixel_size_um,
            physical_size_y_unit=UnitsLength.MICROMETER,
        )
        
        # Add channels
        for c in range(size_c):
            channel = Channel(
                id=f"Channel:{self.name}:{c}",
                name=self.channel_name if size_c == 1 else f"{self.channel_name}_{c}",
                samples_per_pixel=1,
            )
            if self.channel_color:
                # Parse hex color to RGB with validation
                try:
                    # Ensure it's a valid 6-character hex string
                    color_str = str(self.channel_color).lstrip('#')
                    if len(color_str) == 6 and all(c in '0123456789ABCDEFabcdef' for c in color_str):
                        color_int = int(color_str, 16)
                        channel.color = color_int
                    else:
                        # Invalid color format, skip silently
                        pass
                except (ValueError, TypeError) as e:
                    # Log warning but don't fail
                    import logging
                    logging.getLogger(__name__).warning(f"Invalid channel color format: {self.channel_color}, error: {e}")
                    pass
            if self.wavelength_nm:
                channel.emission_wavelength = self.wavelength_nm
                channel.emission_wavelength_unit = UnitsLength.NANOMETER
            pixels.channels.append(channel)
        
        return pixels


@dataclass
class FrameEvent:
    """
    Per-frame metadata event.
    
    Captures metadata at the time of frame acquisition,
    ensuring alignment with actual image data.
    """
    frame_number: int
    timestamp: float = field(default_factory=time.time)
    detector_name: Optional[str] = None
    
    # Positional metadata
    stage_x_um: Optional[float] = None
    stage_y_um: Optional[float] = None
    stage_z_um: Optional[float] = None
    
    # Acquisition settings at trigger time
    exposure_ms: Optional[float] = None
    laser_power_mw: Optional[float] = None
    
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
            'metadata': self.metadata,
        }
    
    def to_ome_plane(self, the_z: int = 0, the_c: int = 0, the_t: int = 0) -> 'Plane':
        """
        Generate OME Plane object for this frame.
        
        Args:
            the_z: Z index
            the_c: Channel index
            the_t: Time index
        
        Returns:
            OME Plane object
        """
        if not HAS_OME_TYPES:
            raise ImportError("ome-types is required for OME metadata generation")
        
        plane = Plane(
            the_z=the_z,
            the_c=the_c,
            the_t=the_t,
        )
        
        # Add positional metadata
        if self.stage_x_um is not None:
            plane.position_x = self.stage_x_um
            plane.position_x_unit = UnitsLength.MICROMETER
        if self.stage_y_um is not None:
            plane.position_y = self.stage_y_um
            plane.position_y_unit = UnitsLength.MICROMETER
        if self.stage_z_um is not None:
            plane.position_z = self.stage_z_um
            plane.position_z_unit = UnitsLength.MICROMETER
        
        # Add timing
        if self.timestamp:
            plane.delta_t = self.timestamp
            plane.delta_t_unit = UnitsTime.SECOND
        
        if self.exposure_ms is not None:
            plane.exposure_time = self.exposure_ms / 1000.0  # Convert to seconds
            plane.exposure_time_unit = UnitsTime.SECOND
        
        return plane


class MetadataHub:
    """
    Central metadata aggregator for ImSwitch.
    
    Provides:
    - Global metadata storage (hardware state)
    - Per-detector metadata contexts
    - Per-frame event queues for acquisition alignment
    - OME-types generation for writers
    
    Thread-safe for concurrent access from multiple controllers.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # Global metadata store: key -> SharedAttrValue
        self._global_metadata: Dict[Tuple[str, ...], SharedAttrValue] = {}
        
        # Detector contexts: detector_name -> DetectorContext
        self._detector_contexts: Dict[str, DetectorContext] = {}
        
        # Per-frame event queues: detector_name -> deque[FrameEvent]
        self._frame_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Frame counters: detector_name -> int
        self._frame_counters: Dict[str, int] = defaultdict(int)
    
    def update(self, key: Tuple[str, ...], value: Any, 
              ts: Optional[float] = None, 
              units: Optional[str] = None,
              source: Optional[str] = None):
        """
        Update global metadata.
        
        Args:
            key: Metadata key tuple
            value: Metadata value
            ts: Optional timestamp (defaults to now)
            units: Optional units string
            source: Optional source identifier
        """
        with self._lock:
            # Normalize using schema
            attr_value = MetadataSchema.normalize_value(key, value, timestamp=ts, source=source)
            if units:
                attr_value.units = units
            self._global_metadata[key] = attr_value
    
    def get(self, key: Tuple[str, ...]) -> Optional[SharedAttrValue]:
        """Get a metadata value."""
        with self._lock:
            return self._global_metadata.get(key)
    
    def get_latest(self, flat: bool = False, 
                   filter_category: Optional[str] = None) -> Dict:
        """
        Get latest global metadata.
        
        Args:
            flat: If True, return flat dict with ':' separated keys
            filter_category: Optional category to filter by
        
        Returns:
            Dictionary of metadata
        """
        with self._lock:
            result = {}
            for key, attr_value in self._global_metadata.items():
                # Apply filter if specified
                if filter_category and key[0] != filter_category:
                    continue
                
                if flat:
                    # Flatten key to string
                    key_str = ':'.join(key)
                    result[key_str] = {
                        'value': attr_value.value,
                        'timestamp': attr_value.timestamp,
                        'units': attr_value.units,
                        'source': attr_value.source,
                    }
                else:
                    # Nested dict
                    current = result
                    for i, segment in enumerate(key[:-1]):
                        if segment not in current:
                            current[segment] = {}
                        current = current[segment]
                    current[key[-1]] = {
                        'value': attr_value.value,
                        'timestamp': attr_value.timestamp,
                        'units': attr_value.units,
                        'source': attr_value.source,
                    }
            return result
    
    def register_detector(self, detector_name: str, context: DetectorContext):
        """Register a detector context."""
        with self._lock:
            self._detector_contexts[detector_name] = context
    
    def get_detector(self, detector_name: str) -> Optional[DetectorContext]:
        """Get a detector context."""
        with self._lock:
            return self._detector_contexts.get(detector_name)
    
    def update_detector(self, detector_name: str, **kwargs):
        """Update detector context fields."""
        with self._lock:
            if detector_name in self._detector_contexts:
                self._detector_contexts[detector_name].update(**kwargs)
    
    def snapshot_global(self) -> Dict[str, Any]:
        """Get a snapshot of all global metadata."""
        return self.get_latest(flat=False)
    
    def snapshot_detector(self, detector_name: str) -> Dict[str, Any]:
        """
        Get a snapshot of detector-specific metadata.
        
        Returns:
            Dictionary with detector context and relevant global metadata
        """
        with self._lock:
            result = {}
            
            # Add detector context
            if detector_name in self._detector_contexts:
                result['detector_context'] = self._detector_contexts[detector_name].to_dict()
            
            # Add relevant global metadata for this detector
            detector_metadata = {}
            for key, attr_value in self._global_metadata.items():
                # Check if this metadata is for this detector
                if len(key) >= 2 and key[1] == detector_name:
                    detector_metadata[':'.join(key)] = {
                        'value': attr_value.value,
                        'timestamp': attr_value.timestamp,
                        'units': attr_value.units,
                    }
            if detector_metadata:
                result['metadata'] = detector_metadata
            
            return result
    
    def push_frame_event(self, detector_name: str, event: Optional[FrameEvent] = None, **kwargs):
        """
        Push a frame event for a detector.
        
        Args:
            detector_name: Detector name
            event: Optional pre-constructed FrameEvent
            **kwargs: If event is None, construct FrameEvent from kwargs
        """
        with self._lock:
            if event is None:
                # Auto-increment frame counter
                frame_number = self._frame_counters[detector_name]
                self._frame_counters[detector_name] += 1
                
                event = FrameEvent(
                    frame_number=frame_number,
                    detector_name=detector_name,
                    **kwargs
                )
            
            self._frame_events[detector_name].append(event)
    
    def pop_frame_events(self, detector_name: str, n: int) -> List[FrameEvent]:
        """
        Pop n frame events for a detector.
        
        Args:
            detector_name: Detector name
            n: Number of events to pop
        
        Returns:
            List of FrameEvent objects (may be fewer than n if queue is short)
        """
        with self._lock:
            events = []
            queue = self._frame_events[detector_name]
            for _ in range(min(n, len(queue))):
                if queue:
                    events.append(queue.popleft())
            return events
    
    def peek_frame_events(self, detector_name: str, n: int = None) -> List[FrameEvent]:
        """
        Peek at frame events without removing them.
        
        Args:
            detector_name: Detector name
            n: Number of events to peek (None = all)
        
        Returns:
            List of FrameEvent objects
        """
        with self._lock:
            queue = self._frame_events[detector_name]
            if n is None:
                return list(queue)
            else:
                return list(queue)[:n]
    
    def clear_frame_events(self, detector_name: str):
        """Clear all frame events for a detector."""
        with self._lock:
            self._frame_events[detector_name].clear()
            self._frame_counters[detector_name] = 0
    
    def create_pre_trigger_snapshot(self, detector_name: str) -> Dict[str, Any]:
        """
        Create a pre-trigger snapshot of the current hardware state.
        
        This method should be called BEFORE triggering image acquisition
        to capture the hardware state at the moment of trigger, avoiding
        race conditions where state changes between trigger and frame receipt.
        
        Following the pattern from octopi-research (CaptureInfo set before trigger).
        
        Args:
            detector_name: Detector name
            
        Returns:
            Dictionary with current hardware state (positions, illumination, etc.)
        """
        with self._lock:
            snapshot = {
                'timestamp': time.time(),
                'detector_name': detector_name,
                'global_metadata': {},
                'detector_context': None,
            }
            
            # Capture global metadata (positioners, illumination, objective)
            for key, attr_value in self._global_metadata.items():
                key_str = ':'.join(key)
                snapshot['global_metadata'][key_str] = {
                    'value': attr_value.value,
                    'timestamp': attr_value.timestamp,
                    'units': attr_value.units,
                }
            
            # Capture detector context
            if detector_name in self._detector_contexts:
                snapshot['detector_context'] = self._detector_contexts[detector_name].to_dict()
            
            return snapshot
    
    def create_frame_event_from_snapshot(self, snapshot: Dict[str, Any], 
                                          frame_number: int = None,
                                          hw_frame_number: int = None) -> FrameEvent:
        """
        Create a FrameEvent from a pre-trigger snapshot.
        
        This method should be called when a frame is received, using the
        snapshot that was captured before the trigger. This ensures metadata
        alignment with actual image data.
        
        Args:
            snapshot: Pre-trigger snapshot from create_pre_trigger_snapshot
            frame_number: Optional override for frame number
            hw_frame_number: Hardware frame number from camera
            
        Returns:
            FrameEvent with aligned metadata
        """
        with self._lock:
            detector_name = snapshot.get('detector_name')
            
            if frame_number is None:
                frame_number = self._frame_counters[detector_name]
                self._frame_counters[detector_name] += 1
            
            # Extract position from global metadata
            global_meta = snapshot.get('global_metadata', {})
            
            # Find stage positions (look for Positioner:*:*:Position keys)
            stage_x = None
            stage_y = None
            stage_z = None
            
            for key, value_dict in global_meta.items():
                parts = key.split(':')
                if len(parts) >= 4 and parts[0] == 'Positioner' and parts[3] == 'Position':
                    axis = parts[2]
                    pos = value_dict.get('value')
                    if axis == 'X':
                        stage_x = pos
                    elif axis == 'Y':
                        stage_y = pos
                    elif axis == 'Z':
                        stage_z = pos
            
            # Get exposure from detector context
            ctx = snapshot.get('detector_context', {})
            exposure_ms = ctx.get('exposure_ms')
            
            # Create event
            event = FrameEvent(
                frame_number=frame_number,
                timestamp=snapshot.get('timestamp', time.time()),
                detector_name=detector_name,
                stage_x_um=stage_x,
                stage_y_um=stage_y,
                stage_z_um=stage_z,
                exposure_ms=exposure_ms,
                metadata={
                    'hw_frame_number': hw_frame_number,
                    'pre_trigger_snapshot': True,
                }
            )
            
            return event
    
    def to_ome(self, detector_names: Optional[List[str]] = None) -> Optional['OME']:
        """
        Generate OME metadata object for registered detectors.
        
        Args:
            detector_names: List of detector names (None = all)
        
        Returns:
            OME object or None if ome-types not available
        """
        if not HAS_OME_TYPES:
            return None
        
        with self._lock:
            if detector_names is None:
                detector_names = list(self._detector_contexts.keys())
            
            ome = OME()
            
            # Create an image for each detector
            for det_name in detector_names:
                if det_name not in self._detector_contexts:
                    continue
                
                context = self._detector_contexts[det_name]
                
                # Create image
                image = Image(
                    id=f"Image:{det_name}",
                    name=det_name,
                )
                
                # Add pixels
                image.pixels = context.to_ome_pixels()
                
                # Add planes from frame events if available
                events = self.peek_frame_events(det_name)
                for event in events:
                    plane = event.to_ome_plane(the_t=event.frame_number)
                    image.pixels.planes.append(plane)
                
                ome.images.append(image)
            
            return ome


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
