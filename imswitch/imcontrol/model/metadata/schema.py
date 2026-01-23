"""
Metadata schema and standardization for ImSwitch.

Provides standardized metadata keys, units, and value normalization
to ensure consistency across controllers and managers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict, Tuple
import time


class MetadataCategory(str, Enum):
    """Standard metadata categories following Micro-Manager conventions."""
    POSITIONER = "Positioner"
    ILLUMINATION = "Illumination"
    OBJECTIVE = "Objective"
    DETECTOR = "Detector"
    ENVIRONMENT = "Environment"
    SYSTEM = "System"
    RECORDING = "Recording"


@dataclass
class SharedAttrValue:
    """
    Typed, timestamped metadata value wrapper.
    
    Similar to Micro-Manager's PropertyValue but adapted for ImSwitch.
    """
    value: Any
    timestamp: float = field(default_factory=time.time)
    units: Optional[str] = None
    dtype: Optional[str] = None
    source: Optional[str] = None  # Controller/Manager name
    valid: bool = True
    
    def __repr__(self):
        return f"SharedAttrValue({self.value} {self.units or ''} @ {self.timestamp:.3f})"


class MetadataSchema:
    """
    Schema registry for standardized metadata keys and units.
    
    Provides validation and normalization of metadata keys/values
    following OME-types and Micro-Manager conventions.
    """
    
    # Standard field definitions: (units, dtype, description)
    POSITIONER_FIELDS = {
        'PositionUm': ('um', 'float', 'Position in micrometers'),
        'SpeedUmS': ('um/s', 'float', 'Speed in micrometers per second'),
        'IsHomed': ('', 'bool', 'Whether axis is homed'),
        'IsMoving': ('', 'bool', 'Whether axis is moving'),
        'SetpointUm': ('um', 'float', 'Target position in micrometers'),
        'AccelerationUmS2': ('um/s^2', 'float', 'Acceleration'),
    }
    
    ILLUMINATION_FIELDS = {
        'Enabled': ('', 'bool', 'Whether illumination is enabled'),
        'WavelengthNm': ('nm', 'float', 'Wavelength in nanometers'),
        'PowerMw': ('mW', 'float', 'Power in milliwatts'),
        'CurrentMa': ('mA', 'float', 'Current in milliamps'),
        'Mode': ('', 'str', 'Operating mode'),
        'IntensityPercent': ('%', 'float', 'Intensity as percentage'),
    }
    
    OBJECTIVE_FIELDS = {
        'Name': ('', 'str', 'Objective name'),
        'Magnification': ('', 'float', 'Magnification factor'),
        'NA': ('', 'float', 'Numerical aperture'),
        'Immersion': ('', 'str', 'Immersion medium'),
        'TurretIndex': ('', 'int', 'Turret position'),
        'WorkingDistanceUm': ('um', 'float', 'Working distance'),
    }
    
    DETECTOR_FIELDS = {
        'ExposureMs': ('ms', 'float', 'Exposure time in milliseconds'),
        'Gain': ('', 'float', 'Detector gain'),
        'Binning': ('', 'int', 'Binning factor'),
        'ROI': ('', 'tuple', 'Region of interest (x, y, w, h)'),
        'TemperatureC': ('C', 'float', 'Detector temperature in Celsius'),
        'PixelSizeUm': ('um', 'float', 'Physical pixel size in micrometers'),
        'ShapePx': ('px', 'tuple', 'Detector shape in pixels (width, height)'),
        'BitDepth': ('', 'int', 'Bit depth'),
        'ReadoutMode': ('', 'str', 'Readout mode'),
    }
    
    ENVIRONMENT_FIELDS = {
        'TemperatureC': ('C', 'float', 'Temperature in Celsius'),
        'HumidityPercent': ('%', 'float', 'Relative humidity'),
        'CO2Percent': ('%', 'float', 'CO2 concentration'),
        'PressurePa': ('Pa', 'float', 'Pressure in Pascals'),
    }
    
    SYSTEM_FIELDS = {
        'Timestamp': ('s', 'float', 'Unix timestamp'),
        'FrameNumber': ('', 'int', 'Frame number in sequence'),
        'ElapsedTimeS': ('s', 'float', 'Elapsed time in seconds'),
    }
    
    # Map categories to their field definitions
    CATEGORY_FIELDS = {
        MetadataCategory.POSITIONER: POSITIONER_FIELDS,
        MetadataCategory.ILLUMINATION: ILLUMINATION_FIELDS,
        MetadataCategory.OBJECTIVE: OBJECTIVE_FIELDS,
        MetadataCategory.DETECTOR: DETECTOR_FIELDS,
        MetadataCategory.ENVIRONMENT: ENVIRONMENT_FIELDS,
        MetadataCategory.SYSTEM: SYSTEM_FIELDS,
    }
    
    @classmethod
    def validate_key(cls, key: Tuple[str, ...]) -> bool:
        """
        Validate a metadata key tuple.
        
        Args:
            key: Tuple like ('Positioner', 'Stage', 'X', 'PositionUm')
                 Format: (category, device, axis_or_sub, field)
                 Minimum 2 elements for compatibility, but 4 is recommended.
        
        Returns:
            True if key is valid
        """
        if not isinstance(key, tuple) or len(key) < 2:
            return False
        
        # Check if category is recognized
        category = key[0]
        try:
            MetadataCategory(category)
        except ValueError:
            return False
        
        return True
    
    @classmethod
    def get_field_info(cls, category: str, field: str) -> Optional[Tuple[str, str, str]]:
        """
        Get field information (units, dtype, description).
        
        Args:
            category: Metadata category
            field: Field name
        
        Returns:
            (units, dtype, description) or None if not found
        """
        try:
            cat = MetadataCategory(category)
            fields = cls.CATEGORY_FIELDS.get(cat, {})
            return fields.get(field)
        except (ValueError, KeyError):
            return None
    
    @classmethod
    def normalize_value(cls, key: Tuple[str, ...], value: Any, 
                       timestamp: Optional[float] = None,
                       source: Optional[str] = None) -> SharedAttrValue:
        """
        Normalize a metadata value to a SharedAttrValue.
        
        Args:
            key: Metadata key tuple
            value: Raw value
            timestamp: Optional timestamp (defaults to now)
            source: Optional source identifier
        
        Returns:
            SharedAttrValue with units and type information
        """
        if timestamp is None:
            timestamp = time.time()
        
        # If already a SharedAttrValue, return it
        if isinstance(value, SharedAttrValue):
            return value
        
        # Extract category and field from key
        if len(key) >= 4:
            category, device, axis_or_sub, field = key[0], key[1], key[2], key[3]
        elif len(key) >= 2:
            category, field = key[0], key[-1]
            device, axis_or_sub = None, None
        else:
            # Invalid key, return raw value wrapped
            return SharedAttrValue(value=value, timestamp=timestamp, source=source)
        
        # Get field info from schema
        field_info = cls.get_field_info(category, field)
        if field_info:
            units, dtype, description = field_info
            return SharedAttrValue(
                value=value,
                timestamp=timestamp,
                units=units if units else None,
                dtype=dtype,
                source=source,
                valid=True
            )
        else:
            # Unknown field, but still wrap it
            return SharedAttrValue(
                value=value,
                timestamp=timestamp,
                source=source,
                valid=True
            )
    
    @classmethod
    def make_key(cls, category: MetadataCategory, device: str, 
                 axis_or_sub: Optional[str], field: str) -> Tuple[str, ...]:
        """
        Construct a standardized metadata key.
        
        Args:
            category: Metadata category
            device: Device name
            axis_or_sub: Axis name or sub-component (can be None)
            field: Field name
        
        Returns:
            Tuple key
        """
        if axis_or_sub:
            return (category.value, device, axis_or_sub, field)
        else:
            return (category.value, device, field)


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
