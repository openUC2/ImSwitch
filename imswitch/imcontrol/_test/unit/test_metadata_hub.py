"""
Unit tests for MetadataHub and related metadata infrastructure.

Tests the MetadataHub, DetectorContext, FrameEvent, and schema functionality.
"""
import time
import numpy as np
from typing import Tuple


def test_metadata_schema_basic():
    """Test basic MetadataSchema functionality."""
    from imswitch.imcontrol.model.metadata import MetadataSchema, MetadataCategory
    
    # Test key validation
    valid_key = ('Positioner', 'Stage', 'X', 'PositionUm')
    assert MetadataSchema.validate_key(valid_key) is True
    
    invalid_key = ('InvalidCategory', 'Device')
    assert MetadataSchema.validate_key(invalid_key) is False
    
    # Test field info retrieval
    field_info = MetadataSchema.get_field_info('Positioner', 'PositionUm')
    assert field_info is not None
    units, dtype, description = field_info
    assert units == 'um'
    assert dtype == 'float'


def test_shared_attr_value():
    """Test SharedAttrValue wrapper."""
    from imswitch.imcontrol.model.metadata import SharedAttrValue
    
    # Create a value with timestamp
    timestamp = time.time()
    attr_value = SharedAttrValue(
        value=123.45,
        timestamp=timestamp,
        units='um',
        dtype='float',
        source='TestController'
    )
    
    assert attr_value.value == 123.45
    assert attr_value.units == 'um'
    assert attr_value.timestamp == timestamp
    assert attr_value.valid is True


def test_metadata_schema_normalize():
    """Test value normalization with schema."""
    from imswitch.imcontrol.model.metadata import MetadataSchema
    
    key = ('Positioner', 'Stage', 'X', 'PositionUm')
    value = 100.5
    
    normalized = MetadataSchema.normalize_value(key, value, source='Test')
    
    assert normalized.value == 100.5
    assert normalized.units == 'um'
    assert normalized.dtype == 'float'
    assert normalized.source == 'Test'


def test_detector_context_basic():
    """Test DetectorContext creation."""
    from imswitch.imcontrol.model.metadata import DetectorContext
    
    context = DetectorContext(
        name='TestCamera',
        shape_px=(1024, 1024),
        pixel_size_um=6.5,
        dtype='uint16'
    )
    
    assert context.name == 'TestCamera'
    assert context.shape_px == (1024, 1024)
    assert context.pixel_size_um == 6.5
    assert context.dtype == 'uint16'
    
    # FOV should be calculated automatically
    expected_fov = (1024 * 6.5, 1024 * 6.5)
    assert context.fov_um == expected_fov


def test_detector_context_update():
    """Test DetectorContext updates."""
    from imswitch.imcontrol.model.metadata import DetectorContext
    
    context = DetectorContext(
        name='TestCamera',
        shape_px=(512, 512),
        pixel_size_um=1.0,
    )
    
    initial_time = context.last_update
    time.sleep(0.01)
    
    context.update(exposure_ms=100.0, gain=2.5)
    
    assert context.exposure_ms == 100.0
    assert context.gain == 2.5
    assert context.last_update > initial_time


def test_detector_context_to_dict():
    """Test DetectorContext serialization."""
    from imswitch.imcontrol.model.metadata import DetectorContext
    
    context = DetectorContext(
        name='TestCamera',
        shape_px=(1024, 768),
        pixel_size_um=5.0,
        exposure_ms=50.0,
        gain=1.5
    )
    
    context_dict = context.to_dict()
    
    assert context_dict['name'] == 'TestCamera'
    assert context_dict['shape_px'] == (1024, 768)
    assert context_dict['pixel_size_um'] == 5.0
    assert context_dict['exposure_ms'] == 50.0
    assert context_dict['gain'] == 1.5


def test_frame_event_basic():
    """Test FrameEvent creation."""
    from imswitch.imcontrol.model.metadata import FrameEvent
    
    event = FrameEvent(
        frame_number=42,
        detector_name='Camera1',
        stage_x_um=100.0,
        stage_y_um=200.0,
        stage_z_um=50.0,
        exposure_ms=100.0
    )
    
    assert event.frame_number == 42
    assert event.detector_name == 'Camera1'
    assert event.stage_x_um == 100.0
    assert event.stage_y_um == 200.0
    assert event.stage_z_um == 50.0
    assert event.exposure_ms == 100.0


def test_frame_event_to_dict():
    """Test FrameEvent serialization."""
    from imswitch.imcontrol.model.metadata import FrameEvent
    
    event = FrameEvent(
        frame_number=10,
        stage_x_um=1.0,
        stage_y_um=2.0,
        metadata={'extra': 'data'}
    )
    
    event_dict = event.to_dict()
    
    assert event_dict['frame_number'] == 10
    assert event_dict['stage_x_um'] == 1.0
    assert event_dict['metadata']['extra'] == 'data'


def test_metadata_hub_creation():
    """Test MetadataHub initialization."""
    from imswitch.imcontrol.model.metadata import MetadataHub
    
    hub = MetadataHub()
    assert hub is not None


def test_metadata_hub_update():
    """Test MetadataHub update functionality."""
    from imswitch.imcontrol.model.metadata import MetadataHub
    
    hub = MetadataHub()
    
    key = ('Positioner', 'Stage', 'X', 'PositionUm')
    hub.update(key, 123.45, source='TestController')
    
    value = hub.get(key)
    assert value is not None
    assert value.value == 123.45
    assert value.units == 'um'


def test_metadata_hub_detector_registration():
    """Test detector registration with hub."""
    from imswitch.imcontrol.model.metadata import MetadataHub, DetectorContext
    
    hub = MetadataHub()
    
    context = DetectorContext(
        name='Camera1',
        shape_px=(1024, 1024),
        pixel_size_um=6.5
    )
    
    hub.register_detector('Camera1', context)
    
    retrieved = hub.get_detector('Camera1')
    assert retrieved is not None
    assert retrieved.name == 'Camera1'
    assert retrieved.shape_px == (1024, 1024)


def test_metadata_hub_detector_update():
    """Test updating detector context via hub."""
    from imswitch.imcontrol.model.metadata import MetadataHub, DetectorContext
    
    hub = MetadataHub()
    
    context = DetectorContext(
        name='Camera1',
        shape_px=(512, 512),
        pixel_size_um=1.0
    )
    
    hub.register_detector('Camera1', context)
    hub.update_detector('Camera1', exposure_ms=200.0, gain=3.0)
    
    retrieved = hub.get_detector('Camera1')
    assert retrieved.exposure_ms == 200.0
    assert retrieved.gain == 3.0


def test_metadata_hub_snapshot():
    """Test metadata snapshot functionality."""
    from imswitch.imcontrol.model.metadata import MetadataHub
    
    hub = MetadataHub()
    
    # Add some metadata
    hub.update(('Positioner', 'Stage', 'X', 'PositionUm'), 100.0)
    hub.update(('Positioner', 'Stage', 'Y', 'PositionUm'), 200.0)
    hub.update(('Detector', 'Camera1', '', 'ExposureMs'), 50.0)
    
    snapshot = hub.snapshot_global()
    
    assert 'Positioner' in snapshot
    assert 'Detector' in snapshot


def test_metadata_hub_frame_events():
    """Test frame event queue functionality."""
    from imswitch.imcontrol.model.metadata import MetadataHub, FrameEvent
    
    hub = MetadataHub()
    
    # Push frame events
    event1 = FrameEvent(frame_number=0, stage_x_um=0.0, stage_y_um=0.0)
    event2 = FrameEvent(frame_number=1, stage_x_um=10.0, stage_y_um=20.0)
    
    hub.push_frame_event('Camera1', event1)
    hub.push_frame_event('Camera1', event2)
    
    # Peek without removing
    events = hub.peek_frame_events('Camera1', n=2)
    assert len(events) == 2
    assert events[0].frame_number == 0
    assert events[1].frame_number == 1
    
    # Pop events
    popped = hub.pop_frame_events('Camera1', n=2)
    assert len(popped) == 2
    assert popped[0].stage_x_um == 0.0
    assert popped[1].stage_x_um == 10.0
    
    # Queue should be empty now
    remaining = hub.peek_frame_events('Camera1')
    assert len(remaining) == 0


def test_metadata_hub_frame_events_auto_increment():
    """Test automatic frame number increment."""
    from imswitch.imcontrol.model.metadata import MetadataHub
    
    hub = MetadataHub()
    
    # Push events without frame numbers
    hub.push_frame_event('Camera1', stage_x_um=1.0)
    hub.push_frame_event('Camera1', stage_x_um=2.0)
    hub.push_frame_event('Camera1', stage_x_um=3.0)
    
    events = hub.peek_frame_events('Camera1')
    
    assert len(events) == 3
    assert events[0].frame_number == 0
    assert events[1].frame_number == 1
    assert events[2].frame_number == 2


def test_shared_attributes_with_typed_values():
    """Test SharedAttributes with SharedAttrValue integration."""
    from imswitch.imcommon.model import SharedAttributes
    try:
        from imswitch.imcontrol.model.metadata import SharedAttrValue
        has_metadata = True
    except ImportError:
        has_metadata = False
    
    if not has_metadata:
        # Skip test if metadata module not available
        return
    
    shared_attrs = SharedAttributes()
    
    # Set a raw value
    key1 = ('Test', 'Device', 'Property')
    shared_attrs[key1] = 123.45
    
    # Should return raw value
    assert shared_attrs[key1] == 123.45
    
    # Set a SharedAttrValue
    key2 = ('Test', 'Device2', 'Property2')
    typed_value = SharedAttrValue(
        value=678.9,
        units='um',
        source='TestSource'
    )
    shared_attrs[key2] = typed_value
    
    # __getitem__ should return unwrapped value
    assert shared_attrs[key2] == 678.9
    
    # get_typed should return full SharedAttrValue
    full_value = shared_attrs.get_typed(key2)
    assert isinstance(full_value, SharedAttrValue)
    assert full_value.value == 678.9
    assert full_value.units == 'um'


def test_shared_attributes_hdf5_with_metadata():
    """Test HDF5 export with metadata."""
    from imswitch.imcommon.model import SharedAttributes
    try:
        from imswitch.imcontrol.model.metadata import SharedAttrValue
        has_metadata = True
    except ImportError:
        has_metadata = False
    
    if not has_metadata:
        return
    
    shared_attrs = SharedAttributes()
    
    # Add typed value
    key = ('Positioner', 'Stage', 'X', 'PositionUm')
    typed_value = SharedAttrValue(
        value=100.5,
        units='um',
        timestamp=time.time(),
        source='Controller'
    )
    shared_attrs[key] = typed_value
    
    # Export to HDF5 format
    hdf5_attrs = shared_attrs.getSharedAttributes()
    
    # Should have main value and metadata keys
    key_str = 'Positioner:Stage:X:PositionUm'
    assert key_str in hdf5_attrs
    assert hdf5_attrs[key_str] == 100.5
    assert f'{key_str}:units' in hdf5_attrs
    assert hdf5_attrs[f'{key_str}:units'] == 'um'
    assert f'{key_str}:source' in hdf5_attrs


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
