# Central Metadata Hub Documentation

## Overview

The Metadata Hub provides a centralized, structured metadata management system for ImSwitch, with OME-types integration for standards-compliant microscopy metadata.

## Architecture

```
Controllers/Managers
       ↓
  SharedAttributes (with typed values)
       ↓
SharedAttrsMetadataBridge (validates & normalizes)
       ↓
  MetadataHub (central aggregator)
       ├─→ Global metadata store
       ├─→ Detector contexts
       └─→ Frame event queues
       ↓
Recording/Writing
  ├─→ HDF5 attributes
  ├─→ OME-Zarr metadata
  └─→ OME-TIFF metadata
```

## Core Components

### 1. MetadataHub

Central aggregator for all metadata:

```python
from imswitch.imcontrol.model.metadata import MetadataHub

# Access the hub (initialized in MasterController)
hub = master_controller.metadataHub

# Update global metadata
hub.update(('Positioner', 'Stage', 'X', 'PositionUm'), 100.5, source='PositionerController')

# Get latest metadata
global_snapshot = hub.snapshot_global()
detector_snapshot = hub.snapshot_detector('Camera1')

# Register a detector
from imswitch.imcontrol.model.metadata import DetectorContext
context = DetectorContext(
    name='Camera1',
    shape_px=(2048, 2048),
    pixel_size_um=6.5,
    dtype='uint16'
)
hub.register_detector('Camera1', context)
```

### 2. DetectorContext

Stores detector-specific metadata:

```python
from imswitch.imcontrol.model.metadata import DetectorContext

context = DetectorContext(
    name='Camera1',
    shape_px=(2048, 2048),
    pixel_size_um=6.5,
    dtype='uint16',
    binning=1,
    channel_name='GFP',
    channel_color='00FF00',  # Hex color
    wavelength_nm=488,
    exposure_ms=100,
    gain=2.5,
    objective_magnification=40.0,
    objective_na=1.3
)

# Update fields
context.update(exposure_ms=200.0, gain=3.0)

# Export for storage
context_dict = context.to_dict()

# Generate OME Pixels object
pixels = context.to_ome_pixels(size_z=10, size_t=5, size_c=1)
```

### 3. Metadata Schema

Standardized keys with units and types:

```python
from imswitch.imcontrol.model.metadata import MetadataSchema, MetadataCategory

# Standard categories
categories = [
    MetadataCategory.POSITIONER,
    MetadataCategory.ILLUMINATION,
    MetadataCategory.OBJECTIVE,
    MetadataCategory.DETECTOR,
    MetadataCategory.ENVIRONMENT,
    MetadataCategory.SYSTEM
]

# Make a standardized key
key = MetadataSchema.make_key(
    category=MetadataCategory.POSITIONER,
    device='Stage',
    axis_or_sub='X',
    field='PositionUm'
)
# Result: ('Positioner', 'Stage', 'X', 'PositionUm')

# Normalize a value
normalized = MetadataSchema.normalize_value(key, 100.5, source='Controller')
# Returns SharedAttrValue with value=100.5, units='um', dtype='float'
```

### 4. Frame Events

Per-frame metadata for acquisition alignment:

```python
from imswitch.imcontrol.model.metadata import FrameEvent

# During acquisition, push events
event = FrameEvent(
    frame_number=42,
    detector_name='Camera1',
    stage_x_um=100.0,
    stage_y_um=200.0,
    stage_z_um=50.0,
    exposure_ms=100.0,
    laser_power_mw=10.0
)
hub.push_frame_event('Camera1', event)

# During writing, pop events
events = hub.pop_frame_events('Camera1', n=10)

# Generate OME Plane objects
plane = event.to_ome_plane(the_z=0, the_c=0, the_t=42)
```

## Standardized Metadata Keys

### Positioner Fields
- `PositionUm` (um, float): Position in micrometers
- `SpeedUmS` (um/s, float): Speed in micrometers per second
- `IsHomed` (bool): Whether axis is homed
- `IsMoving` (bool): Whether axis is moving
- `SetpointUm` (um, float): Target position

### Illumination Fields
- `Enabled` (bool): Whether illumination is enabled
- `WavelengthNm` (nm, float): Wavelength in nanometers
- `PowerMw` (mW, float): Power in milliwatts
- `CurrentMa` (mA, float): Current in milliamps
- `Mode` (str): Operating mode
- `IntensityPercent` (%, float): Intensity as percentage

### Objective Fields
- `Name` (str): Objective name
- `Magnification` (float): Magnification factor
- `NA` (float): Numerical aperture
- `Immersion` (str): Immersion medium
- `TurretIndex` (int): Turret position
- `WorkingDistanceUm` (um, float): Working distance

### Detector Fields
- `ExposureMs` (ms, float): Exposure time
- `Gain` (float): Detector gain
- `Binning` (int): Binning factor
- `ROI` (tuple): Region of interest (x, y, w, h)
- `TemperatureC` (C, float): Detector temperature
- `PixelSizeUm` (um, float): Physical pixel size
- `ShapePx` (px, tuple): Detector shape (width, height)
- `BitDepth` (int): Bit depth
- `ReadoutMode` (str): Readout mode

### Environment Fields
- `TemperatureC` (C, float): Temperature
- `HumidityPercent` (%, float): Relative humidity
- `CO2Percent` (%, float): CO2 concentration
- `PressurePa` (Pa, float): Pressure

## Integration Points

### Controllers/Managers

Controllers should publish metadata using `setSharedAttr` with standardized keys:

```python
from imswitch.imcontrol.model.metadata import MetadataSchema, MetadataCategory

class MyPositionerController:
    def updatePosition(self, positionerName, axis, position):
        # Use standardized key
        key = MetadataSchema.make_key(
            MetadataCategory.POSITIONER,
            positionerName,
            axis,
            'PositionUm'
        )
        self._commChannel.sharedAttrs[key] = position
```

The SharedAttrsMetadataBridge automatically forwards these to the MetadataHub.

### Recording

RecordingController automatically enriches attrs with hub metadata:

```python
# Happens automatically in RecordingController._get_detector_attrs()
attrs = {
    'Positioner:Stage:X:PositionUm': 100.5,
    'Detector:Camera1:ExposureMs': 100.0,
    'Camera1:pixel_size_um': 6.5,
    'Camera1:shape_px': '[2048, 2048]',
    'Camera1:fov_um': '[13312.0, 13312.0]',
    '_metadata_hub_global': '{"Positioner": {...}, "Detector": {...}}'
}
```

### OME Writers

Future OME writers can consume hub metadata:

```python
# Get OME object for all detectors
ome = hub.to_ome(detector_names=['Camera1', 'Camera2'])

# Or per-detector
context = hub.get_detector('Camera1')
pixels = context.to_ome_pixels(size_z=10, size_t=100, size_c=1)

# With frame events
events = hub.peek_frame_events('Camera1', n=100)
planes = [event.to_ome_plane(the_t=i) for i, event in enumerate(events)]
```

## Usage Examples

### Example 1: Publishing Positioner Metadata

```python
class MyPositionerManager:
    def __init__(self):
        self.position = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        
    def move(self, axis, distance):
        self.position[axis] += distance
        
        # Publish position update
        key = ('Positioner', 'MyStage', axis, 'PositionUm')
        commChannel.sharedAttrs[key] = self.position[axis]
```

### Example 2: Registering a Detector

```python
# In MasterController or detector manager init
from imswitch.imcontrol.model.metadata import DetectorContext

context = DetectorContext(
    name='Camera1',
    shape_px=(2048, 2048),
    pixel_size_um=6.5,
    dtype='uint16',
    exposure_ms=100.0,
    gain=1.0,
    binning=1
)

master_controller.metadataHub.register_detector('Camera1', context)
```

### Example 3: Pushing Frame Events

```python
# In acquisition loop
for frame_idx in range(num_frames):
    # Capture frame
    image = detector.getLatestFrame()
    
    # Get current positions
    stage_x = positioner.position['X']
    stage_y = positioner.position['Y']
    
    # Push frame event
    hub.push_frame_event(
        'Camera1',
        stage_x_um=stage_x,
        stage_y_um=stage_y,
        exposure_ms=current_exposure
    )
```

### Example 4: Consuming Metadata in Writers

```python
# In storer write method
def write_chunk(self, detector_name, chunk):
    # Get metadata snapshot
    detector_snapshot = hub.snapshot_detector(detector_name)
    
    # Get detector context
    context = detector_snapshot['detector_context']
    pixel_size = context['pixel_size_um']
    
    # Get frame events
    events = hub.pop_frame_events(detector_name, n=len(chunk))
    
    # Write with metadata
    for i, (frame, event) in enumerate(zip(chunk, events)):
        self.write_frame(
            frame,
            pixel_size=pixel_size,
            stage_x=event.stage_x_um,
            stage_y=event.stage_y_um
        )
```

## Best Practices

1. **Use Standardized Keys**: Always use `MetadataSchema.make_key()` to create keys
2. **Publish on Change**: Update metadata when hardware state changes
3. **Register Detectors Early**: Register detectors during initialization
4. **Push Frame Events**: Push events at trigger time, not after acquisition
5. **Pop Events on Write**: Pop exactly the number of frames being written
6. **Clear Events**: Clear event queues between recordings

## Migration from Legacy Code

### Old Code (Legacy)
```python
# Old way - no units, no types
self._commChannel.sharedAttrs[('Position', 'Stage', 'X')] = 100.5
```

### New Code (Schema-Based)
```python
# New way - with schema
key = MetadataSchema.make_key(
    MetadataCategory.POSITIONER,
    'Stage',
    'X',
    'PositionUm'  # Field name indicates units
)
self._commChannel.sharedAttrs[key] = 100.5
```

## Troubleshooting

### Metadata Not Appearing in Files
- Check that MetadataHub is initialized in MasterController
- Verify SharedAttrsMetadataBridge is connected
- Ensure keys follow standardized format
- Check RecordingController is using `_get_detector_attrs()`

### Frame Event Count Mismatch
- Ensure push_frame_event() is called exactly once per frame
- Verify pop_frame_events() pops correct number
- Clear events between recordings with `clear_frame_events()`

### Missing Detector Metadata
- Verify detector is registered with hub
- Check detector context has required fields
- Ensure registration happens after detectorsManager init

## API Reference

See inline documentation in:
- `imswitch/imcontrol/model/metadata/metadata_hub.py`
- `imswitch/imcontrol/model/metadata/schema.py`
- `imswitch/imcontrol/model/metadata/sharedattrs_bridge.py`

## OME-types Integration

The hub provides direct OME-types object generation:

```python
# Generate complete OME metadata
ome = hub.to_ome(detector_names=['Camera1'])

# Export to OME-XML
xml_string = ome.to_xml()

# Use with OME-Zarr/OME-TIFF writers
# (Integration in progress)
```

---

*For questions or issues, consult the ImSwitch documentation or file an issue on GitHub.*
