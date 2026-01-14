# Experiment Protocol Logging

## Overview

ImSwitch now automatically saves detailed experiment protocols and parameters to JSON files alongside your experiment data. This feature works for both **Normal Mode** and **Performance Mode** experiments, enabling full reproducibility and tracking of experimental conditions.

## File Location

Protocol files are saved in the same directory as your experiment data with the suffix `_protocol.json`:

- **Normal Mode**: `{experiment_dir}/{filename}_t{timepoint:04d}_protocol.json`
- **Performance Mode**: `{experiment_dir}/{timestamp}_FastStageScan_protocol.json`

## Protocol Contents

### Normal Mode Protocol

```json
{
  "timestamp": "2026-01-14T12:30:45.123456",
  "mode": "normal",
  "imswitch_version": "unknown",
  "experiment_name": "my_experiment",
  "experiment_mode": "normal",
  "directory": "/path/to/data",
  "filename": "experiment_001",
  "timepoint": 0,
  "total_timepoints": 5,
  "tile_count": 4,
  "step_count": 123,
  "snake_tiles": [
    [
      {"x": 0, "y": 0, "iX": 0, "iY": 0, "iterator": 0},
      {"x": 100, "y": 0, "iX": 1, "iY": 0, "iterator": 1}
    ]
  ],
  "z_positions": [0, 10, 20],
  "illumination_sources": ["Laser488", "Laser561"],
  "illumination_intensities": [100, 75],
  "exposures": [50, 50],
  "gains": [1.0, 1.0],
  "autofocus": {
    "enabled": true,
    "min": -50,
    "max": 50,
    "step_size": 5,
    "channel": "Laser488",
    "mode": "software",
    "max_attempts": 2,
    "target_focus_setpoint": null
  },
  "workflow_steps": [
    {
      "step_id": 0,
      "name": "Turn on single illumination source",
      "main_func": "set_laser_power",
      "main_params": {"power": 100, "channel": "Laser488"},
      "pre_funcs": [],
      "pre_params": {},
      "post_funcs": ["wait_time"],
      "post_params": {"seconds": 0.05},
      "max_retries": 0
    }
  ]
}
```

### Performance Mode Protocol

```json
{
  "timestamp": "2026-01-14T12:30:45.123456",
  "mode": "performance",
  "imswitch_version": "unknown",
  "experiment_name": "performance_scan",
  "experiment_mode": "performance",
  "trigger_mode": "hardware",
  "data_path": "/FastStageScan.ome.zarr",
  "scan_parameters": {
    "xstart": 0,
    "xstep": 500,
    "nx": 10,
    "ystart": 0,
    "ystep": 500,
    "ny": 10,
    "zstart": 0,
    "zstep": 0,
    "nz": 1,
    "tsettle": 90,
    "tExposure": 50,
    "illumination": [100, 75, 0, 0, 0],
    "led": 0
  },
  "timelapse": {
    "tPeriod": 1,
    "nTimes": 1
  },
  "illumination_sources": ["Laser488", "Laser561"],
  "illumination_intensities": [100, 75],
  "exposures": [50, 50],
  "total_frames": 100
}
```

## Key Features

### Automatic Saving
- Protocols are automatically saved at the end of experiment execution
- No user intervention required
- Works seamlessly with existing experiment workflows

### Complete Parameter Tracking
**Normal Mode tracks:**
- All workflow steps with function names and parameters
- XY stage positions (snake scan pattern)
- Z-stack positions
- Illumination settings per channel
- Exposure times and gain values
- Autofocus configuration
- Tile organization

**Performance Mode tracks:**
- Hardware scan parameters (grid size, step sizes)
- Timing parameters (settle time, exposure)
- Illumination channel configuration (as list)
- Trigger mode (hardware vs software)
- Timelapse settings
- Total frame count

### JSON Format Benefits
- Human-readable and editable
- Easy to parse programmatically
- Can be loaded into Python/MATLAB/R for analysis
- Supports version control (git)
- Compatible with lab notebooks and data management systems

## Usage Examples

### Loading Protocol in Python

```python
import json

# Load protocol file
with open('experiment_001_t0000_protocol.json', 'r') as f:
    protocol = json.load(f)

# Access parameters
print(f"Experiment mode: {protocol['mode']}")
print(f"Total tiles: {protocol['tile_count']}")
print(f"Illumination sources: {protocol['illumination_sources']}")
print(f"Z positions: {protocol['z_positions']}")

# Iterate through workflow steps (normal mode)
for step in protocol['workflow_steps']:
    print(f"Step {step['step_id']}: {step['name']}")
    print(f"  Function: {step['main_func']}")
    print(f"  Parameters: {step['main_params']}")
```

### Comparing Protocols

```python
import json
from deepdiff import DeepDiff

with open('experiment_001_protocol.json', 'r') as f:
    protocol1 = json.load(f)
    
with open('experiment_002_protocol.json', 'r') as f:
    protocol2 = json.load(f)

# Find differences
diff = DeepDiff(protocol1, protocol2, ignore_order=True)
print(diff)
```

### Recreating Experiment

```python
import json

# Load protocol
with open('experiment_protocol.json', 'r') as f:
    protocol = json.load(f)

# Use parameters to configure new experiment
controller.set_exposure_times(protocol['exposures'])
controller.set_gains(protocol['gains'])

for source, intensity in zip(protocol['illumination_sources'], 
                             protocol['illumination_intensities']):
    controller.set_laser_power(power=intensity, channel=source)
```

## Implementation Details

### Base Class Method
The `save_experiment_protocol()` method is implemented in `ExperimentModeBase` and inherited by both mode classes:

```python
def save_experiment_protocol(self, 
                            protocol_data: Dict[str, Any],
                            file_path: str,
                            mode: str = "unknown") -> str:
    """Save experiment protocol and parameters to JSON file."""
```

### Custom JSON Serialization
A custom serializer handles non-standard types:
- NumPy arrays → Python lists
- NumPy integers/floats → Python int/float
- Datetime objects → ISO format strings
- Callable functions → Function name strings
- Objects with `__dict__` → Dictionary representation

### Normal Mode Integration
- Called after workflow creation in `execute_experiment()`
- Serializes all `WorkflowStep` objects with `_serialize_workflow_step()`
- Includes complete tile and illumination configuration

### Performance Mode Integration
- Called after scan completion in `_execute_scan_background()`
- Extracts parameters from `scan_params` and `experiment_params`
- Implemented in `_save_performance_protocol()`

## Troubleshooting

### Protocol File Not Created
- Check that the experiment completed successfully
- Verify write permissions in the save directory
- Check ImSwitch logs for error messages

### Missing Parameters
- Some parameters may be `null` if not set in experiment
- Default values are used when optional parameters are missing

### Large Protocol Files
- Normal mode protocols with many steps can be large (>1MB)
- Consider compressing protocol files for long-term storage
- Performance mode protocols are typically <10KB

## Future Enhancements

Potential future improvements:
- Protocol comparison tools
- Protocol validation
- Experiment replay from protocol file
- Protocol templates for common experiments
- Integration with LIMS systems
- Metadata schema validation

## Related Documentation

- [Experiment Controller](experiment_controller.md)
- [OME-Zarr Storage](ome_zarr_storage.md)
- [Workflow System](workflow_system.md)
