# Native useq-schema MDA Integration - Implementation Summary

## Overview

ImSwitch now supports **native useq-schema `MDASequence` objects** following the same pattern as pymmcore-plus and raman-mda-engine. This enables protocol sharing across different microscopy systems.

## Key Changes

### 1. Refactored MDASequenceManager

The `MDASequenceManager` is now a full-featured MDA engine that:

- **Accepts native useq.MDASequence objects** directly (no simplified wrappers)
- **Supports ALL useq-schema features**:
  - `metadata`: Arbitrary experiment metadata
  - `stage_positions`: AbsolutePosition tuples for multi-position
  - `grid_plan`: GridRowsColumns for automated grid scanning
  - `channels`: Channel configurations with exposure times
  - `time_plan`: TIntervalLoops for time-lapse with automatic timing delays
  - `z_plan`: ZRangeAround for Z-stacks with automatic position generation
  - `autofocus_plan`: Autofocus strategies
  - `axis_order`: Order of acquisition (e.g., "tpcz")
  - `keep_shutter_open_across`: Keep illumination on across certain axes

### How Plans Are Handled

The useq-schema library automatically expands plans into individual `MDAEvent` objects:

- **z_plan**: Each event gets a `z_pos` field with the appropriate Z position
- **time_plan**: Each event gets a `min_start_time` field specifying when it should execute
- **grid_plan**: Each event gets `x_pos` and `y_pos` fields for grid positions

The `MDASequenceManager` executes these events, respecting:
- **Positioning**: Moves stage to specified x_pos, y_pos, z_pos before acquisition
- **Timing**: Waits until `min_start_time` before executing each event (handles time-lapse delays)
- **Channel setup**: Configures illumination and exposure for each channel

This automatic expansion means you just define high-level plans and the engine handles all the details.

### 2. Engine Pattern (like pymmcore-plus)

```python
# Step 1: Create engine
engine = MDASequenceManager()

# Step 2: Register with hardware managers
engine.register(
    detector_manager=detectorsManager,
    positioners_manager=positionersManager,
    lasers_manager=lasersManager,
    autofocus_manager=autofocusManager  # optional
)

# Step 3: Run native useq-schema sequence
engine.run_mda(sequence, output_path="/data/experiment")
```

This follows the **EXACT same pattern** as:
- `CMMCorePlus.register_mda_engine(engine)` + `run_mda(sequence)` in pymmcore-plus
- `RamanEngine()` + `core.register_mda_engine(engine)` in raman-mda-engine

### 3. Hook System

Custom logic can be added via hooks:

```python
def before_event(event):
    """Run before each acquisition."""
    if event.index.get('c', 0) == 0:
        # Run autofocus at first channel
        pass

def after_event(event, image):
    """Run after each acquisition."""
    # Analyze image, log data, etc.
    pass

engine.register_hook_before_event(before_event)
engine.register_hook_after_event(after_event)
```

### 4. Protocol Compatibility

The **SAME** `MDASequence` object works across systems:

```python
from useq import MDASequence, Channel, ZRangeAround

# Define protocol once
protocol = MDASequence(
    channels=[Channel(config="DAPI", exposure=50.0)],
    z_plan=ZRangeAround(range=10.0, step=2.0),
    axis_order="tzcg"
)

# Use on ImSwitch
imswitch_engine.run_mda(protocol)

# Use on pymmcore-plus
from pymmcore_plus import CMMCorePlus
core = CMMCorePlus()
core.run_mda(protocol)

# Save/load as JSON
import json
with open('protocol.json', 'w') as f:
    json.dump(protocol.dict(), f)
```

## Example Usage

### Basic Multi-Dimensional Acquisition

```python
from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround, AbsolutePosition
from imswitch.imcontrol.model.managers.MDASequenceManager import MDASequenceManager

# Create native useq-schema sequence
mda = MDASequence(
    metadata={
        "experiment": "cell_dynamics",
        "user": "researcher_name"
    },
    stage_positions=[
        AbsolutePosition(x=100.0, y=100.0, z=30.0),
        AbsolutePosition(x=200.0, y=150.0, z=35.0)
    ],
    channels=[
        Channel(config="BF"),      # Brightfield
        Channel(config="DAPI")     # DAPI fluorescence
    ],
    time_plan=TIntervalLoops(interval=1, loops=20),  # 20 timepoints, 1 sec apart
    z_plan=ZRangeAround(range=4.0, step=0.5),         # 4µm range, 0.5µm steps
    axis_order="tpcz"  # time, position, channel, z
)

# Execute with ImSwitch
engine = MDASequenceManager()
engine.register(
    detector_manager=controller._master.detectorsManager,
    positioners_manager=controller._master.positionersManager,
    lasers_manager=controller._master.lasersManager
)
engine.run_mda(mda, output_path="/data/experiment")
```

### With Autofocus

```python
def autofocus_hook(event):
    """Run autofocus at first Z position of each position."""
    if event.index.get('z', 0) == 0:
        print(f"Running autofocus at position {event.index.get('p', 0)}")
        # autofocus_manager.runAutofocus()

engine.register_hook_before_event(autofocus_hook)
engine.run_mda(sequence)
```

### Grid-Based Scanning

```python
from useq import GridRowsColumns

mda = MDASequence(
    metadata={"scan_type": "grid"},
    channels=[Channel(config="Brightfield")],
    grid_plan=GridRowsColumns(
        rows=3,
        columns=3,
        fov_width=100.0,   # µm
        fov_height=100.0
    )
)

engine.run_mda(mda)
```

## Files Changed

1. **`imswitch/imcontrol/model/managers/MDASequenceManager.py`**
   - Refactored to be a native useq-schema engine
   - Added `register()` and `run_mda()` methods
   - Full support for all useq-schema features
   - Hook system for custom logic
   - Backward compatible with WorkflowStep approach

2. **`examples/native_useq_mda_example.py`** (NEW)
   - Comprehensive examples following raman-mda-engine pattern
   - Shows all useq-schema features
   - Demonstrates hooks and protocol sharing

3. **`examples/README.md`**
   - Updated to recommend native useq-schema approach
   - Added usage examples and documentation links

## Benefits

1. **Standardization**: Uses the established useq-schema adopted by the microscopy community
2. **Protocol Portability**: Share protocols between ImSwitch, pymmcore-plus, and other systems
3. **Rich Features**: Full access to useq-schema capabilities (metadata, grid plans, autofocus, etc.)
4. **Validation**: Automatic parameter validation via Pydantic
5. **Extensibility**: Hook system for custom acquisition logic
6. **Future Proof**: Built on actively maintained standard

## Migration Guide

### Old Approach (Simplified API)

```python
# Old: Simplified wrapper API
experiment = {
    "channels": [{"name": "DAPI", "exposure": 50.0}],
    "z_range": 10.0,
    "z_step": 2.0,
    "time_points": 5
}
response = requests.post("/api/experimentcontroller/start_mda_experiment", json=experiment)
```

### New Approach (Native useq-schema)

```python
# New: Native useq-schema objects
from useq import MDASequence, Channel, ZRangeAround, TIntervalLoops

sequence = MDASequence(
    channels=[Channel(config="DAPI", exposure=50.0)],
    z_plan=ZRangeAround(range=10.0, step=2.0),
    time_plan=TIntervalLoops(interval=30.0, loops=5)
)

engine = MDASequenceManager()
engine.register(detector_mgr, pos_mgr, laser_mgr)
engine.run_mda(sequence)
```

## Testing

Run the comprehensive example:

```bash
python examples/native_useq_mda_example.py
```

This will show:
- Native useq-schema object creation
- All supported features
- Hook usage patterns
- Protocol sharing examples

## Next Steps

1. Test with actual ImSwitch hardware setup
2. Add more specialized examples (autofocus strategies, drift correction, etc.)
3. Consider adding REST API endpoints that accept native useq-schema JSON
4. Implement OME-TIFF/OME-Zarr writers that preserve useq metadata

## References

- [useq-schema documentation](https://pymmcore-plus.github.io/useq-schema/)
- [raman-mda-engine example](https://github.com/ianhi/raman-mda-engine/blob/main/examples/with-notebook.ipynb)
- [pymmcore-plus MDA guide](https://pymmcore-plus.github.io/pymmcore-plus/guides/mda/)
- [opto-loop-sim](https://github.com/ddd42-star/opto-loop-sim)
