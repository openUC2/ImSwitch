# Using useq-schema MDA Protocol with ImSwitch HAL

This guide explains how to use the useq-schema Multi-Dimensional Acquisition (MDA) protocol directly with ImSwitch's Hardware Abstraction Layer (HAL), similar to how it's used in pymmcore-plus.

## Overview

The useq-schema library provides a standardized way to define microscopy acquisition sequences that can be executed by different acquisition engines. ImSwitch now supports two approaches to using useq-schema:

1. **High-level API** (existing): Use the REST API endpoints to submit MDA experiments
2. **Engine Pattern** (new): Create custom MDA engines that directly control ImSwitch's HAL

The engine pattern gives you fine-grained control and allows you to integrate ImSwitch with useq-based workflows, similar to pymmcore-plus's approach.

## The Engine Pattern

### Concept

The engine pattern separates concerns:

- **useq-schema**: Defines WHAT to acquire (channels, positions, timepoints)
- **MDA Engine**: Defines HOW to acquire using specific hardware
- **ImSwitch HAL**: Provides hardware control primitives

This is the same pattern used by:
- [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus) with Micro-Manager
- [raman-mda-engine](https://github.com/ianhi/raman-mda-engine) for Raman microscopy
- [opto-loop-sim](https://github.com/ddd42-star/opto-loop-sim) for optogenetics

### Architecture

```
┌─────────────────────┐
│  useq.MDASequence   │  ← Protocol definition (software-agnostic)
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  ImSwitchMDAEngine  │  ← Execution engine (ImSwitch-specific)
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   ImSwitch HAL      │  ← Hardware abstraction layer
│  (Managers)         │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│   Physical Hardware │  ← Cameras, stages, lasers, etc.
└─────────────────────┘
```

## Example: Basic MDA Engine

See `examples/mda_engine_wrapper.py` for a complete implementation. Here's a simplified version:

```python
from useq import MDASequence, MDAEvent
import numpy as np

class ImSwitchMDAEngine:
    """Execute useq-schema sequences using ImSwitch hardware."""
    
    def __init__(self, detector_manager, positioners_manager, lasers_manager):
        self.detector_manager = detector_manager
        self.positioners_manager = positioners_manager
        self.lasers_manager = lasers_manager
        
    def run(self, sequence: MDASequence, output_path: str = None):
        """Execute an MDA sequence."""
        # Start detector
        detector = self.detector_manager.getAllDeviceNames()[0]
        self.detector_manager[detector].startAcquisition()
        
        # Execute each event
        for event in sequence:
            image = self._execute_event(event)
            if output_path:
                self._save_image(image, event, output_path)
                
    def _execute_event(self, event: MDAEvent) -> np.ndarray:
        """Execute a single acquisition event."""
        # Move stages
        if event.z_pos is not None:
            self._move_z(event.z_pos)
        if event.x_pos is not None or event.y_pos is not None:
            self._move_xy(event.x_pos, event.y_pos)
            
        # Setup channel
        if event.channel is not None:
            self._setup_channel(event.channel, event.exposure)
            
        # Acquire
        image = self._acquire_image()
        
        # Cleanup
        if event.channel is not None:
            self._cleanup_channel(event.channel)
            
        return image
```

## Using the Engine in ImSwitch

### From a Controller

```python
class MyController:
    def __init__(self, master):
        self._master = master
        
        # Create MDA engine
        self.mda_engine = ImSwitchMDAEngine(
            detector_manager=self._master.detectorsManager,
            positioners_manager=self._master.positionersManager,
            lasers_manager=self._master.lasersManager,
            logger=self._logger
        )
        
    def run_custom_mda(self):
        """Run a custom MDA experiment."""
        from useq import MDASequence, Channel, ZRangeAround
        
        # Define experiment
        sequence = MDASequence(
            channels=[
                Channel(config="DAPI", exposure=50.0),
                Channel(config="FITC", exposure=100.0)
            ],
            z_plan=ZRangeAround(range=10.0, step=2.0)
        )
        
        # Execute with ImSwitch hardware
        self.mda_engine.run(sequence, output_path="/data/experiment")
```

### With Custom Hooks

Add custom behavior before/after each acquisition:

```python
# Setup hooks
def before_acquisition(event: MDAEvent):
    """Called before each image acquisition."""
    print(f"Acquiring: channel={event.channel.config}, z={event.z_pos}")
    # Custom logic: autofocus, update UI, etc.
    
def after_acquisition(event: MDAEvent, image: np.ndarray):
    """Called after each image acquisition."""
    # Custom logic: analyze image, update display, etc.
    mean_intensity = image.mean()
    print(f"Mean intensity: {mean_intensity}")
    
# Register hooks
engine.register_hook_before_event(before_acquisition)
engine.register_hook_after_event(after_acquisition)

# Run sequence
engine.run(sequence)
```

### With Custom Autofocus

```python
class AutofocusHook:
    def __init__(self, autofocus_manager):
        self.autofocus_manager = autofocus_manager
        
    def __call__(self, event: MDAEvent):
        """Run autofocus before certain acquisitions."""
        # Only autofocus at first Z position of each position
        if event.index.get('z', 0) == 0:
            self.autofocus_manager.runAutofocus()
            
# Add to engine
autofocus_hook = AutofocusHook(self._master.autofocusManager)
engine.register_hook_before_event(autofocus_hook)
```

## Advanced: Grid Positions

useq-schema supports arbitrary position grids:

```python
from useq import MDASequence, Channel, GridRowsColumns, RelativePosition

# Define a 3x3 grid with 100μm spacing
sequence = MDASequence(
    channels=[Channel(config="Brightfield", exposure=10.0)],
    grid_plan=GridRowsColumns(
        rows=3,
        columns=3,
        relative_to=RelativePosition.CENTER,
        fov_width=100.0,  # μm
        fov_height=100.0,
    )
)

# Engine automatically handles all positions
engine.run(sequence)
```

## Advanced: Time-Lapse with Position Drift Correction

```python
# Track reference positions
reference_positions = {}

def drift_correction(event: MDAEvent):
    """Correct for stage drift at each position."""
    pos_idx = event.index.get('p', 0)
    
    # Store reference on first visit
    if pos_idx not in reference_positions:
        reference_positions[pos_idx] = (event.x_pos, event.y_pos)
        return
        
    # Calculate drift
    ref_x, ref_y = reference_positions[pos_idx]
    drift_x = event.x_pos - ref_x
    drift_y = event.y_pos - ref_y
    
    # Correct (this is simplified - real implementation would use image analysis)
    if abs(drift_x) > 1.0 or abs(drift_y) > 1.0:  # 1μm threshold
        print(f"Correcting drift: ({drift_x:.2f}, {drift_y:.2f})")
        # Apply correction via stage manager
        
engine.register_hook_before_event(drift_correction)
```

## Integration with Existing ImSwitch Features

### With RecordingManager

```python
class RecordingMDAEngine(ImSwitchMDAEngine):
    """MDA engine that saves via RecordingManager."""
    
    def __init__(self, *args, recording_manager, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording_manager = recording_manager
        
    def _save_image(self, image, event, output_path):
        """Save via RecordingManager for consistent storage."""
        metadata = {
            'channel': event.channel.config if event.channel else None,
            'z_pos': event.z_pos,
            'event_index': dict(event.index)
        }
        
        self.recording_manager.saveFrame(
            image,
            metadata=metadata,
            filename=self._generate_filename(event)
        )
```

### With WorkflowManager

The current implementation already converts MDASequence to WorkflowSteps. You can also use the engine pattern directly:

```python
# Option 1: Current approach (high-level)
# MDASequence → WorkflowSteps → Execute
workflow_steps = mda_manager.convert_sequence_to_workflow_steps(sequence, ...)

# Option 2: Engine approach (direct)
# MDASequence → Direct hardware control
engine.run(sequence)
```

Choose based on your needs:
- **WorkflowSteps**: Better integration with existing workflow UI and controls
- **Direct Engine**: More control, better for custom protocols

## Comparison with pymmcore-plus

| Aspect | pymmcore-plus | ImSwitch MDA Engine |
|--------|---------------|---------------------|
| Protocol | useq-schema | useq-schema (same!) |
| Hardware | Micro-Manager | ImSwitch HAL |
| Engine | PMCEngine | ImSwitchMDAEngine |
| Hooks | Before/after events | Before/after events |
| Autofocus | Hardware AF | AutofocusManager |
| Saving | NDTiff, OME-TIFF | Flexible (RecordingManager) |

**Key advantage**: Protocols written for pymmcore-plus can be adapted to ImSwitch by just changing the engine, since both use useq-schema!

## Example: Shared Protocol

This same sequence definition works with both systems:

```python
from useq import MDASequence, Channel, ZRangeAround, TIntervalLoops

# Define experiment (software-agnostic!)
experiment = MDASequence(
    channels=[
        Channel(config="DAPI", exposure=50.0),
        Channel(config="FITC", exposure=100.0)
    ],
    z_plan=ZRangeAround(range=10.0, step=2.0),
    time_plan=TIntervalLoops(interval=60.0, loops=10)
)

# Execute with pymmcore-plus
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import mda_engine
mmc = CMMCorePlus()
mda_engine.run_mda(mmc, experiment)

# Execute with ImSwitch (same experiment definition!)
from examples.mda_engine_wrapper import ImSwitchMDAEngine
engine = ImSwitchMDAEngine(detector_mgr, pos_mgr, laser_mgr)
engine.run(experiment)
```

## Benefits of the Engine Pattern

1. **Hardware Control**: Direct access to ImSwitch managers for fine-grained control
2. **Flexibility**: Easy to add custom logic, hooks, and behaviors
3. **Extensibility**: Subclass the engine for specific experiment types
4. **Compatibility**: Share protocols with other useq-schema systems
5. **Testability**: Easy to mock hardware for testing

## Next Steps

1. **Try the example**: Run `python examples/mda_engine_wrapper.py`
2. **Adapt to your hardware**: Modify the engine methods to match your specific ImSwitch configuration
3. **Add custom hooks**: Implement autofocus, analysis, or other custom behaviors
4. **Share protocols**: Use the same MDASequence definitions across different systems

## Resources

- [useq-schema documentation](https://pymmcore-plus.github.io/useq-schema/)
- [raman-mda-engine example](https://github.com/ianhi/raman-mda-engine)
- [opto-loop-sim example](https://github.com/ddd42-star/opto-loop-sim)
- [pymmcore-plus MDA guide](https://pymmcore-plus.github.io/pymmcore-plus/guides/mda/)
- ImSwitch MDA integration: `docs/mda_integration.md`

## Contributing

If you develop useful engine variants or hooks, please consider contributing them back to ImSwitch! Common patterns that would benefit the community:

- Autofocus strategies
- Drift correction algorithms
- Custom acquisition patterns
- Hardware-specific optimizations
- File format savers (OME-TIFF, N5, Zarr, etc.)
