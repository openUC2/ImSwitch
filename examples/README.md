# ImSwitch Examples

This directory contains example scripts and demonstrations for ImSwitch functionality.

## MDA Integration Examples

ImSwitch supports the useq-schema standard for multi-dimensional acquisition using **native useq.MDASequence objects**, following the same pattern as pymmcore-plus and raman-mda-engine.

### Key Features

- **Native useq-schema support**: Use native `useq.MDASequence` objects directly
- **Protocol compatibility**: Share protocols with pymmcore-plus, raman-mda-engine, and other useq-compatible systems
- **Full schema support**: metadata, stage_positions, grid_plan, channels, time_plan, z_plan, autofocus_plan, axis_order
- **Hook system**: Add custom logic for autofocus, drift correction, analysis, etc.
- **REST API**: Send MDA sequences from Jupyter notebooks or external scripts

### 1. Native useq-schema Example (Recommended)

`native_useq_mda_example.py` - Demonstrates using native `useq.MDASequence` objects following the **EXACT** pattern from pymmcore-plus and raman-mda-engine.

#### Requirements
```bash
pip install useq-schema
```

#### Usage
```bash
# Run the comprehensive example
python examples/native_useq_mda_example.py
```

#### Example Code

```python
from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround, AbsolutePosition
from imswitch.imcontrol.model.managers.MDASequenceManager import MDASequenceManager

# Create native useq-schema sequence
mda = MDASequence(
    metadata={"experiment": "test"},
    stage_positions=[
        AbsolutePosition(x=100.0, y=100.0, z=30.0),
        AbsolutePosition(x=200.0, y=150.0, z=35.0)
    ],
    channels=[Channel(config="BF"), Channel(config="DAPI")],
    time_plan=TIntervalLoops(interval=1, loops=20),
    z_plan=ZRangeAround(range=4.0, step=0.5),
    axis_order="tpcz"
)

# Register engine with ImSwitch managers
engine = MDASequenceManager()
engine.register(
    detector_manager=detectorsManager,
    positioners_manager=positionersManager,
    lasers_manager=lasersManager
)

# Run the sequence
engine.run_mda(mda, output_path="/data/experiment")
```

This follows the **same pattern** as:
- `core.run_mda(mda)` in pymmcore-plus
- `engine.run_mda(mda)` in raman-mda-engine

### 2. REST API with imswitchclient (NEW)

`mda_imswitchclient_example.py` - Shows how to create MDA sequences in Jupyter notebooks or external scripts and send them to ImSwitch via REST API.

#### Requirements
```bash
pip install useq-schema requests
# Optional: pip install imswitchclient
```

#### Usage
```bash
# Run the example
python examples/mda_imswitchclient_example.py
```

#### Example: XYZ Time-Lapse from Jupyter Notebook

```python
import requests
from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround, AbsolutePosition

# Create MDA sequence in Jupyter notebook
sequence = MDASequence(
    metadata={"experiment": "xyz_timelapse"},
    stage_positions=[
        AbsolutePosition(x=100.0, y=100.0, z=30.0),
        AbsolutePosition(x=200.0, y=150.0, z=35.0)
    ],
    channels=[Channel(config="Brightfield", exposure=10.0)],
    z_plan=ZRangeAround(range=10.0, step=2.0),
    time_plan=TIntervalLoops(interval=60.0, loops=10),
    axis_order="tpzc"
)

# Send to ImSwitch for execution
response = requests.post(
    "http://localhost:8000/api/experimentcontroller/run_native_mda_sequence",
    json=sequence.model_dump()  # or sequence.dict() for older pydantic
)

print(response.json())
# Output: {'status': 'started', 'save_directory': '/data/...', ...}
```

This pattern allows you to:
- **Formulate protocols outside ImSwitch** in Jupyter notebooks
- **Send via REST API** to a running ImSwitch instance
- **Execute remotely** on microscope hardware
- **Compatible with imswitchclient** library

### 3. MDA Demo (REST API)

`mda_demo.py` - Demonstrates the MDA functionality using the REST API endpoints.

### Requirements
```bash
pip install requests useq-schema
```

### Usage
```bash
# Check MDA capabilities and preview a simple Z-stack experiment
python mda_demo.py --server http://localhost:8000 --demo simple

# Preview a time-lapse experiment  
python mda_demo.py --server http://localhost:8000 --demo timelapse

# Preview a full multi-dimensional experiment
python mda_demo.py --server http://localhost:8000 --demo full
```

### Features Demonstrated
- Checking MDA capabilities via API
- Previewing experiments without execution
- Simple Z-stack with multiple channels
- Time-lapse imaging
- Full multi-dimensional acquisition (Z-stack + time-lapse + multi-channel)

The demo script shows the JSON structure for MDA experiments and displays expected sequence information without actually running the experiments. To execute real experiments, uncomment the `start_experiment()` calls in the script.

### MDA Engine Wrapper (Direct HAL Control)

`mda_engine_wrapper.py` - Demonstrates how to use useq-schema directly with ImSwitch's hardware abstraction layer, similar to the pymmcore-plus pattern.

#### Requirements
```bash
pip install useq-schema tifffile numpy
```

#### Usage
```bash
# Run the example to see the concept
python mda_engine_wrapper.py
```

#### Features Demonstrated
- Creating a custom MDA engine for ImSwitch
- Direct hardware control via managers (DetectorsManager, PositionersManager, LasersManager)
- Hook system for custom pre/post acquisition logic
- Integration with useq-schema MDASequence
- Compatible protocol pattern with pymmcore-plus

This pattern allows you to:
- Share experiment protocols with other useq-schema compatible systems (e.g., pymmcore-plus)
- Implement custom acquisition logic and hooks
- Have fine-grained control over hardware during acquisition
- Build complex workflows on top of the useq-schema standard

For detailed documentation on the engine pattern, see [`docs/mda_engine_pattern.md`](../docs/mda_engine_pattern.md).

## Running with ImSwitch

Make sure ImSwitch is running with the API enabled before running these examples:

```bash
python main.py
# or
imswitch
```

The examples assume ImSwitch is running on `http://localhost:8000` by default.
This directory contains example scripts demonstrating various ImSwitch features.

## Logging Demo

**File:** `logging_demo.py`

Demonstrates the enhanced logging system including:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- File logging with timestamp-based filenames
- Multiple loggers from different components

### Running the Demo

```bash
python examples/logging_demo.py
```

This will create temporary log files and demonstrate logging at different levels.

## More Examples

Additional examples will be added here to demonstrate other ImSwitch features.
