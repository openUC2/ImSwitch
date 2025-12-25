# REST API Integration for Native useq-schema MDA Sequences

## Overview

ImSwitch now supports sending native useq-schema `MDASequence` objects to a running ImSwitch instance via REST API. This enables **external protocol formulation** in Jupyter notebooks or Python scripts, followed by **remote execution** on ImSwitch hardware.

## New REST API Endpoint

### `POST /api/experimentcontroller/run_native_mda_sequence`

Accepts a native `useq.MDASequence` serialized as JSON (dict) and executes it using ImSwitch's hardware abstraction layer.

**Endpoint:** `http://localhost:8000/api/experimentcontroller/run_native_mda_sequence`

**Method:** POST

**Content-Type:** application/json

**Request Body:** Native useq.MDASequence serialized to dict/JSON

**Response:** JSON with execution status and sequence information

## Usage Pattern

This follows the workflow requested by @beniroquai:

1. **Formulate protocol** in Jupyter notebook using native useq-schema
2. **Send request** to ImSwitch REST API
3. **Execute remotely** on microscope hardware

## Example 1: Simple XYZ Time-Lapse

```python
import requests
from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround, AbsolutePosition

# Create native useq-schema sequence
sequence = MDASequence(
    metadata={"experiment": "xyz_timelapse", "user": "researcher"},
    stage_positions=[
        AbsolutePosition(x=100.0, y=100.0, z=30.0),
        AbsolutePosition(x=200.0, y=150.0, z=35.0),
        AbsolutePosition(x=150.0, y=200.0, z=32.0)
    ],
    channels=[Channel(config="Brightfield", exposure=10.0)],
    z_plan=ZRangeAround(range=10.0, step=2.0),  # 10µm Z-range, 2µm steps
    time_plan=TIntervalLoops(interval=60.0, loops=10),  # 10 timepoints, 1 min apart
    axis_order="tpzc"  # time, position, z, channel
)

# Send to ImSwitch for execution
response = requests.post(
    "http://localhost:8000/api/experimentcontroller/run_native_mda_sequence",
    json=sequence.model_dump(),  # or sequence.dict() for older pydantic
    timeout=10  # Connection timeout, not execution timeout
)

result = response.json()
print(f"Status: {result['status']}")
print(f"Save directory: {result['save_directory']}")
print(f"Estimated duration: {result['estimated_duration_minutes']:.1f} minutes")
```

**Response:**
```json
{
  "status": "started",
  "sequence_info": {
    "total_events": 180,
    "channels": ["Brightfield"],
    "z_positions": [...],
    "time_points": [0, 1, 2, ..., 9],
    "xy_positions": 3,
    "axis_order": ["t", "p", "z", "c"],
    "metadata": {...},
    "estimated_duration_minutes": 32.5
  },
  "save_directory": "/data/NativeMDA/xyz_timelapse/20250121_195207",
  "estimated_duration_minutes": 32.5,
  "message": "Native MDA sequence started in background thread"
}
```

## Example 2: Multi-Channel Z-Stack

```python
from useq import MDASequence, Channel, ZRangeAround, AbsolutePosition

sequence = MDASequence(
    metadata={"experiment": "multi_channel_zstack"},
    stage_positions=[
        AbsolutePosition(x=0.0, y=0.0, z=10.0),
        AbsolutePosition(x=100.0, y=0.0, z=10.0)
    ],
    channels=[
        Channel(config="DAPI", exposure=50.0),
        Channel(config="FITC", exposure=100.0),
        Channel(config="TRITC", exposure=150.0)
    ],
    z_plan=ZRangeAround(range=20.0, step=2.0),
    axis_order="pczg"  # position, channel, z
)

response = requests.post(
    "http://localhost:8000/api/experimentcontroller/run_native_mda_sequence",
    json=sequence.model_dump()
)
```

## Example 3: Using the MDAClient Helper Class

The example file includes a helper class for easier usage:

```python
from examples.mda_imswitchclient_example import MDAClient
from useq import MDASequence, Channel, ZRangeAround

# Create client
client = MDAClient(base_url="http://localhost:8000")

# Check capabilities
caps = client.check_mda_available()
print(f"Available channels: {caps['available_channels']}")

# Create and run sequence
sequence = MDASequence(...)
result = client.run_mda_sequence(sequence)
```

## Integration with imswitchclient Library

This endpoint can be integrated into the `imswitchclient` library's `experimentController` module:

**Proposed imswitchclient API:**

```python
from imswitchclient import ImSwitchClient
from useq import MDASequence, Channel, ZRangeAround

# Connect to ImSwitch
client = ImSwitchClient('localhost', 8000)

# Create MDA sequence
sequence = MDASequence(
    channels=[Channel(config='DAPI', exposure=50.0)],
    z_plan=ZRangeAround(range=10.0, step=2.0)
)

# Execute via experimentController
result = client.experimentController.run_mda_sequence(sequence)
```

**Implementation in imswitchclient:**

```python
# In imswitchclient/experimentController.py

def run_mda_sequence(self, sequence):
    """
    Execute a native useq-schema MDASequence.
    
    Args:
        sequence: useq.MDASequence object
        
    Returns:
        Response dict with execution status
    """
    sequence_dict = sequence.model_dump() if hasattr(sequence, 'model_dump') else sequence.dict()
    
    response = self.client.post(
        '/api/experimentcontroller/run_native_mda_sequence',
        json=sequence_dict
    )
    return response.json()
```

## Request Format

The request body should be a JSON object with the following structure (native useq-schema format):

```json
{
  "metadata": {
    "experiment": "my_experiment",
    "user": "researcher_name"
  },
  "axis_order": ["t", "p", "c", "z"],
  "stage_positions": [
    {"x": 100.0, "y": 100.0, "z": 30.0},
    {"x": 200.0, "y": 150.0, "z": 35.0}
  ],
  "channels": [
    {"config": "BF", "exposure": 50.0},
    {"config": "DAPI", "exposure": 100.0}
  ],
  "time_plan": {
    "interval": 1,
    "loops": 20
  },
  "z_plan": {
    "range": 4.0,
    "step": 0.5
  },
  "grid_plan": null,
  "autofocus_plan": null,
  "keep_shutter_open_across": []
}
```

## Response Format

```json
{
  "status": "started",
  "sequence_info": {
    "total_events": 720,
    "channels": ["BF", "DAPI"],
    "z_positions": [...],
    "time_points": [0, 1, 2, ..., 19],
    "xy_positions": 2,
    "axis_order": ["t", "p", "c", "z"],
    "metadata": {...},
    "estimated_duration_minutes": 15.2
  },
  "save_directory": "/data/NativeMDA/my_experiment/20250121_195207",
  "estimated_duration_minutes": 15.2,
  "message": "Native MDA sequence started in background thread"
}
```

## Error Handling

**useq-schema not available:**
```json
{
  "detail": "useq-schema not available. Install with: pip install useq-schema"
}
```

**Invalid sequence format:**
```json
{
  "detail": "Error starting native MDA sequence: <error message>"
}
```

## Execution Model

The MDA sequence is executed in a **background thread** to avoid blocking the REST API. This means:

- The API endpoint returns immediately with `"status": "started"`
- The sequence executes asynchronously on ImSwitch hardware
- Images are saved to the specified directory
- Check logs for execution progress and completion

## Benefits

1. **Remote Protocol Formulation**: Create protocols in Jupyter notebooks, separate from ImSwitch runtime
2. **REST API Integration**: Standard HTTP interface, works with any HTTP client
3. **imswitchclient Compatible**: Can be integrated into the imswitchclient library
4. **Protocol Portability**: Same useq-schema format works across different systems
5. **Language Agnostic**: Can be called from Python, JavaScript, curl, etc.

## Use Cases

### 1. Interactive Protocol Development
Develop and test protocols in Jupyter notebooks with immediate feedback before running on hardware.

### 2. Batch Execution
Submit multiple protocols programmatically from a script.

### 3. Remote Control
Control microscope from a different machine (e.g., from a computational server).

### 4. Workflow Automation
Integrate ImSwitch MDA into larger analysis pipelines.

### 5. Multi-System Deployment
Use the same protocol definition on ImSwitch, pymmcore-plus, and other systems.

## Example Files

- **`examples/mda_imswitchclient_example.py`** - Comprehensive examples showing all patterns
- **`docs/NATIVE_USEQ_MDA.md`** - Complete native useq-schema guide
- **`examples/native_useq_mda_example.py`** - Direct engine usage examples

## Testing

Run the example:
```bash
# Start ImSwitch first
python main.py

# In another terminal
python examples/mda_imswitchclient_example.py
```

## Future Enhancements

Possible improvements:
- Add `/get_mda_status` endpoint to check execution progress
- Add `/stop_mda_sequence` endpoint to cancel running sequences
- Support for streaming image data via WebSocket
- Integration with ImSwitch's live view system
- Real-time progress updates via SSE (Server-Sent Events)

## References

- useq-schema: https://pymmcore-plus.github.io/useq-schema/
- imswitchclient: https://github.com/openUC2/imswitchclient
- pymmcore-plus: https://pymmcore-plus.github.io/pymmcore-plus/
- raman-mda-engine: https://github.com/ianhi/raman-mda-engine
