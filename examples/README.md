# ImSwitch Examples

This directory contains example scripts and demonstrations for ImSwitch functionality.

## MDA Integration Examples

ImSwitch supports the useq-schema standard for multi-dimensional acquisition in two ways:

1. **High-level REST API** (`mda_demo.py`) - Submit MDA experiments via HTTP endpoints
2. **Direct Engine Pattern** (`mda_engine_wrapper.py`) - Control ImSwitch HAL directly with useq-schema

### MDA Demo (REST API)

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
