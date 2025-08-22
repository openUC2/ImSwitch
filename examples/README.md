# ImSwitch Examples

This directory contains example scripts and demonstrations for ImSwitch functionality.

## MDA Integration Demo

`mda_demo.py` - Demonstrates the new MDA (Multi-Dimensional Acquisition) functionality using the useq-schema standard.

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

## Running with ImSwitch

Make sure ImSwitch is running with the API enabled before running these examples:

```bash
python main.py
# or
imswitch
```

The examples assume ImSwitch is running on `http://localhost:8000` by default.