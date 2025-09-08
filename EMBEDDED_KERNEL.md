# Embedded Jupyter Kernel for ImSwitch

This feature enables a live Jupyter kernel inside ImSwitch for debugging and interactive access to managers, controllers, and variables while the application is running.

## Overview

The embedded Jupyter kernel allows you to:

- **Debug live experiments** by checking the state of managers
- **Call functions directly** (e.g., stage moves, camera acquisition) 
- **Prototype new analysis code** without restarting the full app
- **Inspect variables** and application state in real-time
- **Access hardware** and processing capabilities directly

## Usage

### Starting ImSwitch with Embedded Kernel

```bash
# Start with embedded kernel (headless mode)
python -m imswitch --with-kernel --headless

# Start with embedded kernel (GUI mode)  
python -m imswitch --with-kernel

# Combine with other options
python -m imswitch --with-kernel --config-file myconfig.json --headless
```

### Find folder of config file 

```py
In [1]: from IPython.lib.kernel import find_connection_file
In [2]: find_connection_file()
# in windows
In [1]: from jupyter_client import find_connection_file
In [2]: find_connection_file()
````

### Connecting to the Kernel

Once ImSwitch is running with `--with-kernel`, you can connect from another terminal:

#### Option 1: Jupyter Console
```bash
jupyter console --existing
```

#### Option 2: JupyterLab
```bash
jupyter lab
# Then select the ImSwitch kernel from the available kernels
```

#### Option 3: Existing Jupyter Notebook
In an existing Jupyter notebook, you can change the kernel to connect to the ImSwitch instance.

## Available Variables

When connected to the kernel, you have access to:

### Core Objects
- `moduleMainControllers` - Dictionary of all module controllers
- `app` - The Qt application object (if GUI mode)
- `mainView` - The main window object (if GUI mode)
- `config` - ImSwitch configuration object

### Individual Controllers
- `imcontrol_controller` - Main ImControl controller
- `{module}_controller` - Other module controllers

### Master Controller & Managers
- `master_controller` - ImControl's master controller
- `detectorsManager` - Camera/detector control
- `lasersManager` - Laser control  
- `stageManager` - Stage/positioning control
- `LEDsManager` - LED control
- `recordingManager` - Recording control
- And other managers depending on your setup...

## Example Usage

### Basic Operations

```python
# Check current detector
detectorsManager.current_detector

# Snap an image
image = detectorsManager.snap_image()

# Set exposure time
detectorsManager.set_exposure(100)  # 100ms

# Move stage
stageManager.move_to(x=100, y=50, z=10)

# Get current position
position = stageManager.get_position()
print(f"Current position: {position}")
```

### Laser Control

```python
# Set laser power
lasersManager.set_power('laser1', 50)  # 50% power

# Enable/disable laser
lasersManager.enable_laser('laser1')
lasersManager.disable_laser('laser1')

# Check laser status
status = lasersManager.get_status('laser1')
```

### Advanced Debugging

```python
# Inspect the full controller structure
print(dir(master_controller))

# Access internal state
print(master_controller.detectorsManager._internal_state)

# Call any method available on managers
result = master_controller.some_manager.some_method()

# Access configuration
print(config.default_config)
print(config.is_headless)
```

### Live Data Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Capture and analyze image
image = detectorsManager.snap_image()
mean_intensity = np.mean(image)
print(f"Mean intensity: {mean_intensity}")

# Plot histogram
plt.hist(image.flatten(), bins=50)
plt.title("Image Intensity Histogram")
plt.show()
```

## Implementation Details

### Security Considerations

The embedded kernel is guarded behind the `--with-kernel` flag for security reasons. Anyone with access to the kernel can execute arbitrary Python code with the same privileges as the ImSwitch process.

**Only enable this feature in trusted environments.**

### Threading

The kernel runs in a separate daemon thread to avoid blocking:
- **GUI mode**: Doesn't interfere with Qt event loop
- **Headless mode**: Doesn't interfere with the main loop

### Dependencies

Requires `ipykernel` package:
```bash
pip install ipykernel
```

If `ipykernel` is not available, ImSwitch will start normally but display a warning message.

## Troubleshooting

### "ipykernel not available"
Install ipykernel:
```bash
pip install ipykernel
```

### "No kernels available" in Jupyter
Make sure ImSwitch is running with `--with-kernel` flag and check the console output for kernel connection information.

### Connection refused
Ensure ImSwitch is still running and the kernel thread hasn't crashed. Check the ImSwitch console for error messages.

### Can't find managers
Some managers are only available if the corresponding hardware is configured in your setup file. Check your ImSwitch configuration.

## Configuration

The embedded kernel feature can be controlled via:

### Command Line
```bash
--with-kernel    # Enable embedded kernel
```

### Configuration File
```python
from imswitch.config import get_config
config = get_config()
config.enable_kernel = True
```

### Programmatically
```python
import imswitch
imswitch.main(with_kernel=True, is_headless=True)
```

## Examples

See `demo_embedded_kernel.py` for a complete demonstration of how the feature works with mock ImSwitch components.

## Implementation Reference

The implementation consists of:

1. **Configuration**: `enable_kernel` option in `ImSwitchConfig`
2. **Argument parsing**: `--with-kernel` flag in `__main__.py`
3. **Kernel function**: `start_embedded_kernel()` in `applaunch.py`
4. **Integration**: Namespace preparation and threading in `launchApp()`

Key files modified:
- `imswitch/config.py` - Added configuration option
- `imswitch/__main__.py` - Added command line argument
- `imswitch/imcommon/applaunch.py` - Added kernel functionality