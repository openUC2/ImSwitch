# Troubleshooting Jupyter Notebook Connection Issues

## Problem Summary

The user is experiencing issues connecting to the ImSwitch embedded kernel from Jupyter notebooks. While `jupyter console --existing` works correctly, the kernel doesn't appear properly in Jupyter notebook/lab interface.

## Root Cause Analysis

1. **User's Manual Kernel Spec is Incorrect**: The user created a kernel spec trying to use `jupyter_console` as a kernel:
   ```json
   {
     "argv": [
       "python", "-m", "jupyter_console", "--existing",
       "/Users/bene/Library/Jupyter/runtime/kernel-7038.json",
       "--Session.key=b''"
     ],
     "display_name": "ImSwitch Embedded Kernel",
     "language": "python"
   }
   ```
   This is fundamentally wrong because:
   - `jupyter_console` is a **client** that connects to kernels
   - Kernel specs need to launch actual **kernel processes**
   - The `--existing` flag only works for console clients, not kernel specs

2. **Correct Approach**: Use a proxy kernel that connects to the embedded kernel and relays messages.

## Solution Implemented

### 1. Proxy Kernel Architecture
- **`imswitch_kernel_proxy.py`**: A proper Jupyter kernel that connects to the embedded ImSwitch kernel
- **Custom Kernel Spec**: Points to the proxy kernel with the correct connection file
- **Automatic Registration**: The notebook server automatically detects and registers the kernel spec

### 2. Key Components

#### Proxy Kernel (`imswitch_kernel_proxy.py`)
```python
class ImSwitchKernelProxy(Kernel):
    def __init__(self, connection_file=None, **kwargs):
        super().__init__(**kwargs)
        self.connection_file = connection_file
        self.embedded_client = None
        self.connect_to_embedded_kernel()

    def connect_to_embedded_kernel(self):
        """Connect to the running ImSwitch embedded kernel"""
        # Uses BlockingKernelClient to connect to embedded kernel
        # Relays messages between notebook and embedded kernel

    def do_execute(self, code, silent, store_history=True, ...):
        """Execute code in the embedded kernel and relay results"""
        # Forwards code execution to embedded kernel
        # Streams back results, output, and errors
```

#### Kernel Spec Creation (`embedded_kernel_spec.py`)
```python
def create_imswitch_kernel_spec():
    """Creates a proper kernel spec that launches the proxy kernel"""
    kernel_json = {
        "argv": [
            sys.executable,
            "imswitch_kernel_proxy.py",
            "--connection-file", "/path/to/actual/kernel-file.json"
        ],
        "display_name": "ImSwitch (Live Connection)",
        "language": "python"
    }
```

#### Automatic Detection (`notebook_process.py`)
```python
def startnotebook(...):
    # Detects if embedded kernel is running
    # Automatically creates and installs kernel spec
    # Provides verification and troubleshooting info
```

### 3. Connection Flow

1. **ImSwitch starts** with `--with-kernel` flag
2. **Embedded kernel** creates connection file (e.g., `kernel-7038.json`)
3. **Notebook server starts** and detects the embedded kernel
4. **Kernel spec is created** pointing to proxy script with connection file
5. **Kernel spec is installed** in Jupyter (`imswitch_embedded`)
6. **User selects kernel** "ImSwitch (Live Connection)" in notebook interface
7. **Proxy kernel launches** and connects to embedded kernel
8. **Code execution** is relayed between notebook and ImSwitch

## Fixed Issues

### 1. Connection File Handling
- **Before**: Used placeholder `{connection_file}` that wasn't replaced
- **After**: Pass actual connection file path: `--connection-file /path/to/kernel-file.json`

### 2. Proxy Kernel Argument Parsing
- **Before**: No proper argument handling
- **After**: Custom `ImSwitchProxyKernelApp` that parses `--connection-file` argument

### 3. Cross-Platform Compatibility
- **Before**: Only checked Linux paths
- **After**: Support for macOS (`~/Library/Jupyter/runtime`), Linux, and Windows

### 4. Error Handling and Debugging
- **Before**: Minimal error information
- **After**: Comprehensive logging, connection verification, and troubleshooting output

### 5. Message Relay Improvements
- **Before**: Basic message forwarding
- **After**: Proper handling of all message types (stream, execute_result, display_data, error, status)

## Verification Steps

When the notebook server starts with an embedded kernel running, you should see:

```
==========================================================
EMBEDDED KERNEL DETECTED
==========================================================
ImSwitch embedded kernel is running
Connection file: /Users/bene/Library/Jupyter/runtime/kernel-7038.json

✅ ImSwitch kernel installed successfully!
   Look for 'ImSwitch (Live Connection)' in the notebook kernel list

✅ Kernel spec verification: FOUND in Jupyter

Alternative connection methods:
1. From Jupyter console (terminal):
   jupyter console --existing kernel-7038.json

2. From notebook cells:
   Use the helper notebook: connect_to_imswitch_kernel.ipynb

Available ImSwitch objects in embedded kernel:
   - detectorsManager, lasersManager, stageManager, etc.
   - master_controller, app, mainView, config
==========================================================
```

## What the User Should See

1. **In Jupyter Notebook/Lab**: "ImSwitch (Live Connection)" kernel option
2. **When selecting the kernel**: Successful connection without errors
3. **In notebook cells**: Direct access to ImSwitch managers:
   ```python
   # Should work immediately:
   lasersManager.getAllDeviceNames()
   detectorsManager.getAllDeviceNames()
   ```

## Common Issues and Solutions

### Issue: Kernel appears but fails to connect
- **Cause**: Connection file path is incorrect or embedded kernel stopped
- **Solution**: Restart ImSwitch with `--with-kernel`, then restart notebook server

### Issue: "ImSwitch (Live Connection)" doesn't appear
- **Cause**: Kernel spec installation failed
- **Solution**: Check console output for error messages, verify permissions

### Issue: Code execution fails with connection errors
- **Cause**: Proxy kernel can't connect to embedded kernel
- **Solution**: Verify embedded kernel is still running, check connection file exists

## Technical Notes

- The proxy kernel uses `BlockingKernelClient` to maintain a persistent connection
- Connection files are automatically detected from Jupyter runtime directory
- Kernel specs are installed in user space (`--user` flag) to avoid permission issues
- The implementation handles kernel restarts and connection recovery