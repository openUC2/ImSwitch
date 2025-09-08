# Connecting to ImSwitch Embedded Kernel from Jupyter Notebooks

This guide explains how to connect to the ImSwitch embedded Jupyter kernel from Jupyter notebooks and JupyterLab.

## The Problem

When you start ImSwitch with `--with-kernel`, it creates an embedded Jupyter kernel that can be accessed via:
```bash
jupyter console --existing  # This works!
```

However, Jupyter notebook and JupyterLab don't support the `--existing` flag:
```bash
jupyter notebook --existing  # ❌ "Unrecognized alias: 'existing'"
jupyter lab --existing       # ❌ "Unrecognized alias: 'existing'"
```

## The Solution

ImSwitch now provides several ways to connect to the embedded kernel from notebooks:

### Method 1: Custom Kernel Spec (Recommended)

When you start the ImSwitch notebook server while an embedded kernel is running, it automatically installs a custom kernel spec called **"ImSwitch (Live Connection)"**.

1. Start ImSwitch with embedded kernel:
   ```bash
   python -m imswitch --with-kernel --headless
   ```

2. Start the ImSwitch notebook server (this will detect and register the embedded kernel):
   ```bash
   # From ImSwitch interface or programmatically
   ```

3. In Jupyter notebook/lab, look for **"ImSwitch (Live Connection)"** in the kernel selection dropdown.

4. Select this kernel to get direct access to all ImSwitch managers:
   ```python
   # These are automatically available:
   detectorsManager.snap_image()
   lasersManager.getAllDeviceNames()
   stageManager.move_to(x=100, y=50)
   master_controller.some_method()
   ```

### Method 2: Jupyter Console (Always Works)

This is the most reliable method:

```bash
# In a separate terminal:
jupyter console --existing

# Or specify the exact kernel file:
jupyter console --existing kernel-XXXXX.json
```

### Method 3: Helper Notebook

Use the provided `connect_to_imswitch_kernel.ipynb` notebook that automatically finds and connects to the embedded kernel.

## How It Works

1. **Kernel State Tracking**: ImSwitch tracks the embedded kernel's connection file using `imswitch/imcommon/kernel_state.py`

2. **Kernel Spec Creation**: When the notebook server starts, it checks for a running embedded kernel and creates a custom kernel specification in `imswitch/imcommon/embedded_kernel_spec.py`

3. **Proxy Kernel**: The custom kernel spec uses `imswitch_kernel_proxy.py` to create a proxy that forwards commands to the actual embedded kernel

4. **Automatic Registration**: The notebook startup process (`imswitch/imnotebook/view/notebook_process.py`) automatically detects and registers the embedded kernel

## Files Modified

- `imswitch/imcommon/kernel_state.py` - Tracks embedded kernel connection info
- `imswitch/imcommon/embedded_kernel_spec.py` - Creates custom kernel specs  
- `imswitch/imcommon/applaunch.py` - Enhanced to store connection file info
- `imswitch/imnotebook/view/notebook_process.py` - Auto-detects and registers embedded kernel
- `imswitch/imnotebook/view/jupyter_notebook_config.py` - Updated for modern Jupyter server
- `imswitch_kernel_proxy.py` - Proxy kernel for connecting to embedded kernel
- `connect_to_imswitch_kernel.ipynb` - Helper notebook for manual connection

## Troubleshooting

### Kernel Not Appearing in Notebook

1. Make sure ImSwitch is running with `--with-kernel`
2. Check that the notebook server shows "EMBEDDED KERNEL DETECTED" on startup
3. Look for "✅ ImSwitch kernel installed successfully!" message
4. Refresh the notebook page and check the kernel dropdown

### Connection Issues

1. Verify the kernel connection file exists:
   ```bash
   ls ~/.local/share/jupyter/runtime/kernel-*.json
   ```

2. Try the console method first to verify the embedded kernel is working:
   ```bash
   jupyter console --existing
   ```

3. Check for error messages in the notebook server logs

### Manual Kernel Spec Installation

If automatic installation fails, you can manually install the kernel spec:

```python
from imswitch.imcommon.embedded_kernel_spec import install_imswitch_kernel_spec
install_imswitch_kernel_spec()
```

## Benefits

- **Live debugging**: Access all ImSwitch managers without stopping the application
- **Interactive development**: Test new code with real hardware connections
- **Visual analysis**: Combine ImSwitch data with Jupyter's rich visualization capabilities
- **Seamless integration**: Works with both GUI and headless ImSwitch modes