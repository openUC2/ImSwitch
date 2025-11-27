# Hologram Controllers Documentation

## Overview

The hologram processing functionality has been split into two separate controllers for better modularity and maintainability:

1. **InLineHoloController** - For inline holography processing
2. **OffAxisHoloController** - For off-axis holography processing

The original `HoloController` now serves as a backward compatibility wrapper that imports `InLineHoloController`.

## Features

### Common Features (Both Controllers)

- ✅ **Backend Processing**: All hologram processing happens in the backend, not frontend
- ✅ **Frame Queue**: Configurable queue-based processing with rate limiting
- ✅ **Pause/Resume**: Pause stream to process last frame continuously, or resume to process live frames
- ✅ **Binning Support**: Automatic pixel size adjustment in reconstruction kernel
- ✅ **ROI Support**: Configurable region of interest
- ✅ **API Control**: Full REST API via `@APIExport` decorators
- ✅ **Signal Emission**: Processed frames emitted via signals for real-time display
- ✅ **Auto Camera Management**: Automatically starts camera if not running
- ✅ **Thread-Safe**: Uses locks for concurrent access protection

### InLineHoloController Specific

- Fresnel propagation for inline holograms
- Simple intensity-based reconstruction
- Fast processing for real-time applications

### OffAxisHoloController Specific

- Cross-correlation based spatial filtering
- Fresnel propagation after filtering
- Configurable CC center and radius
- Better for off-axis holography setups

## API Usage

### Starting Processing

```python
# Inline holography
inline_controller.start_processing()

# Off-axis holography
offaxis_controller.start_processing()
```

### Pause/Resume

```python
# Pause - processes last frame continuously at update_freq
controller.pause_processing()

# Resume - processes incoming frames continuously
controller.resume_processing()
```

### Setting Parameters

```python
# Single parameter updates
controller.set_wavelength(488e-9)  # 488 nm
controller.set_pixelsize(3.45e-6)  # 3.45 µm
controller.set_dz(0.005)  # 5 mm propagation distance
controller.set_binning(2)  # 2x2 binning

# Batch parameter update
controller.set_parameters({
    "wavelength": 488e-9,
    "pixelsize": 3.45e-6,
    "dz": 0.005,
    "roi_center": [512, 512],
    "roi_size": 256,
    "binning": 2,
    "update_freq": 10.0
})
```

### Off-Axis Specific

```python
# Set cross-correlation center and radius
offaxis_controller.set_cc_params(center=(256, 256), radius=80)
```

### Getting State

```python
# Get current state
state = controller.get_state()
# Returns: {
#   "is_processing": True,
#   "is_paused": False,
#   "frame_count": 123,
#   "processed_count": 100,
#   "queue_size": 2,
#   "last_process_time": 1234567890.123
# }

# Get current parameters
params = controller.get_parameters()
```

## Configuration

### Setup File Configuration

```json
{
  "holo": {
    "camera": "camera_name",  // Optional, uses first detector if not specified
    "pixelsize": 3.45e-6,     // meters
    "wavelength": 488e-9,      // meters
    "na": 0.3,
    "roi_center": [512, 512],  // Optional, defaults to image center
    "roi_size": 256,
    "update_freq": 10.0,       // Hz
    "binning": 1,              // 1, 2, 4, etc.
    
    // Off-axis specific:
    "cc_center": [256, 256],   // Optional
    "cc_radius": 100           // Optional
  }
}
```

## Implementation Details

### Frame Queue Mechanism

Both controllers implement a **frame queue** system that:

1. **Receives frames** from camera via `sigUpdateImage` signal
2. **Stores last frame** for pause mode
3. **Queues frames** (max 10) when processing is active and not paused
4. **Background thread** processes frames from queue at `update_freq` rate
5. **Emits results** via `sigHoloImageComputed` signal

### Pause vs Resume Behavior

- **Processing (not paused)**: 
  - Incoming frames → Queue → Process at update_freq → Emit
  - Queue keeps latest 10 frames, older frames dropped
  
- **Paused**: 
  - Incoming frames → Store as last_frame (no queue)
  - Last frame processed continuously at update_freq
  - Useful for parameter adjustment without live stream

### Binning Implementation

When binning > 1:
1. Image is binned (averaged) before processing
2. Pixel size in Fresnel kernel is automatically adjusted: `effective_pixelsize = pixelsize * binning`
3. This maintains correct physical scaling in reconstruction

### TODOs Resolved

All TODOs from the original implementation have been resolved:

✅ **Split inline/off-axis code** into separate controllers  
✅ **Auto-detect first available camera** if not specified in setup  
✅ **Frame queue mechanism** with unique frame handling  
✅ **Pause/resume functionality** for last-frame processing  
✅ **Binning support** with automatic kernel adjustment  
✅ **Sample request examples** in docstrings  
✅ **Removed unused legacy code** (`update()` method)  

## Signals

### sigHoloImageComputed
Emitted when a processed hologram is ready.

**Parameters:**
- `image` (np.ndarray): Processed hologram intensity
- `name` (str): Image name ("inline_holo" or "offaxis_holo")

### sigHoloStateChanged
Emitted when controller state changes.

**Parameters:**
- `state_dict` (dict): Current state dictionary

## Legacy GUI Compatibility

Both controllers maintain backward compatibility with existing GUI widgets:

- `setShowInLineHolo(enabled)` / `setShowOffAxisHolo(enabled)`
- `changeRate(updateRate)`
- `inLineValueChanged(magnitude)` / `offAxisValueChanged(magnitude)`
- `selectCCCenter()` (off-axis only)
- `updateCCCenter()` / `updateCCRadius()` (off-axis only)
- `displayImage(im, name)` for Napari integration

## Migration Guide

### From Old HoloController

The old `HoloController` is now deprecated but still works as an alias to `InLineHoloController`.

**No changes needed** if you only use inline holography.

**For off-axis**, update your setup to use:

```python
# In setup file
"OffAxisHoloController": {
    "camera": "camera_name",
    ...
}
```

### Key Differences

| Old HoloController | New Controllers |
|-------------------|----------------|
| Single `mode` parameter ("inline"/"offaxis") | Separate controllers |
| `start_processing(mode="inline")` | `inline_controller.start_processing()` |
| No pause/resume | `pause_processing()` / `resume_processing()` |
| No binning support | `set_binning(2)` |
| No frame queue | Built-in queue with rate limiting |

## Performance Considerations

- **Update Frequency**: Higher `update_freq` means more CPU usage. Typical: 5-20 Hz
- **ROI Size**: Smaller ROI = faster processing. FFT scales as O(N² log N)
- **Binning**: Higher binning = faster processing + less resolution
- **Queue Size**: Fixed at 10 frames to prevent memory buildup

## Examples

### Basic Inline Holography

```python
controller = InLineHoloController(...)
controller.set_wavelength(488e-9)
controller.set_dz(0.005)  # 5 mm
controller.start_processing()

# Later: adjust focus
controller.set_dz(0.008)  # 8 mm
```

### Off-Axis with Pause

```python
controller = OffAxisHoloController(...)
controller.set_cc_params((256, 256), 80)
controller.start_processing()

# Pause to adjust parameters on frozen frame
controller.pause_processing()
controller.set_dz(0.001)  # Fine-tune while looking at same frame

# Resume live processing
controller.resume_processing()
```

### High-Speed with Binning

```python
controller = InLineHoloController(...)
controller.set_binning(4)  # 4x4 binning for speed
controller.set_parameters({
    "update_freq": 30.0,  # 30 Hz processing
    "roi_size": 128       # Small ROI
})
controller.start_processing()
```

## Troubleshooting

### No frames being processed

- Check `get_state()["is_processing"]` - should be True
- Check `get_state()["is_paused"]` - should be False for live processing
- Verify camera is running: controller auto-starts but may fail
- Check `update_freq` is not too low

### Processing too slow

- Increase binning: `set_binning(2)` or `set_binning(4)`
- Reduce ROI size: `set_roi([cx, cy], 128)`
- Lower update frequency: `set_parameters({"update_freq": 5.0})`

### Off-axis reconstruction fails

- Verify CC center is set: `set_cc_params((x, y), radius)`
- Check CC center is within image bounds
- Try larger CC radius (e.g., 100-150 pixels)

## Copyright

Copyright (C) 2020-2024 ImSwitch developers

Licensed under GPL-3.0. See main project LICENSE for details.
