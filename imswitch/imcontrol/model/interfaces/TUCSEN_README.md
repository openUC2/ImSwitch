# Tucsen Camera Integration for ImSwitch

This implementation provides support for Tucsen cameras in ImSwitch, specifically designed for Linux systems.

## Files Created

1. **tucsencamera.py** - Main camera interface following ImSwitch patterns
2. **tucsencamera_mock.py** - Mock implementation for testing without hardware
3. **TucsenCamManager.py** - Manager class that integrates with ImSwitch architecture

## Features

- Linux-specific implementation using Tucsen SDK
- Mock implementation for development/testing
- Threading-based frame acquisition
- Support for basic camera parameters:
  - Exposure time
  - Gain
  - Black level
  - Binning
  - Trigger modes (Continuous, Software, External)
- Frame buffering
- ROI support (placeholder)
- Flatfield correction support

## Configuration

To use the Tucsen camera in ImSwitch, add the following to your setup configuration:

```json
{
  "detectors": {
    "TucsenCam": {
      "managerName": "TucsenCamManager",
      "managerProperties": {
        "cameraListIndex": 0,
        "cameraEffPixelsize": 3.45,
        "isRGB": false,
        "tucsencam": {
          "exposure_time": 100,
          "gain": 1,
          "blacklevel": 100,
          "binning": 1
        }
      }
    }
  }
}
```

## Parameters

### Manager Properties
- `cameraListIndex`: Index of the camera in the Tucsen camera list (starts at 0)
- `cameraEffPixelsize`: Effective pixel size in micrometers
- `isRGB`: Whether the camera is an RGB camera (default: false)
- `tucsencam`: Dictionary of camera-specific properties

### Camera Properties
- `exposure_time`: Exposure time in milliseconds
- `gain`: Camera gain (arbitrary units)
- `blacklevel`: Black level offset
- `binning`: Binning factor
- `frame_rate`: Target frame rate (fps, -1 for maximum)

## Dependencies

### Required for real hardware
- Tucsen SDK (libTUCam.so.1)
- Proper camera drivers installed
- Linux operating system

### Always required
- Python ctypes
- NumPy
- Threading support

## Usage

The implementation automatically falls back to the mock camera if:
- Not running on Linux
- Tucsen SDK is not available
- Camera hardware is not connected
- Any initialization error occurs

## Implementation Notes

1. **Linux Only**: The real camera interface only works on Linux systems
2. **Threading**: Uses background threads for continuous frame acquisition
3. **Error Handling**: Graceful fallback to mock implementation on errors
4. **Memory Management**: Proper buffer allocation and cleanup
5. **API Compatibility**: Follows ImSwitch DetectorManager patterns

## Mock Camera

The mock implementation generates:
- Random noise patterns
- Circular features for testing
- Realistic frame timing
- All parameter simulation

## Future Improvements

1. Complete ROI implementation
2. More sophisticated binning support
3. Camera model detection and specific optimizations
4. Advanced trigger modes
5. Better error reporting and diagnostics
6. Windows support (if needed)

## Troubleshooting

### Common Issues

1. **SDK Not Found**: Ensure libTUCam.so.1 is in the library path
2. **Permission Errors**: Check USB device permissions
3. **Camera Not Detected**: Verify hardware connection and drivers
4. **Mock Camera Used**: Check logs for initialization errors

### Debugging

Enable debug logging to see detailed camera initialization and operation information.

```python
import logging
logging.getLogger('imswitch').setLevel(logging.DEBUG)
```

## Testing

The implementation can be tested with:
1. Mock camera (always available)
2. Real hardware (Linux + SDK required)
3. Parameter changes and acquisition cycles
4. Trigger mode switching
5. Buffer management
