# Camera Status API - Quick Start Guide

## What was added?

A new comprehensive camera status endpoint that returns all available information about a camera/detector in a single API call.

## Endpoint

```
GET /api/SettingsController/getCameraStatus?detectorName={optional}
```

## Quick Examples

### Get current camera status
```bash
curl http://localhost:8001/api/SettingsController/getCameraStatus
```

### Get specific camera status
```bash
curl http://localhost:8001/api/SettingsController/getCameraStatus?detectorName=Camera1
```

### Response example
```json
{
  "model": "HIK MV-CS016-10UC",
  "cameraType": "HIK",
  "isConnected": true,
  "isRGB": false,
  "sensorWidth": 1440,
  "sensorHeight": 1080,
  "pixelSizeUm": [1.0, 3.45, 3.45],
  "parameters": {
    "exposure": {
      "value": 100,
      "type": "number",
      "units": "ms"
    },
    "gain": {
      "value": 1,
      "type": "number", 
      "units": "arb.u."
    }
  }
}
```

## What information is included?

### Hardware Specs
- Camera model and type
- Sensor dimensions
- Physical pixel size
- Supported binning values

### Status
- Connection status
- Whether it's a mock/dummy camera
- RGB vs monochrome
- Currently acquiring

### Settings
- Current ROI and frame size
- Current binning
- All camera parameters (exposure, gain, etc.)

### Capabilities
- Whether cropping is supported
- Available trigger types
- What the camera is used for (acquisition/focus lock)

## Implementation Details

### Files Modified
1. `DetectorManager.py` - Base implementation
2. `HikCamManager.py` - HIK camera specific
3. `GXPIPYManager.py` - GXIPY camera specific  
4. `PiCamManager.py` - Picamera2 specific
5. `SettingsController.py` - API endpoint
6. `test_detector_api.py` - Tests

### Architecture
- Base method in `DetectorManager` collects common info
- Each camera manager extends with camera-specific details
- Safe handling of missing/optional information
- Structured parameter metadata

## Testing

Run the test:
```bash
cd /Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/ImSwitch
pytest imswitch/imcontrol/_test/api/test_detector_api.py::test_camera_status_endpoint -v
```

## Frontend Integration

See `CAMERA_STATUS_INTEGRATION_EXAMPLE.jsx` for:
- React hook for fetching status
- Full status display component
- Compact status badge component
- Example CSS styling

## Documentation

- **API Reference**: `docs/api/camera_status_endpoint.md`
- **Implementation Details**: `docs/CAMERA_STATUS_API_IMPLEMENTATION.md`
- **Frontend Example**: `microscope-app/DOCS/CAMERA_STATUS_INTEGRATION_EXAMPLE.jsx`

## Benefits

1. **Single API call** instead of multiple calls for different info
2. **Standardized interface** across all camera types
3. **Type metadata** for all parameters
4. **Extensible** - easy to add new camera types
5. **Safe** - handles missing information gracefully

## Comparison with existing endpoints

### Before
```javascript
// Multiple API calls needed
const names = await fetch('/api/SettingsController/getDetectorNames');
const params = await fetch('/api/SettingsController/getDetectorParameters');
// Still missing: sensor size, connection status, pixel size, etc.
```

### Now
```javascript
// Single API call
const status = await fetch('/api/SettingsController/getCameraStatus');
// Has everything: hardware specs, status, settings, capabilities
```

## Next Steps

1. Test with your camera setup
2. Integrate into frontend status displays
3. Use for diagnostics and monitoring
4. Extend with additional camera types as needed

## Questions?

See the detailed documentation in:
- `docs/api/camera_status_endpoint.md` - Full API documentation
- `docs/CAMERA_STATUS_API_IMPLEMENTATION.md` - Implementation details
