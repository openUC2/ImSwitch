# Camera Status API Implementation Summary

## Overview
Added comprehensive camera status reporting capability to ImSwitch, providing a unified interface for retrieving detailed camera information across all detector types.

## Changes Made

### 1. Base Class Implementation (`DetectorManager.py`)

**New Method:** `getCameraStatus() -> Dict[str, Any]`

Provides comprehensive camera status information including:
- Hardware specifications (model, sensor size, pixel size)
- Connection and operational status
- Current frame settings (ROI, binning)
- All detector parameters with metadata
- Capabilities (croppable, forAcquisition, forFocusLock)

**Key Features:**
- Returns structured dictionary with all camera information
- Automatically collects parameter metadata (type, units, options)
- Extensible by subclasses for camera-specific information
- Safe handling of missing/optional parameters

### 2. Camera-Specific Implementations

#### HikCamManager (`HikCamManager.py`)

**Added Fields:**
- `cameraType`: "HIK"
- `isMock`: Based on mocktype setting
- `isConnected`: Checks for camera object and hardware connection
- `isAcquiring`: Current acquisition status
- `isAdjustingParameters`: Parameter adjustment status
- `hardwareParameters`: Raw camera parameters from SDK (if available)
- `currentTriggerSource`: Current trigger configuration
- `availableTriggerTypes`: Supported trigger modes

#### GXPIPYManager (`GXPIPYManager.py`)

**Added Fields:**
- `cameraType`: "GXIPY"
- `isMock`: Always false (real hardware only)
- `isConnected`: Checks camera object existence
- `isAcquiring`: Current acquisition status
- `isAdjustingParameters`: Parameter adjustment status
- `isStreaming`: Active streaming status
- `currentTriggerSource`: Current trigger configuration
- `availableTriggerTypes`: Supported trigger modes

#### PiCamManager (`PiCamManager.py`)

**Added Fields:**
- `cameraType`: "Picamera2"
- `isMock`: Always false
- `isConnected`: Checks camera_is_open status
- `isAcquiring`: Current acquisition status
- `isAdjustingParameters`: Parameter adjustment status
- `cameraOpen`: Camera device open status
- `cameraIndex`: Camera index/number

### 3. API Endpoint (`SettingsController.py`)

**New Endpoint:** `/SettingsController/getCameraStatus`

**Method:** GET

**Parameters:**
- `detectorName` (optional): Detector to query. If omitted, uses current detector.

**Returns:**
- Comprehensive camera status dictionary
- Error object if query fails

**Features:**
- Optional detector name parameter
- Returns current detector status if no name specified
- Graceful error handling with informative error messages
- Proper APIExport decoration for automatic OpenAPI documentation

### 4. Testing (`test_detector_api.py`)

**New Test:** `test_camera_status_endpoint()`

**Tests:**
- Endpoint accessibility
- Response structure validation
- Field presence verification
- Current detector query
- Specific detector query
- Error handling

**Validation:**
- Checks for all expected fields
- Prints key camera information
- Displays sample parameters
- Handles missing detectors gracefully

### 5. Documentation

**New Files:**
- `docs/api/camera_status_endpoint.md`: Complete API documentation
- `docs/CAMERA_STATUS_API_IMPLEMENTATION.md`: This implementation summary

## Data Structure

### Base Status Fields
```python
{
    'model': str,               # Camera model name
    'isMock': bool,            # Mock camera flag
    'isConnected': bool,       # Connection status
    'isRGB': bool,             # Color camera flag
    'sensorWidth': int,        # Full sensor width
    'sensorHeight': int,       # Full sensor height
    'currentWidth': int,       # Current frame width
    'currentHeight': int,      # Current frame height
    'pixelSizeUm': List[float], # [Z, Y, X] pixel size
    'binning': int,            # Current binning
    'supportedBinnings': List[int], # Supported binnings
    'frameStart': Tuple[int, int],  # ROI position
    'croppable': bool,         # ROI support flag
    'forAcquisition': bool,    # Acquisition detector flag
    'forFocusLock': bool,      # Focus lock detector flag
    'parameters': Dict[str, Dict] # All parameters with metadata
}
```

### Parameter Metadata Structure
```python
{
    'parameter_name': {
        'value': Any,          # Current value
        'type': str,           # 'number', 'list', 'boolean'
        'group': str,          # Parameter group
        'editable': bool,      # Editable flag
        'units': str,          # (number only) Units
        'options': List[str]   # (list only) Available options
    }
}
```

## Benefits

1. **Unified Interface**: All camera types provide consistent status information
2. **Comprehensive Information**: Single endpoint provides all camera details
3. **Type Safety**: Structured metadata for all parameters
4. **Extensibility**: Easy to add camera-specific fields
5. **Error Handling**: Graceful degradation when information unavailable
6. **Documentation**: Auto-generated OpenAPI documentation
7. **Testing**: Comprehensive test coverage
8. **Frontend Integration**: Ready for UI status displays

## Usage Examples

### Python API
```python
# Get current detector status
status = settingsController.getCameraStatus()

# Get specific detector status
status = settingsController.getCameraStatus(detectorName="Camera1")
```

### REST API
```bash
# Current detector
curl http://localhost:8001/api/SettingsController/getCameraStatus

# Specific detector
curl http://localhost:8001/api/SettingsController/getCameraStatus?detectorName=Camera1
```

### JavaScript/Frontend
```javascript
// Fetch camera status
const response = await fetch('/api/SettingsController/getCameraStatus');
const status = await response.json();

console.log(`Camera: ${status.model}`);
console.log(`Sensor: ${status.sensorWidth}x${status.sensorHeight}`);
console.log(`Connected: ${status.isConnected}`);
console.log(`Exposure: ${status.parameters.exposure.value} ${status.parameters.exposure.units}`);
```

## Compatibility

- **Backward Compatible**: No breaking changes to existing APIs
- **Complementary**: Works alongside existing parameter endpoints
- **Future-Proof**: Extensible design for new camera types
- **Standards Compliant**: Follows existing ImSwitch patterns

## Integration Points

### Existing Systems
- Uses existing DetectorManager infrastructure
- Leverages existing parameter system
- Integrates with APIExport framework
- Compatible with all detector types

### Frontend Integration
- Ready for microscope-app integration
- Can replace multiple separate API calls
- Provides all info for status displays
- Supports real-time monitoring

## Testing

Run the test with:
```bash
cd /Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/ImSwitch
pytest imswitch/imcontrol/_test/api/test_detector_api.py::test_camera_status_endpoint -v
```

## Future Enhancements

Potential improvements:
1. Add frame rate monitoring
2. Include buffer status information
3. Add temperature readings (if supported)
4. Include last error information
5. Add timestamp of last frame
6. Include performance metrics
7. Add calibration status

## Notes

- All camera-specific implementations use safe attribute checking
- Missing information gracefully handled with try/except blocks
- Debug logging for diagnostic purposes
- No performance impact on acquisition
- Thread-safe implementation

## Files Modified

1. `imswitch/imcontrol/model/managers/detectors/DetectorManager.py`
2. `imswitch/imcontrol/model/managers/detectors/HikCamManager.py`
3. `imswitch/imcontrol/model/managers/detectors/GXPIPYManager.py`
4. `imswitch/imcontrol/model/managers/detectors/PiCamManager.py`
5. `imswitch/imcontrol/controller/controllers/SettingsController.py`
6. `imswitch/imcontrol/_test/api/test_detector_api.py`

## Files Created

1. `docs/api/camera_status_endpoint.md`
2. `docs/CAMERA_STATUS_API_IMPLEMENTATION.md`
