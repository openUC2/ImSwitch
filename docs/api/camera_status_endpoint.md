# Camera Status API Endpoint

## Overview
The `getCameraStatus` endpoint provides comprehensive camera information including hardware specifications, connection status, current settings, and available parameters.

## Endpoint

**URL:** `/SettingsController/getCameraStatus`

**Method:** `GET`

**Parameters:**
- `detectorName` (optional, string): Name of the detector to query. If not provided, returns status for the current detector.

## Response Structure

The endpoint returns a JSON object with the following fields:

### Hardware Information
- `model` (string): Camera model name
- `cameraType` (string): Camera type/driver (e.g., "HIK", "GXIPY", "Picamera2")
- `sensorWidth` (integer): Full sensor width in pixels
- `sensorHeight` (integer): Full sensor height in pixels
- `pixelSizeUm` (array): Physical pixel size in micrometers [Z, Y, X]

### Status Information
- `isMock` (boolean): Whether this is a mock/dummy camera
- `isConnected` (boolean): Current connection status
- `isRGB` (boolean): Whether camera is RGB/color
- `isAcquiring` (boolean): Whether camera is currently acquiring
- `isAdjustingParameters` (boolean): Whether parameters are being adjusted

### Frame Information
- `currentWidth` (integer): Current frame width (after ROI/binning)
- `currentHeight` (integer): Current frame height (after ROI/binning)
- `frameStart` (array): Current ROI position as [x, y]
- `binning` (integer): Current binning value
- `supportedBinnings` (array): List of supported binning values
- `croppable` (boolean): Whether ROI/cropping is supported

### Capabilities
- `forAcquisition` (boolean): Whether detector is used for acquisition
- `forFocusLock` (boolean): Whether detector is used for focus lock

### Parameters
- `parameters` (object): Dictionary of all detector parameters, each containing:
  - `value`: Current parameter value
  - `type`: Parameter type ("number", "list", "boolean")
  - `group`: Parameter group name
  - `editable`: Whether parameter can be edited
  - `units`: (for number parameters) Unit of measurement
  - `options`: (for list parameters) Available options

### Camera-Specific Information
Depending on the camera type, additional fields may be included:

#### HIK Cameras
- `hardwareParameters` (object): Raw hardware parameters from the camera
- `currentTriggerSource` (string): Current trigger source
- `availableTriggerTypes` (array): List of available trigger types

#### GXIPY Cameras
- `isStreaming` (boolean): Whether camera is actively streaming
- `currentTriggerSource` (string): Current trigger source
- `availableTriggerTypes` (array): List of available trigger types

#### Picamera2
- `cameraOpen` (boolean): Whether camera device is open
- `cameraIndex` (integer): Camera index/number

## Example Usage

### Get Current Camera Status
```bash
curl -X GET "http://localhost:8001/api/SettingsController/getCameraStatus"
```

### Get Specific Camera Status
```bash
curl -X GET "http://localhost:8001/api/SettingsController/getCameraStatus?detectorName=Camera1"
```

## Example Response

```json
{
  "model": "HIK MV-CS016-10UC",
  "cameraType": "HIK",
  "isMock": false,
  "isConnected": true,
  "isRGB": false,
  "sensorWidth": 1440,
  "sensorHeight": 1080,
  "currentWidth": 1440,
  "currentHeight": 1080,
  "pixelSizeUm": [1.0, 3.45, 3.45],
  "binning": 1,
  "supportedBinnings": [1],
  "frameStart": [0, 0],
  "croppable": true,
  "forAcquisition": true,
  "forFocusLock": false,
  "isAcquiring": false,
  "isAdjustingParameters": false,
  "currentTriggerSource": "Continous",
  "availableTriggerTypes": ["Continous", "Internal trigger", "External trigger"],
  "parameters": {
    "exposure": {
      "value": 100,
      "type": "number",
      "group": "Misc",
      "editable": true,
      "units": "ms"
    },
    "gain": {
      "value": 1,
      "type": "number",
      "group": "Misc",
      "editable": true,
      "units": "arb.u."
    },
    "blacklevel": {
      "value": 100,
      "type": "number",
      "group": "Misc",
      "editable": true,
      "units": "arb.u."
    },
    "isRGB": {
      "value": false,
      "type": "boolean",
      "group": "Misc",
      "editable": false
    },
    "trigger_source": {
      "value": "Continous",
      "type": "list",
      "group": "Acquisition mode",
      "editable": true,
      "options": ["Continous", "Internal trigger", "External trigger"]
    }
  }
}
```

## Implementation Details

### Base Implementation
The base `getCameraStatus()` method is implemented in `DetectorManager` and collects:
- Basic hardware specs from the manager initialization
- Current frame settings
- All registered parameters with their metadata

### Camera-Specific Implementations
Each detector manager (HIK, GXIPY, Picamera2) extends the base implementation to add:
- Camera type identification
- Connection status checking
- Hardware-specific parameters
- Driver-specific status information

## Use Cases

1. **Frontend Status Display**: Display comprehensive camera information in the UI
2. **Diagnostics**: Troubleshoot camera connection and configuration issues
3. **Configuration Validation**: Verify camera settings before starting acquisition
4. **Multi-Camera Systems**: Query status of multiple cameras independently
5. **Integration Testing**: Automated testing of camera configuration and status

## Error Handling

If an error occurs (e.g., invalid detector name), the endpoint returns:
```json
{
  "error": "Error message",
  "detectorName": "requested_detector_name",
  "status": "error"
}
```

## Related Endpoints

- `/SettingsController/getDetectorNames` - Get list of available detectors
- `/SettingsController/getDetectorParameters` - Get current detector parameters (legacy)
- `/SettingsController/setDetectorParameter` - Set a specific detector parameter
