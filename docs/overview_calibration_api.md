# Overview Camera Calibration API Documentation

This document describes the REST API endpoints for overview-camera-based stage axis identification, illumination mapping, and calibration verification.

## Overview

The overview calibration API provides automatic detection and calibration of:
- Stage axis directions and signs using AprilTag tracking
- Illumination channel to color/wavelength mapping
- Homing polarity verification
- Step size sign correction
- Per-objective reference images

All operations use an overhead/observation camera and AprilTag markers for robust, automated calibration.

## Prerequisites

1. **Observation Camera**: Configure an observation camera in your setup:
   ```json
   {
     "PixelCalibration": {
       "ObservationCamera": "TopCam_1"
     }
   }
   ```

2. **AprilTag Marker**: Place an AprilTag marker (36h11 family) visible to the observation camera. The tag should be fixed to the stage or sample holder.

3. **Dependencies**: Ensure `opencv-contrib-python` is installed for AprilTag detection.

## API Endpoints

All endpoints are accessible via the ImSwitch REST API at `/api/pixelCalibration/overview/...`

### 1. Check Observation Camera Availability (GET)

Check if the observation camera is configured and available.

**Endpoint:** `/overviewIsObservationCameraAvailable`

**Method:** GET

**Parameters:** None

**Example Response (Available):**
```json
{
  "available": true,
  "name": "TopCam_1"
}
```

**Example Response (Not Available):**
```json
{
  "available": false,
  "name": null
}
```

**Notes:**
- Returns HTTP 200 regardless of availability
- Check the `available` field before calling other endpoints

---

### 2. Identify Stage Axes (POST)

Automatically identify stage axis directions and signs using AprilTag tracking.

**Endpoint:** `/overviewIdentifyAxes`

**Method:** POST

**Parameters:**
- `stepUm` (float, optional): Step size in micrometers. Default: 2000.0

**Example Request:**
```json
{
  "stepUm": 2000.0
}
```

**Example Response (Success):**
```json
{
  "mapping": {
    "stageX_to_cam": "width",
    "stageY_to_cam": "height"
  },
  "sign": {
    "X": 1,
    "Y": -1
  },
  "samples": [
    {
      "stageMove": [2000, 0],
      "camShift": [150.5, -2.3]
    },
    {
      "stageMove": [0, 2000],
      "camShift": [1.2, -180.7]
    }
  ]
}
```

**Example Response (Error):**
```json
{
  "error": "No AprilTag detected in initial frame"
}
```

**Notes:**
- This operation takes ~5-10 seconds
- Stage will move +X and +Y by `stepUm` micrometers
- Automatically saves results to configuration
- Returns HTTP 409 if observation camera not available

**Interpretation:**
- `mapping`: Which camera axis (width=u, height=v) corresponds to each stage axis
- `sign`: Direction multiplier (+1 or -1) for each axis
- `samples`: Raw measurement data showing stage movement vs. camera displacement

---

### 3. Map Illumination Channels (POST)

Automatically map illumination channels (lasers/LEDs) to colors and wavelengths.

**Endpoint:** `/overviewMapIlluminationChannels`

**Method:** POST

**Parameters:** None

**Example Response (Success):**
```json
{
  "illuminationMap": [
    {
      "channel": "Laser_635",
      "type": "laser",
      "wavelength_nm": 635.0,
      "color": "red",
      "mean_intensity": 1250.3,
      "max_intensity": 3500.8
    },
    {
      "channel": "LED_matrix",
      "type": "led",
      "wavelength_nm": null,
      "color": "white",
      "mean_intensity": 850.2,
      "max_intensity": 2100.5
    }
  ],
  "darkStats": {
    "mean": 15.3,
    "max": 45,
    "shape": [1280, 720]
  }
}
```

**Example Response (Error):**
```json
{
  "error": "No illumination sources available"
}
```

**Notes:**
- This operation takes ~2-5 seconds per channel
- All illumination sources are turned off initially to capture dark reference
- Each channel is tested at 50% power
- Automatically saves results to configuration
- Color classification:
  - `uv`: < 410 nm
  - `blue`: 410-500 nm
  - `green`: 500-580 nm
  - `yellow`: 580-600 nm
  - `red`: 600-700 nm
  - `far-red`: 700-780 nm
  - `ir`: > 780 nm
  - `white`: Multi-band or unknown wavelength

---

### 4. Verify Homing (POST)

Verify homing behavior and detect inverted motor directions.

**Endpoint:** `/overviewVerifyHoming`

**Method:** POST

**Parameters:**
- `maxTimeS` (float, optional): Maximum time to wait for homing (seconds). Default: 20.0

**Example Request:**
```json
{
  "maxTimeS": 20.0
}
```

**Example Response (Success):**
```json
{
  "X": {
    "inverted": false,
    "evidence": [
      {"time": 0.5, "centroid": [320.5, 240.2], "motion": 5.3},
      {"time": 1.0, "centroid": [315.2, 240.1], "motion": 5.3},
      {"time": 5.5, "centroid": [180.1, 240.5], "motion": 0.8}
    ],
    "lastCheck": "2025-11-08T20:30:00"
  },
  "Y": {
    "inverted": true,
    "evidence": [
      {"time": 0.5, "centroid": [320.5, 240.2], "motion": 4.8},
      {"time": 1.0, "centroid": [320.2, 235.1], "motion": 5.1},
      {"time": 3.0, "centroid": null, "note": "tag_lost"}
    ],
    "lastCheck": "2025-11-08T20:30:05",
    "recommendation": "Invert motor direction"
  }
}
```

**Notes:**
- This operation takes up to `maxTimeS` seconds per axis
- Monitors AprilTag motion during homing
- If tag is lost (moves out of view), suggests motor direction inversion
- Automatically saves results to configuration
- **Use with caution**: Homing may move stage to limits

**Interpretation:**
- `inverted: true`: Motor direction should be inverted in configuration
- `inverted: false`: Motor direction is correct
- `evidence`: Time-series data of tag centroid and motion
- `tag_lost`: Indicates tag moved out of camera field of view

---

### 5. Fix Step Sign (POST)

Determine correct step size sign by visiting rectangle corners.

**Endpoint:** `/overviewFixStepSign`

**Method:** POST

**Parameters:**
- `rectSizeUm` (float, optional): Rectangle size in micrometers. Default: 20000.0

**Example Request:**
```json
{
  "rectSizeUm": 20000.0
}
```

**Example Response (Success):**
```json
{
  "sign": {
    "X": 1,
    "Y": -1
  },
  "samples": [
    {
      "stage_pos": [0.0, 0.0],
      "cam_centroid": [320.5, 240.2],
      "target": [0, 0]
    },
    {
      "stage_pos": [20000.0, 0.0],
      "cam_centroid": [470.3, 240.5],
      "target": [20000, 0]
    },
    {
      "stage_pos": [20000.0, 20000.0],
      "cam_centroid": [470.1, 90.8],
      "target": [20000, 20000]
    },
    {
      "stage_pos": [0.0, 20000.0],
      "cam_centroid": [320.2, 90.5],
      "target": [0, 20000]
    }
  ]
}
```

**Example Response (Error):**
```json
{
  "error": "AprilTag lost at corner 2"
}
```

**Notes:**
- This operation takes ~10-20 seconds
- Stage visits 4 corners of a rectangle
- Compares stage displacement with camera displacement
- Sign of -1 indicates axis needs inversion
- Automatically saves results to configuration
- Ensure sufficient travel range for rectangle size

---

### 6. Capture Objective Image (POST)

Capture and save a reference image for a specific objective slot.

**Endpoint:** `/overviewCaptureObjectiveImage`

**Method:** POST

**Parameters:**
- `slot` (integer, required): Objective slot number

**Example Request:**
```json
{
  "slot": 1
}
```

**Example Response (Success):**
```json
{
  "slot": 1,
  "path": "/path/to/ImSwitchConfig/objective1_calibration.png"
}
```

**Example Response (Error):**
```json
{
  "error": "Failed to save image",
  "slot": 1
}
```

**Notes:**
- Captures current observation camera frame
- Saves as PNG in ImSwitch configuration directory
- Automatically updates configuration with image path
- Useful for documenting objective alignment and calibration state

---

### 7. Get Overview Configuration (GET)

Retrieve current overview calibration configuration.

**Endpoint:** `/overviewGetConfig`

**Method:** GET

**Parameters:** None

**Example Response:**
```json
{
  "axes": {
    "mapping": {
      "stageX_to_cam": "width",
      "stageY_to_cam": "height"
    },
    "sign": {
      "X": 1,
      "Y": -1
    }
  },
  "homing": {
    "X": {
      "inverted": false,
      "lastCheck": "2025-11-08T20:00:00"
    },
    "Y": {
      "inverted": true,
      "lastCheck": "2025-11-08T20:00:05"
    }
  },
  "illuminationMap": [
    {
      "channel": "Laser_635",
      "type": "laser",
      "wavelength_nm": 635.0,
      "color": "red",
      "mean_intensity": 1250.3,
      "max_intensity": 3500.8
    }
  ],
  "objectiveImages": {
    "slot1": "/path/to/objective1_calibration.png",
    "slot2": "/path/to/objective2_calibration.png"
  }
}
```

**Notes:**
- Returns empty object `{}` if no calibrations performed yet
- Reflects current saved configuration
- Use to verify calibration state

---

### 8. MJPEG Stream (GET)

Get real-time MJPEG video stream from observation camera.

**Endpoint:** `/overviewStream`

**Method:** GET

**Parameters:**
- `startStream` (boolean, optional): Whether to start streaming. Default: true

**Response:**
- Content-Type: `multipart/x-mixed-replace;boundary=frame`
- Streaming response with PNG frames at ~30 FPS

**Example Usage (HTML):**
```html
<img src="http://localhost:8001/api/pixelCalibration/overview/stream" 
     alt="Observation Camera Stream" />
```

**Example Usage (Python):**
```python
import requests
from PIL import Image
import io

response = requests.get(
    "http://localhost:8001/api/pixelCalibration/overview/stream",
    stream=True
)

for chunk in response.iter_content(chunk_size=8192):
    # Parse MJPEG frames
    # (Implementation depends on your needs)
    pass
```

**Notes:**
- Stream continues until connection is closed
- Frame rate: ~30 FPS (adjustable in implementation)
- Frames encoded as PNG for quality
- Use for live feedback during calibration

---

## Complete Calibration Workflow

### Recommended Calibration Sequence

```python
import requests
import time

BASE_URL = "http://localhost:8001/api/pixelCalibration/overview"

# 1. Check camera availability
response = requests.get(f"{BASE_URL}/isObservationCameraAvailable")
if not response.json()["available"]:
    print("Error: Observation camera not available")
    exit(1)

print(f"Using camera: {response.json()['name']}")

# 2. Identify stage axes
print("Identifying stage axes...")
response = requests.post(
    f"{BASE_URL}/identifyAxes",
    json={"stepUm": 2000.0}
)
result = response.json()
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Axis mapping: {result['mapping']}")
    print(f"Axis signs: {result['sign']}")

time.sleep(1)

# 3. Map illumination channels
print("\nMapping illumination channels...")
response = requests.post(f"{BASE_URL}/mapIlluminationChannels")
result = response.json()
if "error" in result:
    print(f"Error: {result['error']}")
else:
    for channel in result["illuminationMap"]:
        print(f"  {channel['channel']}: {channel['color']} "
              f"({channel.get('wavelength_nm', 'N/A')} nm)")

# 4. Verify homing (optional - use with caution)
print("\nVerifying homing...")
response = requests.post(
    f"{BASE_URL}/verifyHoming",
    json={"maxTimeS": 20.0}
)
result = response.json()
for axis, data in result.items():
    if "inverted" in data:
        status = "INVERTED" if data["inverted"] else "OK"
        print(f"  {axis}: {status}")

# 5. Fix step signs
print("\nDetermining step signs...")
response = requests.post(
    f"{BASE_URL}/fixStepSign",
    json={"rectSizeUm": 20000.0}
)
result = response.json()
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Step signs: {result['sign']}")

# 6. Capture objective images
print("\nCapturing objective images...")
for slot in [1, 2]:
    response = requests.post(
        f"{BASE_URL}/captureObjectiveImage",
        json={"slot": slot}
    )
    result = response.json()
    if "error" in result:
        print(f"  Slot {slot}: Error - {result['error']}")
    else:
        print(f"  Slot {slot}: Saved to {result['path']}")

# 7. Get final configuration
print("\nFinal configuration:")
response = requests.get(f"{BASE_URL}/getConfig")
config = response.json()
print(json.dumps(config, indent=2))
```

---

## Configuration Storage

All calibration results are automatically saved to the ImSwitch configuration file under `PixelCalibration.overviewCalibration`:

```json
{
  "PixelCalibration": {
    "ObservationCamera": "TopCam_1",
    "overviewCalibration": {
      "axes": {
        "mapping": {
          "stageX_to_cam": "width",
          "stageY_to_cam": "height"
        },
        "sign": {
          "X": 1,
          "Y": -1
        }
      },
      "homing": {
        "X": {
          "inverted": false,
          "lastCheck": "2025-11-08T20:00:00"
        },
        "Y": {
          "inverted": false,
          "lastCheck": "2025-11-08T20:00:00"
        }
      },
      "illuminationMap": [
        {
          "channel": "LED_matrix",
          "wavelength_nm": 530,
          "color": "green"
        },
        {
          "channel": "Laser_635",
          "wavelength_nm": 635,
          "color": "red"
        }
      ],
      "objectiveImages": {
        "slot1": "ImSwitchConfig/objective1_calibration.png",
        "slot2": "ImSwitchConfig/objective2_calibration.png"
      }
    }
  }
}
```

---

## Error Handling

All endpoints follow consistent error handling:

**HTTP Status Codes:**
- `200 OK`: Successful operation
- `409 Conflict`: Observation camera not available or other resource conflict
- `500 Internal Server Error`: Unexpected error during operation

**Error Response Format:**
```json
{
  "error": "Descriptive error message"
}
```

**Common Errors:**
- `"Observation camera not available"`: Configure camera in setup or check hardware
- `"No AprilTag detected"`: Ensure AprilTag is visible and properly illuminated
- `"AprilTag lost after X movement"`: Reduce step size or check tag placement
- `"No illumination sources available"`: No lasers or LEDs configured
- `"No positioner available"`: Stage not configured or not connected

---

## Best Practices

### 1. AprilTag Setup
- Use AprilTag 36h11 family (most robust)
- Print tag at ~50-100mm size for overhead camera
- Mount tag securely on stage or sample holder
- Ensure good illumination and contrast
- Position tag to remain visible during all movements

### 2. Calibration Parameters
- **stepUm**: 
  - Use 1000-3000 µm for initial axis identification
  - Larger steps = more robust detection
  - Ensure tag remains in view after movement
  
- **rectSizeUm**:
  - Use 10000-30000 µm based on stage travel range
  - Ensure all corners are reachable
  - Tag must remain in camera view at all positions

### 3. Workflow
1. Start with camera availability check
2. Perform axis identification first
3. Map illumination channels for documentation
4. Use homing verification carefully (may hit limits)
5. Fix step signs for precise control
6. Capture objective images for reference

### 4. Safety
- **Homing verification**: May move stage to limits - use with caution
- **Rectangle movement**: Ensure clear path before running
- **Power levels**: Illumination mapping uses 50% power - verify safe for camera

---

## Integration with Affine Calibration

Overview calibration complements the affine calibration workflow:

1. **Overview calibration** (this API): Identifies basic axis configuration and signs
2. **Affine calibration** (existing API): Performs precise pixel-to-micron mapping

Typical sequence:
```python
# 1. First time setup - Overview calibration
overview_result = requests.post("/api/pixelCalibration/overview/identifyAxes")

# 2. Precise calibration - Affine calibration
affine_result = requests.post("/api/pixelCalibration/calibrateStageAffine", 
                              json={"objectiveId": "10x", "stepSizeUm": 150})
```

Overview calibration results can inform affine calibration parameter choices.

---

## Troubleshooting

### AprilTag Not Detected
**Symptoms**: `"No AprilTag detected"` error

**Solutions:**
1. Check tag is in camera field of view
2. Improve illumination
3. Ensure tag is flat and undistorted
4. Verify camera focus
5. Use larger tag size

### Tag Lost During Movement
**Symptoms**: `"AprilTag lost after X movement"` error

**Solutions:**
1. Reduce step size
2. Reposition tag closer to center of field
3. Use wider field of view camera
4. Check stage movement limits

### Incorrect Axis Mapping
**Symptoms**: Axes identified incorrectly

**Solutions:**
1. Increase step size for clearer motion
2. Check camera orientation
3. Verify stage is moving correctly
4. Repeat calibration

### Poor Illumination Mapping
**Symptoms**: Colors classified as "unknown"

**Solutions:**
1. Use color camera for better classification
2. Check illumination sources are working
3. Increase settle time
4. Adjust power levels

---

## See Also

- Affine calibration API: `docs/stage_calibration_api.md`
- AprilTag detection: `imswitch/imcontrol/controller/controllers/pixelcalibration/overview_calibrator.py`
- Configuration schema: `imswitch/imcontrol/model/SetupInfo.py`
