# Motor Settings Implementation

## Overview

This document describes the unified motor settings interface implementation that provides:

1. **Backend API endpoints** in `UC2ConfigController` for getting/setting motor parameters
2. **Unified data structures** in `motor_config.py` for consistent configuration management
3. **Frontend React component** `MotorSettingsController` for user interaction
4. **API wrappers** in `backendapi/` for frontend-backend communication

## Architecture

```
Frontend (React)                    Backend (ImSwitch)                    Device (ESP32)
┌─────────────────────┐            ┌─────────────────────┐               ┌──────────────┐
│ MotorSettingsController │  HTTP  │ UC2ConfigController │    Serial     │ ESP32 + TMC  │
│                     │  ─────>   │                     │   ─────>      │   Drivers    │
│ - Per-axis tabs     │           │ - getMotorSettings  │               │              │
│ - Motion settings   │           │ - setMotorSettings  │               │              │
│ - Homing settings   │           │ - setTMCSettings    │               │              │
│ - TMC settings      │           │                     │               │              │
└─────────────────────┘           └─────────────────────┘               └──────────────┘
          │                                  │
          │                                  │
          ▼                                  ▼
    ┌───────────────┐                ┌───────────────────┐
    │ API Wrappers  │                │ ESP32StageManager │
    │ (backendapi/) │                │ - Unified getters │
    │               │                │ - Unified setters │
    └───────────────┘                └───────────────────┘
```

## API Endpoints

### GET `/UC2ConfigController/getMotorSettings`

Returns all motor settings for all axes and global configuration.

**Response:**
```json
{
  "global": {
    "axisOrder": [0, 1, 2, 3],
    "isCoreXY": false,
    "isEnabled": true,
    "enableAuto": true,
    "isDualAxis": false
  },
  "axes": {
    "X": {
      "axis": "X",
      "motion": {
        "stepSize": 1.0,
        "maxSpeed": 10000,
        "speed": 10000,
        "acceleration": 1000000,
        "minPos": null,
        "maxPos": null,
        "backlash": 0
      },
      "homing": {
        "enabled": true,
        "speed": 15000,
        "direction": -1,
        "endstopPolarity": 1,
        "endposRelease": 3000,
        "timeout": 20000,
        "homeOnStart": false,
        "homeSteps": 0
      },
      "limits": {
        "enabled": false
      }
    },
    "Y": { ... },
    "Z": { ... },
    "A": { ... }
  }
}
```

### GET `/UC2ConfigController/getMotorSettingsForAxis?axis=X`

Returns settings for a specific axis.

### POST `/UC2ConfigController/setMotorSettingsForAxis?axis=X`

Sets motor settings for a specific axis.

**Request Body:**
```json
{
  "motion": {
    "stepSize": 1.0,
    "maxSpeed": 10000,
    "backlash": 0
  },
  "homing": {
    "enabled": true,
    "speed": 15000,
    "direction": -1
  }
}
```

### POST `/UC2ConfigController/setTMCSettingsForAxis?axis=X`

Sets TMC stepper driver settings for a specific axis.

**Request Body:**
```json
{
  "msteps": 16,
  "rmsCurrent": 500,
  "sgthrs": 10,
  "semin": 5,
  "semax": 2,
  "blankTime": 24,
  "toff": 3
}
```

### POST `/UC2ConfigController/setGlobalMotorSettings`

Sets global motor system settings.

**Request Body:**
```json
{
  "axisOrder": [0, 1, 2, 3],
  "isCoreXY": false,
  "isEnabled": true,
  "enableAuto": true
}
```

## Files Changed/Created

### UC2-REST (uc2rest/)
- **`motor_config.py`** (NEW) - Unified dataclasses for motor configuration

### ImSwitch
- **`ESP32StageManager.py`** - Added getter/setter methods for unified motor settings interface
- **`UC2ConfigController.py`** - Added API endpoints for motor settings

### Frontend (microscope-app)
- **`MotorSettingsController.jsx`** (NEW) - Main UI component for motor configuration
- **`SettingsMenu.jsx`** - Added Motor Settings menu entry
- **`App.jsx`** - Added route for MotorSettings component
- **`backendapi/apiMotorSettingsGet.js`** (NEW) - API wrapper
- **`backendapi/apiMotorSettingsGetForAxis.js`** (NEW) - API wrapper
- **`backendapi/apiMotorSettingsSet.js`** (NEW) - API wrapper
- **`backendapi/apiMotorSettingsSetForAxis.js`** (NEW) - API wrapper
- **`backendapi/apiTMCSettingsSetForAxis.js`** (NEW) - API wrapper
- **`backendapi/apiMotorSettingsSetGlobal.js`** (NEW) - API wrapper

## Configuration Parameters

### Motion Settings (per axis)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| stepSize | float | 1.0 | Steps per µm (calibration factor) |
| maxSpeed | int | 10000 | Maximum speed in steps/s |
| speed | int | 10000 | Default speed in steps/s |
| acceleration | int | 1000000 | Acceleration in steps/s² |
| minPos | float | -∞ | Minimum position limit |
| maxPos | float | +∞ | Maximum position limit |
| backlash | int | 0 | Backlash compensation in steps |

### Homing Settings (per axis)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | false | Enable homing with endstop |
| speed | int | 15000 | Homing speed in steps/s |
| direction | int | -1 | Homing direction (-1 or 1) |
| endstopPolarity | int | 1 | Endstop polarity (0=NO, 1=NC) |
| endposRelease | int | 3000 | Back-off distance after endstop hit |
| timeout | int | 20000 | Homing timeout in ms |
| homeOnStart | bool | false | Home this axis on startup |
| homeSteps | int | 0 | Steps to move if no endstop (open-loop homing) |

### TMC Driver Settings (per axis)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| msteps | int | 16 | Microsteps (1, 2, 4, 8, 16, 32, 64, 128, 256) |
| rmsCurrent | int | 500 | RMS current in mA |
| sgthrs | int | 10 | StallGuard threshold |
| semin | int | 5 | Minimum CoolStep current |
| semax | int | 2 | Maximum CoolStep current |
| blankTime | int | 24 | Comparator blank time |
| toff | int | 3 | Off time |

### Global Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| axisOrder | int[4] | [0,1,2,3] | Physical axis mapping (A,X,Y,Z) |
| isCoreXY | bool | false | Enable CoreXY kinematics |
| isEnabled | bool | true | Motors powered on |
| enableAuto | bool | true | Auto power-off when idle |
| isDualAxis | bool | false | Link A and Z axes |

## Usage

### From Frontend
1. Navigate to Settings → Motor Settings
2. Select axis tab (X, Y, Z, A)
3. Modify settings in Motion, Homing, or Limits sections
4. Click "Save [Axis] Settings" to apply

### From Python/API
```python
import requests

# Get all settings
response = requests.get("http://localhost:8000/imswitch/api/UC2ConfigController/getMotorSettings")
settings = response.json()

# Set homing direction for X axis
requests.post(
    "http://localhost:8000/imswitch/api/UC2ConfigController/setMotorSettingsForAxis",
    params={"axis": "X"},
    json={"homing": {"direction": 1}}
)

# Apply TMC settings
requests.post(
    "http://localhost:8000/imswitch/api/UC2ConfigController/setTMCSettingsForAxis",
    params={"axis": "X"},
    json={"msteps": 32, "rmsCurrent": 600}
)
```

## Future Improvements

1. **Config Persistence**: Add endpoints to save settings to the ImSwitch configuration JSON file
2. **Live Preview**: Show real-time motor position updates while adjusting settings
3. **Presets**: Allow saving/loading motor configuration presets
4. **Validation**: Add more comprehensive parameter validation
5. **TMC Diagnostics**: Add readback of TMC driver status (temperature, load, etc.)
