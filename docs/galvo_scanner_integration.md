# Galvo Scanner Integration for ImSwitch

This documentation describes how to integrate and use the galvo scanner functionality in ImSwitch with UC2/ESP32 hardware.

## Overview

The galvo scanner integration provides:
- **Backend Manager**: `ESP32GalvoScannerManager` for controlling galvo mirrors via ESP32
- **REST API Controller**: `GalvoScannerController` with full API endpoints
- **Frontend Component**: React-based control panel with scan visualization

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React)                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              GalvoScannerController.jsx                    │  │
│  │  - Scan parameters (nx, ny, x/y ranges, timing)           │  │
│  │  - Visual scan pattern preview                            │  │
│  │  - Start/Stop controls                                    │  │
│  │  - Real-time status polling                               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Backend (Python/ImSwitch)                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            GalvoScannerController.py                       │  │
│  │  @APIExport endpoints:                                     │  │
│  │  - getGalvoScannerNames()                                  │  │
│  │  - getGalvoScannerConfig()                                 │  │
│  │  - getGalvoScannerStatus()                                 │  │
│  │  - setGalvoScanConfig()                                    │  │
│  │  - startGalvoScan()                                        │  │
│  │  - stopGalvoScan()                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           GalvoScannersManager (MultiManager)              │  │
│  │  - Manages multiple galvo scanner devices                  │  │
│  │  - Dynamic loading of scanner managers                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           ESP32GalvoScannerManager                         │  │
│  │  - Interfaces with UC2-REST galvo.py                       │  │
│  │  - Manages scan configuration                              │  │
│  │  - Communicates via RS232/Serial                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ Serial/HTTP
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ESP32 Hardware (UC2-REST)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    galvo.py                                │  │
│  │  - set_galvo_scan(): Configure and start scanning          │  │
│  │  - stop_galvo_scan(): Stop active scan                     │  │
│  │  - get_galvo_status(): Query scanner state                 │  │
│  │  - set_dac(): Direct DAC waveform control                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### JSON Setup Configuration

Add the following to your ImSwitch setup JSON file:

```json
{
  "galvoScanners": {
    "ESP32Galvo": {
      "managerName": "ESP32GalvoScannerManager",
      "managerProperties": {
        "rs232device": "ESP32",
        "nx": 256,
        "ny": 256,
        "x_min": 500,
        "x_max": 3500,
        "y_min": 500,
        "y_max": 3500,
        "sample_period_us": 1,
        "frame_count": 0,
        "bidirectional": false
      },
      "analogChannel": null,
      "digitalLine": null
    }
  },
  
  "rs232devices": {
    "ESP32": {
      "managerName": "ESP32Manager",
      "managerProperties": {
        "serialport": "COM3",
        "debug": 1
      }
    }
  },
  
  "availableWidgets": [
    "GalvoScanner",
    "..."
  ]
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nx` | int | 256 | Number of X samples per line |
| `ny` | int | 256 | Number of Y lines per frame |
| `x_min` | int | 500 | Minimum X position (DAC: 0-4095) |
| `x_max` | int | 3500 | Maximum X position (DAC: 0-4095) |
| `y_min` | int | 500 | Minimum Y position (DAC: 0-4095) |
| `y_max` | int | 3500 | Maximum Y position (DAC: 0-4095) |
| `sample_period_us` | int | 1 | Microseconds per sample (0 = max speed) |
| `frame_count` | int | 0 | Number of frames (0 = infinite) |
| `bidirectional` | bool | false | Enable bidirectional scanning |

## REST API Endpoints

### GET `/api/GalvoScannerController/getGalvoScannerNames`

Returns list of configured galvo scanner device names.

**Response:**
```json
["ESP32Galvo"]
```

### GET `/api/GalvoScannerController/getGalvoScannerConfig`

Get current configuration for a scanner.

**Parameters:**
- `scannerName` (optional): Name of the scanner

**Response:**
```json
{
  "scannerName": "ESP32Galvo",
  "config": {
    "nx": 256,
    "ny": 256,
    "x_min": 500,
    "x_max": 3500,
    "y_min": 500,
    "y_max": 3500,
    "sample_period_us": 1,
    "frame_count": 0,
    "bidirectional": false
  }
}
```

### GET `/api/GalvoScannerController/getGalvoScannerStatus`

Get current scanner status.

**Response:**
```json
{
  "scannerName": "ESP32Galvo",
  "running": true,
  "current_frame": 5,
  "current_line": 128,
  "config": {...}
}
```

### POST `/api/GalvoScannerController/startGalvoScan`

Start a galvo scan with specified parameters.

**Parameters:**
- `scannerName` (optional): Scanner name
- `nx`, `ny`, `x_min`, `x_max`, `y_min`, `y_max`, `sample_period_us`, `frame_count`, `bidirectional` (all optional)

**Example:**
```
POST /api/GalvoScannerController/startGalvoScan?nx=512&ny=512&bidirectional=true
```

### POST `/api/GalvoScannerController/stopGalvoScan`

Stop an active scan.

**Parameters:**
- `scannerName` (optional): Scanner name

### POST `/api/GalvoScannerController/setGalvoScanConfig`

Update configuration without starting a scan.

## Python API Usage

```python
# Access galvo scanner manager
galvo_manager = master.galvoScannersManager['ESP32Galvo']

# Start scan with parameters
result = galvo_manager.start_scan(
    nx=512, 
    ny=512, 
    x_min=1000, 
    x_max=3000,
    bidirectional=True,
    frame_count=10
)

# Get status
status = galvo_manager.get_status()
print(f"Running: {status['running']}, Frame: {status['current_frame']}")

# Stop scan
galvo_manager.stop_scan()

# Update configuration
galvo_manager.update_config(sample_period_us=2)
```

## Frontend Integration

The React component `GalvoScannerController.jsx` provides:

1. **Scanner Selection**: Dropdown for multiple scanner devices
2. **Resolution Controls**: nx/ny pixel configuration
3. **Position Range Sliders**: Visual X/Y range selection (0-4095)
4. **Timing Parameters**: Sample period and frame count
5. **Bidirectional Toggle**: Enable bidirectional scanning
6. **Scan Pattern Preview**: SVG visualization of scan pattern
7. **Status Display**: Running state, current frame/line
8. **Quick Presets**: Common resolution/range presets
9. **Start/Stop Controls**: Large action buttons

### Adding to Navigation

To add the galvo scanner to the ImSwitch frontend navigation, import and add the component to your routes.

## File Structure

```
imswitch/
├── imcontrol/
│   ├── model/
│   │   ├── managers/
│   │   │   ├── galvoscanners/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── GalvoScannerManager.py      # Base class
│   │   │   │   └── ESP32GalvoScannerManager.py # ESP32 implementation
│   │   │   ├── GalvoScannersManager.py         # MultiManager
│   │   │   └── __init__.py                     # Updated exports
│   │   └── SetupInfo.py                        # Added GalvoScannerInfo
│   └── controller/
│       ├── controllers/
│       │   └── GalvoScannerController.py       # API Controller
│       └── MasterController.py                 # Updated initialization
│
└── frontend/
    └── src/
        └── components/
            └── GalvoScannerController.jsx      # React component
```

## Creating Custom Galvo Scanner Managers

To support different hardware, create a new manager class:

```python
from imswitch.imcontrol.model.managers.galvoscanners import GalvoScannerManager

class MyCustomGalvoManager(GalvoScannerManager):
    def __init__(self, galvoScannerInfo, name, **lowLevelManagers):
        super().__init__(galvoScannerInfo, name)
        # Initialize your hardware connection
        
    def start_scan(self, **kwargs):
        # Implement scan start for your hardware
        pass
        
    def stop_scan(self, timeout=1):
        # Implement scan stop
        pass
        
    def get_status(self, timeout=1):
        # Return current status
        pass
```

## Troubleshooting

### Scanner not appearing
- Check that `galvoScanners` is defined in setup JSON
- Verify `rs232device` matches your RS232 device name
- Check serial port connection

### Scan not starting
- Verify ESP32 firmware supports galvo commands
- Check serial communication with debug mode
- Ensure DAC values are within valid range (0-4095)

### Performance issues
- Reduce `nx`/`ny` for faster frame rates
- Set `sample_period_us` to 0 for maximum speed
- Enable `bidirectional` scanning

## Version History

- **v1.0.0** (2025-01-28): Initial galvo scanner integration
  - Base manager and ESP32 implementation
  - REST API controller with full endpoint coverage
  - React frontend component with visualization
