# Lightsheet Controller Enhancement

## Overview

This document describes the enhancements made to the Lightsheet Controller to support:

1. **Go-Stop-Acquire Mode** - A new acquisition mode for high-quality Z-stacks
2. **OME-Zarr Export** - Modern, scalable data format support
3. **Real-time Socket Updates** - Live progress updates from backend to frontend
4. **3D Visualization** - Integrated VizarrViewer for in-browser 3D viewing

## New Features

### Scan Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Continuous** | Stage moves continuously while acquiring | Fast scans, large volumes |
| **Step-Acquire** | Move → Stop → Acquire → Move | High-quality, precise Z-stacks |

### Storage Formats

| Format | Description |
|--------|-------------|
| `tiff` | Traditional TIFF stack |
| `ome_zarr` | OME-Zarr format (chunked, cloud-ready) |
| `both` | Save in both formats |

## Backend API (Python)

### New Endpoints

```python
# Start step-acquire scan with OME-Zarr support
GET /LightsheetController/startStepAcquireScan
    ?minPos=-500
    &maxPos=500
    &stepSize=10
    &axis=A
    &illuSource=laser1
    &illuValue=512
    &storageFormat=ome_zarr
    &experimentName=my_scan

# Start continuous scan with Zarr support
GET /LightsheetController/startContinuousScanWithZarr
    ?minPos=-500
    &maxPos=500
    &speed=1000
    &axis=A
    &illuSource=laser1
    &illuValue=512
    &storageFormat=both
    &experimentName=my_scan

# Get current scan status
GET /LightsheetController/getScanStatus
# Returns: {isRunning, scanMode, currentPosition, totalPositions, currentFrame, progress, zarrPath, tiffPath, errorMessage}

# Get available scan modes
GET /LightsheetController/getAvailableScanModes
# Returns: ["continuous", "step_acquire"]

# Get available storage formats
GET /LightsheetController/getAvailableStorageFormats
# Returns: ["tiff", "ome_zarr", "both"]

# Get latest Zarr path for visualization
GET /LightsheetController/getLatestZarrPath
# Returns: {zarrPath: "/data/recordings/.../scan.zarr", absolutePath: "...", exists: true}
```

### Socket.IO Events

The backend emits real-time scan status updates via Socket.IO:

```javascript
// Event name: 'lightsheet_status'
// Payload:
{
  isRunning: true,
  scanMode: "step_acquire",
  currentPosition: 123.5,
  totalPositions: 100,
  currentFrame: 45,
  progress: 45.0,
  zarrPath: "/data/recordings/2024-01-02_12-30-00/lightsheet_scan.zarr",
  tiffPath: null,
  errorMessage: null
}
```

## Frontend (React)

### Redux State (LightsheetSlice)

New state properties:

```javascript
{
  // Scan configuration
  stepSize: 10,              // Step size for step-acquire mode (µm)
  scanMode: "continuous",    // "continuous" or "step_acquire"
  storageFormat: "ome_zarr", // "tiff", "ome_zarr", or "both"
  experimentName: "lightsheet_scan",

  // Live scan status (updated via socket)
  scanStatus: {
    isRunning: false,
    scanMode: null,
    currentPosition: 0,
    totalPositions: 0,
    currentFrame: 0,
    progress: 0,
    zarrPath: null,
    tiffPath: null,
    errorMessage: null,
  },

  // Available options from backend
  availableScanModes: ["continuous", "step_acquire"],
  availableStorageFormats: ["tiff", "ome_zarr", "both"],

  // Latest zarr path for visualization
  latestZarrPath: null,
  latestZarrAbsolutePath: null,
}
```

### New Actions

```javascript
// Scan mode and storage
setScanMode(mode)
setStorageFormat(format)
setExperimentName(name)
setStepSize(size)

// Scan status (from socket)
setScanStatus(status)
updateScanProgress({progress, currentFrame, currentPosition})

// Options from backend
setAvailableScanModes(modes)
setAvailableStorageFormats(formats)

// Zarr path
setLatestZarrPath({zarrPath, absolutePath})
```

### API Functions

New API functions in `src/backendapi/apiLightsheetController.js`:

```javascript
import {
  apiStartStepAcquireScan,
  apiStartContinuousScanWithZarr,
  apiGetScanStatus,
  apiGetAvailableScanModes,
  apiGetAvailableStorageFormats,
  apiGetLatestZarrPath,
} from "../backendapi/apiLightsheetController.js";
```

## UI Components

### LightsheetController.jsx

New tabs:
1. **Scanning Parameters** - Enhanced with scan mode, storage format, progress bar
2. **Galvo Scanner** - Unchanged
3. **View Latest Stack** - Download buttons
4. **3D Zarr Viewer** - Integrated VizarrViewer for OME-Zarr visualization
5. **VTK Viewer** - Legacy TIFF viewer

New UI elements:
- Scan mode dropdown (Continuous vs Step-Acquire)
- Storage format dropdown (TIFF, OME-Zarr, Both)
- Experiment name input
- Step size input (visible in step-acquire mode)
- Progress bar with frame count and position
- Socket connection status indicator
- Output file paths display
- View Latest Zarr button

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
├─────────────────────────────────────────────────────────────────┤
│  LightsheetController.jsx                                        │
│  ├── Scan Configuration (mode, format, params)                   │
│  ├── Start Scan Button → apiStartStepAcquireScan()              │
│  ├── Socket.IO listener → setScanStatus()                       │
│  ├── Progress Bar (from scanStatus.progress)                    │
│  └── VizarrViewer (from latestZarrPath)                         │
│                                                                  │
│  Redux Store (LightsheetSlice)                                   │
│  ├── scanMode, storageFormat, experimentName                    │
│  ├── scanStatus (updated via socket)                            │
│  └── latestZarrPath                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                    REST API + Socket.IO
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (Python)                          │
├─────────────────────────────────────────────────────────────────┤
│  LightsheetController.py                                         │
│  ├── startStepAcquireScan() → _stepAcquireThread()              │
│  │   ├── Initialize OME-Zarr writer                             │
│  │   ├── For each position:                                     │
│  │   │   ├── Move stage (blocking)                              │
│  │   │   ├── Acquire frame                                      │
│  │   │   ├── Write to Zarr                                      │
│  │   │   └── Emit status via Socket.IO                          │
│  │   └── Finalize (close writer, save TIFF if needed)           │
│  │                                                               │
│  ├── getScanStatus() → Current scan status                      │
│  └── getLatestZarrPath() → Path for frontend visualization      │
│                                                                  │
│  SingleMultiscaleZarrWriter (OME-Zarr)                          │
│  ├── set_metadata(t, c, z, height, width)                       │
│  ├── open_store()                                                │
│  ├── write_tile(frame, t, c, z, y_start, x_start)               │
│  └── close()                                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Storage                                   │
├─────────────────────────────────────────────────────────────────┤
│  /data/recordings/YYYY-MM-DD_HH-MM-SS/                          │
│  ├── lightsheet_scan.zarr/                                       │
│  │   └── 0/                                                      │
│  │       └── 0/  (shape: [1, 1, Z, Height, Width])              │
│  └── lightsheet_scan.tif (optional)                             │
└─────────────────────────────────────────────────────────────────┘
```

## Usage Example

1. Open Lightsheet Controller
2. Select "Step-Acquire (High Quality)" scan mode
3. Select "OME-Zarr" storage format
4. Set parameters:
   - Min Position: -500 µm
   - Max Position: 500 µm
   - Step Size: 10 µm (100 frames)
   - Axis: A
5. Click "Start Step-Acquire"
6. Watch progress bar update in real-time via socket
7. When complete, click "View Latest Zarr" to open 3D viewer
8. Or switch to "3D Zarr Viewer" tab for integrated visualization

## Dependencies

### Backend (Python)
- zarr >= 2.0
- numpy
- tifffile
- python-socketio (already included)

### Frontend (React)
- socket.io-client
- zarrita (for VizarrViewer)
- @mui/material (already included)
