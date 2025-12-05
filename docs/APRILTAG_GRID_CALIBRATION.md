# AprilTag Grid-Based Stage Calibration and Navigation

## Overview

This system enables automatic stage navigation using an overhead observation camera and a printed AprilTag calibration grid. It provides:

1. **Camera-to-stage calibration**: Automatically determine the transformation between camera pixel coordinates and stage micrometers
2. **Tag-based navigation**: Move the stage to center any specific AprilTag ID using closed-loop feedback
3. **Grid layout support**: Handle oblique/trapezoidal views with affine transformations

## Grid Layout

The AprilTag grid is arranged in a 2D matrix with row-major ordering:

```
Grid Layout (example: 17 rows × 25 columns, 40mm pitch):
Row 0:    0,   1,   2, ...,  24
Row 1:   25,  26,  27, ...,  49
Row 2:   50,  51,  52, ...,  74
...
Row 16: 400, 401, 402, ..., 424
```

Each tag is separated by a fixed physical distance (pitch) in both X and Y directions.

## Configuration

### Setup Configuration File

Add the following to your ImSwitch configuration JSON (e.g., `imcontrol_setups/example_config.json`):

```json
{
  "PixelCalibration": {
    "ObservationCamera": "ObservationCam",
    "ObservationCameraFlip": {
      "flipX": true,
      "flipY": true
    },
    "aprilTagGrid": {
      "rows": 17,
      "cols": 25,
      "start_id": 0,
      "pitch_mm": 40.0
    }
  }
}
```

**Parameters:**
- `rows`: Number of rows in the grid
- `cols`: Number of columns in the grid  
- `start_id`: Starting tag ID (usually 0)
- `pitch_mm`: Physical spacing between tag centers in millimeters

### Generating the Calibration Target

Use the included AprilTag generator script:

```python
# See: ImSwitch/ImTools/apriltag/generateAprilTag.py
from generateAprilTag import generate_apriltag_grid

res = generate_apriltag_grid(
    rows=17, 
    cols=25, 
    start_id=0,
    family="DICT_APRILTAG_36h11",
    tag_size_mm=35,      # Individual tag size
    margin_mm=5,          # Gap between tags (pitch = tag_size + margin)
    dpi=300,
    label_ids=True,
    out_png="apriltag_grid.png", 
    out_pdf="apriltag_grid.pdf"
)

print(f"Generated: {res['png']}")
```

**Important:** The `pitch_mm` in configuration should equal `tag_size_mm + margin_mm` from generation.

## API Endpoints

All endpoints are available via REST API at `/api/pixelcalibration/`:

### 1. Configure Grid

```http
GET /api/pixelcalibration/gridSetConfig?rows=17&cols=25&start_id=0&pitch_mm=40.0
```

**Response:**
```json
{
  "success": true,
  "config": {
    "rows": 17,
    "cols": 25,
    "start_id": 0,
    "pitch_mm": 40.0
  },
  "transform_preserved": false
}
```

### 2. Get Current Configuration

```http
GET /api/pixelcalibration/gridGetConfig
```

**Response:**
```json
{
  "success": true,
  "config": {
    "rows": 17,
    "cols": 25,
    "start_id": 0,
    "pitch_mm": 40.0
  },
  "calibrated": true,
  "transform": [[0.0234, -0.0002, 123.45], [0.0001, 0.0231, 456.78]]
}
```

### 3. Detect Tags

```http
GET /api/pixelcalibration/gridDetectTags?save_annotated=false
```

**Response:**
```json
{
  "success": true,
  "num_tags": 12,
  "tags": [
    {
      "id": 50,
      "cx": 512.3,
      "cy": 384.7,
      "grid_position": [2, 0]
    },
    {
      "id": 51,
      "cx": 645.1,
      "cy": 386.2,
      "grid_position": [2, 1]
    }
  ]
}
```

### 4. Calibrate Camera-to-Stage Transformation

**Prerequisites:** 
- At least 3 AprilTags must be visible in the observation camera
- Tags must be from the configured grid (valid IDs)

```http
GET /api/pixelcalibration/gridCalibrateTransform
```

**Response:**
```json
{
  "success": true,
  "T_cam2stage": [[0.0234, -0.0002, 123.45], [0.0001, 0.0231, 456.78]],
  "num_tags": 12,
  "residual_um": 2.45,
  "tag_ids": [50, 51, 52, 75, 76, 77, 100, 101, 102, 125, 126, 127]
}
```

**Interpretation:**
- `T_cam2stage`: 2×3 affine transformation matrix mapping pixel coordinates to stage micrometers
- `num_tags`: Number of tags used for calibration
- `residual_um`: RMS error of the fit in micrometers (should be <5 µm for good calibration)
- `tag_ids`: List of tag IDs used in the calibration

**Validation:**
Run calibration 5 times and verify:
- Residual error is consistent (<5% variation)
- Transform matrix values are stable

### 5. Navigate to Specific Tag

```http
GET /api/pixelcalibration/gridMoveToTag?target_id=101&roi_tolerance_px=8&max_iterations=20&step_fraction=0.8&settle_time=0.3&search_enabled=true
```

**Parameters:**
- `target_id`: Tag ID to navigate to (must be within grid range: 0-424 for 17×25 grid)
- `roi_tolerance_px`: Acceptable pixel offset for convergence (default: 8.0)
- `max_iterations`: Maximum iteration count (default: 20)
- `step_fraction`: Fraction of displacement to apply per step, 0-1 (default: 0.8)
- `settle_time`: Wait time after movement in seconds (default: 0.3)
- `search_enabled`: Enable coarse search if target not initially visible (default: true)

**Response (Success):**
```json
{
  "success": true,
  "final_offset_px": 3.2,
  "iterations": 5,
  "final_tag_id": 101,
  "trajectory": [
    {
      "iteration": 0,
      "mode": "coarse_navigation",
      "current_tag": 50,
      "target_tag": 101,
      "offset_px": 145.6,
      "move_um": [2000.5, 800.3]
    },
    {
      "iteration": 1,
      "mode": "micro_centering",
      "current_tag": 101,
      "offset_px": 8.4,
      "move_um": [5.2, -3.1]
    }
  ]
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Max iterations (20) reached",
  "final_offset_px": 12.3,
  "iterations": 20,
  "final_tag_id": 100,
  "trajectory": [...]
}
```

### 6. Get Tag Information

```http
GET /api/pixelcalibration/gridGetTagInfo?tag_id=101
```

**Response:**
```json
{
  "success": true,
  "tag_id": 101,
  "row": 4,
  "col": 1,
  "position_mm": {
    "x": 40.0,
    "y": 160.0
  }
}
```

## Workflow

### Initial Setup

1. **Print calibration target**
   ```bash
   cd ImSwitch/ImTools/apriltag
   python generateAprilTag.py
   # Print apriltag_grid.pdf at 100% scale (no scaling!)
   ```

2. **Mount target on stage**
   - Ensure target is flat and rigidly mounted
   - Position target under observation camera

3. **Configure grid in ImSwitch**
   ```http
   GET /api/pixelcalibration/gridSetConfig?rows=17&cols=25&pitch_mm=40.0
   ```

### Calibration Procedure

1. **Position stage so multiple tags are visible**
   - Aim for 10-20 tags in view
   - More tags = better calibration

2. **Detect tags to verify**
   ```http
   GET /api/pixelcalibration/gridDetectTags
   ```
   
3. **Run calibration**
   ```http
   GET /api/pixelcalibration/gridCalibrateTransform
   ```
   
4. **Validate calibration**
   - Run calibration 5 times from different positions
   - Verify residual error <5 µm
   - Check transform matrix stability (<5% variation)

### Navigation Usage

1. **Navigate to specific tag**
   ```http
   GET /api/pixelcalibration/gridMoveToTag?target_id=200
   ```

2. **Navigate to grid position**
   - Calculate tag ID: `tag_id = start_id + row * cols + col`
   - Example: Row 8, Col 5 → `tag_id = 0 + 8*25 + 5 = 205`

## Algorithm Details

### Camera-to-Stage Calibration

The calibration computes a 2×3 affine transformation matrix **T**:

```
[stage_x]   [a  b  tx]   [pixel_u]
[stage_y] = [c  d  ty] × [pixel_v]
                         [   1   ]
```

**Process:**
1. Detect N ≥ 3 tags in camera frame
2. For each tag i:
   - Camera position: (u_i, v_i) pixels
   - Grid position: (col_i, row_i)
   - Physical position: (col_i × pitch, row_i × pitch) mm
3. Solve least-squares system for 6 parameters (a, b, c, d, tx, ty)
4. Compute residual error to assess calibration quality

### Move-to-Tag Navigation

**Two-phase approach:**

1. **Coarse navigation** (when current tag ≠ target tag):
   - Detect current tag ID and position
   - Compute grid displacement: Δrow, Δcol = target_pos - current_pos
   - Convert to physical displacement: Δx, Δy (millimeters)
   - Move stage by `step_fraction × displacement`
   - Re-detect and iterate

2. **Micro-centering** (when current tag = target tag):
   - Measure pixel offset from ROI center: (Δu, Δv)
   - Transform to stage displacement using linear part of **T**
   - Move stage to center tag
   - Converge when offset ≤ `roi_tolerance_px`

**Search mode** (when target not initially visible):
- Performs raster search pattern (default 3×3 grid)
- Step size: 5000 µm (configurable)
- Returns to start if search fails

## Troubleshooting

### Calibration Issues

**Problem:** High residual error (>10 µm)
- **Cause:** Oblique camera angle, lens distortion, or tags not flat
- **Solution:** Ensure target is perpendicular to camera; use more tags; check for tag detection errors

**Problem:** Transform matrix changes significantly between calibrations
- **Cause:** Mechanical instability, vibration, or thermal drift
- **Solution:** Stabilize setup; wait for thermal equilibrium; check stage repeatability

**Problem:** "Need at least 3 valid grid tags"
- **Cause:** Not enough tags visible, or wrong grid configuration
- **Solution:** Increase camera field of view; verify grid config matches printed target

### Navigation Issues

**Problem:** Navigation converges to wrong tag
- **Cause:** Initial tag detection error or extreme oblique view
- **Solution:** Increase `step_fraction` to smaller value (0.5-0.7); verify calibration quality

**Problem:** "Max iterations reached"
- **Cause:** Poor calibration, mechanical backlash, or tolerance too tight
- **Solution:** Re-calibrate; increase `roi_tolerance_px`; reduce `step_fraction`

**Problem:** "Camera-to-stage transformation not calibrated"
- **Cause:** Calibration hasn't been run or was lost
- **Solution:** Run `gridCalibrateTransform` endpoint

### Detection Issues

**Problem:** No tags detected
- **Cause:** Poor lighting, focus, or camera exposure
- **Solution:** Adjust illumination; check focus; increase exposure time

**Problem:** Wrong tags detected (IDs don't match physical target)
- **Cause:** AprilTag family mismatch
- **Solution:** Ensure target generated with `DICT_APRILTAG_36h11` family

## Performance Specifications

### Acceptance Criteria

✅ **Calibration stability**: <5% variation in transform parameters over 5 runs  
✅ **Calibration accuracy**: RMS residual error <5 µm  
✅ **Navigation success**: Target centered within `roi_tolerance_px` in <20 iterations  
✅ **Search capability**: Finds tags within 3×3 raster search when not initially visible  
✅ **ROI configurability**: Supports configurable ROI center (default: image center)  
✅ **Persistence**: Grid config and calibration saved and reloaded between sessions  

### Typical Performance

- **Calibration time**: 1-2 seconds for detection + computation
- **Navigation iterations**: 3-8 for typical movements (5-10 tag spacing)
- **Convergence accuracy**: 2-5 pixels final offset
- **Maximum working distance**: Limited by camera field of view and tag visibility

## Code Structure

### Core Modules

```
imswitch/imcontrol/controller/controllers/pixelcalibration/
├── apriltag_grid_calibrator.py    # Grid calibration and navigation logic
│   ├── GridConfig                  # Grid layout configuration
│   └── AprilTagGridCalibrator      # Main calibrator class
│
├── overview_calibrator.py          # General overhead camera utilities
└── __init__.py
```

### Integration Points

- **PixelCalibrationController**: Provides REST API endpoints
- **DetectorManager**: Provides observation camera access via `ObservationCamera` config
- **PositionersManager**: Provides stage movement via first available positioner
- **Setup Configuration**: Stores grid config in `imcontrol_setups/XXX_config.json`

## Python API Usage

For programmatic access:

```python
from imswitch.imcontrol.controller.controllers.pixelcalibration.apriltag_grid_calibrator import (
    AprilTagGridCalibrator, GridConfig
)

# Create grid config
grid = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=40.0)

# Initialize calibrator
calibrator = AprilTagGridCalibrator(grid)

# Detect tags from image
import cv2
img = cv2.imread("frame.png")
tags = calibrator.detect_tags(img)  # Returns {tag_id: (cx, cy)}

# Calibrate from detected tags
result = calibrator.calibrate_from_frame(tags)
print(f"Calibration residual: {result['residual_um']:.2f} µm")

# Navigate to tag (requires camera and positioner objects)
nav_result = calibrator.move_to_tag(
    target_id=101,
    observation_camera=camera,
    positioner=stage,
    roi_tolerance_px=8.0
)
print(f"Navigation success: {nav_result['success']}")
```

## Future Enhancements

Potential improvements:

- [ ] Multi-grid support for large stages
- [ ] Automatic optimal grid positioning
- [ ] Real-time visualization overlay
- [ ] Distortion correction for wide-angle lenses
- [ ] Adaptive step sizing based on convergence rate
- [ ] Trajectory optimization for multi-tag visits

## References

- AprilTag Detection: OpenCV ArUco module (`cv2.aruco`)
- AprilTag Family: `DICT_APRILTAG_36h11` (36h11 tag set)
- Tag Generator: `ImSwitch/ImTools/apriltag/generateAprilTag.py`
- Configuration: `imcontrol_setups/example_config.json`
