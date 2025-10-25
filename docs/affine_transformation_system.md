# Affine Transformation System

## Overview

The stage-to-camera affine calibration system provides a complete 2×3 transformation matrix that maps pixel coordinates to stage micron coordinates. The system is stored in the setup configuration and provides default identity transformation when no calibration exists.

## Storage Location

**Primary Storage**: Setup configuration (`config.json`)
- Path: `PixelCalibration.affineCalibrations[objectiveId]`
- Per-objective storage with metadata
- Persisted across sessions
- No separate JSON file needed

## Affine Matrix Format

The affine matrix is a 2×3 matrix:

```
[a11  a12  tx]
[a21  a22  ty]
```

### Transformation

```
stage_x = a11 * pixel_x + a12 * pixel_y + tx
stage_y = a21 * pixel_x + a22 * pixel_y + ty
```

### Matrix Components

- **a11, a22**: Scaling factors (µm per pixel)
- **a12, a21**: Rotation/shear
- **tx, ty**: Translation (usually 0 for displacement-based calibration)

### Common Transformations

| Transformation | Matrix Values |
|---------------|---------------|
| Identity (1:1) | `[[1, 0, 0], [0, 1, 0]]` |
| Flip X | `[[-1, 0, 0], [0, 1, 0]]` |
| Flip Y | `[[1, 0, 0], [0, -1, 0]]` |
| Flip Both | `[[-1, 0, 0], [0, -1, 0]]` |
| 90° Rotation | `[[0, -1, 0], [1, 0, 0]]` |
| Scale 0.5 µm/px | `[[0.5, 0, 0], [0, 0.5, 0]]` |

## Default Behavior

When no calibration exists, the system returns a default identity matrix:
- `[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]`
- 1:1 mapping with no rotation
- Can be customized in `PixelCalibrationInfo.defaultAffineMatrix`

## Application of Transformations

### Stage Coordinate Mapping
**Handled by**: `PixelCalibrationController` and `CSMExtension`
- Applies affine matrix to convert pixel coordinates to stage coordinates
- Used for stage movement commands
- Accounts for rotation, scaling, and camera-stage misalignment

### Camera Transformations (Rotation/Flip)
**Handled by**: `DetectorManager` (per-camera or centralized)
- Software rotation/flip of images
- Applied before displaying or processing
- Computationally efficient (CPU-free if using GPU)
- Configuration stored per camera in detector settings

### Recommended Approach

1. **Physical Setup**
   - Align camera as close to 90° with stage axes as possible
   - This minimizes rotation in affine matrix

2. **Calibration**
   - Perform affine calibration for each objective
   - System computes full transformation automatically
   - Rotation typically < 5° for well-aligned systems

3. **Image Display**
   - Apply camera rotation/flip in DetectorManager
   - This ensures images display correctly
   - Independent of stage coordinate system

4. **Stage Movement**
   - Use affine matrix for pixel-to-stage coordinate conversion
   - Accounts for any remaining misalignment
   - Provides high precision (< 1 µm RMSE)

## Calibration Storage Format

```json
{
  "format_version": "2.0",
  "PixelCalibration": {
    "affineCalibrations": {
      "10x": {
        "affine_matrix": [
          [0.500, 0.010, 0.0],
          [-0.010, 0.500, 0.0]
        ],
        "metrics": {
          "rmse_um": 0.234,
          "quality": "excellent",
          "rotation_deg": 1.15,
          "scale_x_um_per_pixel": 0.500,
          "scale_y_um_per_pixel": 0.500
        },
        "timestamp": "2025-10-14T23:16:55",
        "objective_info": {
          "name": "10x",
          "detector": "VirtualCamera"
        }
      }
    },
    "defaultAffineMatrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
  }
}
```

## API Usage

### Save Calibration
```python
# Performed automatically during calibration
setupInfo.setAffineCalibration("10x", {
    "affine_matrix": [[0.5, 0.01, 0.0], [-0.01, 0.5, 0.0]],
    "metrics": {...},
    "timestamp": "2025-10-14T23:16:55",
    "objective_info": {...}
})
```

### Load Calibration
```python
# Get full calibration data
calib = setupInfo.getAffineCalibration("10x")

# Get just the matrix (returns default if not found)
matrix = setupInfo.getAffineMatrix("10x")
```

### REST API
```bash
# Perform calibration
curl -X POST http://localhost:8001/calibrateStageAffine \
  -H "Content-Type: application/json" \
  -d '{"objectiveId": "10x", "stepSizeUm": 150.0, "pattern": "cross"}'

# Get calibration data
curl http://localhost:8001/getCalibrationData?objectiveId=10x

# List calibrated objectives
curl http://localhost:8001/getCalibrationObjectives

# Delete calibration
curl -X POST http://localhost:8001/deleteCalibration \
  -d '{"objectiveId": "10x"}'
```

## Best Practices

1. **Calibrate Each Objective**: Different magnifications have different pixel sizes
2. **Use Structured Sample**: Grid pattern or features for reliable correlation
3. **Check Quality Metrics**: RMSE should be < 2.0 µm for "good" quality
4. **Recalibrate When**: 
   - Changing objectives
   - After significant hardware changes
   - If accuracy degrades
5. **Set Default Matrix**: Configure appropriate default for your system
6. **Apply Camera Transforms**: Handle rotation/flip in DetectorManager for efficiency

## Troubleshooting

### High RMSE (> 5 µm)
- Check sample is in focus
- Ensure sufficient features for correlation
- Reduce step size
- Check stage mechanics

### Large Rotation (> 10°)
- Physically realign camera to stage
- Check mounting hardware
- Verify stage axes are orthogonal

### Flip Required
- Set in DetectorManager camera settings
- OR adjust default affine matrix
- Prefer camera-side flip for efficiency

## Integration with Other Systems

### Position Manager
- Similar to `saveStageOffset` pattern
- Stored in setup configuration
- Persisted automatically

### Objective Changer
- Automatically switch calibration when changing objectives
- Access via `setupInfo.getAffineMatrix(currentObjective)`

### Detector Manager
- Apply camera transformations (rotation/flip)
- Independent of stage coordinate system
- Stored per-camera in detector configuration
