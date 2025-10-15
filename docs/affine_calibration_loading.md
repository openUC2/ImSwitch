# Affine Calibration Loading System

## Overview

Affine calibrations are automatically loaded from the setup configuration when ImSwitch starts. This document explains the loading mechanism, how to access calibrations, and how to integrate them into your components.

## Automatic Loading on Startup

The `PixelCalibrationController` loads all affine calibrations during initialization:

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # ... other initialization ...
    
    # Load affine calibrations from setup info
    self.affineCalibrations = {}
    self.currentObjective = "default"
    self._loadAffineCalibrations()
```

### What Gets Loaded

For each objective with calibration data:
- **Affine Matrix**: 2×3 transformation matrix  
- **Pixel Size**: Extracted from scale parameters (scale_x_um_per_pixel, scale_y_um_per_pixel)
- **Rotation**: Camera rotation relative to stage axes in degrees
- **Quality Metrics**: RMSE, correlation, condition number, quality classification
- **Timestamp**: When the calibration was performed
- **Objective Info**: Name and detector used

### Log Output Example

```
[INFO] Loaded 2 affine calibration(s) from setup configuration:
  - default: scale=(1.000, 1.000) µm/px, rotation=0.00°, quality=excellent, calibrated=2025-10-15T21:34:20
  - 10x: scale=(0.500, 0.500) µm/px, rotation=1.15°, quality=excellent, calibrated=2025-10-15T22:00:00
[INFO] Set 'default' as active calibration
```

If no calibrations are found:
```
[INFO] No affine calibrations found in setup configuration
```

## Accessing Calibrations

### Get Affine Matrix

```python
# Get matrix for current objective
matrix = pixelCalibrationController.getAffineMatrix()

# Get matrix for specific objective
matrix_10x = pixelCalibrationController.getAffineMatrix("10x")

# Returns 2x3 numpy array:
# [[a11, a12, tx],
#  [a21, a22, ty]]
```

### Get Pixel Size

```python
# Get pixel size for current objective
scale_x, scale_y = pixelCalibrationController.getPixelSize()

# Get pixel size for specific objective
scale_x, scale_y = pixelCalibrationController.getPixelSize("10x")

# Returns tuple: (scale_x_um_per_pixel, scale_y_um_per_pixel)
```

### Switch Active Objective

```python
# Set current objective
pixelCalibrationController.setCurrentObjective("10x")

# Log output:
# [INFO] Switched to objective '10x'
# [INFO] Calibration loaded: pixel size = (0.500, 0.500) µm/px
```

If no calibration exists for the objective:
```
# [INFO] Switched to objective '20x'
# [INFO] No calibration for '20x' - using default identity matrix
```

### List Calibrated Objectives

```python
# Get all objectives with calibration data
objectives = pixelcalibration_helper.list_calibrated_objectives()
# Returns: ['default', '10x', '20x', ...]
```

## Configuration File Format

Calibrations are stored in `config.json`:

```json
{
  "PixelCalibration": {
    "affineCalibrations": {
      "default": {
        "affine_matrix": [
          [-1.0, 0.0, 0.0],
          [0.0, -1.0, 0.0]
        ],
        "metrics": {
          "rmse_um": 0.0,
          "max_error_um": 0.0,
          "mean_error_um": 0.0,
          "n_inliers": 9,
          "n_outliers": 0,
          "rotation_deg": 0.0,
          "scale_x_um_per_pixel": 1.0,
          "scale_y_um_per_pixel": 1.0,
          "condition_number": 1.0,
          "mean_correlation": 0.98,
          "min_correlation": 0.95,
          "quality": "excellent"
        },
        "timestamp": "2025-10-15T21:34:20",
        "objective_info": {
          "name": "default",
          "detector": "VirtualCamera"
        }
      },
      "10x": {
        "affine_matrix": [
          [0.5, 0.01, 0.0],
          [-0.01, 0.5, 0.0]
        ],
        "metrics": {
          "rmse_um": 0.234,
          "rotation_deg": 1.15,
          "scale_x_um_per_pixel": 0.5,
          "scale_y_um_per_pixel": 0.5,
          "quality": "excellent"
        },
        "timestamp": "2025-10-15T22:00:00",
        "objective_info": {
          "name": "10x",
          "detector": "VirtualCamera"
        }
      }
    },
    "defaultAffineMatrix": [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0]
    ]
  }
}
```

## Integration with Other Components

### ObjectiveController

Update calibration when objective changes:

```python
class ObjectiveController:
    def onObjectiveChanged(self, objective_id):
        """Called when user switches objectives."""
        # Update active calibration
        self._master.pixelCalibrationController.setCurrentObjective(objective_id)
        
        # Get pixel size for this objective
        scale_x, scale_y = self._master.pixelCalibrationController.getPixelSize(objective_id)
        
        # Update detector if needed
        self.detector.setPixelSizeUm([scale_x, scale_y])
        
        # Log info
        self._logger.info(f"Objective {objective_id}: pixel size = ({scale_x:.3f}, {scale_y:.3f}) µm/px")
```

### DetectorManager

Apply camera transformations from calibration:

```python
class DetectorManager:
    def applyAffineTransform(self, image, objective_id=None):
        """Apply rotation/flip from affine calibration."""
        # Get affine matrix
        affine_matrix = self._master.pixelCalibrationController.getAffineMatrix(objective_id)
        
        # Extract rotation
        rotation_rad = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        
        # Apply rotation if significant (use GPU when available)
        if abs(rotation_deg) > 0.1:
            image = self._rotateImage(image, -rotation_deg)
        
        # Apply flips
        if affine_matrix[0, 0] < 0:  # X-axis flipped
            image = np.flip(image, axis=1)
        if affine_matrix[1, 1] < 0:  # Y-axis flipped
            image = np.flip(image, axis=0)
        
        return image
```

### StageController

Convert pixel coordinates to stage movements:

```python
class StageController:
    def moveToPixelCoordinates(self, pixel_x, pixel_y, objective_id=None):
        """Move stage to clicked pixel coordinates."""
        # Get affine matrix
        affine_matrix = self._master.pixelCalibrationController.getAffineMatrix(objective_id)
        
        # Transform pixel to stage coordinates
        pixel_coords = np.array([pixel_x, pixel_y])
        stage_coords = affine_matrix[:, :2] @ pixel_coords + affine_matrix[:, 2]
        
        # Move stage
        self.moveAbsolute(stage_coords[0], stage_coords[1])
```

## Using in Custom Controllers

To access calibrations in your controller:

```python
class MyController(Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get reference to PixelCalibrationController
        self.pixelCalibController = self._master.pixelCalibrationController
    
    def useCalibration(self, objective_id="default"):
        # Get affine matrix
        matrix = self.pixelCalibController.getAffineMatrix(objective_id)
        
        # Get pixel size
        scale_x, scale_y = self.pixelCalibController.getPixelSize(objective_id)
        
        # Access raw calibration data
        if objective_id in self.pixelCalibController.affineCalibrations:
            calib_data = self.pixelCalibController.affineCalibrations[objective_id]
            metrics = calib_data['metrics']
            timestamp = calib_data['timestamp']
            
            self._logger.info(f"Using calibration from {timestamp}")
            self._logger.info(f"Quality: {metrics['quality']}, RMSE: {metrics['rmse_um']:.3f} µm")
```

## Default Behavior

If no calibration exists for an objective, the system returns a default identity matrix:

```python
[[1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0]]
```

This ensures the system works without calibration, though coordinate mapping will be 1:1 (not accurate).

Pixel size defaults to `(1.0, 1.0)` µm/px when no calibration exists.

## Saving Calibrations

Calibrations are saved automatically after each calibration:

```python
# Performed internally in calibrate_affine()
self._setupInfo.setAffineCalibration(objective_id, calibration_data)

import imswitch.imcontrol.model.configfiletools as configfiletools
config_file_path, _ = configfiletools.loadOptions()
configfiletools.saveSetupInfo(config_file_path, self._setupInfo)
```

The save mechanism follows the same pattern as `PositionerManager.saveStageOffset()`.

## Troubleshooting

### Calibrations Not Loading

**Check 1**: Verify `PixelCalibration` exists in `config.json`

```json
{
  "PixelCalibration": {
    "affineCalibrations": {...},
    "defaultAffineMatrix": [[1, 0, 0], [0, 1, 0]]
  }
}
```

**Check 2**: Check log output on startup

Look for:
```
[INFO] Loaded N affine calibration(s) from setup configuration
```

If not present:
```
[INFO] No PixelCalibration in setup configuration - using default identity matrix
```

**Check 3**: Verify JSON formatting

Ensure all brackets, commas, and quotes are correct. Use a JSON validator if needed.

### Wrong Pixel Size

**Cause**: Calibration metrics may not include scale parameters

**Solution**: Recalibrate using the new system. Old calibrations may not have `scale_x_um_per_pixel` and `scale_y_um_per_pixel` in metrics.

### Calibration for Wrong Detector

**Cause**: Calibration performed with different camera

**Solution**: Check `objective_info.detector` in calibration data. Recalibrate if camera changed.

## Best Practices

1. **Calibrate all objectives**: Each objective should have its own calibration
2. **Use descriptive IDs**: "10x", "20x", "40x" are clearer than "slot_1", "slot_2"
3. **Check quality metrics**: Verify RMSE < 2 µm and correlation > 0.8
4. **Recalibrate after changes**: Any mechanical adjustment requires recalibration
5. **Monitor on startup**: Check log output to confirm calibrations loaded
6. **Switch objectives properly**: Use `setCurrentObjective()` when changing
7. **Document configuration**: Note which objectives are calibrated in your setup notes

## API Reference

### PixelCalibrationController Methods

```python
# Loading (automatic on startup)
_loadAffineCalibrations()

# Accessing
getAffineMatrix(objective_id=None) -> np.ndarray
getPixelSize(objective_id=None) -> tuple
setCurrentObjective(objective_id: str)

# Properties
affineCalibrations: dict  # All loaded calibrations
currentObjective: str     # Active objective ID
```

### PixelCalibrationClass (Helper) Methods

```python
# Calibration
calibrate_affine(objective_id, step_size_um, pattern, ...) -> dict

# Retrieval
get_affine_matrix(objective_id) -> np.ndarray
list_calibrated_objectives() -> list
get_metrics(objective_id) -> dict
```

### SetupInfo Methods

```python
# Storage
setAffineCalibration(objective_id, calibration_data)
getAffineCalibration(objective_id) -> dict
getAffineMatrix(objective_id) -> np.ndarray
```

## Example: Complete Workflow

```python
# On startup (automatic)
# PixelCalibrationController loads all calibrations from config.json

# Switch to 10x objective (in ObjectiveController)
pixelCalibController.setCurrentObjective("10x")

# Get calibration info
matrix = pixelCalibController.getAffineMatrix("10x")
scale_x, scale_y = pixelCalibController.getPixelSize("10x")

# Use in stage movement (in StageController)
pixel_coords = np.array([clicked_x, clicked_y])
stage_coords = matrix[:, :2] @ pixel_coords + matrix[:, 2]
stage.moveTo(stage_coords[0], stage_coords[1])

# Switch to 20x objective
pixelCalibController.setCurrentObjective("20x")

# If not calibrated, returns default identity matrix
matrix_20x = pixelCalibController.getAffineMatrix("20x")
# [[1, 0, 0], [0, 1, 0]]
```

This system ensures calibrations are loaded once at startup and available throughout the application lifecycle.
