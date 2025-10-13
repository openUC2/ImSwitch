# Robust Affine Stage-to-Camera Calibration

## Overview

This implementation provides a production-ready automated calibration system that determines the relationship between microscope stage movement and camera pixel coordinates. The system computes a full 2×3 affine transformation matrix and supports per-objective calibration persistence.

## Key Features

### 1. Robust Calibration Protocol
- **Phase Correlation**: Uses FFT-based phase correlation for sub-pixel displacement accuracy (0.01 pixel precision)
- **Outlier Detection**: RANSAC-like robust fitting with automatic outlier rejection
- **Quality Metrics**: Comprehensive validation including RMSE, correlation quality, and condition number
- **Structured Patterns**: Supports both "cross" and "grid" movement patterns

### 2. Per-Objective Storage
- **Multiple Objectives**: Store separate calibration for each objective (10x, 20x, etc.)
- **JSON Format**: Human-readable calibration files with versioning
- **Backward Compatible**: Automatically migrates legacy calibration data
- **Metadata**: Stores timestamp, quality metrics, and objective information

### 3. Production Ready
- **Error Handling**: Comprehensive validation and error reporting
- **Low-Power Friendly**: Efficient algorithms suitable for Raspberry Pi
- **Logging**: Detailed progress logging for debugging
- **Thread-Safe**: Can run in background threads

## Architecture

### Core Modules

1. **affine_stage_calibration.py**
   - `calibrate_affine_transform()`: Main calibration routine
   - `compute_displacement_phase_correlation()`: Sub-pixel image displacement
   - `robust_affine_from_correspondences()`: Outlier-resistant affine fitting
   - `validate_calibration()`: Quality validation
   - `apply_affine_transform()`: Transform pixel coordinates to stage units

2. **calibration_storage.py**
   - `CalibrationStorage`: Manages per-objective calibration data
   - JSON file format with versioning
   - Backward compatibility layer
   - CRUD operations for calibration data

3. **OFMStageMapping.py** (updated)
   - `calibrate_affine()`: New calibration method
   - `get_affine_matrix()`: Retrieve per-objective transformation
   - `list_calibrated_objectives()`: List available calibrations
   - `move_in_image_coordinates_affine()`: Move using affine transform

4. **PixelCalibrationController.py** (updated)
   - `stageCalibrationAffine()`: UI hook for new calibration
   - Integration with existing controller architecture

## Usage

### Basic Calibration

```python
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.OFMStageMapping import OFMStageScanClass

# Initialize stage mapping
stage_mapping = OFMStageScanClass(
    calibration_file_path="my_calibration.json",
    effPixelsize=1.0,  # Effective pixel size in microns
    stageStepSize=1.0,  # Stage step size in microns
    mDetector=detector,
    mStage=stage
)

# Perform calibration for 10x objective
result = stage_mapping.calibrate_affine(
    objective_id="10x",
    step_size_um=100.0,  # Move 100 microns per step
    pattern="cross",      # Use cross pattern
    n_steps=4,           # 4 steps in each direction
    validate=True        # Run validation checks
)

# Check calibration quality
print(f"Quality: {result['metrics']['quality']}")
print(f"RMSE: {result['metrics']['rmse_um']:.3f} µm")
print(f"Rotation: {result['metrics']['rotation_deg']:.2f}°")
```

### Using Calibration for Movement

```python
# Get calibration matrix
affine_matrix = stage_mapping.get_affine_matrix("10x")

# Move 50 pixels right, 30 pixels up
pixel_displacement = np.array([50, 30])
stage_mapping.move_in_image_coordinates_affine(
    pixel_displacement,
    objective_id="10x"
)
```

### Managing Multiple Objectives

```python
# Calibrate multiple objectives
for objective_id, step_size in [("10x", 150), ("20x", 75), ("40x", 40)]:
    stage_mapping.calibrate_affine(
        objective_id=objective_id,
        step_size_um=step_size
    )

# List all calibrated objectives
objectives = stage_mapping.list_calibrated_objectives()
print(f"Calibrated objectives: {objectives}")

# Switch between objectives
current_objective = "20x"
affine_matrix = stage_mapping.get_affine_matrix(current_objective)
```

## Calibration File Format

The calibration data is stored in JSON format:

```json
{
    "format_version": "2.0",
    "objectives": {
        "10x": {
            "affine_matrix": [
                [0.500, 0.010, 0.0],
                [-0.010, 0.500, 0.0]
            ],
            "metrics": {
                "rmse_um": 0.234,
                "rotation_deg": 1.15,
                "scale_x_um_per_pixel": 0.500,
                "scale_y_um_per_pixel": 0.500,
                "quality": "excellent"
            },
            "timestamp": "2024-10-13T19:00:00",
            "objective_info": {
                "name": "10x",
                "magnification": 10
            }
        }
    },
    "legacy_data": {}
}
```

## Calibration Protocol Details

### Movement Pattern

**Cross Pattern** (default, recommended):
```
    5
    |
1---0---2
    |
    3
    (+ diagonals: 6, 7, 8, 9)
```
- Center position (reference)
- 4 cardinal directions (±X, ±Y)
- 4 diagonal positions for better conditioning
- Total: 9 positions

**Grid Pattern**:
- Regular grid of n_steps × n_steps positions
- More measurements but takes longer
- Use for high-precision requirements

### Algorithm Steps

1. **Initialize**: Capture reference image at starting position
2. **Move**: Execute structured movement pattern
3. **Measure**: Use phase correlation to compute sub-pixel displacement for each position
4. **Fit**: Least-squares fit of 2×3 affine matrix with outlier rejection
5. **Validate**: Check RMSE, correlation, condition number, scaling consistency
6. **Store**: Save calibration data with metadata

### Quality Metrics

The system computes comprehensive quality metrics:

- **RMSE**: Root mean square error in microns
- **Mean/Max Error**: Statistical error measures
- **Correlation**: Image correlation quality (0-1)
- **Rotation**: Estimated rotation angle in degrees
- **Scale X/Y**: Microns per pixel in each direction
- **Condition Number**: Matrix conditioning (stability)
- **Inliers/Outliers**: Number of valid/rejected measurements

### Quality Classification

- **Excellent**: RMSE < 1.0 µm, correlation > 0.5
- **Good**: RMSE < 2.0 µm, correlation > 0.3
- **Acceptable**: RMSE < 5.0 µm
- **Poor**: RMSE ≥ 5.0 µm (calibration should be redone)

## Best Practices

### Sample Preparation
- Use structured calibration sample (e.g., grid pattern, dots)
- Ensure sample is in focus
- Adequate illumination (auto-exposure will adjust if available)
- Sample should have sufficient features for correlation

### Step Size Selection
- **10x objective**: 100-150 µm recommended
- **20x objective**: 50-100 µm recommended
- **40x objective**: 25-50 µm recommended
- General rule: step size should be ~10-20% of field of view

### Troubleshooting

**High RMSE / Poor Quality**:
- Check focus
- Verify sample has sufficient features
- Try different step size
- Check for mechanical issues (loose stage, vibration)
- Ensure stage axes are correctly configured

**Low Correlation**:
- Improve illumination
- Check sample contrast
- Verify camera is capturing images correctly
- Try different region of sample

**Anisotropic Scaling**:
- May indicate camera/stage misalignment
- Check that stage axes are perpendicular
- Verify camera sensor is not skewed

## API Reference

### calibrate_affine_transform()

```python
def calibrate_affine_transform(
    tracker: Tracker,
    move: Callable,
    step_size_um: float = 100.0,
    pattern: str = "cross",
    n_steps: int = 4,
    settle_time: float = 0.2,
    logger: Optional[logging.Logger] = None
) -> Dict
```

**Returns**: Dictionary with keys:
- `affine_matrix`: 2×3 numpy array
- `metrics`: Quality metrics dictionary
- `pixel_displacements`: Measured pixel shifts
- `stage_displacements`: Commanded stage shifts
- `correlation_values`: Correlation quality for each measurement
- `inlier_mask`: Boolean mask of valid measurements

### CalibrationStorage

```python
class CalibrationStorage:
    def save_calibration(objective_id, affine_matrix, metrics, objective_info)
    def load_calibration(objective_id) -> Optional[Dict]
    def list_objectives() -> List[str]
    def delete_calibration(objective_id) -> bool
    def get_affine_matrix(objective_id) -> Optional[np.ndarray]
```

## Migration from Legacy Calibration

The system automatically migrates legacy calibration data:

1. Detects old format (v1.0) with `image_to_stage_displacement` field
2. Converts 2×2 matrix to 2×3 by adding zero translation
3. Stores in `legacy_data` section
4. Creates default objective entry if possible

Legacy code accessing `image_to_stage_displacement_matrix` continues to work:
- First tries to use new affine calibration
- Falls back to legacy data if available
- Raises error if no calibration found

## Performance Considerations

- **Calibration Time**: ~30-60 seconds depending on pattern and number of steps
- **Memory**: Minimal (<10 MB for calibration data)
- **CPU**: Efficient FFT-based correlation works well on Raspberry Pi
- **Storage**: ~5 KB per objective in JSON format

## Future Enhancements

Potential improvements for future versions:
- Auto-exposure integration (hardware-dependent)
- Multi-scale calibration for zoom systems
- Temperature compensation
- Real-time drift correction
- Machine learning-based quality prediction
- Interactive calibration wizard UI

## License

Copyright 2024, released under GNU GPL v3

## See Also

- `camera_stage_calibration_1d.py`: Legacy 1D calibration
- `camera_stage_tracker.py`: Image tracking utilities
- `fft_image_tracking.py`: FFT-based correlation
