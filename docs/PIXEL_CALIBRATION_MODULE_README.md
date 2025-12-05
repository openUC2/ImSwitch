"""
Pixel Calibration Module

This module provides camera-to-stage calibration and navigation capabilities
using both traditional affine calibration and AprilTag grid-based methods.

## Modules

### overview_calibrator.py
General overhead camera calibration utilities:
- Stage axis identification using AprilTag tracking
- Illumination channel mapping
- Homing verification
- Step size sign determination

### apriltag_grid_calibrator.py
AprilTag grid-based stage calibration and navigation:
- GridConfig: Stores grid layout (rows, cols, pitch, start_id)
- AprilTagGridCalibrator: Main calibration and navigation class
  - Camera-to-stage affine transformation calibration
  - Closed-loop navigation to specific tag IDs
  - Search patterns for initially invisible tags

## Usage

See documentation:
- docs/APRILTAG_GRID_CALIBRATION.md - Full documentation
- docs/APRILTAG_GRID_QUICKREF.md - Quick API reference
- docs/APRILTAG_IMPLEMENTATION_SUMMARY.md - Implementation details

## API Endpoints

Access via PixelCalibrationController:

### Grid Calibration
- gridSetConfig(rows, cols, start_id, pitch_mm)
- gridGetConfig()
- gridDetectTags(save_annotated)
- gridCalibrateTransform()
- gridMoveToTag(target_id, ...)
- gridGetTagInfo(tag_id)

### Overview Calibration
- overviewIdentifyAxes(...)
- overviewMapIllumination(...)
- overviewVerifyHoming(...)
- overviewFixStepSign(...)
- overviewCaptureObjective(...)

## Testing

Run unit tests:
```bash
pytest imswitch/imcontrol/_test/unit/pixelcalibration/ -v
```

## Dependencies

- OpenCV (opencv-contrib-python) for AprilTag detection
- NumPy for linear algebra
- ImSwitch core modules
"""
