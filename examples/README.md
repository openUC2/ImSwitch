# ImSwitch Examples

This directory contains example scripts demonstrating how to use various features of ImSwitch.

## Affine Stage-to-Camera Calibration

### affine_calibration_examples.py

Comprehensive examples showing how to use the new robust affine calibration system for microscope stage-to-camera coordinate mapping.

**Features demonstrated:**
- Basic calibration for a single objective
- Calibrating multiple objectives with appropriate parameters
- Using calibration for precise stage movements
- Managing calibration data storage
- High-precision grid pattern calibration
- Validating existing calibrations

**Prerequisites:**
- ImSwitch installed and configured
- Microscope with camera and motorized stage
- Calibration sample (grid pattern or structured features)

**Usage:**
1. Edit the example file to provide your detector and stage objects
2. Uncomment the examples you want to run
3. Run: `python examples/affine_calibration_examples.py`

**Quick Start:**

```python
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.OFMStageMapping import OFMStageScanClass

# Initialize
stage_mapping = OFMStageScanClass(
    calibration_file_path="my_calibration.json",
    effPixelsize=1.0,
    stageStepSize=1.0,
    mDetector=your_detector,
    mStage=your_stage
)

# Calibrate
result = stage_mapping.calibrate_affine(
    objective_id="10x",
    step_size_um=150.0,
    pattern="cross"
)

# Use calibration
stage_mapping.move_in_image_coordinates_affine(
    np.array([100, 50]),  # Move 100px right, 50px up
    objective_id="10x"
)
```

**Documentation:**
See `docs/affine_calibration_guide.md` for complete documentation.

## Contributing Examples

If you have useful example scripts, please consider contributing them!

1. Create a descriptive filename
2. Add comprehensive comments
3. Include error handling
4. Document prerequisites
5. Update this README

## License

All examples are released under the same license as ImSwitch (GNU GPL v3).
