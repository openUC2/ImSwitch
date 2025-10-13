"""
Example: Using the Affine Stage-to-Camera Calibration System

This script demonstrates how to use the new robust affine calibration
system for microscope stage-to-camera coordinate mapping.

Note: This example assumes you have a microscope with a camera and motorized stage.
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_basic_calibration():
    """
    Example 1: Basic calibration for a single objective
    """
    print("\n" + "="*70)
    print("Example 1: Basic Affine Calibration")
    print("="*70)
    
    from imswitch.imcontrol.controller.controllers.camera_stage_mapping.OFMStageMapping import OFMStageScanClass
    
    # Initialize the stage mapping class
    # In a real setup, you would pass actual detector and stage objects
    stage_mapping = OFMStageScanClass(
        calibration_file_path="microscope_calibration.json",
        effPixelsize=1.0,     # 1.0 micron effective pixel size
        stageStepSize=1.0,    # 1.0 micron stage step size
        IS_CLIENT=False,
        # mDetector=your_detector,  # Replace with actual detector
        # mStage=your_stage          # Replace with actual stage
    )
    
    # Perform calibration for 10x objective
    print("\nCalibrating 10x objective...")
    print("This will move the stage in a cross pattern and measure displacements.")
    
    try:
        result = stage_mapping.calibrate_affine(
            objective_id="10x",
            step_size_um=150.0,   # 150 microns per step (suitable for 10x)
            pattern="cross",       # Cross pattern (faster)
            n_steps=4,            # 4 steps per direction (9 positions total)
            validate=True         # Enable validation
        )
        
        # Display results
        print("\nCalibration Results:")
        print(f"  Quality: {result['metrics']['quality']}")
        print(f"  RMSE: {result['metrics']['rmse_um']:.3f} µm")
        print(f"  Rotation: {result['metrics']['rotation_deg']:.2f}°")
        print(f"  Scale X: {result['metrics']['scale_x_um_per_pixel']:.3f} µm/pixel")
        print(f"  Scale Y: {result['metrics']['scale_y_um_per_pixel']:.3f} µm/pixel")
        print(f"  Inliers: {result['metrics']['n_inliers']}/{result['metrics']['n_inliers'] + result['metrics']['n_outliers']}")
        
        # Check validation
        if result['validation']['is_valid']:
            print("\n✓ Calibration passed validation!")
        else:
            print(f"\n⚠ Calibration validation warnings:")
            print(f"  {result['validation']['message']}")
        
        # The affine matrix is automatically saved to the calibration file
        print(f"\nCalibration saved to: microscope_calibration.json")
        
    except Exception as e:
        print(f"Calibration failed: {e}")
        logger.exception("Calibration error details:")


def example_multiple_objectives():
    """
    Example 2: Calibrate multiple objectives with appropriate parameters
    """
    print("\n" + "="*70)
    print("Example 2: Calibrating Multiple Objectives")
    print("="*70)
    
    from imswitch.imcontrol.controller.controllers.camera_stage_mapping.OFMStageMapping import OFMStageScanClass
    
    stage_mapping = OFMStageScanClass(
        calibration_file_path="microscope_calibration.json",
        effPixelsize=1.0,
        stageStepSize=1.0,
        IS_CLIENT=False,
    )
    
    # Define objectives with appropriate step sizes
    objectives_config = [
        {"id": "4x",  "step_um": 200.0, "description": "4x objective"},
        {"id": "10x", "step_um": 150.0, "description": "10x objective"},
        {"id": "20x", "step_um": 75.0,  "description": "20x objective"},
        {"id": "40x", "step_um": 40.0,  "description": "40x objective"},
    ]
    
    for obj_config in objectives_config:
        print(f"\nCalibrating {obj_config['description']}...")
        
        try:
            result = stage_mapping.calibrate_affine(
                objective_id=obj_config['id'],
                step_size_um=obj_config['step_um'],
                pattern="cross",
                validate=True
            )
            
            quality = result['metrics']['quality']
            rmse = result['metrics']['rmse_um']
            print(f"  ✓ {obj_config['id']}: {quality} (RMSE={rmse:.3f}µm)")
            
        except Exception as e:
            print(f"  ✗ {obj_config['id']}: Failed - {e}")
    
    # List all calibrated objectives
    objectives = stage_mapping.list_calibrated_objectives()
    print(f"\nCalibrated objectives: {', '.join(objectives)}")


def example_using_calibration():
    """
    Example 3: Using calibration for precise movements
    """
    print("\n" + "="*70)
    print("Example 3: Using Calibration for Movement")
    print("="*70)
    
    from imswitch.imcontrol.controller.controllers.camera_stage_mapping.OFMStageMapping import OFMStageScanClass
    
    stage_mapping = OFMStageScanClass(
        calibration_file_path="microscope_calibration.json",
        effPixelsize=1.0,
        stageStepSize=1.0,
        IS_CLIENT=False,
    )
    
    # Select objective
    objective_id = "10x"
    print(f"\nUsing calibration for {objective_id} objective")
    
    try:
        # Get the affine matrix
        affine_matrix = stage_mapping.get_affine_matrix(objective_id)
        print(f"\nAffine transformation matrix:")
        print(affine_matrix)
        
        # Example: Move 100 pixels to the right, 50 pixels up in the image
        pixel_displacement = np.array([100, 50])
        print(f"\nMoving {pixel_displacement} pixels in image coordinates...")
        
        # This will calculate the required stage movement and execute it
        stage_mapping.move_in_image_coordinates_affine(
            pixel_displacement,
            objective_id=objective_id
        )
        
        print("✓ Movement completed")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure the objective is calibrated first!")


def example_calibration_storage():
    """
    Example 4: Working with calibration storage directly
    """
    print("\n" + "="*70)
    print("Example 4: Managing Calibration Data")
    print("="*70)
    
    from imswitch.imcontrol.controller.controllers.camera_stage_mapping.calibration_storage import CalibrationStorage
    
    # Open calibration storage
    storage = CalibrationStorage("microscope_calibration.json")
    
    # List all calibrated objectives
    objectives = storage.list_objectives()
    print(f"\nCalibrated objectives: {objectives}")
    
    # Get detailed information for each objective
    for obj_id in objectives:
        print(f"\n{obj_id}:")
        
        # Load calibration data
        calib = storage.load_calibration(obj_id)
        
        if calib:
            print(f"  Timestamp: {calib['timestamp']}")
            print(f"  Quality: {calib['metrics'].get('quality', 'N/A')}")
            print(f"  RMSE: {calib['metrics'].get('rmse_um', 0):.3f} µm")
            print(f"  Rotation: {calib['metrics'].get('rotation_deg', 0):.2f}°")
            
            # Show affine matrix
            print(f"  Affine matrix shape: {calib['affine_matrix'].shape}")
            print(f"  Scale X: {calib['metrics'].get('scale_x_um_per_pixel', 0):.3f} µm/px")
            print(f"  Scale Y: {calib['metrics'].get('scale_y_um_per_pixel', 0):.3f} µm/px")
    
    # Export to legacy format for backward compatibility
    if objectives:
        print(f"\nExporting {objectives[0]} to legacy format...")
        legacy_data = storage.export_to_legacy_format(objectives[0])
        print(f"Legacy format keys: {list(legacy_data.keys())}")


def example_custom_pattern():
    """
    Example 5: Using a grid pattern for high-precision calibration
    """
    print("\n" + "="*70)
    print("Example 5: High-Precision Grid Calibration")
    print("="*70)
    
    from imswitch.imcontrol.controller.controllers.camera_stage_mapping.OFMStageMapping import OFMStageScanClass
    
    stage_mapping = OFMStageScanClass(
        calibration_file_path="microscope_calibration.json",
        effPixelsize=1.0,
        stageStepSize=1.0,
        IS_CLIENT=False,
    )
    
    print("\nPerforming high-precision calibration with grid pattern...")
    print("This takes longer but provides more measurement points.")
    
    try:
        result = stage_mapping.calibrate_affine(
            objective_id="20x_precision",
            step_size_um=75.0,
            pattern="grid",      # Grid pattern instead of cross
            n_steps=4,           # 4x4 = 16 positions
            validate=True
        )
        
        print(f"\nResults:")
        print(f"  Quality: {result['metrics']['quality']}")
        print(f"  RMSE: {result['metrics']['rmse_um']:.3f} µm")
        print(f"  Number of measurements: {result['metrics']['n_inliers'] + result['metrics']['n_outliers']}")
        print(f"  Outliers rejected: {result['metrics']['n_outliers']}")
        
    except Exception as e:
        print(f"Calibration failed: {e}")


def example_validation_only():
    """
    Example 6: Load and validate existing calibration
    """
    print("\n" + "="*70)
    print("Example 6: Validating Existing Calibration")
    print("="*70)
    
    from imswitch.imcontrol.controller.controllers.camera_stage_mapping.calibration_storage import CalibrationStorage
    from imswitch.imcontrol.controller.controllers.camera_stage_mapping.affine_stage_calibration import validate_calibration
    
    storage = CalibrationStorage("microscope_calibration.json")
    objectives = storage.list_objectives()
    
    if not objectives:
        print("No calibrations found.")
        return
    
    for obj_id in objectives:
        print(f"\nValidating {obj_id}...")
        
        calib = storage.load_calibration(obj_id)
        if calib:
            is_valid, message = validate_calibration(
                calib['affine_matrix'],
                calib['metrics']
            )
            
            if is_valid:
                print(f"  ✓ Valid calibration")
            else:
                print(f"  ⚠ Validation issues:")
                print(f"    {message}")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Affine Stage-to-Camera Calibration Examples")
    print("="*70)
    print("\nNote: These examples demonstrate the API usage.")
    print("To run on real hardware, you need to provide actual detector and stage objects.")
    print("\nUncomment the examples you want to run:\n")
    
    # Uncomment to run specific examples:
    
    # example_basic_calibration()
    # example_multiple_objectives()
    # example_using_calibration()
    # example_calibration_storage()
    # example_custom_pattern()
    # example_validation_only()
    
    print("\n" + "="*70)
    print("To use these examples:")
    print("1. Uncomment the example functions above")
    print("2. Provide actual detector and stage objects")
    print("3. Ensure you have a calibration sample in focus")
    print("4. Run the script")
    print("="*70 + "\n")
