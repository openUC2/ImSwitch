#!/usr/bin/env python3
"""
Example script demonstrating the renewed STORM controller functionality.

This script shows how to use the enhanced STORM controller with:
- Pydantic parameter models
- Local microeye processing
- Enhanced data storage
- REST API endpoints
- No widget dependencies
"""

import asyncio
import json
import time
from pathlib import Path

# Import the renewed STORM controller (would normally be done via ImSwitch)
# from imswitch.imcontrol.controller.controllers.STORMReconController import STORMReconController
# from imswitch.imcontrol.model.storm_models import STORMProcessingParameters, STORMAcquisitionParameters

def example_basic_local_processing():
    """
    Example 1: Basic local STORM processing with microeye
    """
    print("=== Example 1: Basic Local STORM Processing ===")
    
    # This would normally be obtained from ImSwitch controller manager
    # controller = get_storm_controller()
    
    # Set processing parameters using pydantic model
    processing_params = {
        'threshold': 0.3,
        'fit_roi_size': 15,
        'fitting_method': '2D_Phasor_CPU',
        'update_rate': 10
    }
    
    print(f"Setting processing parameters: {processing_params}")
    # result = controller.setSTORMProcessingParameters(**processing_params)
    # print(f"Parameters set: {result}")
    
    # Start local reconstruction
    acquisition_params = {
        'session_id': 'example_local_session',
        'process_locally': True,
        'save_enabled': True,
        'save_format': 'tiff'
    }
    
    print(f"Starting local reconstruction: {acquisition_params}")
    # result = controller.startSTORMReconstructionLocal(**acquisition_params)
    # print(f"Started: {result}")
    
    # Simulate acquisition time
    print("Simulating 10 seconds of acquisition...")
    time.sleep(10)
    
    # Stop reconstruction
    # result = controller.stopSTORMReconstructionLocal()
    # print(f"Stopped: {result}")
    
    # Get final reconstruction path
    # final_path = controller.getLastReconstructedImagePath()
    # print(f"Final reconstruction saved at: {final_path}")
    
    print("Example 1 completed.\n")


def example_parameter_validation():
    """
    Example 2: Demonstrate pydantic parameter validation
    """
    print("=== Example 2: Parameter Validation ===")
    
    # Valid parameters
    valid_params = {
        'threshold': 0.25,
        'fit_roi_size': 13,
        'fitting_method': '2D_Gauss_MLE_fixed_sigma',
        'update_rate': 15,
        'pixel_size_nm': 100.0
    }
    
    print(f"Valid parameters: {json.dumps(valid_params, indent=2)}")
    # result = controller.setSTORMProcessingParameters(**valid_params)
    
    # Invalid parameters (would be caught by pydantic)
    invalid_params = {
        'threshold': 1.5,  # > 1.0, should fail
        'fit_roi_size': 6,  # < 7, should fail
        'fitting_method': 'invalid_method'  # not in enum
    }
    
    print(f"Invalid parameters: {json.dumps(invalid_params, indent=2)}")
    try:
        # result = controller.setSTORMProcessingParameters(**invalid_params)
        pass
    except Exception as e:
        print(f"Validation error (expected): {e}")
    
    print("Example 2 completed.\n")


def example_enhanced_data_storage():
    """
    Example 3: Enhanced data storage following experimentcontroller pattern
    """
    print("=== Example 3: Enhanced Data Storage ===")
    
    # Configure acquisition with enhanced storage
    acquisition_config = {
        'session_id': 'enhanced_storage_session',
        'process_locally': True,
        'save_enabled': True,
        'save_format': 'tiff',
        'max_frames': 100
    }
    
    print(f"Storage configuration: {json.dumps(acquisition_config, indent=2)}")
    
    # This would create directory structure like:
    # UserFileDirs.Data/STORMController/20240115_143022/enhanced_storage_session/
    #   ├── raw_frames/
    #   ├── reconstructed_frames/
    #   ├── localizations/
    #   └── final_reconstruction.tif
    
    print("Directory structure would be created:")
    print("  Data/STORMController/YYYYMMDD_HHMMSS/session_id/")
    print("    ├── raw_frames/")
    print("    ├── reconstructed_frames/") 
    print("    ├── localizations/")
    print("    └── final_reconstruction.tif")
    
    print("Example 3 completed.\n")


def example_rest_api_usage():
    """
    Example 4: REST API endpoint usage
    """
    print("=== Example 4: REST API Endpoints ===")
    
    # These endpoints would be available via ImSwitch REST API
    api_endpoints = {
        'set_parameters': '/storm/setSTORMProcessingParameters',
        'start_local': '/storm/startSTORMReconstructionLocal', 
        'stop_local': '/storm/stopSTORMReconstructionLocal',
        'get_status': '/storm/getSTORMReconstructionStatus',
        'get_last_image': '/storm/getLastReconstructedImagePath',
        'trigger_single': '/storm/triggerSingleFrameReconstruction'
    }
    
    print("Available REST API endpoints:")
    for name, endpoint in api_endpoints.items():
        print(f"  {name}: {endpoint}")
    
    # Example API call structure:
    api_call_examples = {
        'set_parameters': {
            'method': 'POST',
            'data': {
                'threshold': 0.3,
                'fit_roi_size': 15,
                'fitting_method': '2D_Phasor_CPU'
            }
        },
        'start_local': {
            'method': 'POST', 
            'data': {
                'session_id': 'api_session',
                'processing': {'threshold': 0.25},
                'acquisition': {'save_enabled': True}
            }
        },
        'get_status': {
            'method': 'GET',
            'response_format': {
                'acquisition_active': False,
                'local_processing_active': False,
                'last_reconstruction_path': None,
                'microeye_worker_available': True
            }
        }
    }
    
    print("\nExample API calls:")
    for endpoint, example in api_call_examples.items():
        print(f"  {endpoint}:")
        print(f"    Method: {example['method']}")
        if 'data' in example:
            print(f"    Data: {json.dumps(example['data'], indent=6)}")
        if 'response_format' in example:
            print(f"    Response: {json.dumps(example['response_format'], indent=6)}")
    
    print("Example 4 completed.\n")


def example_signal_updates():
    """
    Example 5: Signal updates for frontend consumption
    """
    print("=== Example 5: Signal Updates ===")
    
    # The renewed controller provides signals for frontend updates
    available_signals = [
        'sigFrameAcquired(int)',  # frame_number
        'sigFrameProcessed(STORMReconstructionResult)',  # result object
        'sigAcquisitionStateChanged(bool)',  # active state
        'sigProcessingStateChanged(bool)',  # processing state
        'sigErrorOccurred(str)'  # error message
    ]
    
    print("Available signals for frontend updates:")
    for signal in available_signals:
        print(f"  - {signal}")
    
    # Example signal connection (would be done in ImSwitch frontend)
    print("\nExample signal connections:")
    print("  controller.sigFrameAcquired.connect(update_frame_counter)")
    print("  controller.sigFrameProcessed.connect(update_reconstruction_display)")
    print("  controller.sigAcquisitionStateChanged.connect(update_ui_state)")
    print("  controller.sigErrorOccurred.connect(show_error_message)")
    
    # Example result object structure
    result_structure = {
        'frame_number': 123,
        'timestamp': '2024-01-15T14:30:22.123456',
        'session_id': 'example_session',
        'num_localizations': 45,
        'raw_frame_path': '/data/.../raw_frame_000123.tif',
        'reconstructed_frame_path': '/data/.../recon_frame_000123.tif',
        'processing_parameters': '...',
        'acquisition_parameters': '...'
    }
    
    print(f"\nExample result object structure:")
    print(json.dumps(result_structure, indent=2))
    
    print("Example 5 completed.\n")


async def example_async_workflow():
    """
    Example 6: Asynchronous workflow for integration
    """
    print("=== Example 6: Asynchronous Workflow ===")
    
    print("Simulating async workflow...")
    
    # Start acquisition
    print("1. Starting acquisition...")
    await asyncio.sleep(1)
    
    # Process frames in background
    print("2. Processing frames...")
    for i in range(5):
        print(f"   Frame {i+1} processed")
        await asyncio.sleep(0.5)
    
    # Get intermediate results
    print("3. Getting intermediate results...")
    await asyncio.sleep(1)
    
    # Stop acquisition
    print("4. Stopping acquisition...")
    await asyncio.sleep(1)
    
    # Finalize results
    print("5. Finalizing results...")
    await asyncio.sleep(1)
    
    print("Async workflow completed.\n")


def main():
    """
    Main function demonstrating all examples
    """
    print("STORM Controller Renewal Examples")
    print("=" * 50)
    print()
    
    # Run synchronous examples
    example_basic_local_processing()
    example_parameter_validation()
    example_enhanced_data_storage()
    example_rest_api_usage()
    example_signal_updates()
    
    # Run asynchronous example
    print("Running async example...")
    asyncio.run(example_async_workflow())
    
    print("All examples completed!")
    print("\nKey improvements in the renewed STORM controller:")
    print("✓ Removed widget dependencies")
    print("✓ Enhanced microeye integration with pydantic parameters")
    print("✓ Improved local data storage following experimentcontroller pattern")
    print("✓ Modern async processing without Qt threading")
    print("✓ REST API endpoints for all functionality")
    print("✓ Signal updates for frontend consumption")
    print("✓ Backward compatibility with legacy API")


if __name__ == '__main__':
    main()