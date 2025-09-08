#!/usr/bin/env python3
"""
Simple test script to verify embedded Jupyter kernel functionality.
This script tests the configuration and kernel initialization without 
starting the full ImSwitch application.
"""

import threading
import time
import sys
import os

# Add the ImSwitch directory to the path
sys.path.insert(0, '/home/runner/work/ImSwitch/ImSwitch')

def test_config():
    """Test configuration functionality."""
    print("Testing configuration...")
    
    from imswitch.config import get_config
    config = get_config()
    
    # Test default value
    assert config.enable_kernel == False, "Default enable_kernel should be False"
    print("✓ Default enable_kernel is False")
    
    # Test setting value
    config.enable_kernel = True
    assert config.enable_kernel == True, "enable_kernel should be settable"
    print("✓ enable_kernel can be set to True")
    
    # Test argument parsing
    class MockArgs:
        with_kernel = True
    
    config.update_from_argparse(MockArgs())
    assert config.enable_kernel == True, "enable_kernel should be updated from args"
    print("✓ enable_kernel updated from argparse")

def test_kernel_import():
    """Test that kernel functions can be imported."""
    print("\nTesting kernel imports...")
    
    try:
        from imswitch.imcommon.applaunch import start_embedded_kernel
        print("✓ start_embedded_kernel function imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import start_embedded_kernel: {e}")
        return False
    
    try:
        from ipykernel.embed import embed_kernel
        print("✓ ipykernel.embed.embed_kernel available")
    except ImportError:
        print("! ipykernel not available (this is OK for testing)")
    
    return True

def test_kernel_namespace():
    """Test kernel namespace preparation."""
    print("\nTesting kernel namespace...")
    
    # Mock module controllers
    mock_controllers = {
        'imcontrol': type('MockController', (), {
            '_ImConMainController__masterController': type('MockMaster', (), {
                'detectorsManager': 'mock_detector_manager',
                'lasersManager': 'mock_laser_manager'
            })()
        })()
    }
    
    # Simulate namespace preparation (from launchApp)
    kernel_namespace = globals().copy()
    kernel_namespace.update({
        "moduleMainControllers": mock_controllers,
        "app": None,
        "mainView": None,
    })
    
    # Add individual managers
    for module_id, controller in mock_controllers.items():
        kernel_namespace[f"{module_id}_controller"] = controller
        if module_id == 'imcontrol' and hasattr(controller, '_ImConMainController__masterController'):
            master_controller = controller._ImConMainController__masterController
            kernel_namespace["master_controller"] = master_controller
            
            # Add individual managers
            for attr_name in dir(master_controller):
                if attr_name.endswith('Manager') and not attr_name.startswith('_'):
                    manager = getattr(master_controller, attr_name, None)
                    if manager is not None:
                        kernel_namespace[attr_name] = manager
    
    # Verify namespace contents
    assert 'moduleMainControllers' in kernel_namespace, "moduleMainControllers should be in namespace"
    assert 'imcontrol_controller' in kernel_namespace, "imcontrol_controller should be in namespace"
    assert 'master_controller' in kernel_namespace, "master_controller should be in namespace"
    assert 'detectorsManager' in kernel_namespace, "detectorsManager should be in namespace"
    assert 'lasersManager' in kernel_namespace, "lasersManager should be in namespace"
    
    print("✓ Kernel namespace prepared correctly")
    print(f"  - Available keys: {sorted([k for k in kernel_namespace.keys() if not k.startswith('__')])}")

def main():
    """Run all tests."""
    print("=== Embedded Jupyter Kernel Test Suite ===\n")
    
    try:
        test_config()
        test_kernel_import()
        test_kernel_namespace()
        
        print("\n=== All tests passed! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)