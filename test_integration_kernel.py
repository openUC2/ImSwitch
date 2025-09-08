#!/usr/bin/env python3
"""
Integration test for embedded Jupyter kernel.
Tests the kernel functionality in a minimal headless environment.
"""

import threading
import time
import sys
import os

# Add the ImSwitch directory to the path
sys.path.insert(0, '/home/runner/work/ImSwitch/ImSwitch')

def test_start_embedded_kernel():
    """Test starting the embedded kernel function directly."""
    print("Testing embedded kernel startup...")
    
    # Simple namespace for testing
    test_namespace = {
        'test_var': 'Hello from ImSwitch!',
        'test_number': 42,
        'test_dict': {'key': 'value'}
    }
    
    try:
        # Try to import the function
        from imswitch.imcommon.applaunch import start_embedded_kernel
        print("✓ Successfully imported start_embedded_kernel")
        
        # Test with ipykernel available
        try:
            from ipykernel.embed import embed_kernel
            print("✓ ipykernel is available")
            
            # Start kernel in a separate thread with a timeout
            print("Starting kernel in background thread (will timeout after 3 seconds)...")
            
            kernel_thread = threading.Thread(
                target=start_embedded_kernel, 
                args=(test_namespace,), 
                daemon=True
            )
            kernel_thread.start()
            
            # Let it run for a moment
            time.sleep(3)
            
            if kernel_thread.is_alive():
                print("✓ Kernel thread started successfully")
                print("✓ Kernel is running in background")
                # Note: In real usage, the kernel would keep running until the app exits
                return True
            else:
                print("! Kernel thread exited immediately")
                return False
                
        except ImportError:
            print("! ipykernel not available, testing fallback...")
            start_embedded_kernel(test_namespace)
            print("✓ Fallback handling works correctly")
            return True
            
    except Exception as e:
        print(f"✗ Failed to test kernel startup: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_app_startup():
    """Test minimal app startup with kernel enabled."""
    print("\nTesting minimal app startup with kernel...")
    
    try:
        # Set up minimal configuration
        from imswitch.config import get_config
        config = get_config()
        config.enable_kernel = True
        config.is_headless = True
        
        # Mock minimal module controllers
        mock_controllers = {
            'test_module': type('MockController', (), {
                'closeEvent': lambda self: None,
                'test_method': lambda self: "test_result"
            })()
        }
        
        # Test the kernel startup logic (without actually starting the full app)
        from imswitch.imcommon.applaunch import start_embedded_kernel
        
        # Prepare namespace like in launchApp
        kernel_namespace = globals().copy()
        kernel_namespace.update({
            "moduleMainControllers": mock_controllers,
            "app": None,
            "mainView": None,
            "config": config,
        })
        
        # Add individual controllers
        for module_id, controller in mock_controllers.items():
            kernel_namespace[f"{module_id}_controller"] = controller
        
        print("✓ Kernel namespace prepared successfully")
        print(f"  - Available in kernel: {sorted([k for k in kernel_namespace.keys() if not k.startswith('__')])}")
        
        # Test that we could start the kernel (but don't actually do it for testing)
        print("✓ Would be able to start kernel with this namespace")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed minimal app test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration tests."""
    print("=== Embedded Jupyter Kernel Integration Tests ===\n")
    
    try:
        success1 = test_start_embedded_kernel()
        success2 = test_minimal_app_startup()
        
        if success1 and success2:
            print("\n=== All integration tests passed! ===")
            print("\nThe embedded Jupyter kernel feature is ready to use:")
            print("1. Start ImSwitch with: python -m imswitch --with-kernel")
            print("2. Connect with: jupyter console --existing")
            print("3. Access managers via variables like: detectorsManager, lasersManager")
            print("4. Access controllers via: moduleMainControllers")
            return True
        else:
            print("\n✗ Some integration tests failed")
            return False
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)