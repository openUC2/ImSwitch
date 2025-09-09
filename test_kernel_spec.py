#!/usr/bin/env python3
"""
Test script to debug ImSwitch kernel spec installation
"""

import os
import sys

def test_kernel_spec():
    """Test if the ImSwitch kernel spec can be installed correctly"""
    print("=== ImSwitch Kernel Spec Test ===")
    
    # Add ImSwitch to path
    imswitch_path = os.path.dirname(__file__)
    if imswitch_path not in sys.path:
        sys.path.insert(0, imswitch_path)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from imswitch.imcommon.kernel_state import get_embedded_kernel_connection_file, is_embedded_kernel_running
        from imswitch.imcommon.embedded_kernel_spec import ensure_kernel_spec_available, create_imswitch_kernel_spec
        print("   ✅ Imports successful")
        
        # Check for embedded kernel
        print("2. Checking for embedded kernel...")
        if is_embedded_kernel_running():
            connection_file = get_embedded_kernel_connection_file()
            print(f"   ✅ Embedded kernel found: {connection_file}")
            
            # Test kernel spec creation
            print("3. Testing kernel spec creation...")
            spec_dir = create_imswitch_kernel_spec()
            if spec_dir:
                print(f"   ✅ Kernel spec created: {spec_dir}")
                
                # Check if kernel.json was created properly
                kernel_json = os.path.join(spec_dir, 'kernel.json')
                if os.path.exists(kernel_json):
                    print(f"   ✅ kernel.json exists: {kernel_json}")
                    with open(kernel_json, 'r') as f:
                        import json
                        spec = json.load(f)
                        print(f"   ✅ Kernel spec: {spec}")
                else:
                    print(f"   ❌ kernel.json not found")
                    
                # Test installation
                print("4. Testing kernel spec installation...")
                success = ensure_kernel_spec_available()
                if success:
                    print("   ✅ Kernel spec installation successful")
                    
                    # Verify installation
                    try:
                        from jupyter_client.kernelspec import KernelSpecManager
                        ksm = KernelSpecManager()
                        specs = ksm.get_all_specs()
                        if 'imswitch_embedded' in specs:
                            print("   ✅ Kernel spec found in Jupyter")
                            print(f"   Spec details: {specs['imswitch_embedded']}")
                        else:
                            print("   ❌ Kernel spec not found in Jupyter")
                            print(f"   Available specs: {list(specs.keys())}")
                    except Exception as e:
                        print(f"   ❌ Error checking specs: {e}")
                else:
                    print("   ❌ Kernel spec installation failed")
            else:
                print("   ❌ Failed to create kernel spec")
        else:
            print("   ❌ No embedded kernel running")
            print("   Start ImSwitch with --with-kernel flag first")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_kernel_spec()