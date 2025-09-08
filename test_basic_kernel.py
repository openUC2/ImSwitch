#!/usr/bin/env python3
"""
Simple test script to verify basic embedded Jupyter kernel functionality.
This script tests only the core functionality without framework dependencies.
"""

import sys
import os

# Add the ImSwitch directory to the path
sys.path.insert(0, '/home/runner/work/ImSwitch/ImSwitch')

def test_basic_config():
    """Test basic configuration functionality."""
    print("Testing basic configuration...")
    
    from imswitch.config import ImSwitchConfig
    
    # Test default value
    config = ImSwitchConfig()
    assert config.enable_kernel == False, "Default enable_kernel should be False"
    print("✓ Default enable_kernel is False")
    
    # Test setting value
    config.enable_kernel = True
    assert config.enable_kernel == True, "enable_kernel should be settable"
    print("✓ enable_kernel can be set to True")
    
    # Test argument mapping
    config.update_from_args(enable_kernel=True)
    assert config.enable_kernel == True, "enable_kernel should be updated from args"
    print("✓ enable_kernel updated from kwargs")

def test_ipykernel_availability():
    """Test if ipykernel is available."""
    print("\nTesting ipykernel availability...")
    
    try:
        from ipykernel.embed import embed_kernel
        print("✓ ipykernel.embed.embed_kernel is available")
        return True
    except ImportError:
        print("! ipykernel not available - install with: pip install ipykernel")
        return False

def test_jupyter_availability():
    """Test if jupyter is available."""
    print("\nTesting jupyter availability...")
    
    try:
        import jupyter
        print("✓ jupyter is available")
        return True
    except ImportError:
        print("! jupyter not available")
        return False

def test_argparse_simulation():
    """Test command line argument parsing simulation."""
    print("\nTesting argparse simulation...")
    
    from imswitch.config import ImSwitchConfig
    
    # Simulate argparse namespace
    class MockArgs:
        def __init__(self):
            self.with_kernel = True
            self.headless = False
            self.http_port = 8001
    
    config = ImSwitchConfig()
    config.update_from_argparse(MockArgs())
    
    assert config.enable_kernel == True, "enable_kernel should be True from argparse"
    assert config.is_headless == False, "is_headless should be False from argparse"
    print("✓ Argparse simulation works correctly")

def main():
    """Run all tests."""
    print("=== Basic Embedded Jupyter Kernel Test Suite ===\n")
    
    try:
        test_basic_config()
        test_ipykernel_availability()
        test_jupyter_availability()
        test_argparse_simulation()
        
        print("\n=== All basic tests passed! ===")
        print("\nNext steps:")
        print("- The embedded kernel feature should work with --with-kernel")
        print("- To connect: jupyter console --existing")
        print("- Or select the kernel in JupyterLab")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)