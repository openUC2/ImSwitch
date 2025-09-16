#!/usr/bin/env python3
"""
Direct test of the updated CameraTucsen class
"""

import sys
import os
import time

# Add the parent directory to path so we can import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import without going through ImSwitch framework
try:
    from imswitch.imcontrol.model.interfaces.tucsencamera import CameraTucsen
    print("CameraTucsen imported successfully")
except Exception as e:
    print(f"Failed to import CameraTucsen: {e}")
    sys.exit(1)

def test_camera_direct():
    """Test camera without full ImSwitch framework"""
    print("Testing CameraTucsen directly...")
    
    try:
        # Force cleanup first
        CameraTucsen.force_cleanup()
        
        # Initialize camera
        print("Creating camera instance...")
        camera = CameraTucsen(cameraNo=0)
        
        print(f"Camera connected: {camera.is_connected}")
        if not camera.is_connected:
            print("Camera not connected, exiting")
            return
            
        # Test streaming
        print("Starting live streaming...")
        camera.start_live()
        
        print("Waiting for frames...")
        time.sleep(5)  # Let it capture some frames
        
        # Check if we got any frames
        frame = camera.getLast(returnFrameNumber=False, timeout=2.0)
        if frame is not None:
            print(f"SUCCESS! Got frame with shape: {frame.shape}")
        else:
            print("NO FRAMES CAPTURED")
        
        # Stop streaming
        print("Stopping streaming...")
        camera.stop_live()
        
        # Close camera
        print("Closing camera...")
        camera.close()
        
        print("Test completed")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_camera_direct()