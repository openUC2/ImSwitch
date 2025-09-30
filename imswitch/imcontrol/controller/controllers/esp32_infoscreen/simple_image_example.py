#!/usr/bin/env python3
"""
Simple example of sending images to ESP32 display

This shows the basic usage patterns for the new image display feature.
"""

import numpy as np
from uc2_serial_controller import UC2SerialController

def main():
    print("ESP32 Image Display - Simple Example")
    print("=" * 40)
    
    # Connect to ESP32
    controller = UC2SerialController()
    
    try:
        if not controller.connect():
            print("❌ Failed to connect to ESP32!")
            print("Make sure ESP32 is connected and has the updated firmware.")
            return
        
        print("✅ Connected to ESP32!")
        
        # Example 1: Send a simple test pattern
        print("\n📸 Sending a test pattern...")
        
        # Create a simple red/blue gradient
        test_image = np.zeros((80, 120, 3), dtype=np.uint8)
        for y in range(80):
            for x in range(120):
                test_image[y, x] = [
                    255 * x // 120,  # Red increases left to right
                    0,               # No green
                    255 * y // 80    # Blue increases top to bottom
                ]
        
        success = controller.send_image(test_image, "Test Pattern")
        if success:
            print("✅ Test pattern sent! Check your ESP32 display.")
        else:
            print("❌ Failed to send test pattern.")
        
        # Example 2: Set up automatic image capture
        print("\n🎯 Setting up automatic capture...")
        
        def capture_function():
            """This function will be called when snap button is pressed"""
            print("📷 Snap button pressed! Generating new image...")
            
            # Generate a random pattern
            import random
            pattern = np.random.randint(0, 256, (60, 80, 3), dtype=np.uint8)
            
            # Add some structure - create a circle
            center_x, center_y = 40, 30
            for y in range(60):
                for x in range(80):
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if dist < 20:
                        intensity = int(255 * (1 - dist / 20))
                        pattern[y, x] = [intensity, intensity, 0]  # Yellow circle
            
            return pattern
        
        controller.send_image_on_snap(capture_function)
        print("✅ Auto-capture configured!")
        print("Now press the 'CAPTURE IMAGE' button on your ESP32 display.")
        print("A new image tab should appear each time you press it!")
        
        # Keep the program running to handle snap button presses
        print("\n⌚ Listening for snap button presses...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        controller.disconnect()
        print("🔌 Disconnected from ESP32")

if __name__ == "__main__":
    main()