#!/usr/bin/env python3
"""
Demo script showing how to send images to ESP32 display

This example demonstrates:
1. Sending a generated test image to the display
2. Automatically sending images when snap button is pressed
3. Different ways to provide image data (numpy array, file path)
"""

import numpy as np
import time
from uc2_serial_controller import UC2SerialController

def create_test_image(width=240, height=160):
    """Create a colorful test pattern image"""
    # Create gradient test pattern
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create rainbow gradient
    for y in range(height):
        for x in range(width):
            # Create a rainbow color based on position
            hue = (x / width) * 360  # Hue varies horizontally
            saturation = (y / height)  # Saturation varies vertically
            
            # Simple HSV to RGB conversion
            c = saturation
            h_prime = hue / 60
            x_val = c * (1 - abs((h_prime % 2) - 1))
            
            if h_prime < 1:
                r, g, b = c, x_val, 0
            elif h_prime < 2:
                r, g, b = x_val, c, 0
            elif h_prime < 3:
                r, g, b = 0, c, x_val
            elif h_prime < 4:
                r, g, b = 0, x_val, c
            elif h_prime < 5:
                r, g, b = x_val, 0, c
            else:
                r, g, b = c, 0, x_val
            
            image[y, x] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return image

def create_camera_simulator():
    """Simulate a camera that captures different images"""
    counter = 0
    
    def capture_frame():
        nonlocal counter
        counter += 1
        
        # Create different test patterns
        if counter % 3 == 1:
            # Red gradient
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            for y in range(120):
                for x in range(160):
                    image[y, x] = [int(255 * x / 160), 0, int(255 * y / 120)]
        elif counter % 3 == 2:
            # Green checker pattern
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            for y in range(120):
                for x in range(160):
                    if (x // 20 + y // 20) % 2:
                        image[y, x] = [0, 255, 0]
                    else:
                        image[y, x] = [0, 100, 0]
        else:
            # Blue circle
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            center_x, center_y = 80, 60
            radius = 40
            for y in range(120):
                for x in range(160):
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if dist < radius:
                        intensity = int(255 * (1 - dist / radius))
                        image[y, x] = [0, 0, intensity]
        
        print(f"Camera captured frame {counter}")
        return image
    
    return capture_frame

def main():
    print("ESP32 Image Display Demo")
    print("=" * 40)
    
    # Initialize controller
    controller = UC2SerialController()
    
    try:
        # Connect to ESP32
        print("Connecting to ESP32...")
        if not controller.connect():
            print("Failed to connect to ESP32!")
            return
        
        print("Connected successfully!")
        
        # Demo 1: Send a test image immediately
        print("\n1. Sending test rainbow image...")
        test_image = create_test_image()
        success = controller.send_image(test_image, "Rainbow Test")
        if success:
            print("✓ Test image sent successfully!")
        else:
            print("✗ Failed to send test image")
        
        time.sleep(2)
        
        # Demo 2: Send a smaller pattern
        print("\n2. Sending small checker pattern...")
        checker = np.zeros((80, 120, 3), dtype=np.uint8)
        for y in range(80):
            for x in range(120):
                if (x // 15 + y // 15) % 2:
                    checker[y, x] = [255, 255, 255]  # White
                else:
                    checker[y, x] = [255, 0, 255]    # Magenta
        
        success = controller.send_image(checker, "Checker")
        if success:
            print("✓ Checker pattern sent successfully!")
        
        # Demo 3: Setup automatic image capture on snap button press
        print("\n3. Setting up automatic image capture...")
        print("Now when you press the 'CAPTURE IMAGE' button on the display,")
        print("a new image will be automatically generated and displayed!")
        
        camera_sim = create_camera_simulator()
        controller.send_image_on_snap(camera_sim)
        
        # Demo 4: Manual snap button simulation
        print("\n4. Simulating snap button presses every 5 seconds...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(5)
                # Simulate what happens when snap button is pressed
                print("Simulating snap button press...")
                controller._notify_callbacks('snap_image_command', {})
                
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
    
    except Exception as e:
        print(f"Error during demo: {e}")
    
    finally:
        controller.disconnect()
        print("Disconnected from ESP32")

if __name__ == "__main__":
    main()