#!/usr/bin/env python3
"""
Example demonstrating base64 image transmission to ESP32
This example shows how images are encoded and sent to the ESP32 for display.
"""

import numpy as np
import base64
import json
import sys
import os

# Add the current directory to Python path to import our module
sys.path.insert(0, os.path.dirname(__file__))

def create_test_pattern():
    """Create a simple test pattern image"""
    # Create a 100x100 RGB image with a color gradient
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    for y in range(100):
        for x in range(100):
            # Create a gradient pattern
            image[y, x, 0] = min(255, x * 2.55)     # Red gradient
            image[y, x, 1] = min(255, y * 2.55)     # Green gradient  
            image[y, x, 2] = min(255, (x+y) * 1.28) # Blue gradient
    
    return image

def convert_to_rgb565(image_rgb: np.ndarray) -> bytes:
    """Convert RGB888 image to RGB565 format for ESP32 display"""
    height, width = image_rgb.shape[:2]
    rgb565_data = []
    
    for y in range(height):
        for x in range(width):
            # Convert to int to avoid numpy uint8 overflow issues
            r, g, b = int(image_rgb[y, x, 0]), int(image_rgb[y, x, 1]), int(image_rgb[y, x, 2])
            
            # Convert 8-bit RGB to 5-6-5 bit RGB565
            r5 = (r >> 3) & 0x1F  # 5 bits
            g6 = (g >> 2) & 0x3F  # 6 bits  
            b5 = (b >> 3) & 0x1F  # 5 bits
            
            # Pack into 16-bit RGB565 format (little endian for ESP32)
            rgb565 = (r5 << 11) | (g6 << 5) | b5
            rgb565_data.append(rgb565 & 0xFF)        # Low byte first
            rgb565_data.append((rgb565 >> 8) & 0xFF) # High byte
    
    return bytes(rgb565_data)

def main():
    print("ESP32 Image Display Base64 Encoding Example")
    print("=" * 50)
    
    # Create test image
    print("1. Creating test pattern image...")
    image = create_test_pattern()
    print(f"   Image shape: {image.shape}")
    print(f"   Image dtype: {image.dtype}")
    
    # Convert to RGB565
    print("\n2. Converting RGB888 to RGB565 format...")
    rgb565_data = convert_to_rgb565(image)
    print(f"   RGB565 data size: {len(rgb565_data)} bytes")
    print(f"   Expected size: {image.shape[0] * image.shape[1] * 2} bytes")
    
    # Base64 encode
    print("\n3. Base64 encoding for JSON transmission...")
    image_b64 = base64.b64encode(rgb565_data).decode('utf-8')
    print(f"   Base64 encoded length: {len(image_b64)} characters")
    print(f"   First 100 characters: {image_b64[:100]}...")
    
    # Create the command message that would be sent to ESP32
    print("\n4. Creating display_image_command message...")
    command_message = {
        "type": "display_image_command",
        "data": {
            "tab_name": "Test Pattern",
            "width": image.shape[1],
            "height": image.shape[0],
            "format": "rgb565",
            "image_data": image_b64
        }
    }
    
    # Convert to JSON
    json_message = json.dumps(command_message)
    print(f"   JSON message size: {len(json_message)} bytes")
    print(f"   Required buffer size: {len(json_message) + 1024} bytes (with safety margin)")
    
    # Verify decoding works
    print("\n5. Verifying base64 decoding...")
    try:
        decoded_data = base64.b64decode(image_b64)
        print(f"   Successfully decoded {len(decoded_data)} bytes")
        print(f"   Matches original: {decoded_data == rgb565_data}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return 1
    
    print("\n6. ESP32 Firmware Processing Summary:")
    print("   ✓ SerialApi.cpp receives JSON with 16KB buffer")
    print("   ✓ Calls uc2ui_controller::displayImage()")  
    print("   ✓ Base64 decodes the image_data field")
    print("   ✓ Creates LVGL image descriptor with RGB565 format")
    print("   ✓ Displays image in new tab with close button")
    print("   ✓ Memory allocated in SPIRAM if available")
    
    print(f"\n🎉 Example complete! The ESP32 should receive and display a {image.shape[1]}x{image.shape[0]} test pattern.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())