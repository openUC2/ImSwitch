#!/usr/bin/env python3
"""
Integration test for ESP32 image display protocol
Tests the complete workflow from Python image conversion to base64 encoding
that gets sent to ESP32 via the display_image_command message.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json
import base64
import numpy as np
from unittest.mock import Mock

# Mock the uc2_serial_controller for testing without hardware
class MockUC2SerialController:
    def __init__(self):
        self.is_connected = False
        self.sent_messages = []
        
    def connect(self, port=None, baudrate=115200):
        self.is_connected = True
        return True
    
    def disconnect(self):
        self.is_connected = False
    
    def _send_message(self, message):
        """Mock message sender that captures sent messages"""
        self.sent_messages.append(message)
        return True
    
    def _convert_to_rgb565(self, image_rgb: np.ndarray) -> bytes:
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
    
    def send_image(self, image_data, tab_name="Image"):
        """Send image to ESP32 display as new tab"""
        try:
            # Convert numpy array to RGB format if needed
            if isinstance(image_data, np.ndarray):
                if len(image_data.shape) == 2:  # Grayscale
                    image_array = np.stack([image_data] * 3, axis=2)
                elif len(image_data.shape) == 3 and image_data.shape[2] == 4:  # RGBA
                    image_array = image_data[:, :, :3]  # Remove alpha channel
                else:
                    image_array = image_data
            else:
                raise ValueError("Unsupported image format")
            
            # Validate image dimensions
            height, width = image_array.shape[:2]
            max_pixels = 240 * 160  # ESP32 memory constraint
            if width * height > max_pixels:
                # Resize image to fit constraints
                ratio = (max_pixels / (width * height)) ** 0.5
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                # Simulate resize by creating smaller image
                image_array = image_array[:new_height, :new_width]
                print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Convert RGB888 to RGB565 format for ESP32
            rgb565_data = self._convert_to_rgb565(image_array)
            
            # Encode as base64 for JSON transmission
            image_b64 = base64.b64encode(rgb565_data).decode('utf-8')
            
            # Truncate tab name if too long
            tab_name = tab_name[:15]
            
            # Send image display command
            return self._send_message({
                "type": "display_image_command",
                "data": {
                    "tab_name": tab_name,
                    "width": image_array.shape[1],
                    "height": image_array.shape[0],
                    "format": "rgb565",
                    "image_data": image_b64
                }
            })
            
        except Exception as e:
            print(f"Failed to send image: {e}")
            return False


def test_image_display_protocol():
    """Test the complete image display protocol"""
    print("Testing ESP32 image display protocol...")
    
    # Create mock controller
    controller = MockUC2SerialController()
    controller.connect()
    
    # Create a simple test image (red square)
    test_image = np.zeros((50, 50, 3), dtype=np.uint8)
    test_image[:, :, 0] = 255  # Red channel
    
    print(f"Created test image: {test_image.shape}")
    
    # Send image
    success = controller.send_image(test_image, "Test Image")
    print(f"Image send result: {success}")
    
    # Verify the message was sent
    assert len(controller.sent_messages) == 1, "Should have sent exactly one message"
    
    message = controller.sent_messages[0]
    print(f"Message type: {message.get('type')}")
    
    # Verify message structure
    assert message["type"] == "display_image_command", "Should be display_image_command"
    assert "data" in message, "Should have data field"
    
    data = message["data"]
    required_fields = ["tab_name", "width", "height", "format", "image_data"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    print(f"Tab name: {data['tab_name']}")
    print(f"Dimensions: {data['width']}x{data['height']}")
    print(f"Format: {data['format']}")
    print(f"Base64 data length: {len(data['image_data'])}")
    
    # Verify image data can be decoded
    try:
        decoded_data = base64.b64decode(data['image_data'])
        expected_size = data['width'] * data['height'] * 2  # RGB565 = 2 bytes per pixel
        print(f"Decoded data size: {len(decoded_data)} bytes (expected: {expected_size})")
        assert len(decoded_data) == expected_size, "Decoded size should match expected RGB565 size"
    except Exception as e:
        raise AssertionError(f"Failed to decode base64 image data: {e}")
    
    print("✅ All tests passed!")
    return True


def test_large_image_resize():
    """Test that large images are properly resized"""
    print("\nTesting large image resize...")
    
    controller = MockUC2SerialController()
    controller.connect()
    
    # Create a large test image (larger than 240x160)
    large_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    print(f"Created large test image: {large_image.shape}")
    
    success = controller.send_image(large_image, "Large Image")
    assert success, "Should successfully send large image"
    
    message = controller.sent_messages[0]
    data = message["data"]
    
    # Verify image was resized to fit constraints
    total_pixels = data['width'] * data['height']
    max_pixels = 240 * 160
    print(f"Final image pixels: {total_pixels} (max allowed: {max_pixels})")
    assert total_pixels <= max_pixels, "Image should be resized to fit constraints"
    
    print("✅ Large image resize test passed!")


if __name__ == "__main__":
    try:
        test_image_display_protocol()
        test_large_image_resize()
        print("\n🎉 All ESP32 image display protocol tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)