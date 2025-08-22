#!/usr/bin/env python3
"""
Complete image display integration test and example

This example shows:
1. How to send images to ESP32 display
2. How to set up automatic image capture on snap button press
3. Complete workflow from image generation to display
"""

import numpy as np
import time
import json
from uc2_serial_controller import UC2SerialController

class MockESP32:
    """Mock ESP32 for testing the image display functionality"""
    
    def __init__(self):
        self.received_messages = []
        self.simulated_responses = []
    
    def write(self, data):
        """Mock serial write method"""
        message = data.decode('utf-8').strip()
        self.received_messages.append(message)
    
    def flush(self):
        """Mock serial flush method"""
        pass
    
    def add_response(self, message_type, response_data):
        """Add a simulated response for a specific message type"""
        self.simulated_responses.append({
            "trigger": message_type,
            "response": response_data
        })
    
    def process_message(self, message):
        """Process incoming message and return appropriate response"""
        self.received_messages.append(message)
        
        try:
            msg_dict = json.loads(message)
            msg_type = msg_dict.get("type")
            
            # Find matching response
            for resp in self.simulated_responses:
                if resp["trigger"] == msg_type:
                    return json.dumps(resp["response"])
            
            # Default responses
            if msg_type == "status_request":
                return json.dumps({
                    "type": "status_update",
                    "data": {"connected": True, "modules": {"display": True}}
                })
            elif msg_type == "display_image_command":
                return json.dumps({
                    "type": "image_display_result", 
                    "data": {
                        "tab_name": msg_dict["data"]["tab_name"],
                        "width": msg_dict["data"]["width"], 
                        "height": msg_dict["data"]["height"],
                        "success": True
                    }
                })
        except:
            pass
        
        return None

def create_test_images():
    """Create various test images for demonstration"""
    images = {}
    
    # 1. Simple solid color image
    images["red_square"] = np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)
    
    # 2. Gradient image
    gradient = np.zeros((80, 120, 3), dtype=np.uint8)
    for y in range(80):
        for x in range(120):
            gradient[y, x] = [
                int(255 * x / 120),    # Red gradient horizontally
                int(255 * y / 80),     # Green gradient vertically  
                128                     # Constant blue
            ]
    images["gradient"] = gradient
    
    # 3. Checkerboard pattern
    checker = np.zeros((60, 80, 3), dtype=np.uint8)
    for y in range(60):
        for x in range(80):
            if (x // 10 + y // 10) % 2:
                checker[y, x] = [255, 255, 255]  # White
            else:
                checker[y, x] = [0, 0, 0]        # Black
    images["checkerboard"] = checker
    
    # 4. Simple synthetic microscopy image
    microscopy = np.zeros((100, 100, 3), dtype=np.uint8)
    center = 50
    for y in range(100):
        for x in range(100):
            # Create concentric circles
            dist = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
            intensity = max(0, 255 - int(dist * 3))
            if dist < 40:
                microscopy[y, x] = [intensity, intensity // 2, 0]  # Orange-ish
    images["microscopy"] = microscopy
    
    return images

def main():
    print("🔬 ESP32 Image Display Integration Example")
    print("=" * 50)
    
    # Create test images
    test_images = create_test_images()
    print(f"Created {len(test_images)} test images")
    
    # Create controller with mock ESP32 for testing
    controller = UC2SerialController()
    mock_esp32 = MockESP32()
    
    # Set up mock responses
    mock_esp32.add_response("display_image_command", {
        "type": "image_display_result",
        "data": {"success": True, "message": "Image displayed successfully"}
    })
    
    print("\n📡 Simulating connection to ESP32...")
    # In real usage, you would call: controller.connect()
    # For this demo, we'll simulate the connection
    controller.is_connected = True
    controller.serial_conn = mock_esp32  # Mock serial connection
    
    # Demo 1: Send individual images
    print("\n🖼️  Demo 1: Sending individual images...")
    for name, image in test_images.items():
        print(f"  Sending {name} image ({image.shape[1]}x{image.shape[0]})")
        
        # This would normally send to real ESP32
        success = controller.send_image(image, name.replace("_", " ").title())
        if success:
            print(f"  ✅ {name} sent successfully")
            
            # Simulate ESP32 processing time
            time.sleep(0.5)
        else:
            print(f"  ❌ Failed to send {name}")
    
    # Demo 2: Set up automatic image capture
    print("\n📸 Demo 2: Setting up automatic image capture...")
    
    # Create a camera simulator
    camera_counter = 0
    def simulate_camera_capture():
        nonlocal camera_counter
        camera_counter += 1
        
        # Return different images in rotation
        image_names = list(test_images.keys())
        selected_name = image_names[camera_counter % len(image_names)]
        selected_image = test_images[selected_name]
        
        print(f"  📷 Camera captured: {selected_name}")
        return selected_image
    
    # Set up automatic capture on snap button press
    controller.send_image_on_snap(simulate_camera_capture)
    print("  ✅ Automatic capture configured")
    
    # Demo 3: Simulate snap button presses
    print("\n🎯 Demo 3: Simulating snap button presses...")
    for i in range(3):
        print(f"  Simulating snap button press {i+1}...")
        
        # This simulates what happens when user presses snap on ESP32 display
        controller._notify_callbacks('snap_image_command', {})
        
        time.sleep(1)
    
    # Demo 4: Show message statistics
    print("\n📊 Demo Results:")
    print(f"  Total mock ESP32 messages received: {len(mock_esp32.received_messages)}")
    for i, msg in enumerate(mock_esp32.received_messages[-3:], 1):  # Show last 3
        try:
            parsed = json.loads(msg)
            msg_type = parsed.get('type', 'unknown')
            print(f"  Message {i}: {msg_type}")
            if msg_type == 'display_image_command':
                data = parsed.get('data', {})
                print(f"    Tab: {data.get('tab_name', 'N/A')}")
                print(f"    Size: {data.get('width', 0)}x{data.get('height', 0)}")
                print(f"    Data length: {len(data.get('image_data', ''))}")
        except:
            print(f"  Message {i}: [parsing error]")
    
    # Demo 5: Test error handling
    print("\n🔧 Demo 5: Testing error handling...")
    
    # Test with invalid image
    try:
        success = controller.send_image("nonexistent_file.jpg", "Error Test")
        print(f"  Invalid file handling: {'✅ Handled gracefully' if not success else '❌ Should have failed'}")
    except Exception as e:
        print(f"  Exception handling: ❌ {e}")
    
    # Test with None image
    def return_none():
        return None
    
    controller.send_image_on_snap(return_none)
    controller._notify_callbacks('snap_image_command', {})
    print("  Null image handling: ✅ Handled gracefully")
    
    print("\n🎉 Integration test completed!")
    print("\nTo use with real ESP32:")
    print("1. Flash the updated firmware to your ESP32")
    print("2. Connect ESP32 via USB serial")
    print("3. Replace mock with: controller.connect()")
    print("4. Images will appear as new tabs on the ESP32 display!")

if __name__ == "__main__":
    main()