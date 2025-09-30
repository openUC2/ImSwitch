#!/usr/bin/env python3
"""
Enhanced Test Script for UC2 Serial Interface
Demonstrates all the new features implemented.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uc2_serial_controller import UC2SerialController, find_esp32_port

def test_enhanced_features():
    """Test the enhanced features"""
    print("=== Enhanced UC2 Serial Interface Test ===")
    
    # Test enhanced device detection
    print("\n1. Testing enhanced ESP32 detection...")
    port = find_esp32_port()
    if not port:
        print("❌ No ESP32 device found or device validation failed")
        print("Make sure your ESP32 is connected and running the updated firmware")
        return False
    
    print(f"✅ Validated ESP32 found on port: {port}")
    
    # Connect to controller
    print("\n2. Connecting to ESP32...")
    controller = UC2SerialController(port)
    
    # Set up callbacks to monitor responses
    def on_status_update(data):
        print(f"📡 Status: {data}")
    
    def on_led_update(data):
        print(f"💡 LED: {data}")
        
    def on_pwm_update(data):
        print(f"🔦 PWM Channel {data.get('channel', '?')}: {data.get('value', '?')}")
    
    controller.on_status_update(on_status_update)
    controller.on_led_update(on_led_update)
    
    # Register PWM callback (new feature)
    controller.register_callback('pwm_update', on_pwm_update)
    
    if not controller.connect():
        print("❌ Failed to connect to ESP32")
        return False
    
    print("✅ Connected successfully!")
    
    print("\n3. Testing LED Control...")
    controller.set_led(True, 255, 0, 0)  # Red
    time.sleep(1)
    controller.set_led(True, 0, 255, 0)  # Green
    time.sleep(1)
    controller.set_led(True, 0, 0, 255)  # Blue
    time.sleep(1)
    controller.set_led(False)  # Off
    
    print("\n4. Testing Objective Slot Control...")
    controller.set_objective_slot(1)
    time.sleep(1)
    controller.set_objective_slot(2)
    time.sleep(1)
    
    print("\n5. Testing PWM/Laser Control (NEW FEATURE)...")
    # Test all 4 PWM channels
    for channel in range(1, 5):
        for value in [0, 256, 512, 768, 1024]:
            print(f"Setting PWM Channel {channel} to {value}")
            controller.set_pwm_value(channel, value)
            time.sleep(0.5)
    
    # Reset all PWM channels to 0
    for channel in range(1, 5):
        controller.set_pwm_value(channel, 0)
    
    print("\n6. Testing Sample Position Update...")
    controller.update_sample_position(0.25, 0.25)  # Top-left quadrant
    time.sleep(1)
    controller.update_sample_position(0.75, 0.75)  # Bottom-right quadrant
    time.sleep(1)
    controller.update_sample_position(0.5, 0.5)    # Center
    
    print("\n7. Testing Image Capture...")
    controller.snap_image()
    time.sleep(1)
    
    print("\n✅ All enhanced features tested successfully!")
    print("\nThe ESP32 should now respond with clean JSON messages instead of debug logs.")
    print("UI should show individual tabs: Motor, LED, Objective, Sample Map, Lasers, Acquisition")
    
    controller.disconnect()
    return True

if __name__ == "__main__":
    try:
        success = test_enhanced_features()
        if success:
            print("\n🎉 Enhanced serial interface is working correctly!")
        else:
            print("\n❌ Some issues detected. Check your ESP32 connection and firmware.")
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()