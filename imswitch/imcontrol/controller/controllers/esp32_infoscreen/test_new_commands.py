#!/usr/bin/env python3
"""
Test script specifically for new command message types from ESP32 display interactions.

This test verifies that the Python backend can handle all the new command-type messages
that are sent from the ESP32 when users interact with the display interface.
"""

import json
import time
from uc2_serial_controller import UC2SerialController

def test_command_message_handling():
    """Test that all command messages from the problem statement are handled correctly"""
    
    print("🧪 Testing new command message handling...")
    
    # Create controller (no connection needed for message testing)
    controller = UC2SerialController()
    
    # Track which callbacks were called
    callback_results = {
        'objective_slot_command': [],
        'motor_command': [],
        'motor_xy_command': [],
        'led_command': [],
        'pwm_command': [],
        'snap_image_command': []
    }
    
    # Register callbacks for all new command types
    def on_objective_slot_command(data):
        callback_results['objective_slot_command'].append(data)
        print(f"✅ Objective slot command: {data}")
    
    def on_motor_command(data):
        callback_results['motor_command'].append(data)
        print(f"✅ Motor command: {data}")
    
    def on_motor_xy_command(data):
        callback_results['motor_xy_command'].append(data)
        print(f"✅ Motor XY command: {data}")
    
    def on_led_command(data):
        callback_results['led_command'].append(data)
        print(f"✅ LED command: {data}")
    
    def on_pwm_command(data):
        callback_results['pwm_command'].append(data)
        print(f"✅ PWM command: {data}")
    
    def on_snap_image_command(data):
        callback_results['snap_image_command'].append(data)
        print(f"✅ Snap image command: {data}")
    
    # Register callbacks
    controller.on_objective_slot_command(on_objective_slot_command)
    controller.on_motor_command(on_motor_command)
    controller.on_motor_xy_command(on_motor_xy_command)
    controller.on_led_command(on_led_command)
    controller.on_pwm_command(on_pwm_command)
    controller.on_snap_image_command(on_snap_image_command)
    
    # Test messages from the problem statement
    test_messages = [
        '{"type":"objective_slot_command","data":{"slot":1}}',
        '{"type":"objective_slot_command","data":{"slot":2}}',
        '{"type":"motor_command","data":{"motor":0,"speed":2}}',
        '{"type":"motor_command","data":{"motor":0,"speed":4}}',
        '{"type":"motor_command","data":{"motor":0,"speed":50}}',
        '{"type":"motor_command","data":{"motor":0,"speed":100}}',
        '{"type":"motor_command","data":{"motor":0,"speed":0}}',
        '{"type":"motor_xy_command","data":{"speedX":190,"speedY":500}}',
        '{"type":"motor_xy_command","data":{"speedX":250,"speedY":470}}',
        '{"type":"motor_xy_command","data":{"speedX":510,"speedY":430}}',
        '{"type":"motor_xy_command","data":{"speedX":640,"speedY":440}}',
        '{"type":"motor_xy_command","data":{"speedX":740,"speedY":450}}',
        '{"type":"motor_xy_command","data":{"speedX":790,"speedY":460}}',
        '{"type":"motor_xy_command","data":{"speedX":0,"speedY":0}}',
        '{"type":"motor_step_update","data":{"motor":1,"steps":-1000}}',
        '{"type":"motor_step_update","data":{"motor":1,"steps":-10}}',
        '{"type":"motor_step_update","data":{"motor":1,"steps":10}}',
        '{"type":"motor_step_update","data":{"motor":1,"steps":1000}}',
        '{"type":"motor_step_update","data":{"motor":2,"steps":10}}',
        '{"type":"motor_step_update","data":{"motor":2,"steps":-10}}',
        '{"type":"motor_step_update","data":{"motor":2,"steps":-1000}}',
        '{"type":"motor_step_update","data":{"motor":3,"steps":-1000}}',
        '{"type":"motor_step_update","data":{"motor":3,"steps":-10}}',
        '{"type":"motor_step_update","data":{"motor":3,"steps":10}}',
        '{"type":"motor_step_update","data":{"motor":3,"steps":1000}}',
        '{"type":"led_command","data":{"r":76,"g":255,"b":255}}',
        '{"type":"led_command","data":{"r":76,"g":87,"b":255}}',
        '{"type":"led_command","data":{"r":76,"g":87,"b":110}}',
        '{"type":"led_command","data":{"enabled":true,"r":76,"g":87,"b":110}}',
        '{"type":"sample_map_click","data":{"pixel_x":414,"pixel_y":68,"sample_number":12}}',
        '{"type":"sample_map_click","data":{"pixel_x":415,"pixel_y":58,"sample_number":12}}',
        '{"type":"pwm_command","data":{"channel":1,"value":423}}',
        '{"type":"pwm_command","data":{"channel":2,"value":563}}',
        '{"type":"pwm_command","data":{"channel":3,"value":727}}',
        '{"type":"pwm_command","data":{"channel":4,"value":847}}',
        '{"type":"snap_image_command"}'
    ]
    
    print(f"\n📨 Processing {len(test_messages)} command messages...")
    
    for i, message in enumerate(test_messages):
        print(f"\nMessage {i+1}/{len(test_messages)}: {message}")
        controller._handle_message(message)
    
    # Verify all callback types were called
    print(f"\n📊 Callback Results Summary:")
    for event_type, results in callback_results.items():
        print(f"  {event_type}: {len(results)} calls")
        
    # Verify specific counts based on test messages
    expected_counts = {
        'objective_slot_command': 2,  # 2 slot commands
        'motor_command': 5,  # 5 motor commands (different speeds)
        'motor_xy_command': 7,  # 7 XY motor commands
        'led_command': 4,  # 4 LED commands
        'pwm_command': 4,  # 4 PWM commands
        'snap_image_command': 1  # 1 snap command
    }
    
    all_passed = True
    for event_type, expected_count in expected_counts.items():
        actual_count = len(callback_results[event_type])
        if actual_count == expected_count:
            print(f"✅ {event_type}: Expected {expected_count}, got {actual_count}")
        else:
            print(f"❌ {event_type}: Expected {expected_count}, got {actual_count}")
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All command message tests passed!")
        return True
    else:
        print(f"\n❌ Some command message tests failed!")
        return False

def test_existing_functionality_preserved():
    """Test that existing message types still work"""
    
    print(f"\n🔄 Testing that existing functionality is preserved...")
    
    controller = UC2SerialController()
    
    # Test existing message types
    existing_messages = [
        '{"type": "status_update", "data": {"connected": true, "modules": {"motor": true, "led": true}}}',
        '{"type": "led_update", "data": {"enabled": true, "r": 255, "g": 0, "b": 0}}',
        '{"type": "motor_update", "data": {"positions": {"x": 1000, "y": 2000, "z": 500}}}',
        '{"type": "motor_step_update", "data": {"motor": 1, "steps": 100}}',
        '{"type": "pwm_update", "data": {"channel": 1, "value": 512}}',
        '{"type": "sample_map_click", "data": {"pixel_x": 414, "pixel_y": 68, "sample_number": 12}}',
        '{"type": "image_captured", "data": {}}',
    ]
    
    for message in existing_messages:
        print(f"Processing existing message: {message}")
        try:
            controller._handle_message(message)
            print("✅ Message processed successfully")
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            return False
    
    print("✅ All existing message types still work")
    return True

if __name__ == "__main__":
    print("🚀 Testing New ESP32 Command Messages")
    print("=" * 60)
    
    try:
        # Test new command message handling
        commands_passed = test_command_message_handling()
        
        # Test existing functionality still works
        existing_passed = test_existing_functionality_preserved()
        
        print("\n" + "=" * 60)
        if commands_passed and existing_passed:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Python backend now handles all ESP32 display command messages")
            print("✅ Existing functionality is preserved")
        else:
            print("❌ SOME TESTS FAILED!")
            exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)