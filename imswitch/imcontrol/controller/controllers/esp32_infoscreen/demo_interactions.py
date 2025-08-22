#!/usr/bin/env python3
"""
Demonstration script showing the new ESP32 command message handling.

This script simulates the messages that would be received when users interact
with the ESP32 display interface, demonstrating how the Python backend handles
all the different types of user interactions.
"""

import time
import threading
from uc2_serial_controller import UC2SerialController

def demo_display_interactions():
    """Demonstrate handling of ESP32 display interactions"""
    
    print("🚀 ESP32 Display Interaction Demo")
    print("=" * 50)
    print("This demo simulates messages sent from the ESP32 when users")
    print("interact with buttons, sliders, and other UI elements on the display.\n")
    
    # Create controller (no actual serial connection needed for demo)
    controller = UC2SerialController()
    
    # Set up event tracking
    events_received = []
    
    # Register callbacks for all command events
    def on_objective_slot_command(data):
        slot = data.get('slot', 1)
        events_received.append(f"🎯 Objective slot {slot} selected")
        print(f"🎯 User selected objective slot {slot} on display")
    
    def on_motor_command(data):
        motor = data.get('motor', 0)
        speed = data.get('speed', 0)
        events_received.append(f"🔧 Motor {motor} speed set to {speed}")
        print(f"🔧 User set motor {motor} to speed {speed} on display")
    
    def on_motor_xy_command(data):
        speed_x = data.get('speedX', 0)
        speed_y = data.get('speedY', 0)
        events_received.append(f"🕹️  XY joystick: X={speed_x}, Y={speed_y}")
        print(f"🕹️  User moved XY joystick: X={speed_x}, Y={speed_y}")
    
    def on_led_command(data):
        enabled = data.get('enabled', False)
        r = data.get('r', 0)
        g = data.get('g', 0)
        b = data.get('b', 0)
        events_received.append(f"🌈 LED: enabled={enabled}, RGB=({r},{g},{b})")
        print(f"🌈 User changed LED: enabled={enabled}, RGB=({r}, {g}, {b})")
    
    def on_pwm_command(data):
        channel = data.get('channel', 0)
        value = data.get('value', 0)
        events_received.append(f"⚡ PWM channel {channel} set to {value}")
        print(f"⚡ User set PWM channel {channel} to {value} on display")
    
    def on_snap_image_command(data):
        events_received.append("📸 Snap image button pressed")
        print("📸 User pressed snap image button on display!")
    
    # Register all the callbacks
    controller.on_objective_slot_command(on_objective_slot_command)
    controller.on_motor_command(on_motor_command) 
    controller.on_motor_xy_command(on_motor_xy_command)
    controller.on_led_command(on_led_command)
    controller.on_pwm_command(on_pwm_command)
    controller.on_snap_image_command(on_snap_image_command)
    
    # Simulate user interactions with realistic pauses
    simulated_interactions = [
        ('{"type":"objective_slot_command","data":{"slot":1}}', "User switches to objective slot 1"),
        ('{"type":"led_command","data":{"enabled":true,"r":255,"g":0,"b":0}}', "User enables red LED"),
        ('{"type":"motor_command","data":{"motor":0,"speed":50}}', "User adjusts motor 0 speed to 50"),
        ('{"type":"motor_xy_command","data":{"speedX":300,"speedY":200}}', "User moves XY joystick"),
        ('{"type":"pwm_command","data":{"channel":1,"value":512}}', "User sets PWM channel 1 to middle"),
        ('{"type":"led_command","data":{"r":0,"g":255,"b":0}}', "User changes LED to green"),
        ('{"type":"motor_xy_command","data":{"speedX":0,"speedY":0}}', "User centers XY joystick"),
        ('{"type":"objective_slot_command","data":{"slot":2}}', "User switches to objective slot 2"),
        ('{"type":"snap_image_command"}', "User presses snap image button"),
        ('{"type":"led_command","data":{"enabled":false}}', "User turns off LED"),
    ]
    
    print("📱 Simulating user interactions with ESP32 display...\n")
    
    for i, (message, description) in enumerate(simulated_interactions):
        print(f"[Interaction {i+1}/{len(simulated_interactions)}] {description}")
        controller._handle_message(message)
        print() # Empty line for readability
        time.sleep(0.8)  # Realistic pause between interactions
    
    print("=" * 50)
    print(f"✅ Demo completed! Processed {len(events_received)} user interactions:")
    for event in events_received:
        print(f"  • {event}")
    
    print(f"\n🎉 The Python backend successfully handled all ESP32 display interactions!")
    print(f"💡 In a real application, you would use these callbacks to:")
    print(f"   • Log user actions for analysis")
    print(f"   • Trigger automated sequences")
    print(f"   • Update other connected systems")
    print(f"   • Provide feedback to users")

if __name__ == "__main__":
    demo_display_interactions()