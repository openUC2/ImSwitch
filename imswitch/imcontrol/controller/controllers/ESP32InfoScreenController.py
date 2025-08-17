import numpy as np
try:
    import NanoImagingPack as nip
    isNIP = True
except:
    isNIP = False
import time
import threading
import collections

from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex, Timer
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger
from ..basecontrollers import ImConWidgetController
from imswitch import IS_HEADLESS

import time
import logging

from .esp32_infoscreen.uc2_serial_controller import UC2SerialController, find_esp32_port

class ESP32InfoScreenController(ImConWidgetController):
    """ Linked to HoliSheetWidget."""

    sigImageReceived = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        detoctorName = allDetectorNames[0]
        self.detector = self._master.detectorsManager[detoctorName]
        
        # select stage 
        allPositionerNames = self._master.positionersManager.getAllDeviceNames()
        positionerName = allPositionerNames[0]
        self.positioner = self._master.positionersManager[positionerName]

        # get all Lasers
        self.lasers = self._master.lasersManager.getAllDeviceNames()
        self.laser = self.lasers[0]
        try:
            self._master.lasersManager[self.laser].setGalvo(channel=1, frequency=10, offset=0, amplitude=1, clk_div=0, phase=0, invert=1, timeout=1)
        except  Exception as e:
            self._logger.error(e)
            
        # get LEDMatrix # TODO: Improve
        allLEDMatrixNames = self._master.LEDMatrixsManager.getAllDeviceNames()
        if len(allLEDMatrixNames) == 0:
            if not IS_HEADLESS: self._widget.replaceWithError('No LEDMatrix found in your setup file.')
            return
        self.ledMatrix = self._master.LEDMatrixsManager[allLEDMatrixNames[0]]



        # assign signals 
        # TODO: We need to assign this objective to the button click on the remote display (on_objective_slot_command,on_objective_slot_update )
        # self._commChannel.sigToggleObjective.emit(1)
        
        # TODO: We need to assign the stage motion self.positioner.move to the button clicks (on_motor_xy_command, on_motor_command, on_sample_position_update

        # TODO: We need to assign the led intensity setting to the infoscreen buttons (on_led_update),                 self.ledMatrix.mLEDmatrix.setAll(0)

        # TODO: We need to assign the lasers to the pwm updates (on_pwm_update)


'''
def main():
    """Main example function"""
    
    # Find ESP32 automatically
    port = find_esp32_port()
    if not port:
        print("❌ No ESP32 device found. Please check connections.")
        print("Available ports:")
        import serial.tools.list_ports
        for port_info in serial.tools.list_ports.comports():
            print(f"  - {port_info.device}: {port_info.description}")
        return
    
    print(f"🔍 Found ESP32 on port: {port}")
    
    # Create controller
    controller = UC2SerialController(port)
    
    # Set up callback functions
    def on_status_update(data):
        print(f"📊 Status Update: {data}")
    
    def on_led_update(data):
        print(f"💡 LED Update: enabled={data.get('enabled')}, "
              f"RGB=({data.get('r', 0)}, {data.get('g', 0)}, {data.get('b', 0)})")
    
    def on_motor_update(data):
        positions = data.get('positions', {})
        print(f"🔧 Motor Update: X={positions.get('x', 0)}, "
              f"Y={positions.get('y', 0)}, Z={positions.get('z', 0)}")
    
    def on_pwm_update(data):
        channel = data.get('channel', 0)
        value = data.get('value', 0)
        print(f"🔌 PWM Update: Channel {channel}, Value {value}"
              )
    def on_objective_slot_update(data):
        slot = data.get('current_slot', 1)
        print(f"🔬 Objective Slot: {slot}")
    
    def on_sample_position_update(data):
        x, y = data.get('x', 0), data.get('y', 0)
        print(f"📍 Sample Position: ({x:.2f}, {y:.2f})")
    
    def on_image_captured(data):
        print("📸 Image captured!")
    
    def on_connection_changed(data):
        connected = data.get('connected', False)
        status = "✅ Connected" if connected else "❌ Disconnected"
        print(f"🔌 Connection: {status}")
    
    # New command event callbacks (ESP32 display interactions)
    def on_objective_slot_command(data):
        slot = data.get('slot', 1)
        print(f"🎯 User selected objective slot {slot} on display")
        
    def on_motor_command(data):
        motor = data.get('motor', 0)
        speed = data.get('speed', 0)
        print(f"🔧 User set motor {motor} to speed {speed} on display")
        
    def on_motor_xy_command(data):
        speed_x = data.get('speedX', 0)
        speed_y = data.get('speedY', 0)
        print(f"🕹️  User moved XY joystick: X={speed_x}, Y={speed_y}")
        
    def on_led_command(data):
        enabled = data.get('enabled', False)
        r = data.get('r', 0)
        g = data.get('g', 0) 
        b = data.get('b', 0)
        print(f"🌈 User changed LED on display: enabled={enabled}, RGB=({r}, {g}, {b})")
        
    def on_pwm_command(data):
        channel = data.get('channel', 0)
        value = data.get('value', 0)
        print(f"⚡ User set PWM channel {channel} to {value} on display")
        
    def on_snap_image_command(data):
        print(f"📸 User pressed snap button on display!")
    
    # Register all callbacks
    controller.on_status_update(on_status_update)
    controller.on_led_update(on_led_update)
    controller.on_motor_update(on_motor_update)
    controller.on_pwm_update(on_pwm_update)
    controller.on_objective_slot_update(on_objective_slot_update)
    controller.on_sample_position_update(on_sample_position_update)
    controller.on_image_captured(on_image_captured)
    controller.on_connection_changed(on_connection_changed)
    
    # Register new command event callbacks (ESP32 display interactions)
    controller.on_objective_slot_command(on_objective_slot_command)
    controller.on_motor_command(on_motor_command)
    controller.on_motor_xy_command(on_motor_xy_command)
    controller.on_led_command(on_led_command)
    controller.on_pwm_command(on_pwm_command)
    controller.on_snap_image_command(on_snap_image_command)
    
    # Connect to ESP32
    print("🔄 Connecting to ESP32...")
    if not controller.connect():
        print("❌ Failed to connect to ESP32")
        return
    
    print("✅ Connected successfully!")
    
    try:
        # Demonstrate various functions
        print("\n🚀 Starting demonstration sequence...")
        
        # 1. LED Control Demo
        print("\n💡 LED Control Demo:")
        print("Setting LED to Red...")
        controller.set_led(True, 255, 0, 0)
        time.sleep(2)
        
        print("Setting LED to Green...")
        controller.set_led(True, 0, 255, 0)
        time.sleep(2)
        
        print("Setting LED to Blue...")
        controller.set_led(True, 0, 0, 255)
        time.sleep(2)
        
        print("Turning LED off...")
        controller.set_led(False)
        time.sleep(1)
        
        # 2. Motor Control Demo
        print("\n🔧 Motor Control Demo:")
        print("Moving X motor forward...")
        controller.move_motor(1, 1000)  # Motor 1 (X), speed 1000
        time.sleep(2)
        
        print("Stopping X motor...")
        controller.move_motor(1, 0)  # Stop motor
        time.sleep(1)
        
        print("Moving XY motors diagonally...")
        controller.move_xy_motors(500, 500)
        time.sleep(2)
        
        print("Stopping XY motors...")
        controller.move_xy_motors(0, 0)
        time.sleep(1)
        
        # 3. Objective Slot Demo  
        print("\n🔬 Objective Slot Demo:")
        print("Switching to slot 1...")
        controller.set_objective_slot(1)
        time.sleep(2)
        
        print("Switching to slot 2...")
        controller.set_objective_slot(2)
        time.sleep(2)
        
        # 4. Sample Position Demo
        print("\n📍 Sample Position Demo:")
        positions = [
            (0.1, 0.1), (0.9, 0.1),  # Top corners
            (0.9, 0.9), (0.1, 0.9),  # Bottom corners
            (0.5, 0.5)               # Center
        ]
        
        for x, y in positions:
            print(f"Moving to position ({x}, {y})")
            controller.update_sample_position(x, y)
            time.sleep(1.5)
        
        # 5. Image Capture Demo
        print("\n📸 Image Capture Demo:")
        for i in range(3):
            print(f"Taking picture {i+1}/3...")
            controller.snap_image()
            time.sleep(1)
        
        print("\n✨ Demonstration complete!")
        print("Current device state:")
        print(f"  LED: {controller.led_status}")
        print(f"  Motors: {controller.motor_positions}")
        print(f"  Objective Slot: {controller.current_objective_slot}")
        print(f"  Sample Position: {controller.current_sample_position}")
        
        # Keep running to monitor updates
        print(f"\n👂 Monitoring for ESP32 display interactions and updates (Press Ctrl+C to exit)...")
        print("   Try pressing buttons or moving sliders on the ESP32 display to see command events!")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopping demonstration...")
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
    finally:
        # Clean shutdown
        print("🔌 Disconnecting...")
        controller.disconnect()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
    '''