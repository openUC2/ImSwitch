#!/usr/bin/env python3
"""
Advanced OpenUC2 Serial Interface Demo

This script demonstrates advanced features of the UC2 Serial Controller including:
- Real-time monitoring of microscope state
- Interactive control via console commands
- Automated scanning patterns
- Data logging and analysis
"""

import time
import json
import threading
import logging
from datetime import datetime
from uc2_serial_controller import UC2SerialController, find_esp32_port

class UC2AdvancedDemo:
    """Advanced demonstration of UC2 Serial Controller features"""
    
    def __init__(self):
        self.controller = None
        self.running = False
        self.scan_active = False
        self.log_file = f"uc2_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.data_log = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('uc2_demo.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_callbacks(self):
        """Setup event callbacks for monitoring"""
        
        def on_status_update(data):
            self.logger.info(f"Status: {data}")
            self.log_event("status_update", data)
        
        def on_led_update(data):
            self.logger.info(f"LED changed: enabled={data.get('enabled')}, "
                           f"RGB=({data.get('r', 0)}, {data.get('g', 0)}, {data.get('b', 0)})")
            self.log_event("led_update", data)
        
        def on_motor_update(data):
            positions = data.get('positions', {})
            self.logger.info(f"Motors: X={positions.get('x', 0)}, "
                           f"Y={positions.get('y', 0)}, Z={positions.get('z', 0)}")
            self.log_event("motor_update", data)
        
        def on_objective_slot_update(data):
            slot = data.get('current_slot', 1)
            self.logger.info(f"Objective slot: {slot}")
            self.log_event("objective_slot_update", data)
        
        def on_sample_position_update(data):
            x, y = data.get('x', 0), data.get('y', 0)
            self.logger.info(f"Sample position: ({x:.2f}, {y:.2f})")
            self.log_event("sample_position_update", data)
        
        def on_image_captured(data):
            self.logger.info("📸 Image captured!")
            self.log_event("image_captured", data)
        
        def on_connection_changed(data):
            connected = data.get('connected', False)
            self.logger.info(f"Connection: {'Connected' if connected else 'Disconnected'}")
            self.log_event("connection_changed", data)
        
        # Register all callbacks
        self.controller.on_status_update(on_status_update)
        self.controller.on_led_update(on_led_update)
        self.controller.on_motor_update(on_motor_update)
        self.controller.on_objective_slot_update(on_objective_slot_update)
        self.controller.on_sample_position_update(on_sample_position_update)
        self.controller.on_image_captured(on_image_captured)
        self.controller.on_connection_changed(on_connection_changed)
    
    def log_event(self, event_type, data):
        """Log event to data file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.data_log.append(log_entry)
        
        # Save to file periodically
        if len(self.data_log) % 10 == 0:
            self.save_log()
    
    def save_log(self):
        """Save data log to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.data_log, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save log: {e}")
    
    def connect_to_microscope(self):
        """Connect to ESP32 microscope"""
        print("🔍 Searching for ESP32 microscope...")
        
        port = find_esp32_port()
        if not port:
            print("❌ No ESP32 found. Please check connections.")
            return False
        
        print(f"✅ Found ESP32 on {port}")
        
        self.controller = UC2SerialController(port)
        self.setup_callbacks()
        
        if self.controller.connect():
            print("🎯 Connected successfully!")
            return True
        else:
            print("❌ Failed to connect")
            return False
    
    def interactive_mode(self):
        """Interactive console control"""
        print("\n🎮 Interactive Control Mode")
        print("=" * 50)
        print("Commands:")
        print("  led <r> <g> <b>     - Set LED color (0-255)")
        print("  led off             - Turn LED off")
        print("  motor <id> <speed>  - Move motor (id: 1-4, speed: -10 to 10)")
        print("  xy <x> <y>          - Move XY stage (speed: -10 to 10)")
        print("  slot <1|2>          - Switch objective slot")
        print("  snap                - Take image")
        print("  scan                - Start grid scan")
        print("  stop                - Stop scan")
        print("  pos <x> <y>         - Set sample position (0.0-1.0)")
        print("  status              - Request status")
        print("  quit                - Exit")
        print("=" * 50)
        
        while self.running:
            try:
                cmd = input("\nUC2> ").strip().lower()
                if not cmd:
                    continue
                
                self.process_command(cmd)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def process_command(self, cmd):
        """Process interactive commands"""
        parts = cmd.split()
        command = parts[0]
        
        if command == "quit":
            self.running = False
        
        elif command == "led":
            if len(parts) == 2 and parts[1] == "off":
                self.controller.set_led(False)
                print("💡 LED turned off")
            elif len(parts) == 4:
                r, g, b = map(int, parts[1:4])
                self.controller.set_led(True, r, g, b)
                print(f"💡 LED set to RGB({r}, {g}, {b})")
            else:
                print("Usage: led <r> <g> <b> or led off")
        
        elif command == "motor":
            if len(parts) == 3:
                motor_id, speed = int(parts[1]), int(parts[2])
                self.controller.move_motor(motor_id, speed)
                print(f"🔧 Motor {motor_id} speed set to {speed}")
            else:
                print("Usage: motor <id> <speed>")
        
        elif command == "xy":
            if len(parts) == 3:
                x, y = int(parts[1]), int(parts[2])
                self.controller.move_xy_motors(x, y)
                print(f"↗️ XY motors: X={x}, Y={y}")
            else:
                print("Usage: xy <x_speed> <y_speed>")
        
        elif command == "slot":
            if len(parts) == 2:
                slot = int(parts[1])
                if slot in [1, 2]:
                    self.controller.set_objective_slot(slot)
                    print(f"🔬 Switched to objective slot {slot}")
                else:
                    print("Slot must be 1 or 2")
            else:
                print("Usage: slot <1|2>")
        
        elif command == "snap":
            self.controller.snap_image()
            print("📸 Image capture triggered")
        
        elif command == "scan":
            self.start_grid_scan()
        
        elif command == "stop":
            self.stop_scan()
        
        elif command == "pos":
            if len(parts) == 3:
                x, y = float(parts[1]), float(parts[2])
                if 0 <= x <= 1 and 0 <= y <= 1:
                    self.controller.update_sample_position(x, y)
                    print(f"📍 Sample position: ({x:.2f}, {y:.2f})")
                else:
                    print("Position values must be between 0.0 and 1.0")
            else:
                print("Usage: pos <x> <y>")
        
        elif command == "status":
            self.controller.request_status()
            print("📊 Status requested")
        
        else:
            print(f"Unknown command: {command}")
    
    def start_grid_scan(self):
        """Start automated grid scanning"""
        if self.scan_active:
            print("⚠️ Scan already active")
            return
        
        print("🔍 Starting 3x3 grid scan...")
        self.scan_active = True
        
        # Start scan in background thread
        scan_thread = threading.Thread(target=self._grid_scan_worker, daemon=True)
        scan_thread.start()
    
    def stop_scan(self):
        """Stop current scan"""
        if self.scan_active:
            self.scan_active = False
            print("⏹️ Scan stopped")
        else:
            print("ℹ️ No scan active")
    
    def _grid_scan_worker(self):
        """Background worker for grid scanning"""
        try:
            positions = [
                (0.2, 0.2), (0.5, 0.2), (0.8, 0.2),  # Top row
                (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),  # Middle row
                (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)   # Bottom row
            ]
            
            for i, (x, y) in enumerate(positions):
                if not self.scan_active:
                    break
                
                print(f"🎯 Scan position {i+1}/9: ({x:.1f}, {y:.1f})")
                
                # Move to position
                self.controller.update_sample_position(x, y)
                time.sleep(1)
                
                # Take image
                self.controller.snap_image()
                time.sleep(2)  # Wait for image capture
            
            print("✅ Grid scan completed!")
            
        except Exception as e:
            self.logger.error(f"Scan error: {e}")
        finally:
            self.scan_active = False
    
    def real_time_monitoring(self):
        """Display real-time system status"""
        print("\n📊 Real-time Monitoring (Press Enter to return to menu)")
        
        monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        monitor_thread.start()
        
        input()  # Wait for user input
        print("📊 Monitoring stopped")
    
    def _monitor_worker(self):
        """Background worker for real-time monitoring"""
        while True:
            try:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
                print("📊 OpenUC2 Real-time Status")
                print("=" * 40)
                print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                print(f"Connected: {'✅' if self.controller.connected else '❌'}")
                
                # LED Status
                led = self.controller.led_status
                led_color = f"RGB({led.r}, {led.g}, {led.b})" if led.enabled else "OFF"
                print(f"LED: {'🟢' if led.enabled else '🔴'} {led_color}")
                
                # Motor Positions
                motors = self.controller.motor_positions
                print(f"Motors: X={motors.x:.1f} Y={motors.y:.1f} Z={motors.z:.1f} A={motors.a:.1f}")
                
                # Objective Slot
                slot = self.controller.current_objective_slot
                print(f"Objective: Slot {slot}")
                
                # Sample Position
                pos = self.controller.current_sample_position
                print(f"Sample: ({pos.x:.2f}, {pos.y:.2f})")
                
                # Scan Status
                scan_status = "🔍 ACTIVE" if self.scan_active else "⏸️ INACTIVE"
                print(f"Scan: {scan_status}")
                
                print("\nPress Enter to exit monitoring...")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                break
    
    def run_demo(self):
        """Main demo runner"""
        print("🚀 OpenUC2 Advanced Serial Interface Demo")
        print("=" * 50)
        
        if not self.connect_to_microscope():
            return
        
        self.running = True
        
        while self.running:
            print("\n📋 Demo Options:")
            print("1. Interactive Control")
            print("2. Automated Demo Sequence") 
            print("3. Real-time Monitoring")
            print("4. Grid Scan Demo")
            print("5. View Data Log")
            print("6. Quit")
            
            try:
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == "1":
                    self.interactive_mode()
                elif choice == "2":
                    self.run_automated_demo()
                elif choice == "3":
                    self.real_time_monitoring()
                elif choice == "4":
                    self.start_grid_scan()
                    time.sleep(20)  # Let scan run
                elif choice == "5":
                    self.show_data_log()
                elif choice == "6":
                    self.running = False
                else:
                    print("Invalid option")
                    
            except KeyboardInterrupt:
                self.running = False
        
        # Cleanup
        self.save_log()
        if self.controller:
            self.controller.disconnect()
        
        print("👋 Demo finished!")
    
    def run_automated_demo(self):
        """Run automated demonstration sequence"""
        print("\n🎭 Running Automated Demo...")
        
        # LED color sequence
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
        for i, (r, g, b) in enumerate(colors):
            print(f"💡 LED Color {i+1}/4: RGB({r}, {g}, {b})")
            self.controller.set_led(True, r, g, b)
            time.sleep(2)
        
        self.controller.set_led(False)
        print("💡 LED off")
        
        # Objective slot switching
        for slot in [1, 2, 1]:
            print(f"🔬 Switching to slot {slot}")
            self.controller.set_objective_slot(slot)
            time.sleep(2)
        
        # Sample positions
        positions = [(0.1, 0.1), (0.9, 0.9), (0.5, 0.5)]
        for x, y in positions:
            print(f"📍 Moving to ({x}, {y})")
            self.controller.update_sample_position(x, y)
            time.sleep(1.5)
        
        # Take some images
        for i in range(3):
            print(f"📸 Taking image {i+1}/3")
            self.controller.snap_image()
            time.sleep(1)
        
        print("✨ Automated demo complete!")
    
    def show_data_log(self):
        """Display recent data log entries"""
        print("\n📋 Recent Data Log Entries:")
        print("=" * 50)
        
        recent_entries = self.data_log[-10:]  # Last 10 entries
        
        for entry in recent_entries:
            timestamp = entry['timestamp']
            event_type = entry['event_type']
            data = entry['data']
            print(f"{timestamp} | {event_type} | {data}")
        
        print(f"\nTotal entries: {len(self.data_log)}")
        print(f"Log file: {self.log_file}")
        
        input("Press Enter to continue...")

if __name__ == "__main__":
    demo = UC2AdvancedDemo()
    demo.run_demo()