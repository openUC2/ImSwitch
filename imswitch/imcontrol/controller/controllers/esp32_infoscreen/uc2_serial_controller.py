#!/usr/bin/env python3
"""
OpenUC2 ESP32 Serial Interface

A Python package for communicating with the OpenUC2 ESP32 controller via USB Serial.
Provides bidirectional communication for microscope control including motors, LEDs,
objective slots, and sample positioning.
"""

import serial
import json
import threading
import time
import logging
import base64
import os
import numpy as np
from typing import Callable, Optional, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class MotorPosition:
    """Motor position data"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    a: float = 0.0

@dataclass
class LEDState:
    """LED state data"""
    enabled: bool = False
    r: int = 0
    g: int = 0
    b: int = 0

@dataclass
class ObjectiveSlot:
    """Objective slot information"""
    current_slot: int = 1
    available_slots: list = None
    
    def __post_init__(self):
        if self.available_slots is None:
            self.available_slots = [1, 2]

@dataclass
class SamplePosition:
    """Sample position on the map"""
    x: float = 0.5  # Normalized coordinates 0-1
    y: float = 0.5

class UC2SerialController:
    """
    Main controller class for ESP32 serial communication
    """
    
    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        self.read_thread: Optional[threading.Thread] = None
        self.running = False
        
        # State tracking
        self.motor_position = MotorPosition()
        self.led_state = LEDState()
        self.objective_slot = ObjectiveSlot()
        self.sample_position = SamplePosition()
        
        # Callbacks
        self.callbacks: Dict[str, list] = {
            'status_update': [],
            'motor_update': [],
            'led_update': [],
            'objective_slot_update': [],
            'sample_position_update': [],
            'image_captured': [],
            'connection_changed': [],
            'pwm_update': [],
            # New command event callbacks (ESP32 -> Python when user interacts with display)
            'objective_slot_command': [],
            'motor_command': [],
            'motor_xy_command': [],
            'led_command': [],
            'pwm_command': [],
            'snap_image_command': []
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def connect(self, port: str = None) -> bool:
        """
        Connect to ESP32 via serial port
        
        Args:
            port: Serial port name (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            
        Returns:
            bool: True if connection successful
        """
        if port:
            self.port = port
            
        if not self.port:
            self.logger.error("No serial port specified")
            return False
            
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for ESP32 to reset
            
            # Start reading thread
            self.running = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            self.is_connected = True
            self.logger.info(f"Connected to ESP32 on {self.port}")
            
            # Request initial status
            self.request_status()
            
            self._notify_callbacks('connection_changed', {'connected': True})
            return True
            
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        self.running = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1)
            
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
            
        self.is_connected = False
        self.logger.info("Disconnected from ESP32")
        self._notify_callbacks('connection_changed', {'connected': False})
    
    def _read_loop(self):
        """Background thread for reading serial messages"""
        buffer = ""
        
        while self.running and self.serial_conn:
            try:
                if self.serial_conn.in_waiting:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._handle_message(line)
                
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in read loop: {e}")
                break
    
    def _handle_message(self, message: str):
        """Handle incoming message from ESP32"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            # Handle update messages (ESP32 status/state changes)
            if msg_type == 'status_update':
                self._handle_status_update(data.get('data', {}))
            elif msg_type == 'motor_update':
                self._handle_motor_update(data.get('data', {}))
            elif msg_type == 'led_update':
                self._handle_led_update(data.get('data', {}))
            elif msg_type == 'objective_slot_update':
                self._handle_objective_slot_update(data.get('data', {}))
            elif msg_type == 'sample_position_update':
                self._handle_sample_position_update(data.get('data', {}))
            elif msg_type == 'image_captured':
                self._handle_image_captured(data.get('data', {}))
            elif msg_type == 'motor_step_update':
                self._handle_motor_step_update(data.get('data', {}))
            elif msg_type == 'pwm_update':
                self._handle_pwm_update(data.get('data', {}))
            elif msg_type == 'sample_map_click':
                self._handle_sample_map_click(data.get('data', {}))
            
            # Handle command messages (user interactions with ESP32 display)
            elif msg_type == 'objective_slot_command':
                self._handle_objective_slot_command(data.get('data', {}))
            elif msg_type == 'motor_command':
                self._handle_motor_command(data.get('data', {}))
            elif msg_type == 'motor_xy_command':
                self._handle_motor_xy_command(data.get('data', {}))
            elif msg_type == 'led_command':
                self._handle_led_command(data.get('data', {}))
            elif msg_type == 'pwm_command':
                self._handle_pwm_command(data.get('data', {}))
            elif msg_type == 'snap_image_command':
                self._handle_snap_image_command(data.get('data', {}))
            else:
                self.logger.debug(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON message: {message} - {e}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    def _handle_status_update(self, data: Dict[str, Any]):
        """Handle status update from ESP32"""
        self.logger.info("Received status update")
        self._notify_callbacks('status_update', data)
    
    def _handle_motor_update(self, data: Dict[str, Any]):
        """Handle motor position update"""
        positions = data.get('positions', {})
        self.motor_position.x = positions.get('x', self.motor_position.x)
        self.motor_position.y = positions.get('y', self.motor_position.y)
        self.motor_position.z = positions.get('z', self.motor_position.z)
        self.motor_position.a = positions.get('a', self.motor_position.a)
        
        self._notify_callbacks('motor_update', data)
    
    def _handle_led_update(self, data: Dict[str, Any]):
        """Handle LED state update"""
        self.led_state.enabled = data.get('enabled', self.led_state.enabled)
        self.led_state.r = data.get('r', self.led_state.r)
        self.led_state.g = data.get('g', self.led_state.g)
        self.led_state.b = data.get('b', self.led_state.b)
        
        self._notify_callbacks('led_update', data)
    
    def _handle_objective_slot_update(self, data: Dict[str, Any]):
        """Handle objective slot update"""
        self.objective_slot.current_slot = data.get('current_slot', self.objective_slot.current_slot)
        
        self._notify_callbacks('objective_slot_update', data)
    
    def _handle_sample_position_update(self, data: Dict[str, Any]):
        """Handle sample position update"""
        self.sample_position.x = data.get('x', self.sample_position.x)
        self.sample_position.y = data.get('y', self.sample_position.y)
        
        self._notify_callbacks('sample_position_update', data)
    
    def _handle_image_captured(self, data: Dict[str, Any]):
        """Handle image capture notification"""
        self.logger.info("Image captured")
        self._notify_callbacks('image_captured', data)
    
    def _handle_motor_step_update(self, data: Dict[str, Any]):
        """Handle motor step command acknowledgment"""
        motor = data.get('motor', 0)
        steps = data.get('steps', 0)
        self.logger.info(f"Motor {motor} moved {steps} steps")
        self._notify_callbacks('motor_update', data)
        
    def _handle_pwm_update(self, data: Dict[str, Any]):
        """Handle PWM update"""
        channel = data.get('channel', 0)
        value = data.get('value', 0)
        self.logger.info(f"PWM Channel {channel} set to {value}")
        self._notify_callbacks('pwm_update', data)
        
    def _handle_sample_map_click(self, data: Dict[str, Any]):
        """Handle sample map click"""
        pixel_x = data.get('pixel_x', 0)
        pixel_y = data.get('pixel_y', 0)  
        sample_number = data.get('sample_number', 0)
        self.logger.info(f"Sample map clicked at ({pixel_x}, {pixel_y}), sample #{sample_number}")
        self._notify_callbacks('sample_position_update', data)
    
    # Command message handlers (user interactions with ESP32 display)
    def _handle_objective_slot_command(self, data: Dict[str, Any]):
        """Handle objective slot command from display"""
        slot = data.get('slot', 1)
        self.logger.info(f"User selected objective slot {slot} on display")
        self._notify_callbacks('objective_slot_command', data)
    
    def _handle_motor_command(self, data: Dict[str, Any]):
        """Handle motor command from display"""
        motor = data.get('motor', 0)
        speed = data.get('speed', 0)
        self.logger.info(f"User set motor {motor} to speed {speed} on display")
        self._notify_callbacks('motor_command', data)
    
    def _handle_motor_xy_command(self, data: Dict[str, Any]):
        """Handle XY motor command from display"""
        speed_x = data.get('speedX', 0)
        speed_y = data.get('speedY', 0)
        self.logger.info(f"User set XY motors to speeds X={speed_x}, Y={speed_y} on display")
        self._notify_callbacks('motor_xy_command', data)
    
    def _handle_led_command(self, data: Dict[str, Any]):
        """Handle LED command from display"""
        enabled = data.get('enabled', False)
        r = data.get('r', 0)
        g = data.get('g', 0)
        b = data.get('b', 0)
        self.logger.info(f"User set LED on display: enabled={enabled}, RGB=({r}, {g}, {b})")
        self._notify_callbacks('led_command', data)
    
    def _handle_pwm_command(self, data: Dict[str, Any]):
        """Handle PWM command from display"""
        channel = data.get('channel', 0)
        value = data.get('value', 0)
        self.logger.info(f"User set PWM channel {channel} to {value} on display")
        self._notify_callbacks('pwm_command', data)
    
    def _handle_snap_image_command(self, data: Dict[str, Any]):
        """Handle snap image command from display"""
        self.logger.info("User pressed snap image button on display")
        self._notify_callbacks('snap_image_command', data)
    
    def _send_message(self, message_dict: Dict[str, Any]):
        """Send message to ESP32"""
        if not self.is_connected or not self.serial_conn:
            self.logger.error("Not connected to ESP32")
            return False
            
        try:
            message = json.dumps(message_dict)
            self.serial_conn.write((message + '\n').encode('utf-8'))
            self.serial_conn.flush()
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in callback for {event_type}: {e}")
    
    # Callback registration methods
    def on_status_update(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for status updates"""
        self.callbacks['status_update'].append(callback)
    
    def on_motor_update(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for motor updates"""
        self.callbacks['motor_update'].append(callback)
    
    def on_pwm_update(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for PWM updates"""
        self.callbacks['pwm_update'].append(callback)

    def on_led_update(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for LED updates"""
        self.callbacks['led_update'].append(callback)
    
    def on_objective_slot_update(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for objective slot updates"""
        self.callbacks['objective_slot_update'].append(callback)
    
    def on_sample_position_update(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for sample position updates"""
        self.callbacks['sample_position_update'].append(callback)
    
    def on_image_captured(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for image capture events"""
        self.callbacks['image_captured'].append(callback)
    
    def on_connection_changed(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for connection changes"""
        self.callbacks['connection_changed'].append(callback)
    
    # Command event callback registration (user interactions with ESP32 display)
    def on_objective_slot_command(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for objective slot commands from display"""
        self.callbacks['objective_slot_command'].append(callback)
    
    def on_motor_command(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for motor commands from display"""
        self.callbacks['motor_command'].append(callback)
    
    def on_motor_xy_command(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for XY motor commands from display"""
        self.callbacks['motor_xy_command'].append(callback)
    
    def on_led_command(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for LED commands from display"""
        self.callbacks['led_command'].append(callback)
    
    def on_pwm_command(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for PWM commands from display"""
        self.callbacks['pwm_command'].append(callback)
    
    def on_snap_image_command(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for snap image commands from display"""
        self.callbacks['snap_image_command'].append(callback)
    
    def send_image_on_snap(self, image_source_callback: Callable[[], Union[np.ndarray, str, None]]):
        """
        Convenience method to automatically send an image when snap button is pressed
        
        Args:
            image_source_callback: Function that returns image data when called.
                                 Should return numpy array, file path, or None
        
        Example:
            def get_current_camera_frame():
                # Your camera capture logic here
                return camera.capture()  # Returns numpy array
            
            controller.send_image_on_snap(get_current_camera_frame)
        """
        def snap_handler(data):
            try:
                image = image_source_callback()
                if image is not None:
                    timestamp = time.strftime("%H:%M:%S")
                    success = self.send_image(image, f"Snap {timestamp}")
                    if success:
                        self.logger.info("Image sent to display after snap command")
                    else:
                        self.logger.error("Failed to send image to display")
                else:
                    self.logger.warning("Image source callback returned None")
            except Exception as e:
                self.logger.error(f"Error in snap image handler: {e}")
        
        self.on_snap_image_command(snap_handler)
    
    # Command methods
    def request_status(self):
        """Request status update from ESP32"""
        return self._send_message({"type": "status_request"})
    
    def set_led(self, enabled: bool, r: int = 0, g: int = 0, b: int = 0):
        """Control LED: {"type": "led_command","data": { "enabled": 1, "r": 0, "g": 255, "b": 255} }"""
        return self._send_message({
            "type": "led_command",
            "data": {
                "enabled": enabled,
                "r": r,
                "g": g,
                "b": b
            }
        })
    
    def move_motor(self, motor_id: int, speed: int):
        """Move individual motor"""
        return self._send_message({
            "type": "motor_command",
            "data": {
                "motor": motor_id,
                "speed": speed
            }
        })
    
    def move_xy_motors(self, speed_x: int, speed_y: int):
        """Move X and Y motors simultaneously"""
        return self._send_message({
            "type": "motor_xy_command",
            "data": {
                "speedX": speed_x,
                "speedY": speed_y
            }
        })
    
    def set_objective_slot(self, slot: int):
        """Set objective slot (1 or 2)"""
        return self._send_message({
            "type": "objective_slot_command",
            "data": {
                "slot": slot
            }
        })
    
    def snap_image(self):
        """Trigger image capture"""
        return self._send_message({
            "type": "snap_image_command"
        })
    
    def update_sample_position(self, x: float, y: float):
        """Update sample position on map (normalized coordinates 0-1)"""
        return self._send_message({
            "type": "sample_position_command",
            "data": {
                "x": x,
                "y": y
            }
        })
    
    def set_pwm_value(self, channel: int, value: int):
        """Set PWM value for laser channel (1-4, value 0-1024)"""
        if not (1 <= channel <= 4):
            raise ValueError("Channel must be between 1 and 4")
        if not (0 <= value <= 1024):
            raise ValueError("Value must be between 0 and 1024")
            
        return self._send_message({
            "type": "pwm_command",
            "data": {
                "channel": channel,
                "value": value
            }
        })
    
    def move_motor_steps(self, motor_id: int, steps: int):
        """Move motor by specific number of steps (position-based control)"""
        return self._send_message({
            "type": "motor_step_command", 
            "data": {
                "motor": motor_id,
                "steps": steps
            }
        })
    
    def send_image(self, image: Union[np.ndarray, str], tab_name: str = "Captured Image", 
                   max_width: int = 240, max_height: int = 160):
        """
        Send an image to display on ESP32 as a new tab
        
        Args:
            image: Numpy array (H, W, 3) RGB image, or path to image file
            tab_name: Name for the image tab (max 15 characters)
            max_width: Maximum width to resize image to fit ESP32 memory
            max_height: Maximum height to resize image to fit ESP32 memory
        
        Returns:
            bool: True if message sent successfully
        """
        try:
            # Import PIL for image processing
            try:
                from PIL import Image
            except ImportError:
                self.logger.error("PIL (Pillow) is required for image processing. Install with: pip install Pillow")
                return False
            
            # Handle file path input
            if isinstance(image, str):
                if not os.path.exists(image):
                    self.logger.error(f"Image file not found: {image}")
                    return False
                pil_image = Image.open(image).convert('RGB')
                image_array = np.array(pil_image)
            else:
                # Handle numpy array input
                if not isinstance(image, np.ndarray):
                    self.logger.error("Image must be a numpy array or file path")
                    return False
                    
                if image.dtype != np.uint8:
                    # Convert to uint8 if needed
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # Ensure RGB format
                if len(image.shape) == 2:
                    # Grayscale to RGB
                    image_array = np.stack([image, image, image], axis=-1)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    # RGBA to RGB
                    image_array = image[:, :, :3]
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    image_array = image
                else:
                    self.logger.error(f"Unsupported image shape: {image.shape}")
                    return False
            
            height, width = image_array.shape[:2]
            
            # Resize if image is too large
            if width > max_width or height > max_height:
                pil_image = Image.fromarray(image_array)
                # Calculate aspect ratio preserving resize
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                image_array = np.array(pil_image)
                self.logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
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
            self.logger.error(f"Failed to send image: {e}")
            return False
    
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
    
    # Property accessors
    @property
    def connected(self) -> bool:
        """Check if connected to ESP32"""
        return self.is_connected
    
    @property
    def motor_positions(self) -> MotorPosition:
        """Get current motor positions"""
        return self.motor_position
    
    @property  
    def led_status(self) -> LEDState:
        """Get current LED state"""
        return self.led_state
    
    @property
    def current_objective_slot(self) -> int:
        """Get current objective slot"""
        return self.objective_slot.current_slot
    
    @property
    def current_sample_position(self) -> SamplePosition:
        """Get current sample position"""
        return self.sample_position


def find_esp32_port() -> Optional[str]:
    """
    Automatically find ESP32 serial port with device validation
    
    Returns:
        str: Port name if found and validated, None otherwise
    """
    import serial.tools.list_ports
    
    # Common ESP32 device identifiers
    esp32_identifiers = [
        "CP210x",  # Silicon Labs CP210x
        "CH340",   # WCH CH340
        "UART",    # Generic UART
        "ESP32",   # ESP32 specific
        "USB Serial Device"  # Generic USB serial
    ]
    
    ports = serial.tools.list_ports.comports()
    
    # First try ports that match ESP32 identifiers
    candidate_ports = []
    for port in ports:
        for identifier in esp32_identifiers:
            if identifier.lower() in port.description.lower():
                candidate_ports.append(port.device)
                break
    
    # If no specific match, add all available ports
    if not candidate_ports:
        candidate_ports = [port.device for port in ports]
    
    # Test each candidate port
    for port_device in candidate_ports:
        if _test_esp32_communication(port_device):
            return port_device
                
    return None


def _test_esp32_communication(port_device: str, timeout: float = 2.0) -> bool:
    """
    Test if the device on the given port is an ESP32 running our firmware
    
    Args:
        port_device: Serial port device name
        timeout: Timeout for communication test
        
    Returns:
        bool: True if ESP32 is detected and responds correctly
    """
    try:
        with serial.Serial(port_device, 115200, timeout=timeout) as ser:
            # Clear any pending data
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Send a status request and wait for JSON response
            test_message = '{"type": "status_request"}\n'
            ser.write(test_message.encode())
            ser.flush()
            
            # Wait for response
            start_time = time.time()
            response_buffer = ""
            
            while time.time() - start_time < timeout:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    response_buffer += data
                    
                    # Look for JSON response
                    lines = response_buffer.split('\n')
                    for line in lines:
                        line = line.strip()
                        print(line)
                        if line.startswith('{') and line.endswith('}'):
                            try:
                                response = json.loads(line)
                                # Check if it's a valid status response from our ESP32
                                if (response.get('type') == 'status_update' and 
                                    'data' in response and
                                    response['data'].get('connected')):
                                    return True
                            except json.JSONDecodeError:
                                continue
                
                time.sleep(0.1)
                
    except (serial.SerialException, OSError):
        pass
    
    return False


if __name__ == "__main__":
    # Example usage
    def on_led_update(data):
        print(f"LED Update: {data}")
    
    def on_motor_update(data):
        print(f"Motor Update: {data}")
    
    def on_status_update(data):
        print(f"Status Update: {data}")
    
    # Find ESP32 port automatically
    port = find_esp32_port()
    if not port:
        print("No ESP32 device found")
        exit(1)
    
    print(f"Found ESP32 on port: {port}")
    
    # Create controller and connect
    controller = UC2SerialController(port)
    
    # Register callbacks
    controller.on_led_update(on_led_update)
    controller.on_motor_update(on_motor_update)
    controller.on_status_update(on_status_update)
    
    # Connect
    if controller.connect():
        print("Connected successfully!")
        
        try:
            # Test commands
            time.sleep(1)
            controller.set_led(True, 255, 0, 0)  # Red
            time.sleep(1)
            controller.set_led(True, 0, 255, 0)  # Green
            time.sleep(1) 
            controller.set_led(False)  # Off
            
            # Keep running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            controller.disconnect()
    else:
        print("Failed to connect!")