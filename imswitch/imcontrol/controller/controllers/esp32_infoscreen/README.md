# OpenUC2 ESP32 Serial Interface

A Python package for communicating with the OpenUC2 ESP32 controller via USB Serial. This package provides bidirectional communication for microscope control including motors, LEDs, objective slots, sample positioning, laser/PWM control, and **image display**.

## Features

- **Bidirectional Serial Communication**: Send commands to ESP32 and receive status updates
- **Enhanced Device Detection**: Validates ESP32 communication before connecting
- **Motor Control**: Control individual stepper motors and XY stage movements  
- **LED Control**: Set LED colors and brightness
- **Objective Slot Management**: Switch between objective slots (1 and 2)
- **Sample Position Tracking**: Display and update sample position on a map
- **PWM/Laser Control**: Control 4 PWM channels with 0-1024 range for laser intensity
- **Image Capture**: Trigger image capture with snap button
- **🆕 Image Display**: Send numpy arrays/images to display as new tabs on ESP32 screen
- **🆕 Automatic Image Capture**: Auto-send images when snap button is pressed
- **Callback System**: Register callbacks for various events (motor updates, LED changes, PWM updates, etc.)
- **Auto Port Detection**: Automatically find and validate ESP32 serial port
- **Clean JSON Protocol**: Proper separation of debug logs and command messages

## Enhanced UI Structure

The ESP32 interface now features individual tabs instead of nested structure:
- **Motor**: Individual motor control and XY stage movements
- **LED**: Color picker and brightness controls  
- **Objective**: Slot switching and snap button
- **Sample Map**: Visual position display with red indicator
- **Lasers**: PWM channels 1-4 with 0-1024 sliders
- **Acquisition**: Image capture settings and controls
- **🆕 Image Tabs**: Dynamically created tabs showing images sent from Python

## Installation

```bash
# Install from requirements (includes numpy and Pillow for image processing)
pip install -r requirements.txt

# Or install the package in development mode
pip install -e .
```

## Quick Start

```python
from uc2_serial_controller import UC2SerialController, find_esp32_port

# Auto-detect ESP32 port
port = find_esp32_port()
controller = UC2SerialController(port)

# Register callbacks
def on_led_update(data):
    print(f"LED changed: {data}")

def on_motor_update(data):
    print(f"Motor moved: {data}")

controller.on_led_update(on_led_update)
controller.on_motor_update(on_motor_update)

# Connect and control
if controller.connect():
    # Control LED
    controller.set_led(True, 255, 0, 0)  # Red
    
    # Move motors
    controller.move_motor(1, 1000)  # Motor 1, speed 1000
    controller.move_xy_motors(500, -500)  # XY movement
    
    # Switch objective slots
    controller.set_objective_slot(2)
    
    # Control PWM/Lasers (NEW)
    controller.set_pwm_value(1, 512)  # Channel 1, half power
    controller.set_pwm_value(2, 1024) # Channel 2, full power
    
    # Capture image
    controller.snap_image()
    
    # Update sample position
    controller.update_sample_position(0.3, 0.7)  # Normalized coordinates
```

## API Reference

### UC2SerialController Class

#### Connection Methods
- `connect(port=None)` - Connect to ESP32
- `disconnect()` - Disconnect from ESP32
- `connected` (property) - Check connection status

#### Control Methods
- `set_led(enabled, r, g, b)` - Control LED state and color
- `move_motor(motor_id, speed)` - Move individual motor
- `move_xy_motors(speed_x, speed_y)` - Move XY motors simultaneously  
- `set_objective_slot(slot)` - Switch objective slot (1 or 2)
- `snap_image()` - Trigger image capture
- `update_sample_position(x, y)` - Update sample position (0-1 coordinates)

#### Callback Registration
- `on_status_update(callback)` - Status change events
- `on_motor_update(callback)` - Motor position updates
- `on_led_update(callback)` - LED state changes
- `on_objective_slot_update(callback)` - Objective slot changes
- `on_sample_position_update(callback)` - Sample position updates
- `on_image_captured(callback)` - Image capture events
- `on_connection_changed(callback)` - Connection status changes

**New: ESP32 Display Interaction Callbacks** (User interactions with ESP32 display)
- `on_objective_slot_command(callback)` - User pressed objective slot buttons
- `on_motor_command(callback)` - User moved motor speed sliders 
- `on_motor_xy_command(callback)` - User moved XY joystick
- `on_led_command(callback)` - User changed LED settings
- `on_pwm_command(callback)` - User moved PWM sliders
- `on_snap_image_command(callback)` - User pressed snap image button

#### State Properties
- `motor_positions` - Current motor positions
- `led_status` - Current LED state
- `current_objective_slot` - Active objective slot
- `current_sample_position` - Current sample position

### Utility Functions
- `find_esp32_port()` - Automatically detect ESP32 serial port

## Message Protocol

The serial communication uses JSON messages for bidirectional communication:

### Commands (Python → ESP32)
```json
// LED control
{"type": "led_command", "data": {"enabled": true, "r": 255, "g": 0, "b": 0}}

// Motor control
{"type": "motor_command", "data": {"motor": 1, "speed": 1000}}
{"type": "motor_xy_command", "data": {"speedX": 500, "speedY": -500}}

// Objective slot
{"type": "objective_slot_command", "data": {"slot": 2}}

// Image capture
{"type": "snap_image_command"}

// Sample position
{"type": "sample_position_command", "data": {"x": 0.3, "y": 0.7}}
```

### Updates (ESP32 → Python)
```json
// Status update
{"type": "status_update", "data": {"connected": true, "modules": {"motor": true, "led": true}}}

// Motor update
{"type": "motor_update", "data": {"positions": {"x": 1000, "y": 2000, "z": 500}}}

// LED update  
{"type": "led_update", "data": {"enabled": true, "r": 255, "g": 0, "b": 0}}

// Objective slot update
{"type": "objective_slot_update", "data": {"current_slot": 2}}

// Sample position update
{"type": "sample_position_update", "data": {"x": 0.3, "y": 0.7}}

// Image captured
{"type": "image_captured", "data": {}}
```

### Command Events (ESP32 → Python when user interacts with display)
```json
// User pressed objective slot button
{"type": "objective_slot_command", "data": {"slot": 2}}

// User moved motor speed slider
{"type": "motor_command", "data": {"motor": 1, "speed": 100}}

// User moved XY joystick
{"type": "motor_xy_command", "data": {"speedX": 500, "speedY": 300}}

// User changed LED settings
{"type": "led_command", "data": {"enabled": true, "r": 255, "g": 0, "b": 0}}

// User moved PWM slider  
{"type": "pwm_command", "data": {"channel": 1, "value": 512}}

// User pressed snap image button
{"type": "snap_image_command"}

// User pressed motor step buttons
{"type": "motor_step_update", "data": {"motor": 1, "steps": 100}}

// User clicked on sample map
{"type": "sample_map_click", "data": {"pixel_x": 414, "pixel_y": 68, "sample_number": 12}}
```

## 🆕 Image Display Feature

Send images from Python to display as new tabs on the ESP32 screen!

### Quick Start - Image Display

```python
import numpy as np
from uc2_serial_controller import UC2SerialController

controller = UC2SerialController()
controller.connect()

# Send a simple test image
test_image = np.zeros((80, 120, 3), dtype=np.uint8)
test_image[:, :, 0] = 255  # Red image
controller.send_image(test_image, "Red Test")

# Set up automatic image capture on snap button press
def capture_camera_frame():
    # Your camera capture code here
    return camera.get_frame()  # Should return numpy array

controller.send_image_on_snap(capture_camera_frame)
```

### Image Display Methods

#### `send_image(image, tab_name, max_width=240, max_height=160)`
Send an image to display on ESP32 as a new tab.

**Parameters:**
- `image`: Numpy array (H, W, 3) RGB image, or path to image file
- `tab_name`: Name for the image tab (max 15 characters) 
- `max_width`: Maximum width to resize image (default: 240)
- `max_height`: Maximum height to resize image (default: 160)

**Returns:** `bool` - True if sent successfully

**Examples:**
```python
# Send numpy array
image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
controller.send_image(image, "Random Pattern")

# Send image file
controller.send_image("captured_image.jpg", "Camera Shot")

# Send grayscale image (auto-converted to RGB)
gray_img = np.ones((50, 50), dtype=np.uint8) * 128
controller.send_image(gray_img, "Gray Square")
```

#### `send_image_on_snap(image_source_callback)`
Automatically send images when the snap button is pressed on ESP32 display.

**Parameters:**
- `image_source_callback`: Function that returns image data when called

**Example:**
```python
def get_microscope_image():
    """Capture current microscope view"""
    # Connect to your camera/microscope
    frame = microscope.capture()
    return frame  # numpy array

# Now when user presses snap button on ESP32, 
# a new image tab will automatically appear!
controller.send_image_on_snap(get_microscope_image)
```

### Supported Image Formats

- **Numpy arrays**: RGB (H,W,3), RGBA (H,W,4), Grayscale (H,W)
- **Data types**: uint8, float32/float64 (0.0-1.0 range)
- **File formats**: JPG, PNG, BMP, TIFF (via PIL/Pillow)
- **Automatic conversion**: Grayscale→RGB, RGBA→RGB, Float→uint8
- **Automatic resizing**: Large images resized to fit ESP32 memory

### Image Display Protocol

```json
// Send image command (Python → ESP32)
{
  "type": "display_image_command",
  "data": {
    "tab_name": "Camera View",
    "width": 120,
    "height": 80, 
    "format": "rgb565",
    "image_data": "base64_encoded_rgb565_data..."
  }
}

// Image display result (ESP32 → Python)
{
  "type": "image_display_result",
  "data": {
    "tab_name": "Camera View",
    "width": 120,
    "height": 80,
    "success": true
  }
}
```

### Technical Details

- **Format**: Images converted to RGB565 format for ESP32 efficiency
- **Encoding**: Base64 encoding for JSON transmission
- **Memory**: ESP32 memory constraints limit image size (~240x160 max)
- **Tabs**: New tab created for each image with close button
- **Performance**: Optimized for small-to-medium microscopy images

## Hardware Requirements

- OpenUC2 ESP32 with serial interface firmware
- USB connection between PC and ESP32
- Compatible stepper motors and LED array
- Objective lens slots (optional)

## Troubleshooting

### Connection Issues
- Check USB cable connection
- Verify correct serial port (use `find_esp32_port()`)
- Ensure ESP32 is not connected to other applications
- Check baud rate (default: 115200)

### Communication Issues
- Monitor serial output for error messages
- Verify JSON message format
- Check for firmware compatibility
- Ensure proper grounding and connections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.