# ESP32InfoScreen Integration Guide

The ESP32InfoScreen controller provides bidirectional communication between ImSwitch and an ESP32 device with an attached display. This enables users to control microscope hardware directly from the ESP32 display and see real-time status updates.

## Features

### Hardware Control from ESP32 Display
- **Motor/Stage Control**: Joystick controls for XY movement, individual motor speed control
- **LED Matrix Control**: RGB color picker with on/off control
- **Objective Management**: Switch between objective slots (1, 2)
- **Laser Control**: PWM channel sliders for laser intensity (0-1024 range)
- **Image Capture**: Snap button to trigger image acquisition
- **Sample Position**: Visual map showing current stage position

### Status Updates to ESP32 Display
- **Live Images**: Captured images sent to ESP32 as new display tabs
- **Position Tracking**: Stage movements reflected on sample position map
- **Hardware Status**: LED states, objective slot, motor positions synchronized

## Setup Requirements

### Hardware
- ESP32 with display and serial interface firmware
- USB connection between PC and ESP32
- Compatible hardware components:
  - Stepper motors/stage (ESP32StageManager)
  - LED matrix (ESP32LEDMatrixManager) 
  - Laser diodes (ESP32LEDLaserManager)
  - Optional: Objective changer

### Software Configuration

1. **Add to Setup File**: Include `"ESP32InfoScreen"` in `availableWidgets` list:

```json
{
  "availableWidgets": [
    "Settings",
    "View", 
    "Recording",
    "Image",
    "Laser",
    "Positioner",
    "LEDMatrix",
    "ESP32InfoScreen"
  ]
}
```

2. **Hardware Managers**: Ensure your setup includes the required managers:
   - `ESP32Manager` for serial communication
   - `ESP32StageManager` for stage control
   - `ESP32LEDMatrixManager` for LED matrix
   - `ESP32LEDLaserManager` for laser control

## Usage

### Automatic Operation
The ESP32InfoScreen controller automatically:
1. Detects and connects to ESP32 on startup
2. Registers callbacks for all ESP32 display interactions
3. Sets up bidirectional communication
4. Synchronizes hardware state with display

### Manual Controls
The ImSwitch widget provides manual controls for:
- **Connect/Disconnect**: Manual ESP32 connection management
- **Test LED**: Send red LED test command
- **Send Test Image**: Send colored test pattern to display

### ESP32 Display Interactions

#### Motor Control
```python
# User moves joystick on ESP32 → ImSwitch moves stage
# speedX, speedY values from -1000 to +1000
positioner.move(speedX/1000.0, 'X', is_absolute=False, is_blocking=False)
positioner.move(speedY/1000.0, 'Y', is_absolute=False, is_blocking=False)
```

#### LED Control  
```python
# User changes color on ESP32 → ImSwitch updates LED matrix
# RGB values 0-255, enabled true/false
if enabled:
    ledMatrix.setAll((r/255.0, g/255.0, b/255.0))
else:
    ledMatrix.setAll((0, 0, 0))
```

#### Laser Control
```python
# User adjusts PWM slider → ImSwitch sets laser intensity
# channel 1-4, value 0-1024
laser_manager.setValue(int((value / 1024.0) * max_value))
```

#### Image Capture
```python
# User presses snap button → ImSwitch captures and sends image
_commChannel.sigSnapImg.emit()
esp32_controller.send_image(latest_frame, f"Snap {timestamp}")
```

## Technical Details

### Controller Architecture
- **Base Class**: `ImConWidgetController` (standard ImSwitch pattern)
- **Communication**: Uses `UC2SerialController` for ESP32 protocol
- **Signals**: Integrates with ImSwitch `CommunicationChannel` signals
- **Threading**: Non-blocking serial communication in separate thread

### Error Handling
- Graceful degradation when hardware components unavailable
- Connection retry logic with status updates
- Comprehensive logging for debugging
- Safe cleanup on shutdown

### Message Protocol
ESP32 communication uses JSON messages:

```python
# Motor command from ESP32
{
    "type": "motor_xy_command",
    "data": {"speedX": 500, "speedY": -300}
}

# LED update to ESP32  
{
    "type": "led_command", 
    "data": {"enabled": true, "r": 255, "g": 0, "b": 0}
}

# Image data to ESP32
{
    "type": "image_command",
    "data": {
        "tab_name": "Live 14:30:25",
        "width": 240,
        "height": 160, 
        "image_data": "base64_encoded_rgb565_data"
    }
}
```

## Troubleshooting

### Connection Issues
- Check USB cable and ESP32 power
- Verify ESP32 firmware compatibility
- Check serial port availability (not used by other apps)
- Monitor log output for connection errors

### Hardware Integration Issues
- Verify hardware managers in setup configuration
- Check hardware component initialization in logs
- Test individual hardware components separately
- Ensure proper signal connections in ImSwitch

### Performance Considerations
- Image sending limited by serial bandwidth (~115200 baud)
- Images automatically resized to fit ESP32 memory (240x160 max)
- Motor commands are non-blocking for responsive control
- Status updates throttled to prevent ESP32 overload

## Development Notes

The integration follows ImSwitch patterns:
- Controller inherits from `ImConWidgetController`
- Widget inherits from base `Widget` class  
- Uses standard signal/slot communication
- Follows established hardware manager interfaces
- Includes proper finalization for cleanup

This enables seamless integration with existing ImSwitch workflows while adding powerful ESP32 display control capabilities.