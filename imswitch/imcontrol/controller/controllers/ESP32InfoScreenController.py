import numpy as np
try:
    import NanoImagingPack as nip
    isNIP = True
except:
    isNIP = False
import time
import threading
import collections
import logging

from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex, Timer
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger
from ..basecontrollers import ImConWidgetController
from imswitch import IS_HEADLESS

class ESP32InfoScreenController(ImConWidgetController):
    """ Controller for ESP32 InfoScreen display integration """

    sigImageReceived = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize ESP32 serial controller
        self.esp32_controller = None
        self._connected = False
        
        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        if len(allDetectorNames) > 0:
            detectorName = allDetectorNames[0]
            self.detector = self._master.detectorsManager[detectorName]
        else:
            self.detector = None
            self._logger.warning("No detectors found")
        
        # select stage 
        allPositionerNames = self._master.positionersManager.getAllDeviceNames()
        if len(allPositionerNames) > 0:
            positionerName = allPositionerNames[0]
            self.positioner = self._master.positionersManager[positionerName]
        else:
            self.positioner = None
            self._logger.warning("No positioners found")

        # get all Lasers
        self.lasers = self._master.lasersManager.getAllDeviceNames()
        self.laser = None
        if len(self.lasers) > 0:
            self.laser = self.lasers[0]
            try:
                self._master.lasersManager[self.laser].setGalvo(channel=1, frequency=10, offset=0, amplitude=1, clk_div=0, phase=0, invert=1, timeout=1)
            except Exception as e:
                self._logger.error(e)
        else:
            self._logger.warning("No lasers found")
            
        # get LEDMatrix
        allLEDMatrixNames = self._master.LEDMatrixsManager.getAllDeviceNames()
        if len(allLEDMatrixNames) > 0:
            self.ledMatrix = self._master.LEDMatrixsManager[allLEDMatrixNames[0]]
        else:
            self.ledMatrix = None
            if not IS_HEADLESS: 
                self._widget.replaceWithError('No LEDMatrix found in your setup file.')
            self._logger.warning("No LEDMatrix found")
            return

        # Initialize ESP32 connection
        self._initializeESP32Connection()
        
        # Connect ImSwitch signals to send updates to ESP32
        self._connectImSwitchSignals()
        
        # Connect widget signals if not headless
        if not IS_HEADLESS:
            self._connectWidgetSignals()

    def _initializeESP32Connection(self):
        """Initialize ESP32 serial connection and setup callbacks"""
        try:
            # Find and connect to ESP32
            from .esp32_infoscreen.uc2_serial_controller import UC2SerialController, find_esp32_port
            
            port = find_esp32_port()
            if not port:
                self._logger.warning("No ESP32 device found for InfoScreen")
                return
                
            self.esp32_controller = UC2SerialController(port)
            
            # Register callbacks for ESP32 display interactions
            self._setupESP32Callbacks()
            
            # Connect to ESP32
            if self.esp32_controller.connect():
                self._connected = True
                self._logger.info(f"Connected to ESP32 InfoScreen on port: {port}")
                # Update widget status if not headless
                if not IS_HEADLESS:
                    self._widget.setConnectionStatus(self._connected)
            else:
                self._logger.error("Failed to connect to ESP32 InfoScreen")
                
        except Exception as e:
            self._logger.error(f"Error initializing ESP32 connection: {e}")

    def _setupESP32Callbacks(self):
        """Setup callbacks for ESP32 display interactions"""
        if not self.esp32_controller:
            return
            
        # ESP32 -> ImSwitch command callbacks (user interactions with display)
        self.esp32_controller.on_objective_slot_command(self._handleObjectiveSlotCommand)
        self.esp32_controller.on_motor_command(self._handleMotorCommand)
        self.esp32_controller.on_motor_xy_command(self._handleMotorXYCommand)
        self.esp32_controller.on_led_command(self._handleLEDCommand)
        self.esp32_controller.on_pwm_command(self._handlePWMCommand)
        self.esp32_controller.on_snap_image_command(self._handleSnapImageCommand)
        
        # ESP32 -> ImSwitch status callbacks (for monitoring)
        self.esp32_controller.on_connection_changed(self._handleConnectionChanged)
        
    def _handleObjectiveSlotCommand(self, data):
        """Handle objective slot change command from ESP32 display"""
        try:
            slot = data.get('slot', 1)
            self._logger.info(f"ESP32: User selected objective slot {slot}")
            # Emit signal to switch objective
            self._commChannel.sigToggleObjective.emit(slot)
        except Exception as e:
            self._logger.error(f"Error handling objective slot command: {e}")
            
    def _handleMotorCommand(self, data):
        """Handle individual motor movement command from ESP32 display"""
        try:
            if not self.positioner:
                return
                
            motor = data.get('motor', 0)
            speed = data.get('speed', 0)
            self._logger.info(f"ESP32: User set motor {motor} to speed {speed}")
            
            # Map motor ID to axis (assuming 0=X, 1=Y, 2=Z, 3=A)
            axis_map = {0: 'X', 1: 'Y', 2: 'Z', 3: 'A'}
            axis = axis_map.get(motor, 'X')
            
            if hasattr(self.positioner, 'move') and axis in self.positioner.axes:
                # Convert speed to relative movement
                # Note: This is a simplified mapping - actual implementation may need calibration
                if speed != 0:
                    step_size = speed / 1000.0  # Convert speed to step size
                    self.positioner.move(step_size, axis, is_absolute=False, is_blocking=False)
                    
        except Exception as e:
            self._logger.error(f"Error handling motor command: {e}")
            
    def _handleMotorXYCommand(self, data):
        """Handle XY joystick movement command from ESP32 display"""
        try:
            if not self.positioner:
                return
                
            speed_x = data.get('speedX', 0)
            speed_y = data.get('speedY', 0)
            self._logger.info(f"ESP32: User moved XY joystick: X={speed_x}, Y={speed_y}")
            
            # Convert speeds to relative movements
            if speed_x != 0 and 'X' in self.positioner.axes:
                step_x = speed_x / 1000.0
                self.positioner.move(step_x, 'X', is_absolute=False, is_blocking=False)
                
            if speed_y != 0 and 'Y' in self.positioner.axes:
                step_y = speed_y / 1000.0 
                self.positioner.move(step_y, 'Y', is_absolute=False, is_blocking=False)
                
        except Exception as e:
            self._logger.error(f"Error handling XY motor command: {e}")
            
    def _handleLEDCommand(self, data):
        """Handle LED control command from ESP32 display"""
        try:
            if not self.ledMatrix:
                return
                
            enabled = data.get('enabled', False)
            r = data.get('r', 0)
            g = data.get('g', 0)
            b = data.get('b', 0)
            self._logger.info(f"ESP32: User changed LED: enabled={enabled}, RGB=({r}, {g}, {b})")
            
            if hasattr(self.ledMatrix, 'setAll'):
                if enabled:
                    # Set LED matrix to specified color (normalized to 0-1 range)
                    color = (r/255.0, g/255.0, b/255.0)
                    self.ledMatrix.setAll(color)
                else:
                    # Turn off LEDs
                    self.ledMatrix.setAll((0, 0, 0))
                    
        except Exception as e:
            self._logger.error(f"Error handling LED command: {e}")
            
    def _handlePWMCommand(self, data):
        """Handle PWM/laser intensity command from ESP32 display"""
        try:
            channel = data.get('channel', 0)
            value = data.get('value', 0)
            self._logger.info(f"ESP32: User set PWM channel {channel} to {value}")
            
            # Map PWM channel to laser (assuming channel 1-4 maps to laser indices)
            if channel > 0 and channel <= len(self.lasers):
                laser_name = self.lasers[channel - 1]
                laser_manager = self._master.lasersManager[laser_name]
                
                if hasattr(laser_manager, 'setValue'):
                    # Convert 0-1024 range to laser's value range
                    max_value = getattr(laser_manager, 'valueRangeMax', 1024)
                    normalized_value = int((value / 1024.0) * max_value)
                    laser_manager.setValue(normalized_value)
                    
        except Exception as e:
            self._logger.error(f"Error handling PWM command: {e}")
            
    def _handleSnapImageCommand(self, data):
        """Handle image capture command from ESP32 display"""
        try:
            self._logger.info("ESP32: User pressed snap button")
            # Trigger image capture
            self._commChannel.sigSnapImg.emit()
            
            # Optionally send the captured image back to ESP32
            if self.detector and hasattr(self.detector, 'getLatestFrame'):
                try:
                    latest_frame = self.detector.getLatestFrame()
                    if latest_frame is not None and self.esp32_controller:
                        # Send image to ESP32 display
                        timestamp = time.strftime("%H:%M:%S")
                        self.esp32_controller.send_image(latest_frame, f"Snap {timestamp}")
                except Exception as e:
                    self._logger.error(f"Error sending image to ESP32: {e}")
                    
        except Exception as e:
            self._logger.error(f"Error handling snap command: {e}")
            
    def _handleConnectionChanged(self, data):
        """Handle ESP32 connection status changes"""
        connected = data.get('connected', False)
        self._connected = connected
        status = "Connected" if connected else "Disconnected"
        self._logger.info(f"ESP32 InfoScreen: {status}")
        
        # Update widget status
        if not IS_HEADLESS:
            self._widget.setConnectionStatus(self._connected)

    def _connectImSwitchSignals(self):
        """Connect ImSwitch signals to send updates to ESP32 display"""
        if not self._connected or not self.esp32_controller:
            return
            
        # Connect to relevant signals to update ESP32 display
        try:
            # Update LED status when laser/LED state changes
            if hasattr(self._commChannel, 'sigUpdateImage'):
                self._commChannel.sigUpdateImage.connect(self._onImageUpdate)
                
            # Update position when stage moves  
            if hasattr(self._commChannel, 'sigUpdateMotorPosition'):
                self._commChannel.sigUpdateMotorPosition.connect(self._onMotorPositionUpdate)
                
            # Update objective slot when changed
            if hasattr(self._commChannel, 'sigToggleObjective'): 
                self._commChannel.sigToggleObjective.connect(self._onObjectiveSlotUpdate)
                
        except Exception as e:
            self._logger.error(f"Error connecting ImSwitch signals: {e}")
            
    def _onImageUpdate(self, detectorName, image, *args):
        """Send updated image to ESP32 display"""
        try:
            if self.esp32_controller and self._connected and image is not None:
                timestamp = time.strftime("%H:%M:%S")
                self.esp32_controller.send_image(image, f"Live {timestamp}")
        except Exception as e:
            self._logger.error(f"Error sending image update to ESP32: {e}")
            
    def _onMotorPositionUpdate(self, positions):
        """Send motor position update to ESP32 display"""
        try:
            if self.esp32_controller and self._connected:
                # Update sample position based on X,Y coordinates (normalize to 0-1)
                if len(positions) >= 2:
                    # Simple normalization - may need calibration for actual setup
                    x_norm = min(max((positions[0] + 50) / 100.0, 0), 1)  # Assuming ±50mm range
                    y_norm = min(max((positions[1] + 50) / 100.0, 0), 1)
                    self.esp32_controller.update_sample_position(x_norm, y_norm)
        except Exception as e:
            self._logger.error(f"Error sending position update to ESP32: {e}")
            
    def _onObjectiveSlotUpdate(self, slot):
        """Send objective slot update to ESP32 display"""
        try:
            if self.esp32_controller and self._connected:
                self.esp32_controller.set_objective_slot(slot)
        except Exception as e:
            self._logger.error(f"Error sending objective update to ESP32: {e}")
            
    def _connectWidgetSignals(self):
        """Connect widget button signals"""
        if not IS_HEADLESS and hasattr(self._widget, 'connectButton'):
            self._widget.connectButton.clicked.connect(self._onConnectClicked)
            self._widget.disconnectButton.clicked.connect(self._onDisconnectClicked)
            self._widget.testLEDButton.clicked.connect(self._onTestLEDClicked)
            self._widget.sendTestImageButton.clicked.connect(self._onSendTestImageClicked)
            
            # Update initial widget status
            self._widget.setConnectionStatus(self._connected)
    
    def _onConnectClicked(self):
        """Handle connect button click"""
        if not self._connected:
            self._initializeESP32Connection()
            if not IS_HEADLESS:
                self._widget.setConnectionStatus(self._connected)
                
    def _onDisconnectClicked(self):
        """Handle disconnect button click"""
        if self._connected and self.esp32_controller:
            self.esp32_controller.disconnect()
            self._connected = False
            if not IS_HEADLESS:
                self._widget.setConnectionStatus(self._connected)
                
    def _onTestLEDClicked(self):
        """Handle test LED button click"""
        if self._connected and self.esp32_controller:
            # Send red LED test
            self.esp32_controller.set_led(True, 255, 0, 0)
            # Turn off after 2 seconds
            threading.Timer(2.0, lambda: self.esp32_controller.set_led(False)).start()
            
    def _onSendTestImageClicked(self):
        """Handle send test image button click"""
        if self._connected and self.esp32_controller:
            try:
                # Create a simple test image
                import numpy as np
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                # Add some colored patterns
                test_image[20:40, 20:40] = [255, 0, 0]  # Red square
                test_image[60:80, 20:40] = [0, 255, 0]  # Green square
                test_image[20:40, 60:80] = [0, 0, 255]  # Blue square
                test_image[60:80, 60:80] = [255, 255, 0]  # Yellow square
                
                timestamp = time.strftime("%H:%M:%S")
                self.esp32_controller.send_image(test_image, f"Test {timestamp}")
                self._logger.info("Sent test image to ESP32")
            except Exception as e:
                self._logger.error(f"Error sending test image: {e}")
            
    def finalize(self):
        """Clean up ESP32 connection on shutdown"""
        if self.esp32_controller and self._connected:
            self.esp32_controller.disconnect()
            self._logger.info("Disconnected from ESP32 InfoScreen")