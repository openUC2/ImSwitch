
import time
import threading
import collections
import logging

from imswitch.imcommon.framework import Signal, Timer
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import APIExport

class ESP32InfoScreenController(ImConWidgetController):
    """ Controller for ESP32 InfoScreen display integration """

    sigImageReceived = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize ESP32 serial controller
        self.esp32_controller = None
        self._connected = False
        
        # Rate limiting for ESP32 communication (0.5s minimum interval)
        self._last_update_time = 0
        self._update_rate_limit = 0.5  # seconds
        self._rate_limit_timer = Timer()
        self._rate_limit_timer.setSingleShot(True)
        self._pending_updates = {}
        
        # Recursion prevention
        self._in_objective_update = False
        
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
        
        # Map lasers to ESP32 PWM channels (1-4 are typical ESP32 PWM channels)
        # This creates a mapping from laser name to ESP32 channel number
        self.laser_to_channel_map = {}
        self.channel_to_laser_map = {}
        
        # Default channel mapping (can be overridden by setup configuration)
        default_channels = [1, 2, 3, 4]  # ESP32 PWM channels 1-4
        
        # Create bidirectional mapping between lasers and channels
        for i, laser_name in enumerate(self.lasers[:len(default_channels)]):
            channel = default_channels[i]
            self.laser_to_channel_map[laser_name] = channel
            self.channel_to_laser_map[channel] = laser_name
            self._logger.info(f"Mapped laser '{laser_name}' to ESP32 PWM channel {channel}")
        
        # Warn if we have more lasers than available channels
        if len(self.lasers) > len(default_channels):
            unmapped_lasers = self.lasers[len(default_channels):]
            self._logger.warning(f"More lasers than ESP32 channels available. Unmapped lasers: {unmapped_lasers}")
            
        # get LEDMatrix
        allLEDMatrixNames = self._master.LEDMatrixsManager.getAllDeviceNames()
        if len(allLEDMatrixNames) > 0:
            self.ledMatrix = self._master.LEDMatrixsManager[allLEDMatrixNames[0]]
        else:
            self.ledMatrix = None
            self._logger.warning("No LEDMatrix found")
            

        # Initialize ESP32 connection
        self._initializeESP32Connection()
        
        # Connect ImSwitch signals to send updates to ESP32
        self._connectImSwitchSignals()
        

    def _initializeESP32Connection(self):
        """Initialize ESP32 serial connection and setup callbacks"""
        try:
            # Find and connect to ESP32
            from .esp32_infoscreen.uc2_serial_controller import UC2SerialController, find_esp32_port
            
            port = find_esp32_port()
            if not port:
                self._logger.error("No ESP32 device found for InfoScreen")
                return
                
            self.esp32_controller = UC2SerialController(port)
            
            # Register callbacks for ESP32 display interactions
            self._setupESP32Callbacks()
            
            # Connect to ESP32
            if self.esp32_controller.connect():
                self._connected = True
                self._logger.info(f"Connected to ESP32 InfoScreen on port: {port}")
            else:
                self._logger.error("Failed to connect to ESP32 InfoScreen")
                
        except Exception as e:
            self._logger.error(f"Error initializing ESP32 connection: {e}")

    @APIExport()
    def reconnectESP32(self):
        """Reconnect to the ESP32 InfoScreen"""
        if self.esp32_controller:
            self.esp32_controller.disconnect()
        self._initializeESP32Connection()
        self._connectImSwitchSignals()

    @APIExport()
    def getESP32ConnectionStatus(self):
        """Get the current connection status of the ESP32 InfoScreen"""
        if self.esp32_controller:
            return self.esp32_controller.is_connected()
        return False

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
        self.esp32_controller.on_goto_position_command(self._handleGotoPositionCommand)

        # ESP32 -> ImSwitch status callbacks (for monitoring)
        self.esp32_controller.on_connection_changed(self._handleConnectionChanged)
        
    def _handleObjectiveSlotCommand(self, data):
        """Handle objective slot change command from ESP32 display"""
        try:
            slot = data.get('slot', 1)
            self._logger.info(f"ESP32: User selected objective slot {slot}")
            # Emit signal to switch objective - using direct method call to avoid recursion
            if hasattr(self._master, 'objectiveManager'):
                self._master.objectiveManager.setCurrentObjective(slot)
            else:
                # Fallback: emit signal but check to prevent recursion
                if not getattr(self, '_in_objective_update', False):
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
            position = data.get('position', None)
            steps = data.get('steps', None)
            self._logger.info(f"ESP32: User set motor {motor} to speed {speed}, position {position}, steps {steps}")

            # Map motor ID to axis (assuming 0=X, 1=Y, 2=Z, 3=A)
            axis_map = {1: 'X', 2: 'Y', 3: 'Z', 0: 'A'}
            axis = axis_map.get(motor, 'X')
            
            if hasattr(self.positioner, 'move') and axis in self.positioner.axes:

                # If we have position data, use it for relative moves 
                if position is not None:
                    # Position-based movement in ImSwitch units
                    self.positioner.move(position, axis, is_absolute=True, is_blocking=False)
                elif steps is not None:
                    # Step-based movement
                    self.positioner.move(steps, axis, is_absolute=False, is_blocking=False)
                else:
                    # Speed-based movement - continuous motion
                    if hasattr(self.positioner, 'setSpeed'):
                        # Set continuous speed if supported
                        self.positioner.setSpeed(speed, axis)
                        if hasattr(self.positioner, 'moveForever'):
                            # map axis to speed (e.g. axis x => 1 => (0,speed,0,0)), etc
                            # map axis to number A,X,Y,Z=>0,1,2,3 
                            _speed = [0, 0, 0, 0]
                            _speed[axis] = speed
                            self.positioner.moveForever( speed=_speed, is_stop=(abs(speed)!=0))
                    else:
                        # Fallback: convert speed to small relative steps
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
            
            # Handle continuous movement (speed-based)
            for axis, speed in [('X', speed_x), ('Y', speed_y)]:
                if axis in self.positioner.axes:
                    if speed == 0:
                        # Stop movement
                        if hasattr(self.positioner, 'stop'):
                            self.positioner.stop(axis)
                    else:
                        # Continuous movement
                        if hasattr(self.positioner, 'setSpeed') and hasattr(self.positioner, 'moveForever'):
                            self.positioner.setSpeed(abs(speed), axis)
                            direction = 1 if speed > 0 else -1
                            self.positioner.moveForever(axis, direction)
                        else:
                            # Fallback: convert to relative movement
                            step_size = speed / 1000.0
                            self.positioner.move(step_size, axis, is_absolute=False, is_blocking=False)
                
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
            
            # Map ESP32 channel to laser using our channel mapping
            if channel in self.channel_to_laser_map:
                laser_name = self.channel_to_laser_map[channel]
                laser_manager = self._master.lasersManager[laser_name]
                
                # Check if laser needs to be enabled first
                if hasattr(laser_manager, 'enabled') and not laser_manager.enabled:
                    if value > 0:  # Only enable if we're setting a positive value
                        laser_manager.setEnabled(True)
                        self._logger.info(f"Enabled laser {laser_name} (channel {channel})")
                
                if hasattr(laser_manager, 'setValue'):
                    # Convert 0-1024 range to laser's value range
                    max_value = getattr(laser_manager, 'valueRangeMax', 1024)
                    normalized_value = int((value / 1024.0) * max_value)
                    self._logger.debug(f"Set laser {laser_name} (channel {channel}) to {normalized_value}/{max_value}")
                    self._commChannel.sigUpdateLaserValue.emit(laser_name, normalized_value)
                    # If value is 0, we can optionally disable the laser
                    if value == 0 and hasattr(laser_manager, 'setEnabled'):
                        laser_manager.setEnabled(False)
                        self._logger.info(f"Disabled laser {laser_name} (channel {channel})")
            else:
                self._logger.warning(f"ESP32 PWM channel {channel} not mapped to any laser. Available channels: {list(self.channel_to_laser_map.keys())}")
                    
        except Exception as e:
            self._logger.error(f"Error handling PWM command: {e}")
    
    def _handleGotoPositionCommand(self, data):
        """Handle go-to absolute positions for xyz coordinates"""
        try:
            x = data.get('x', None)
            y = data.get('y', None)
            z = data.get('z', None)
            self._logger.info(f"ESP32: User requested goto position x={x}, y={y}, z={z}")
            # Trigger the stage movement
            if x is not None:  self.positioner.move(x, axis="X", is_absolute=True, is_blocking=False)
            if y is not None:  self.positioner.move(y, axis="Y", is_absolute=True, is_blocking=False)
            if z is not None:  self.positioner.move(z, axis="Z", is_absolute=True, is_blocking=False)
        except Exception as e:
            self._logger.error(f"Error handling goto position command: {e}")

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
        

    def _connectImSwitchSignals(self):
        """Connect ImSwitch signals to send updates to ESP32 display"""
        if not self._connected or not self.esp32_controller:
            return
            
        # Connect to relevant signals to update ESP32 display
        try:
            # Update position when stage moves  
            if hasattr(self._commChannel, 'sigUpdateMotorPosition'):
                self._commChannel.sigUpdateMotorPosition.connect(self._onMotorPositionUpdate)
                
            # Update objective slot when changed - prevent recursion
            if hasattr(self._commChannel, 'sigToggleObjective'): 
                self._commChannel.sigToggleObjective.connect(self._onObjectiveSlotUpdate)
                
            # Connect to laser manager signals for PWM updates
            try:
                for laser_name in self.lasers:
                    laser_manager = self._master.lasersManager[laser_name]
                    if hasattr(laser_manager, 'sigLaserValueChanged'):
                        laser_manager.sigLaserValueChanged.connect(
                            lambda value, name=laser_name: self._onLaserValueUpdate(name, value)
                        )
                    if hasattr(laser_manager, 'sigLaserEnabledChanged'):
                        laser_manager.sigLaserEnabledChanged.connect(
                            lambda enabled, name=laser_name: self._onLaserEnabledUpdate(name, enabled)
                        )
            except Exception as e:
                self._logger.warning(f"Could not connect to laser signals: {e}")
                
            # Connect to LED Matrix manager signals
            try:
                if self.ledMatrix and hasattr(self.ledMatrix, 'sigLEDMatrixValueChanged'):
                    self.ledMatrix.sigLEDMatrixValueChanged.connect(self._onLEDMatrixUpdate)
            except Exception as e:
                self._logger.warning(f"Could not connect to LED matrix signals: {e}")
            
        except Exception as e:
            self._logger.error(f"Error connecting ImSwitch signals: {e}")
            
    def _onMotorPositionUpdate(self, positions):
        """Send motor position update to ESP32 display with rate limiting"""
        try:
            if not self.esp32_controller or not self._connected:
                return
                
            current_time = time.time()
            if current_time - self._last_update_time < self._update_rate_limit:
                # Store pending update for rate-limited sending
                self._pending_updates['motor_position'] = positions
                if not self._rate_limit_timer.isActive():
                    remaining_time = self._update_rate_limit - (current_time - self._last_update_time)
                    self._rate_limit_timer.timeout.connect(lambda: self._sendPendingUpdate('motor_position'))
                    self._rate_limit_timer.start(int(remaining_time * 1000))
                return
            
            self._sendMotorPositionUpdate(positions)
            
        except Exception as e:
            self._logger.error(f"Error handling position update: {e}")
            
    def _sendMotorPositionUpdate(self, positions):
        """Actually send the motor position update"""
        try:
            # Update sample position based on X,Y coordinates (normalize to 0-1)
            # {"type":"position_update","data":{"x":1.2,"y":0,"z":0.1}}
            xposition = positions.get(self.positioner.name)["X"]
            yposition = positions.get(self.positioner.name)["Y"]
            zposition = positions.get(self.positioner.name)["Z"]
            # Simple normalization - may need calibration for actual setup
            self.esp32_controller.update_sample_position(xposition, yposition, zposition)
            self._last_update_time = time.time()
        except Exception as e:
            self._logger.error(f"Error sending position update to ESP32: {e}")
            
    def _onObjectiveSlotUpdate(self, slot):
        """Send objective slot update to ESP32 display"""
        try:
            # Prevent recursion when this controller triggers the signal
            if getattr(self, '_in_objective_update', False):
                return
                
            self._in_objective_update = True
            if self.esp32_controller and self._connected:
                self.esp32_controller.set_objective_slot(slot)
            self._in_objective_update = False
        except Exception as e:
            self._logger.error(f"Error sending objective update to ESP32: {e}")
            self._in_objective_update = False
            
    def _onLaserValueUpdate(self, laser_name, value):
        """Send laser value update to ESP32 display"""
        try:
            if self.esp32_controller and self._connected:
                # Map laser name to ESP32 channel using our mapping
                if laser_name in self.laser_to_channel_map:
                    channel = self.laser_to_channel_map[laser_name]
                    # Convert value to 0-1024 range expected by ESP32
                    laser_manager = self._master.lasersManager[laser_name]
                    max_value = getattr(laser_manager, 'valueRangeMax', 1024)
                    normalized_value = int((value / max_value) * 1024)
                    self.esp32_controller.set_pwm(channel, normalized_value)
                    self._logger.debug(f"Updated ESP32 channel {channel} for laser {laser_name}: {normalized_value}/1024")
                else:
                    self._logger.warning(f"Laser {laser_name} not mapped to any ESP32 channel. Available mappings: {self.laser_to_channel_map}")
        except Exception as e:
            self._logger.error(f"Error sending laser value update to ESP32: {e}")
            
    def _onLaserEnabledUpdate(self, laser_name, enabled):
        """Send laser enabled state update to ESP32 display"""
        try:
            if self.esp32_controller and self._connected:
                # If disabled, set PWM to 0 using our channel mapping
                if not enabled and laser_name in self.laser_to_channel_map:
                    channel = self.laser_to_channel_map[laser_name]
                    self.esp32_controller.set_pwm(channel, 0)
                    self._logger.debug(f"Disabled laser {laser_name} on ESP32 channel {channel}")
        except Exception as e:
            self._logger.error(f"Error sending laser enabled update to ESP32: {e}")
            
    def _onLEDMatrixUpdate(self, color_data):
        """Send LED matrix update to ESP32 display"""
        try:
            if self.esp32_controller and self._connected:
                # Extract RGB values from color data
                if isinstance(color_data, (tuple, list)) and len(color_data) >= 3:
                    r, g, b = int(color_data[0] * 255), int(color_data[1] * 255), int(color_data[2] * 255)
                    enabled = any(c > 0 for c in color_data[:3])
                    self.esp32_controller.set_led(enabled, r, g, b)
        except Exception as e:
            self._logger.error(f"Error sending LED matrix update to ESP32: {e}")
            
    def _sendPendingUpdate(self, update_type):
        """Send pending rate-limited updates"""
        try:
            if update_type in self._pending_updates:
                if update_type == 'motor_position':
                    self._sendMotorPositionUpdate(self._pending_updates[update_type])
                del self._pending_updates[update_type]
        except Exception as e:
            self._logger.error(f"Error sending pending update: {e}")
        finally:
            # Disconnect the timeout signal
            self._rate_limit_timer.timeout.disconnect()
            

    def _onConnectClicked(self):
        """Handle connect button click"""
        if not self._connected:
            self._initializeESP32Connection()

    def _onDisconnectClicked(self):
        """Handle disconnect button click"""
        if self._connected and self.esp32_controller:
            self.esp32_controller.disconnect()
            self._connected = False

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
        """Clean up ESP32 connection and timers on shutdown"""
        try:
            if self._rate_limit_timer and self._rate_limit_timer.isActive():
                self._rate_limit_timer.stop()
        except Exception as e:
            self._logger.warning(f"Error stopping rate limit timer: {e}")
            
        if self.esp32_controller and self._connected:
            self.esp32_controller.disconnect()
            self._logger.info("Disconnected from ESP32 InfoScreen")