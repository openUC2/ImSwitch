import os
import time
import datetime
import subprocess
import platform
import numpy as np
import cv2
import json
import shutil
import struct
from datetime import timedelta
from threading import Thread, Event
from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcommon.framework import Signal
from ..basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS
from typing import Dict, List, Union, Optional
from fastapi import Response
import io

# Lepmon hardware dependencies
'''
pip install RPi.GPIO
pip install luma.oled
pip install smbus2
'''
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("RPi.GPIO not available - running in simulation mode")

try:
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import sh1106
    from PIL import ImageFont, ImageDraw, Image
    HAS_OLED = True
except ImportError:
    HAS_OLED = False
    print("OLED libraries not available - running in simulation mode")

try:
    import smbus
    HAS_I2C = True
except ImportError:
    HAS_I2C = False
    print("I2C libraries (smbus) not available - running in simulation mode")

# Lepmon Hardware Configuration (from LepmonOS GPIO_Setup.py)
LED_PINS = {
    'gelb': 22,    # GPIO 22 for yellow LED
    'blau': 6,     # GPIO 6 for blue LED  
    'rot': 17      # GPIO 17 for red LED
}

BUTTON_PINS = {
    'oben': 23,     # GPIO 23 for up button
    'unten': 24,    # GPIO 24 for down button
    'rechts': 8,    # GPIO 8 for right button
    'enter': 7      # GPIO 7 for enter button
}

# OLED Display Configuration  
OLED_I2C_PORT = 1
OLED_I2C_ADDRESS = 0x3C

# I2C Sensor Configuration
I2C_BUS = 1  # I2C bus number
SENSOR_ADDRESSES = {
    "temperature": 0x48,  # Example temperature sensor address
    "humidity": 0x40,     # Example humidity sensor address  
    "pressure": 0x77      # Example pressure sensor address
}

# We map FastAPI GET -> @APIExport()
# and FastAPI POST -> @APIExport(requestType="POST")

# Minimal default config matching the React needs:
#  - exposureTime, gain, timelapsePeriod, storagePath, isRunning
#  - isFocusMode, freeSpace, currentImageCount, etc.
DEFAULT_CONFIG = {
    "exposureTime": 100.0,
    "gain": 0.0,
    "timelapsePeriod": 60,  # in seconds
    "storagePath": "/mnt/usb_drive",  # example
    "isRunning": False,
    "wasRunning": False,
    "numberOfFrames": 10,
    "experimentName": "LepMonTest",
    "axislepmon": "Z",
    "axisFocus": "X",
    "isRecordVideo": True,
    "fileFormat": "JPG",
    "frameRate": 1,
    "delayTimeAfterRestart": 1.0,
    "time": "00:00:00",
    "date": "2021-01-01",
}

class LepmonController(LiveUpdatedController):
    """
    Example Lepmon Controller which provides:
    - GET endpoints to retrieve status (isRunning, imageCount, freeSpace, etc.)
    - POST endpoints to start/stop an experiment, set camera exposure/gain, etc.
    - A background thread capturing frames (lepmonExperimentThread).
    - Minimal code to illustrate a FastAPI-like structure using the @APIExport decorator.
    """

    # Signals -> used to broadcast to WebSocket in the background
    sigImagesTaken = Signal(int)      # e.g. "imageCounter" WS message
    sigIsRunning = Signal(bool)       # e.g. "isRunning" WS message
    sigFocusSharpness = Signal(float) # e.g. "focusSharpness" WS message
    temperatureUpdate = Signal(dict)  # e.g. "temperatureUpdate" WS message
    sigLCDDisplayUpdate = Signal(str) # LCD display updates
    sigButtonPressed = Signal(dict)   # Button press events
    sigLightStateChanged = Signal(dict) # Light state changes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # Mock reading config
        self._master.LepmonManager.defaultConfig = DEFAULT_CONFIG
        self.mExperimentParameters = self._master.LepmonManager.defaultConfig

        self.is_measure = False
        self.imagesTaken = 0

        # Detector (camera)
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detectorlepmonCam = self._master.detectorsManager[allDetectorNames[0]]

        # Possibly set default exposure/gain
        self.changeAutoExposureTime("auto")

        # If was running, start automatically
        if self.mExperimentParameters["wasRunning"]:
            self._logger.debug("Resuming experiment because 'wasRunning' was True.")

        # start thread that pulls sensor data
        self.sensorThread = Thread(target=self._pullSensorData, args=(10,))
        self.sensorThread.start()

        # initialize temperature and humidity
        self.innerTemp = np.round(np.random.uniform(20, 25), 2)
        self.outerTemp = np.round(np.random.uniform(15, 20), 2)
        self.humidity = np.round(np.random.uniform(40, 50), 2)

        # Initialize hardware control states
        self.lightStates = {}  # Track light on/off states
        self.lcdDisplay = {"line1": "", "line2": "", "line3": "", "line4": ""}  # LCD display content
        self.buttonStates = {"oben": False, "unten": False, "rechts": False, "enter": False}
        
        # Initialize timing configurations
        self.timingConfig = {
            "acquisitionInterval": 60,  # seconds between acquisitions
            "stabilizationTime": 5,     # seconds to wait for stabilization
            "preAcquisitionDelay": 2,   # seconds before each acquisition
            "postAcquisitionDelay": 1   # seconds after each acquisition
        }

        # Initialize Lepmon hardware directly
        self._initializeLepmonHardware()

        # LepmonOS state variables
        self.lepmon_config = {}
        self.sensor_data = {}
        self.version = "V1.0"
        self.date = "2024"
        self.serial_number = "LEPMON001"
        
        # LepmonOS experiment control
        self.stop_event = Event()
        self.uv_led_active = False
        self.visible_led_active = False
        self.experiment_start_time = None
        self.experiment_end_time = None
        self.lepiled_end_time = None
        
        # LepmonOS menu system
        self.menu_open = False
        self.current_menu_state = "main"
        self.hmi_stop_event = Event()  # Event to stop continuous button monitoring
        
        # Initialize LepmonOS system
        self._initializeLepmonOS()
        
        # Start continuous button monitoring thread for HMI
        self.buttonMonitoringThread = Thread(target=self._continuous_button_monitoring, daemon=True)
        self.buttonMonitoringThread.start()
        
        # Automatically start HMI menu system
        self._open_hmi_menu()

    def _initializeLepmonHardware(self):
        """Initialize Lepmon hardware directly (GPIO LEDs, OLED display, buttons)"""
        try:
            self._logger.info("Initializing Lepmon hardware directly")
            
            # Initialize GPIO if available
            if HAS_GPIO:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # Setup LED pins as outputs
                for color, pin in LED_PINS.items():
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.HIGH)  # LEDs off initially
                    time.sleep(0.1)
                    GPIO.output(pin, GPIO.LOW)  # LEDs off initially
                    self.lightStates[color] = False                
                # Setup PWM for LED dimming
                self.led_pwm = {}
                for color, pin in LED_PINS.items():
                    pwm = GPIO.PWM(pin, 1000)  # 1kHz PWM
                    pwm.start(1)  # Turn on
                    time.sleep(0.1)
                    pwm.start(0)  # Start at 0% duty cycle (off)
                    self.led_pwm[color] = pwm
                    
                # Setup button pins as inputs with pull-up resistors
                for button, pin in BUTTON_PINS.items():
                    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                    self.buttonStates[button] = False
                    
                self._logger.info("GPIO initialized successfully")
            else:
                self.led_pwm = {}
                # Initialize LED states for simulation
                for color in LED_PINS.keys():
                    self.lightStates[color] = False
                self._logger.warning("GPIO not available - using simulation mode")
                
            # Initialize OLED display if available
            if HAS_OLED:
                try:
                    display_interface = i2c(port=OLED_I2C_PORT, address=OLED_I2C_ADDRESS)
                    self.oled = sh1106(display_interface)
                    
                    # Try to load font
                    try:
                        font_path = os.path.join(os.path.dirname(__file__), 'FreeSans.ttf')
                        self.oled_font = ImageFont.truetype(font_path, 14)
                    except (OSError, IOError):
                        try:
                            self.oled_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                        except (OSError, IOError):
                            self.oled_font = ImageFont.load_default()
                            
                    self._logger.info("OLED display initialized successfully")
                except Exception as e:
                    self.oled = None
                    self.oled_font = None
                    self._logger.warning(f"OLED initialization failed: {e}")
            else:
                self.oled = None
                self.oled_font = None
                self._logger.warning("OLED libraries not available - using simulation mode")
                
            # Initialize I2C for sensors if available
            if HAS_I2C:
                try:
                    self.i2c_bus = smbus.SMBus(I2C_BUS)
                    self._logger.info("I2C bus initialized for sensor communication")
                except Exception as e:
                    self.i2c_bus = None
                    self._logger.warning(f"I2C bus initialization failed: {e}")
            else:
                self.i2c_bus = None
                self._logger.warning("I2C libraries not available - using simulation mode")
                
        except Exception as e:
            self._logger.error(f"Lepmon hardware initialization failed: {e}")
            # Set simulation mode
            self.led_pwm = {}
            self.oled = None
            self.oled_font = None
            self.i2c_bus = None
            for color in LED_PINS.keys():
                self.lightStates[color] = False

    def _cleanupHardware(self):
        """Cleanup GPIO and hardware resources"""
        try:
            # Stop continuous button monitoring
            if hasattr(self, 'hmi_stop_event'):
                self.hmi_stop_event.set()
            
            if HAS_GPIO:
                # Stop PWM and cleanup GPIO
                for pwm in self.led_pwm.values():
                    pwm.stop()
                GPIO.cleanup()
                self._logger.info("GPIO cleaned up successfully")
            
            if HAS_I2C and self.i2c_bus:
                # Close I2C bus
                self.i2c_bus.close()
                self._logger.info("I2C bus closed successfully")
                
        except Exception as e:
            self._logger.warning(f"Hardware cleanup failed: {e}")

    def __del__(self):
        """Destructor to ensure hardware cleanup"""
        self._cleanupHardware()

    def _initializeLepmonOS(self):
        """Initialize LepmonOS system components - equivalent to 00_start_up.py"""
        try:
            self._logger.info("Starting LepmonOS initialization sequence")
            
            # Equivalent to dim_down() and turn_off_led("blau")
            self._dim_down()
            self._turn_off_led("blau")
            
            # Display startup sequence
            self._display_startup_sequence()
            
            # Read configuration
            self._read_lepmon_config()
            
            # Initialize system components
            self._initialize_system_components()
            
            # Calculate sun and power times
            self._calculate_times()
            
            self._logger.info("LepmonOS initialization completed successfully")
            
        except Exception as e:
            self._logger.error(f"LepmonOS initialization failed: {e}")

    # ---------------------- LepmonOS Core Functions (from 00_start_up.py) ---------------------- #

    def _dim_down(self):
        """Equivalent to utils.Lights.dim_down() - dim all LEDs to minimum"""
        try:
            for color in LED_PINS.keys():
                self._dim_led(color, 0)  # Set to 0% brightness
            self._logger.debug("Dimmed down all lights")
        except Exception as e:
            self._logger.warning(f"Could not dim down lights: {e}")

    def _dim_led(self, color: str, brightness: int):
        """Dim the specified LED to given brightness (0-100)"""
        try:
            if color in LED_PINS:
                if HAS_GPIO and color in self.led_pwm:
                    self.led_pwm[color].ChangeDutyCycle(brightness)
                # Update state based on brightness
                self.lightStates[color] = brightness > 0
                self._logger.debug(f"Set LED {color} to {brightness}% brightness")
        except Exception as e:
            self._logger.warning(f"Could not dim LED {color}: {e}")

    def _turn_off_led(self, led_name: str):
        """Equivalent to utils.GPIO_Setup.turn_off_led()"""
        try:
            if led_name in LED_PINS:
                self._dim_led(led_name, 0)  # Set to 0% brightness
                self.lightStates[led_name] = False
                self.sigLightStateChanged.emit({"lightName": led_name, "state": False})
                self._logger.debug(f"Turned off LED: {led_name}")
        except Exception as e:
            self._logger.warning(f"Could not turn off LED {led_name}: {e}")

    def _turn_on_led(self, led_name: str, brightness: int = 100):
        """Equivalent to utils.GPIO_Setup.turn_on_led()"""
        try:
            if led_name in LED_PINS:
                self._dim_led(led_name, brightness)  # Set to specified brightness
                self.lightStates[led_name] = True
                self.sigLightStateChanged.emit({"lightName": led_name, "state": True})
                self._logger.debug(f"Turned on LED: {led_name} at {brightness}%")
        except Exception as e:
            self._logger.warning(f"Could not turn on LED {led_name}: {e}")

    def _display_text(self, line1: str = "", line2: str = "", line3: str = "", line4: str = "", duration: float = 0):
        """Equivalent to utils.OLED_panel.display_text() - direct OLED control"""
        try:
            # Update internal state
            self.lcdDisplay["line1"] = line1[:20]
            self.lcdDisplay["line2"] = line2[:20] 
            self.lcdDisplay["line3"] = line3[:20]
            self.lcdDisplay["line4"] = line4[:20]
            
            # Direct OLED display if available
            if self.oled and HAS_OLED:
                with canvas(self.oled) as draw:
                    # Clear background
                    draw.rectangle(self.oled.bounding_box, outline="white", fill="black")
                    # Draw text lines (OLED typically 3 lines, we'll adapt)
                    if line1:
                        draw.text((5, 5), line1[:20], font=self.oled_font, fill="white")
                    if line2:
                        draw.text((5, 20), line2[:20], font=self.oled_font, fill="white")
                    if line3:
                        draw.text((5, 35), line3[:20], font=self.oled_font, fill="white")
                    # line4 may not fit on small OLED, skip or combine
            
            # Emit signal for UI updates
            display_content = f"{line1}\n{line2}\n{line3}\n{line4}"
            self.sigLCDDisplayUpdate.emit(display_content)
            
            # Wait if duration specified
            if duration > 0:
                time.sleep(duration)
                
            self._logger.debug(f"Display updated: {line1} | {line2} | {line3}")
        except Exception as e:
            self._logger.warning(f"Could not update display: {e}")

    def _display_image(self, image_path: str):
        """Equivalent to utils.OLED_panel.display_image() - direct OLED control"""
        try:
            if self.oled and HAS_OLED and os.path.exists(image_path):
                # Load and convert image to 1-bit mode for OLED
                logo = Image.open(image_path).convert("1")
                with canvas(self.oled) as draw:
                    draw.rectangle(self.oled.bounding_box, outline="white", fill="black")
                    draw.bitmap((0, 0), logo, fill="white")
                self._logger.info(f"Displayed image on OLED: {image_path}")
            else:
                # Fallback for simulation mode
                self._logger.info(f"Displaying image (simulation): {image_path}")
                
            # Emit signal for UI
            self.sigLCDDisplayUpdate.emit(f"IMAGE: {os.path.basename(image_path)}")
        except Exception as e:
            self._logger.warning(f"Could not display image {image_path}: {e}")

    def _display_startup_sequence(self):
        """Display startup sequence like LepmonOS"""
        try:
            # Display manual link
            self._display_text("Beachte", "Anleitung", "", "", 3)
            
            # Logo startup sequence (simulated)
            for i in range(1, 10):
                self._display_text("LepMon", f"Loading {i}/9", "", "", 1)
            
            self._display_text("Willkommen", f"Version {self.version}", "", "", 3)
            
        except Exception as e:
            self._logger.warning(f"Startup sequence display failed: {e}")

    def _read_lepmon_config(self):
        """Read configuration from LepmonOS config files"""
        try:
            # Simulate reading from JSON config
            self.lepmon_config = {
                "software": {
                    "version": self.version,
                    "date": self.date
                },
                "general": {
                    "serielnumber": self.serial_number
                },
                "capture_mode": {
                    "dusk_treshold": 50,
                    "interval": 60,
                    "initial_exposure": 100
                }
            }
            self._logger.debug(f"Loaded LepmonOS configuration")
        except Exception as e:
            self._logger.warning(f"Could not read LepmonOS config: {e}")

    def _initialize_system_components(self):
        """Initialize system components like LepmonOS"""
        try:
            # Equivalent to erstelle_ordner(), initialisiere_logfile()
            self._logger.info("System components initialized")
            
            # Send startup message (equivalent to send_lora)
            self._send_lora("Starte Lepmon Software")
            
        except Exception as e:
            self._logger.warning(f"System component initialization failed: {e}")

    def _send_lora(self, message: str):
        """Equivalent to utils.lora.send_lora()"""
        try:
            self._logger.info(f"LoRa message: {message}")
            # Could emit signal for actual LoRa transmission
        except Exception as e:
            self._logger.warning(f"LoRa transmission failed: {e}")

    def _button_pressed(self, button_name: str) -> bool:
        """Equivalent to utils.GPIO_Setup.button_pressed() - check if button is pressed"""
        try:
            if button_name not in BUTTON_PINS:
                available_buttons = ", ".join(BUTTON_PINS.keys())
                raise ValueError(f"Invalid button name '{button_name}'. Available: {available_buttons}")
            
            if HAS_GPIO:
                # Button pressed when GPIO reads LOW (pull-up configuration)
                is_pressed = GPIO.input(BUTTON_PINS[button_name]) == GPIO.LOW
                if is_pressed != self.buttonStates[button_name]:
                    self.buttonStates[button_name] = is_pressed
                    if is_pressed:  # Only emit on press, not release
                        self.sigButtonPressed.emit({"buttonName": button_name, "state": is_pressed})
                        self._logger.debug(f"Button {button_name} pressed")
                return is_pressed
            else:
                # Simulation mode - return stored state
                return self.buttonStates[button_name]
                
        except Exception as e:
            self._logger.warning(f"Could not read button {button_name}: {e}")
            return False

    def _read_all_buttons(self) -> Dict[str, bool]:
        """Read all button states"""
        button_states = {}
        for button_name in BUTTON_PINS.keys():
            button_states[button_name] = self._button_pressed(button_name)
        return button_states

    def _simulate_button_press(self, button_name: str):
        """Simulate button press for testing purposes"""
        try:
            if button_name in BUTTON_PINS:
                self.buttonStates[button_name] = True
                self.sigButtonPressed.emit({"buttonName": button_name, "state": True})
                self._logger.debug(f"Simulated button press: {button_name}")
                # Auto-release after short delay
                time.sleep(0.1)
                self.buttonStates[button_name] = False
        except Exception as e:
            self._logger.warning(f"Could not simulate button press {button_name}: {e}")

    def _read_i2c_sensor(self, sensor_name: str) -> Optional[float]:
        """Read data from I2C sensor"""
        try:
            if not HAS_I2C or self.i2c_bus is None:
                # Return simulated data
                if sensor_name == "temperature":
                    return np.round(np.random.uniform(20, 25), 2)
                elif sensor_name == "humidity":
                    return np.round(np.random.uniform(40, 60), 2)
                elif sensor_name == "pressure":
                    return np.round(np.random.uniform(1000, 1020), 2)
                return None
                
            if sensor_name not in SENSOR_ADDRESSES:
                self._logger.warning(f"Unknown sensor: {sensor_name}")
                return None
                
            address = SENSOR_ADDRESSES[sensor_name]
            
            # Basic I2C read - implementation depends on specific sensor
            # This is a generic example that would need to be customized for each sensor type
            data = self.i2c_bus.read_byte(address)
            
            # Convert raw data to meaningful value (sensor-specific conversion)
            if sensor_name == "temperature":
                # Example temperature conversion (sensor-specific)
                return round((data * 0.1) + 15.0, 2)  # Example conversion
            elif sensor_name == "humidity":
                # Example humidity conversion  
                return round((data / 255.0) * 100.0, 2)  # Example conversion
            elif sensor_name == "pressure":
                # Example pressure conversion
                return round(1000.0 + (data * 0.1), 2)  # Example conversion
                
            return float(data)
            
        except Exception as e:
            self._logger.warning(f"Failed to read I2C sensor {sensor_name}: {e}")
            # Return simulated data on error
            if sensor_name == "temperature":
                return np.round(np.random.uniform(20, 25), 2)
            elif sensor_name == "humidity":
                return np.round(np.random.uniform(40, 60), 2)
            elif sensor_name == "pressure":
                return np.round(np.random.uniform(1000, 1020), 2)
            return None

    def _read_all_sensors(self) -> Dict[str, float]:
        """Read all available sensors"""
        sensor_data = {}
        for sensor_name in SENSOR_ADDRESSES.keys():
            value = self._read_i2c_sensor(sensor_name)
            if value is not None:
                sensor_data[sensor_name] = value
        return sensor_data

    def _calculate_times(self):
        """Calculate sun times and power times like LepmonOS"""
        try:
            # Simulate sun time calculation
            now = datetime.datetime.now()
            sunset = now.replace(hour=18, minute=30, second=0)
            sunrise = now.replace(hour=6, minute=30, second=0)
            
            self.experiment_start_time = sunset.strftime('%H:%M:%S')
            self.experiment_end_time = sunrise.strftime('%H:%M:%S')
            self.lepiled_end_time = (sunset + timedelta(hours=6)).strftime('%H:%M:%S')
            
            self._logger.info(f"Sonnenuntergang: {sunset.strftime('%H:%M:%S')}")
            self._logger.info(f"Sonnenaufgang: {sunrise.strftime('%H:%M:%S')}")
            
            self._send_lora(f"Sonnenuntergang: {sunset.strftime('%H:%M:%S')}\nSonnenaufgang: {sunrise.strftime('%H:%M:%S')}")
            
        except Exception as e:
            self._logger.warning(f"Time calculation failed: {e}")

    # ---------------------- LepmonOS HMI Functions (from 02_trap_hmi.py) ---------------------- #

    def _open_hmi_menu(self):
        """Equivalent to the HMI menu system from 02_trap_hmi.py"""
        try:
            self.menu_open = True
            self._turn_on_led("blau")
            self._display_text("Menü bereit", "Tasten nutzen:", "rechts=Fokus", "enter=Menü")
            self._logger.info("HMI menu opened and running continuously")
            
        except Exception as e:
            self._logger.error(f"Failed to open HMI menu: {e}")

    def _continuous_button_monitoring(self):
        """Continuously monitor button presses for menu navigation"""
        self._logger.info("Starting continuous button monitoring thread")
        
        while not self.hmi_stop_event.is_set():
            try:
                # Read all button states continuously
                current_states = self._read_all_buttons()
                
                # Only process if menu is open
                if self.menu_open:
                    # Handle button presses based on current menu state
                    if current_states.get("enter", False) and not self.buttonStates.get("enter", False):
                        self._handle_menu_enter()
                    elif current_states.get("rechts", False) and not self.buttonStates.get("rechts", False):
                        self._handle_focus_menu()
                    elif current_states.get("oben", False) and not self.buttonStates.get("oben", False):
                        self._handle_update_menu()
                    elif current_states.get("unten", False) and not self.buttonStates.get("unten", False):
                        self._handle_navigation_down()
                
                # Update stored button states for edge detection
                self.buttonStates.update(current_states)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self._logger.error(f"Continuous button monitoring error: {e}")
                time.sleep(1)  # Longer delay on error
        
        self._logger.info("Continuous button monitoring thread stopped")

    def _monitor_menu_buttons(self):
        """Monitor button presses for menu navigation - legacy method kept for compatibility"""
        try:
            # This method is now handled by _continuous_button_monitoring
            # Keep for backward compatibility but just log
            self._logger.info("Legacy menu monitoring called - using continuous monitoring instead")
                
        except Exception as e:
            self._logger.error(f"Menu monitoring failed: {e}")

    def _handle_menu_enter(self):
        """Handle enter button press in menu"""
        try:
            if self.current_menu_state == "main":
                self._display_text("Hauptmenü", "Eingabe", "ausgewählt", "")
                # Could expand to show submenu options
                time.sleep(1)
                self._display_text("Menü bereit", "Tasten nutzen:", "rechts=Fokus", "enter=Menü")
            elif self.current_menu_state == "settings":
                self._display_text("Einstellungen", "Eingabe", "ausgewählt", "")
                time.sleep(1)
                self._display_text("Einstellungen", "Menü", "oben=Zurück", "enter=Auswahl")
            self._logger.info("Menu enter pressed")
        except Exception as e:
            self._logger.error(f"Menu enter handling failed: {e}")

    def _handle_focus_menu(self):
        """Handle focus menu - equivalent to focus() function"""
        try:
            self._display_text("Fokussierhilfe", "aktiviert", "15 Sekunden", "")
            self._logger.info("Focus mode activated")
            # Implement focus assistance logic
            self._run_focus_mode()
            # Return to main menu display after focus mode
            time.sleep(1)
            if self.menu_open:
                self._display_text("Menü bereit", "Tasten nutzen:", "rechts=Fokus", "enter=Menü")
        except Exception as e:
            self._logger.error(f"Focus menu handling failed: {e}")

    def _run_focus_mode(self):
        """Run focus assistance mode"""
        try:
            start_time = time.time()
            while time.time() - start_time < 15.0:
                # Calculate focus sharpness
                frame = self.detectorlepmonCam.getLatestFrame()
                if frame is not None:
                    # Calculate Laplacian variance as sharpness measure
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    self.sigFocusSharpness.emit(float(sharpness))
                time.sleep(0.5)
        except Exception as e:
            self._logger.error(f"Focus mode failed: {e}")

    def _handle_update_menu(self):
        """Handle update menu"""
        try:
            self._display_text("Update Menü", "geöffnet", "")
            self._logger.info("Update menu opened")
            # Update logic would go here
        except Exception as e:
            self._logger.error(f"Update menu handling failed: {e}")

    def _handle_navigation_down(self):
        """Handle down button press for menu navigation"""
        try:
            if self.current_menu_state == "main":
                self.current_menu_state = "settings"
                self._display_text("Einstellungen", "Menü", "oben=Zurück", "enter=Auswahl")
            elif self.current_menu_state == "settings":
                self.current_menu_state = "main"
                self._display_text("Hauptmenü", "rechts=Fokus", "enter=Eingabe", "unten=Settings")
            self._logger.info(f"Menu navigation: {self.current_menu_state}")
        except Exception as e:
            self._logger.error(f"Navigation down handling failed: {e}")

    # ---------------------- LepmonOS LED Control Functions ---------------------- #

    def _lepiled_start(self):
        """Equivalent to utils.Lights.LepiLED_start()"""
        try:
            self.uv_led_active = True
            self._turn_on_led("UV_LED")
            self._logger.info("UV LED (LepiLED) started")
            self._send_lora("LepiLED eingeschaltet")
        except Exception as e:
            self._logger.error(f"Failed to start UV LED: {e}")

    def _lepiled_ende(self):
        """Equivalent to utils.Lights.LepiLED_ende()"""
        try:
            self.uv_led_active = False
            self._turn_off_led("UV_LED")
            self._logger.info("UV LED (LepiLED) stopped")
            self._send_lora("LepiLED ausgeschaltet")
        except Exception as e:
            self._logger.error(f"Failed to stop UV LED: {e}")

    def _visible_led_start(self):
        """Equivalent to VisibleLED_start()"""
        try:
            self.visible_led_active = True
            self._turn_on_led("Visible_LED")
            self._logger.info("Visible LED started")
        except Exception as e:
            self._logger.error(f"Failed to start Visible LED: {e}")

    def _visible_led_ende(self):
        """Equivalent to VisibleLED_ende()"""
        try:
            self.visible_led_active = False
            self._turn_off_led("Visible_LED")
            self._logger.info("Visible LED stopped")
        except Exception as e:
            self._logger.error(f"Failed to stop Visible LED: {e}")

    # ---------------------- LepmonOS Sensor and Data Functions ---------------------- #

    def _read_sensor_data(self, code: str, local_time: str):
        """Equivalent to utils.sensor_data.read_sensor_data()"""
        try:
            # Simulate sensor data reading
            self.sensor_data = {
                "LUX": np.round(np.random.uniform(0, 100), 2),
                "Temp_in": self.innerTemp,
                "Temp_out": self.outerTemp,
                "humidity": self.humidity,
                "bus_voltage": np.round(np.random.uniform(11.5, 12.5), 2),
                "power": np.round(np.random.uniform(500, 1500), 2),
                "timestamp": local_time,
                "code": code
            }
            return self.sensor_data
        except Exception as e:
            self._logger.error(f"Failed to read sensor data: {e}")
            return {}

    def _get_disk_space(self):
        """Equivalent to utils.service.get_disk_space()"""
        try:
            usage = dirtools.getDiskusage()
            total_space_gb = 100.0  # Mock value
            used_space_gb = total_space_gb * usage
            free_space_gb = total_space_gb - used_space_gb
            used_percent = usage * 100
            free_percent = 100 - used_percent
            
            return total_space_gb, used_space_gb, free_space_gb, used_percent, free_percent
        except Exception as e:
            self._logger.error(f"Failed to get disk space: {e}")
            return 0, 0, 0, 0, 0

    def _zeit_aktualisieren(self):
        """Equivalent to utils.times.Zeit_aktualisieren()"""
        try:
            now = datetime.datetime.now()
            utc_time = now.strftime("%Y-%m-%d %H:%M:%S")
            local_time = now.strftime("%H:%M:%S")
            return utc_time, local_time
        except Exception as e:
            self._logger.error(f"Failed to update time: {e}")
            return "", ""


    # ---------------------- GET-Like Endpoints ---------------------- #

    @APIExport(requestType="POST")
    def setSensorData(self, sensorData: dict) -> dict:
        """
        A GET-like endpoint that sets the inner and outer temperature and humidity.
        {"innerTemp": 25.0, "outerTemp": 20.0, "humidity": 45.0}
        """
        try:
            innerTemp = sensorData["innerTemp"]
            outerTemp = sensorData["outerTemp"]
            humidity = sensorData["humidity"]
            self.innerTemp = innerTemp
            self.outerTemp = outerTemp
            self.humidity = humidity
            sensor_data = {"innerTemp": self.innerTemp, "outerTemp": self.outerTemp, "humidity": self.humidity}
            self.temperatureUpdate.emit(sensor_data)
            return {"success": True, "message": "Sensor data updated."}
        except Exception as e:
            self._logger.error(f"Could not update sensor data: {e}")
            return {"success": False, "message": "Could not update sensor data:"}


    @APIExport()
    def getStatus(self) -> dict:
        """
        A GET-like endpoint that returns a dict with isRunning, currentImageCount, freeSpace, serverTime, etc.
        """
        free_space_str = self._computeFreeSpace()
        status = {
            "isRunning": self.is_measure,
            "currentImageCount": self.imagesTaken,
            "serverTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "freeSpace": free_space_str
        }
        return status

    @APIExport()
    def getInitialParams(self) -> dict:
        """
        Another GET-like endpoint that returns camera/timelapse settings, storage path, etc.
        """
        result = {
            "exposureTime": self.mExperimentParameters["exposureTime"],
            "gain": self.mExperimentParameters["gain"],
            "timelapsePeriod": self.mExperimentParameters["timelapsePeriod"],
            "storagePath": self.mExperimentParameters["storagePath"],
        }
        return result

    @APIExport()
    def getHardwareStatus(self) -> dict:
        """
        Returns the current state of all hardware components including LEDs, OLED display, and buttons.
        """
        return {
            "lightStates": self.lightStates,
            "lcdDisplay": self.lcdDisplay,
            "buttonStates": self.buttonStates,
            "timingConfig": self.timingConfig,
            "availableLEDs": list(LED_PINS.keys()),
            "availableButtons": list(BUTTON_PINS.keys()),
            "hardwareStatus": {
                "gpio_available": HAS_GPIO,
                "oled_available": HAS_OLED and self.oled is not None,
                "i2c_available": HAS_I2C and self.i2c_bus is not None,
                "simulation_mode": not HAS_GPIO
            },
            "availableSensors": list(SENSOR_ADDRESSES.keys())
        }

    @APIExport()
    def getTimingConfig(self) -> dict:
        """
        Returns the current timing configuration.
        """
        return self.timingConfig

    @APIExport()
    def getSensorData(self) -> dict:
        """
        Returns current sensor data including temperature and humidity.
        """
        return {
            "innerTemp": self.innerTemp,
            "outerTemp": self.outerTemp,
            "humidity": self.humidity,
            "timestamp": datetime.datetime.now().isoformat()
        }

    # ---------------------- POST-Like Endpoints --------------------- #

    @APIExport(requestType="POST")
    def startExperiment(self,
                        deviceTime: str,
                        deviceLat: float = None,
                        deviceLng: float = None,
                        exposureTime: float = 100.0,
                        gain: float = 0.0,
                        timelapsePeriod: int = 60,
                        time: str = None,
                        date: str = None) -> dict:
        """
        Called by the frontend to start an experiment.
        Here we store the user deviceTime, lat/lng, exposure, etc.
        Then we call self.startLepmonExperiment(...) in a thread.
        """
        self._logger.debug(f"startExperiment from deviceTime={deviceTime}, lat={deviceLat}, lng={deviceLng}")

        # We can set camera exposure/gain
        self.changeAutoExposureTime("manual")
        self.changeExposureTime(exposureTime)
        self.changeGain(gain)

        # Also set timelapse period
        self.mExperimentParameters["timelapsePeriod"] = timelapsePeriod
        self.mExperimentParameters["time"] = time
        self.mExperimentParameters["date"] = date


        self.mExperimentParameters["timeStamp"] = (time + "_" + date)
        self.mExperimentParameters["storagePath"] = "/mnt/usb_drive"
        self.mExperimentParameters["numImages"] = -1
        self.mExperimentParameters["fileFormat"] = "TIF"
        self.mExperimentParameters["frameRate"] = timelapsePeriod
        self.mExperimentParameters["experimentName"] = "LepMonTest"
        self.mExperimentParameters["uniqueId"] = np.random.randint(0, 1000),

        # Start thread
        self.mExperimentThread = Thread(target=self.lepmonExperimentThread, args=(
            self.mExperimentParameters["timeStamp"],
            self.mExperimentParameters["experimentName"],
            self.mExperimentParameters["uniqueId"],
            self.mExperimentParameters["numImages"],
            self.mExperimentParameters["frameRate"],
            self.mExperimentParameters["storagePath"],
            self.mExperimentParameters["fileFormat"],
        ), daemon=True)
        self.mExperimentThread.start()


        # Actually start the experiment logic
        self.is_measure = True
        self.imagesTaken = 0
        self.sigIsRunning.emit(True)  # Websocket signal
        return {"success": True, "message": "Experiment started", "lat": deviceLat, "lng": deviceLng}

    @APIExport(requestType="POST")
    def stopLepmonExperiment(self) -> dict:
        """
        Called by the frontend to stop any running experiment.
        """
        self._logger.debug("Experiment stopped by user.")
        self.is_measure = False
        self.sigIsRunning.emit(False)
        return {"success": True, "message": "Experiment stopped"}

    @APIExport(requestType="POST")
    def focusMode(self) -> dict:
        """
        The user triggers a 15s focus mode in the backend.
        We simulate sending sharpness values via self.sigFocusSharpness.emit(...) in a small thread.
        """
        self._logger.debug("Focus Mode requested.")
        def focus_thread():
            start_time = time.time()
            while time.time() - start_time < 15.0:
                # Example: generate a random "sharpness" or measure it from camera
                # For now, let's just do random
                import random
                sharp_val = random.uniform(10, 300)
                self.sigFocusSharpness.emit(sharp_val)
                time.sleep(0.5)
        t = Thread(target=focus_thread, daemon=True)
        t.start()
        return {"success": True, "message": "Focus mode started for 15s"}

    @APIExport(requestType="POST")
    def reboot(self) -> dict:
        """
        The user triggers a device reboot.
        """
        self._logger.debug("Reboot requested.")
        # e.g. os.system("sudo reboot")
        return {"success": True, "message": "System is rebooting (mock)"}

    # ---------------------- LepmonOS API Endpoints (matching 03_capturing.py) ---------------------- #

    @APIExport(requestType="POST")
    def lepmonStartup(self) -> dict:
        """Equivalent to 00_start_up.py main execution"""
        try:
            self._initializeLepmonOS()
            return {"success": True, "message": "LepmonOS startup sequence completed"}
        except Exception as e:
            self._logger.error(f"LepmonOS startup failed: {e}")
            return {"success": False, "message": f"Startup failed: {str(e)}"}

    @APIExport(requestType="POST") 
    def lepmonWelcome(self) -> dict:
        """Equivalent to 01_start_up.py"""
        try:
            self._display_text("Willkommen", "Laden... 2/2", "")
            self._logger.info("Welcome message displayed")
            time.sleep(2)
            return {"success": True, "message": "Welcome sequence completed"}
        except Exception as e:
            self._logger.error(f"Welcome sequence failed: {e}")
            return {"success": False, "message": f"Welcome failed: {str(e)}"}

    @APIExport(requestType="POST")
    def lepmonOpenHMI(self) -> dict:
        """Equivalent to 02_trap_hmi.py main menu system"""
        try:
            self._open_hmi_menu()
            return {"success": True, "message": "HMI menu opened", "menu_state": self.current_menu_state}
        except Exception as e:
            self._logger.error(f"HMI menu opening failed: {e}")
            return {"success": False, "message": f"HMI menu failed: {str(e)}"}

    @APIExport(requestType="POST")
    def lepmonCloseHMI(self) -> dict:
        """Close HMI menu system"""
        try:
            self.menu_open = False
            self._turn_off_led("blau")
            self._display_text("HMI geschlossen", "", "", "")
            self._logger.info("HMI menu closed")
            return {"success": True, "message": "HMI menu closed"}
        except Exception as e:
            self._logger.error(f"HMI menu closing failed: {e}")
            return {"success": False, "message": f"HMI menu close failed: {str(e)}"}

    @APIExport()
    def getHMIStatus(self) -> dict:
        """Get HMI menu status"""
        try:
            return {
                "success": True,
                "hmi_open": self.menu_open,
                "current_menu_state": self.current_menu_state,
                "button_states": self.buttonStates,
                "monitoring_active": not self.hmi_stop_event.is_set() if hasattr(self, 'hmi_stop_event') else False
            }
        except Exception as e:
            self._logger.error(f"HMI status retrieval failed: {e}")
            return {"success": False, "message": f"HMI status failed: {str(e)}"}

    @APIExport(requestType="POST")
    def lepmonStartCapturing(self, override_timecheck: bool = True) -> dict:
        """Equivalent to 03_capturing.py main capturing loop"""
        try:
            # Initialize capturing parameters from config
            dusk_threshold = self.lepmon_config.get("capture_mode", {}).get("dusk_treshold", 50)
            interval = self.lepmon_config.get("capture_mode", {}).get("interval", 60)
            initial_exposure = self.lepmon_config.get("capture_mode", {}).get("initial_exposure", 100)
            
            # Set camera exposure
            self.changeExposureTime(initial_exposure)
            
            # Start main capturing thread
            capture_thread = Thread(target=self._lepmon_main_loop, 
                                  args=(dusk_threshold, interval, initial_exposure, override_timecheck), 
                                  daemon=True)
            capture_thread.start()
            
            return {"success": True, "message": "LepmonOS capturing started", 
                   "parameters": {"dusk_threshold": dusk_threshold, "interval": interval, "exposure": initial_exposure}}
        except Exception as e:
            self._logger.error(f"LepmonOS capturing start failed: {e}")
            return {"success": False, "message": f"Capturing start failed: {str(e)}"}

    @APIExport(requestType="POST")
    def lepmonShutdown(self) -> dict:
        """Equivalent to 04_end.py shutdown sequence"""
        try:
            self._logger.info("Starting LepmonOS shutdown sequence")
            
            # Send shutdown message
            self._send_lora("Falle fährt in 1 Minute herunter und startet dann neu. Letzte Nachricht im aktuellen Run")
            
            # Countdown display
            for i in range(60, 0, -1):
                self._display_text("Falle startet", "neu in", f"{i} Sekunden")
                time.sleep(1)
                if i % 10 == 0:  # Update every 10 seconds
                    self._logger.info(f"Shutdown in {i} seconds")
            
            # Stop all activities
            self.stop_event.set()
            self.is_measure = False
            
            # Turn off all LEDs
            self._lepiled_ende()
            self._visible_led_ende()
            
            self._logger.info("LepmonOS shutdown sequence completed")
            return {"success": True, "message": "LepmonOS shutdown completed"}
            
        except Exception as e:
            self._logger.error(f"LepmonOS shutdown failed: {e}")
            return {"success": False, "message": f"Shutdown failed: {str(e)}"}

    @APIExport()
    def lepmonGetConfig(self) -> dict:
        """Get complete LepmonOS configuration"""
        return {
            "config": self.lepmon_config,
            "lightStates": self.lightStates,
            "lcdDisplay": self.lcdDisplay,
            "buttonStates": self.buttonStates,
            "timingConfig": self.timingConfig,
            "version": self.version,
            "serial_number": self.serial_number,
            "experiment_times": {
                "start": self.experiment_start_time,
                "end": self.experiment_end_time,
                "lepiled_end": self.lepiled_end_time
            }
        }

    @APIExport(requestType="POST")
    def lepmonUVLed(self, action: str) -> dict:
        """Control UV LED (LepiLED) - equivalent to LepiLED_start/ende"""
        try:
            if action == "on":
                self._lepiled_start()
                return {"success": True, "message": "UV LED turned on", "state": True}
            elif action == "off":
                self._lepiled_ende()
                return {"success": True, "message": "UV LED turned off", "state": False}
            else:
                return {"success": False, "message": "Invalid action. Use 'on' or 'off'"}
        except Exception as e:
            self._logger.error(f"UV LED control failed: {e}")
            return {"success": False, "message": f"UV LED control failed: {str(e)}"}

    @APIExport(requestType="POST")
    def lepmonVisibleLed(self, action: str) -> dict:
        """Control Visible LED - equivalent to VisibleLED_start/ende"""
        try:
            if action == "on":
                self._visible_led_start()
                return {"success": True, "message": "Visible LED turned on", "state": True}
            elif action == "off":
                self._visible_led_ende()
                return {"success": True, "message": "Visible LED turned off", "state": False}
            else:
                return {"success": False, "message": "Invalid action. Use 'on' or 'off'"}
        except Exception as e:
            self._logger.error(f"Visible LED control failed: {e}")
            return {"success": False, "message": f"Visible LED control failed: {str(e)}"}

    @APIExport(requestType="POST")
    def lepmonSnapImage(self, format: str = "tiff", log_type: str = "log", error_count: int = 0, exposure: float = None) -> dict:
        """Take image - equivalent to snap_image function"""
        try:
            if exposure:
                self.changeExposureTime(exposure)
            
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lepmon_{current_time}_{self.imagesTaken}"
            
            # Take the image
            self.mFrame = self.snapImagelepmonCam(filename, fileFormat=format.upper())
            
            self.imagesTaken += 1
            self.sigImagesTaken.emit(self.imagesTaken)
            
            return {
                "success": True, 
                "message": "Image captured successfully",
                "filename": filename,
                "format": format,
                "image_count": self.imagesTaken,
                "exposure": exposure or self.mExperimentParameters["exposureTime"]
            }
        except Exception as e:
            error_count += 1
            self._logger.error(f"Image capture failed: {e}")
            return {
                "success": False, 
                "message": f"Image capture failed: {str(e)}",
                "error_count": error_count
            }


    @APIExport(runOnUIThread=True)
    def returnLastSnappedImage(self) -> Response:
        """Returns the last captured image from the camera."""
        try:
            try:
                arr = self.mFrame
            except:
                arr = self.detectorlepmonCam.getLatestFrame()

            im = Image.fromarray(arr.astype(np.uint8))
            with io.BytesIO() as buf:
                im = im.convert("L")  # ensure grayscale
                im.save(buf, format="PNG")
                im_bytes = buf.getvalue()
            headers = {"Content-Disposition": 'inline; filename="crop.png"'}
            return Response(im_bytes, headers=headers, media_type="image/png")
        except Exception as e:
            raise RuntimeError("No cropped image available. Please run update() first.") from e


    @APIExport()
    def lepmonGetSensorData(self) -> dict:
        """Get sensor data - equivalent to read_sensor_data function"""
        try:
            _, local_time = self._zeit_aktualisieren()
            sensor_data = self._read_sensor_data("api_request", local_time)
            return {
                "success": True,
                "sensor_data": sensor_data,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            self._logger.error(f"Sensor data retrieval failed: {e}")
            return {"success": False, "message": f"Sensor data failed: {str(e)}"}

    @APIExport()
    def lepmonGetStatus(self) -> dict:
        """Get complete LepmonOS status"""
        try:
            total_space, used_space, free_space, used_pct, free_pct = self._get_disk_space()
            _, local_time = self._zeit_aktualisieren()
            
            return {
                "success": True,
                "status": {
                    "is_running": self.is_measure,
                    "uv_led_active": self.uv_led_active,
                    "visible_led_active": self.visible_led_active,
                    "menu_open": self.menu_open,
                    "images_taken": self.imagesTaken,
                    "current_time": local_time,
                    "disk_space": {
                        "total_gb": total_space,
                        "free_gb": free_space,
                        "used_percent": used_pct
                    },
                    "experiment_times": {
                        "start": self.experiment_start_time,
                        "end": self.experiment_end_time,
                        "lepiled_end": self.lepiled_end_time
                    }
                }
            }
        except Exception as e:
            self._logger.error(f"Status retrieval failed: {e}")
            return {"success": False, "message": f"Status retrieval failed: {str(e)}"}

    def _lepmon_main_loop(self, dusk_threshold: float, interval: int, initial_exposure: float, override_timecheck: bool):
        """Main capturing loop - equivalent to the main loop in 03_capturing.py"""
        try:
            self._logger.info("Starting LepmonOS main capturing loop")
            
            # Store USB space information (equivalent to FRAM writing)
            total_space, used_space, free_space, used_pct, free_pct = self._get_disk_space()
            self._logger.info(f"USB Storage - Total: {total_space}GB, Free: {free_space}GB ({free_pct:.1f}%)")
            
            fang_begonnen = False
            kamera_fehlerserie = 0
            
            while not self.stop_event.is_set():
                _, local_time = self._zeit_aktualisieren()
                sensors = self._read_sensor_data("check_lux", local_time)
                ambient_light = sensors["LUX"]
                
                # Check if we should capture (time and light conditions)
                should_capture = (
                    (ambient_light <= dusk_threshold and not self._is_in_time_range(self.experiment_end_time, self.experiment_start_time, local_time)) or
                    (ambient_light > dusk_threshold and not self._is_in_time_range(self.experiment_start_time, self.experiment_start_time, local_time)) or
                    override_timecheck
                )
                
                if should_capture:
                    if not fang_begonnen:
                        self._lepiled_start()
                        fang_begonnen = True
                    
                    # Capture sequence
                    exposure = initial_exposure
                    
                    # Adjust exposure based on time
                    if self._is_in_first_hour(local_time):
                        exposure -= 30
                    
                    if self._is_after_lepiled_end(local_time):
                        exposure -= 30
                        if self.uv_led_active:
                            self._lepiled_ende()
                    
                    # Take image
                    result = self.lepmonSnapImage("tiff", "log", kamera_fehlerserie, exposure)
                    if not result["success"]:
                        kamera_fehlerserie = result.get("error_count", kamera_fehlerserie)
                    else:
                        kamera_fehlerserie = 0
                    
                    # Update sensor data
                    sensors = self._read_sensor_data(f"img_{self.imagesTaken}", local_time)
                    sensors.update({"Status_Kamera": result["success"], "Exposure": exposure})
                    
                    # Update display
                    self._display_text(
                        f"Image: {self.imagesTaken}",
                        f"Time: {local_time}",
                        f"Exp: {exposure}ms",
                        f"Lux: {ambient_light}"
                    )
                    
                    # Check for camera error series
                    if kamera_fehlerserie >= 3:
                        self._logger.error("Camera error series detected, stopping")
                        break
                    
                    # Wait for next image
                    self._logger.info(f"Waiting {interval} seconds until next image")
                    for _ in range(interval):
                        if self.stop_event.is_set():
                            break
                        time.sleep(1)
                
                else:
                    self._logger.info("Conditions not met for capturing, ending loop")
                    break
                    
        except Exception as e:
            self._logger.error(f"Main loop error: {e}")
        finally:
            self._logger.info("LepmonOS main loop ended")

    def _is_in_time_range(self, start_time: str, end_time: str, current_time: str) -> bool:
        """Check if current time is within a time range"""
        try:
            start = datetime.datetime.strptime(start_time, "%H:%M:%S").time()
            end = datetime.datetime.strptime(end_time, "%H:%M:%S").time()
            current = datetime.datetime.strptime(current_time, "%H:%M:%S").time()
            
            if start <= end:
                return start <= current <= end
            else:  # Time range crosses midnight
                return current >= start or current <= end
        except:
            return False

    def _is_in_first_hour(self, current_time: str) -> bool:
        """Check if we're in the first hour of experiment"""
        try:
            exp_start = datetime.datetime.strptime(self.experiment_start_time, "%H:%M:%S")
            current = datetime.datetime.strptime(current_time, "%H:%M:%S")
            diff = (current - exp_start).total_seconds()
            return 0 <= diff <= 3600  # First hour
        except:
            return False

    def _is_after_lepiled_end(self, current_time: str) -> bool:
        """Check if we're after the LepiLED end time"""
        try:
            lepiled_end = datetime.datetime.strptime(self.lepiled_end_time, "%H:%M:%S").time()
            exp_end = datetime.datetime.strptime(self.experiment_end_time, "%H:%M:%S").time()
            current = datetime.datetime.strptime(current_time, "%H:%M:%S").time()
            
            return lepiled_end <= current < exp_end
        except:
            return False

    # ---------------------- Hardware Control Endpoints (existing ImSwitch integration) ---------------------- #

    @APIExport(requestType="POST")
    def setLightState(self, lightName: str, state: bool) -> dict:
        """
        Turn a specific light on or off using direct GPIO control.
        """
        try:
            # Use direct GPIO control for Lepmon LEDs
            if lightName in LED_PINS:
                if state:
                    self._turn_on_led(lightName)
                else:
                    self._turn_off_led(lightName)
                return {"success": True, "message": f"LED '{lightName}' turned {'on' if state else 'off'}"}
            else:
                # Check for aliases or special LED names
                led_mapping = {
                    "UV_LED": "gelb",  # Map UV LED to yellow
                    "Visible_LED": "blau",  # Map visible LED to blue  
                    "Status_LED": "rot"  # Map status LED to red
                }
                
                if lightName in led_mapping:
                    actual_led = led_mapping[lightName]
                    if state:
                        self._turn_on_led(actual_led)
                    else:
                        self._turn_off_led(actual_led)
                    return {"success": True, "message": f"LED '{lightName}' ({actual_led}) turned {'on' if state else 'off'}"}
                else:
                    available_leds = list(LED_PINS.keys()) + list(led_mapping.keys())
                    return {"success": False, "message": f"LED '{lightName}' not found. Available: {available_leds}"}
                    
        except Exception as e:
            self._logger.error(f"Failed to set light state: {e}")
            return {"success": False, "message": f"Failed to control LED: {str(e)}"}

    @APIExport(requestType="POST")
    def setAllLightsState(self, state: bool) -> dict:
        """
        Turn all LEDs on or off using direct GPIO control.
        """
        try:
            results = []
            # Control all physical LEDs
            for led_name in LED_PINS.keys():
                if state:
                    self._turn_on_led(led_name)
                else:
                    self._turn_off_led(led_name)
                results.append({"led": led_name, "state": state, "success": True})
            
            return {
                "success": True, 
                "message": f"All LEDs turned {'on' if state else 'off'}",
                "individual_results": results
            }
        except Exception as e:
            self._logger.error(f"Failed to set all lights state: {e}")
            return {"success": False, "message": f"Failed to control all LEDs: {str(e)}"}

    @APIExport(requestType="POST")
    def updateLCDDisplay(self, line1: str = "", line2: str = "", line3: str = "", line4: str = "") -> dict:
        """
        Update the OLED display content using direct hardware control.
        """
        try:
            # Use the new direct display method
            self._display_text(line1, line2, line3, line4)
            
            return {
                "success": True,
                "message": "Display updated",
                "content": {
                    "line1": line1[:20],
                    "line2": line2[:20],
                    "line3": line3[:20],
                    "line4": line4[:20]
                }
            }
        except Exception as e:
            self._logger.error(f"Failed to update LCD display: {e}")
            return {"success": False, "message": f"Failed to update display: {str(e)}"}

    @APIExport(requestType="POST")
    def simulateButtonPress(self, buttonName: str) -> dict:
        """
        Simulate a button press event for testing purposes.
        """
        try:
            if buttonName in BUTTON_PINS:
                self._simulate_button_press(buttonName)
                return {"success": True, "message": f"Button '{buttonName}' pressed (simulated)"}
            else:
                available_buttons = list(BUTTON_PINS.keys())
                return {"success": False, "message": f"Unknown button: {buttonName}. Available: {available_buttons}"}
        except Exception as e:
            self._logger.error(f"Failed to simulate button press: {e}")
            return {"success": False, "message": f"Failed to simulate button press: {str(e)}"}

    @APIExport()
    def getButtonStates(self) -> dict:
        """
        Get current button states by reading GPIO pins directly.
        """
        try:
            current_states = self._read_all_buttons()
            return {
                "success": True,
                "buttonStates": current_states,
                "availableButtons": list(BUTTON_PINS.keys()),
                "hardwareMode": "GPIO" if HAS_GPIO else "simulation"
            }
        except Exception as e:
            self._logger.error(f"Failed to read button states: {e}")
            return {"success": False, "message": f"Failed to read button states: {str(e)}"}

    @APIExport()
    def getSensorDataLive(self) -> dict:
        """
        Get live sensor data from I2C sensors.
        """
        try:
            sensor_data = self._read_all_sensors()
            return {
                "success": True,
                "sensorData": sensor_data,
                "timestamp": time.time(),
                "availableSensors": list(SENSOR_ADDRESSES.keys()),
                "hardwareMode": "I2C" if HAS_I2C and self.i2c_bus else "simulation"
            }
        except Exception as e:
            self._logger.error(f"Failed to read sensor data: {e}")
            return {"success": False, "message": f"Failed to read sensor data: {str(e)}"}

    @APIExport(requestType="POST")
    def setTimingConfig(self, config: dict) -> dict:
        """
        Update timing configuration parameters.
        """
        try:
            valid_keys = set(self.timingConfig.keys())
            updated_keys = []
            
            for key, value in config.items():
                if key in valid_keys and isinstance(value, (int, float)) and value >= 0:
                    self.timingConfig[key] = value
                    updated_keys.append(key)
                else:
                    self._logger.warning(f"Invalid timing config key or value: {key}={value}")
            
            # Update the LepmonManager config
            self._master.LepmonManager.updateConfig("timingConfig", self.timingConfig)
            
            return {
                "success": True,
                "message": f"Updated timing configuration for: {', '.join(updated_keys)}",
                "updated_config": {k: self.timingConfig[k] for k in updated_keys}
            }
        except Exception as e:
            self._logger.error(f"Failed to update timing config: {e}")
            return {"success": False, "message": f"Failed to update timing config: {str(e)}"}

    # ---------------------- Main experiment thread ------------------- #
    def lepmonExperimentThread(self,
                             timeStamp: str,
                             experimentName: str,
                             uniqueId: str,
                             numImages: int,
                             frameRate: float,
                             filePath: str,
                             fileFormat: str):

        """
        Enhanced background thread that acquires images with proper timing controls and hardware integration.
        """

        self.is_measure = True
        self.imagesTaken = 0
        self.sigIsRunning.emit(True)

        # Update LCD display with experiment start
        self.updateLCDDisplay(
            line1="Experiment Running",
            line2=f"Name: {experimentName[:12]}",
            line3=f"Images: 0/{numImages if numImages > 0 else 'inf'}",
            line4="Press button to stop"
        )

        # Possibly create a folder
        dirPath = os.path.join(filePath, timeStamp)
        if not os.path.exists(dirPath):
            try:
                os.makedirs(dirPath)
            except Exception as e:
                self._logger.error(e)

        # Pre-experiment stabilization
        self._logger.info(f"Stabilization period: {self.timingConfig['stabilizationTime']} seconds")
        time.sleep(self.timingConfig['stabilizationTime'])

        while self.is_measure and (self.imagesTaken < numImages or numImages == -1):
            # Pre-acquisition delay
            if self.timingConfig['preAcquisitionDelay'] > 0:
                time.sleep(self.timingConfig['preAcquisitionDelay'])
            
            currentTime = time.time()
            self.imagesTaken += 1

            # Update LCD with current progress
            self.updateLCDDisplay(
                line1="Acquiring...",
                line2=f"Image: {self.imagesTaken}",
                line3=f"Time: {time.strftime('%H:%M:%S')}",
                line4=f"Interval: {frameRate}s"
            )

            # Notify WebSocket about new image count
            self.sigImagesTaken.emit(self.imagesTaken)

            # Snap image
            filename = os.path.join(dirPath, f"{timeStamp}_{experimentName}_{uniqueId}_{self.imagesTaken}")
            try:
                self.snapImagelepmonCam(filename, fileFormat=fileFormat)
            except Exception as e:
                self._logger.error(f"Could not snap image: {e}")

            # Post-acquisition delay
            if self.timingConfig['postAcquisitionDelay'] > 0:
                time.sleep(self.timingConfig['postAcquisitionDelay'])

            # Sleep to maintain framerate using timing configuration
            effective_interval = max(frameRate, self.timingConfig['acquisitionInterval'])
            elapsed = time.time() - currentTime
            remaining_time = effective_interval - elapsed
            
            if remaining_time > 0:
                # Break remaining time into small chunks to allow for quick stop
                while remaining_time > 0 and self.is_measure:
                    sleep_chunk = min(0.1, remaining_time)
                    time.sleep(sleep_chunk)
                    remaining_time -= sleep_chunk
            
            if not self.is_measure:
                break
                
        self.is_measure = False
        self.sigIsRunning.emit(False)
        
        # Update LCD display with experiment end
        self.updateLCDDisplay(
            line1="Experiment Complete",
            line2=f"Images taken: {self.imagesTaken}",
            line3=f"Saved to: {dirPath[:15]}...",
            line4="Ready for next run"
        )
        
        self._logger.debug("lepmonExperimentThread done.")

    # ----------------------- Snap single image ----------------------- #
    @APIExport(runOnUIThread=True)
    def snapImagelepmonCam(self, fileName=None, fileFormat="JPG"):
        """Just captures the latest frame from the camera and saves it."""
        if not fileName:
            fileName = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        frame = self.detectorlepmonCam.getLatestFrame()
        if frame is None:
            self._logger.warning("No frame received from the camera.")
            return

        if fileFormat.upper() == "TIF":
            import tifffile as tif
            tif.imwrite(fileName + ".tif", frame, append=False)
        elif fileFormat.upper() == "JPG":
            cv2.imwrite(fileName + ".jpg", frame)
        elif fileFormat.upper() == "PNG":
            cv2.imwrite(fileName + ".png", frame)
        else:
            self._logger.warning(f"No valid fileFormat selected: {fileFormat}")
        return frame
    # ----------------------- Camera controls ------------------------- #
    @APIExport(runOnUIThread=True)
    def changeExposureTime(self, value):
        """Change exposure time (manual mode)."""
        try:
            self.mExperimentParameters["exposureTime"] = value
            self.detectorlepmonCam.setParameter(name="exposure", value=value)
        except Exception as e:
            self._logger.error(f"Could not set exposure: {e}")

    @APIExport(runOnUIThread=True)
    def changeAutoExposureTime(self, value):
        """Enable/disable auto-exposure, e.g. 'auto' or 'manual'."""
        try:
            self.detectorlepmonCam.setParameter(name="exposure_mode", value=value)
        except Exception as e:
            self._logger.error(f"Could not set auto exposure mode: {e}")

    @APIExport(runOnUIThread=True)
    def changeGain(self, value):
        """Change camera gain."""
        try:
            self.mExperimentParameters["gain"] = value
            self.detectorlepmonCam.setGain(value)
        except Exception as e:
            self._logger.error(f"Could not set gain: {e}")

    def closeEvent(self):
        self._pullSensorDataActive = False
        if hasattr(super(), '__del__'):
            super().__del__()
    # ---------------------- Helper functions -------------------------- #

    def _pullSensorData(self, interval):
        self._pullSensorDataActive = True
        while self._pullSensorDataActive:
            # Get sensor data
            # e.g. temperature, humidity, pressure, etc.
            # sensor_data = getSensorData()
            # self.sigSensorData.emit(sensor_data)
            time.sleep(interval)
            # simulate inner/outer temperature and humidity
            # join them in dictionary
            sensor_data = {"innerTemp": self.innerTemp, "outerTemp": self.outerTemp, "humidity": self.humidity}
            self.temperatureUpdate.emit(sensor_data)

    def _computeFreeSpace(self) -> str:
        # Simplistic approach or call your existing function
        usage = dirtools.getDiskusage()  # returns fraction used, e.g. 0.8 => 80%
        used_prc = usage * 100
        free_prc = 100.0 - used_prc
        return f"{free_prc:.1f}% free"

    def detect_external_drives(self):
        """If you want to keep an external drive detection method, do so here."""
        system = platform.system()
        external_drives = []
        if system in ["Linux", "Darwin"]:
            df_result = subprocess.run(['df', '-h'], stdout=subprocess.PIPE)
            output = df_result.stdout.decode('utf-8')
            lines = output.splitlines()
            for line in lines:
                if '/media/' in line or '/Volumes/' in line:
                    drive_info = line.split()
                    mount_point = " ".join(drive_info[5:])
                    external_drives.append(mount_point)
        elif system == "Windows":
            wmic_result = subprocess.run(['wmic', 'logicaldisk', 'get', 'caption,description'],
                                         stdout=subprocess.PIPE)
            output = wmic_result.stdout.decode('utf-8')
            lines = output.splitlines()
            for line in lines:
                if 'Removable Disk' in line:
                    drive_info = line.split()
                    drive_letter = drive_info[0]
                    external_drives.append(drive_letter)
        return external_drives


