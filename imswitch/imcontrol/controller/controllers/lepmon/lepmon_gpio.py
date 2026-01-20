"""
LepmonOS GPIO Control Module

This module handles GPIO setup, LED control, and button reading for Raspberry Pi 4.
Mirrors GPIO_Setup.py from LepmonOS_update.

Note: This is specifically for RPi4 where RPi.GPIO works. For RPi5, different
GPIO handling would be needed (e.g., lgpio or gpiozero with pigpio backend).

Hardware Configuration (Pro_Gen_2 / Pro_Gen_3):
- LED Pins: gelb=22, blau=6, rot=17, Heizung=27
- Button Pins: oben=23, unten=24, rechts=8, enter=7
"""

import time
import threading
from typing import Dict, Callable, Optional
from enum import Enum


# Try to import GPIO library
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("RPi.GPIO not available - running in GPIO simulation mode")


class HardwareGeneration(Enum):
    """Hardware generation enum matching LepmonOS hardware.py"""
    PRO_GEN_1 = "Pro_Gen_1"
    PRO_GEN_2 = "Pro_Gen_2"
    PRO_GEN_3 = "Pro_Gen_3"


# Pin configurations for different hardware generations
PIN_CONFIGS = {
    HardwareGeneration.PRO_GEN_1: {
        "leds": {
            'gelb': 27,
            'blau': 22,
            'rot': 17,
            'Heizung': 13,
        },
        "buttons": {
            'oben': 24,
            'unten': 23,
            'rechts': 7,
            'enter': 8,
        }
    },
    HardwareGeneration.PRO_GEN_2: {
        "leds": {
            'gelb': 22,
            'blau': 6,
            'rot': 17,
            'Heizung': 27,
        },
        "buttons": {
            'oben': 23,
            'unten': 24,
            'rechts': 8,
            'enter': 7,
        }
    },
    HardwareGeneration.PRO_GEN_3: {
        "leds": {
            'gelb': 22,
            'blau': 6,
            'rot': 17,
            'Heizung': 27,
        },
        "buttons": {
            'oben': 23,
            'unten': 24,
            'rechts': 8,
            'enter': 7,
        }
    },
}

# Default to Pro_Gen_2/3 configuration
LED_PINS = PIN_CONFIGS[HardwareGeneration.PRO_GEN_2]["leds"]
BUTTON_PINS = PIN_CONFIGS[HardwareGeneration.PRO_GEN_2]["buttons"]


class LepmonGPIO:
    """
    GPIO controller for Lepmon hardware.
    
    Handles:
    - LED control with PWM dimming
    - Button state reading with debouncing
    - Terminal input simulation for testing
    
    This mirrors GPIO_Setup.py from LepmonOS_update.
    """
    
    PWM_FREQUENCY = 1000  # 1kHz PWM frequency
    DEBOUNCE_TIME = 0.05  # 50ms debounce
    
    def __init__(self, 
                 hardware_generation: HardwareGeneration = HardwareGeneration.PRO_GEN_2,
                 button_callback: Optional[Callable[[str, bool], None]] = None):
        """
        Initialize GPIO controller.
        
        Args:
            hardware_generation: Hardware version for pin configuration
            button_callback: Optional callback(button_name, is_pressed) for button events
        """
        self.hardware_generation = hardware_generation
        self.button_callback = button_callback
        
        # Get pin configuration for hardware generation
        config = PIN_CONFIGS.get(hardware_generation, PIN_CONFIGS[HardwareGeneration.PRO_GEN_2])
        self.led_pins = config["leds"].copy()
        self.button_pins = config["buttons"].copy()
        
        # State tracking
        self.led_states: Dict[str, bool] = {name: False for name in self.led_pins}
        self.led_brightness: Dict[str, int] = {name: 0 for name in self.led_pins}
        self.button_states: Dict[str, bool] = {name: False for name in self.button_pins}
        
        # PWM instances
        self.led_pwm: Dict[str, any] = {}
        
        # Terminal input simulation (for testing)
        self._terminal_input: Optional[str] = None
        self._terminal_listener_thread: Optional[threading.Thread] = None
        self._stop_terminal_listener = threading.Event()
        
        # Initialize hardware
        self._initialized = False
        self.initialize()
    
    def initialize(self):
        """Initialize GPIO hardware"""
        if HAS_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # Setup LED pins as outputs with PWM
                for name, pin in self.led_pins.items():
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)  # Start off
                    
                    # Create PWM instance
                    pwm = GPIO.PWM(pin, self.PWM_FREQUENCY)
                    pwm.start(0)  # Start at 0% duty cycle
                    self.led_pwm[name] = pwm
                    self.led_states[name] = False
                    self.led_brightness[name] = 0
                
                # Setup button pins as inputs with pull-up resistors
                for name, pin in self.button_pins.items():
                    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                    self.button_states[name] = False
                
                self._initialized = True
                print("GPIO initialized successfully")
                
            except Exception as e:
                print(f"GPIO initialization failed: {e}")
                self._initialized = False
        else:
            # Simulation mode - no hardware
            print("Running in GPIO simulation mode")
            self._initialized = True
    
    def cleanup(self):
        """Cleanup GPIO resources"""
        # Stop terminal listener
        self._stop_terminal_listener.set()
        
        if HAS_GPIO and self._initialized:
            try:
                # Stop all PWM
                for pwm in self.led_pwm.values():
                    pwm.stop()
                
                # Cleanup GPIO
                GPIO.cleanup()
                print("GPIO cleaned up successfully")
            except Exception as e:
                print(f"GPIO cleanup failed: {e}")
        
        self._initialized = False
    
    def __del__(self):
        """Destructor - cleanup GPIO"""
        self.cleanup()
    
    # ---------------------- LED Control ---------------------- #
    
    def dim_led(self, led_name: str, brightness: int):
        """
        Set LED brightness using PWM.
        
        Equivalent to LepmonOS dim_led().
        
        Args:
            led_name: LED name ('gelb', 'blau', 'rot', 'Heizung')
            brightness: Brightness level 0-100
        """
        if led_name not in self.led_pins:
            print(f"Unknown LED: {led_name}. Available: {list(self.led_pins.keys())}")
            return
        
        brightness = max(0, min(100, brightness))  # Clamp to 0-100
        
        if HAS_GPIO and led_name in self.led_pwm:
            self.led_pwm[led_name].ChangeDutyCycle(brightness)
        
        self.led_brightness[led_name] = brightness
        self.led_states[led_name] = brightness > 0
    
    def turn_on_led(self, led_name: str, brightness: int = 100):
        """
        Turn on LED at specified brightness.
        
        Equivalent to LepmonOS turn_on_led().
        
        Args:
            led_name: LED name
            brightness: Brightness level 0-100 (default: 100)
        """
        self.dim_led(led_name, brightness)
    
    def turn_off_led(self, led_name: str):
        """
        Turn off LED.
        
        Equivalent to LepmonOS turn_off_led().
        
        Args:
            led_name: LED name
        """
        self.dim_led(led_name, 0)
    
    def dim_down_all(self):
        """Turn off all LEDs"""
        for led_name in self.led_pins:
            self.turn_off_led(led_name)
    
    def get_led_state(self, led_name: str) -> bool:
        """Get current LED on/off state"""
        return self.led_states.get(led_name, False)
    
    def get_led_brightness(self, led_name: str) -> int:
        """Get current LED brightness level"""
        return self.led_brightness.get(led_name, 0)
    
    def get_all_led_states(self) -> Dict[str, bool]:
        """Get all LED states"""
        return self.led_states.copy()
    
    def set_heater(self, enabled: bool):
        """
        Control heater LED/relay.
        
        Args:
            enabled: True to turn on heater
        """
        if enabled:
            self.turn_on_led('Heizung')
        else:
            self.turn_off_led('Heizung')
    
    # ---------------------- Button Reading ---------------------- #
    
    def button_pressed(self, button_name: str) -> bool:
        """
        Check if button is currently pressed.
        
        Equivalent to LepmonOS button_pressed().
        
        Args:
            button_name: Button name ('oben', 'unten', 'rechts', 'enter')
            
        Returns:
            True if button is pressed (GPIO LOW due to pull-up)
        """
        if button_name not in self.button_pins:
            available = ", ".join(self.button_pins.keys())
            raise ValueError(f"Invalid button '{button_name}'. Available: {available}")
        
        # Check GPIO
        if HAS_GPIO:
            is_pressed = GPIO.input(self.button_pins[button_name]) == GPIO.LOW
        else:
            is_pressed = self.button_states.get(button_name, False)
        
        # Check terminal simulation input
        if self._terminal_input == button_name.lower():
            is_pressed = True
            self._terminal_input = None  # Clear after reading
        
        # Update state and call callback if changed
        if is_pressed != self.button_states.get(button_name, False):
            self.button_states[button_name] = is_pressed
            if self.button_callback and is_pressed:  # Only on press, not release
                self.button_callback(button_name, is_pressed)
        
        return is_pressed
    
    def read_all_buttons(self) -> Dict[str, bool]:
        """
        Read all button states.
        
        Returns:
            Dict of button_name -> is_pressed
        """
        states = {}
        for button_name in self.button_pins:
            states[button_name] = self.button_pressed(button_name)
        return states
    
    def wait_for_button(self, button_name: str, timeout: float = 10.0) -> bool:
        """
        Wait for a specific button press.
        
        Args:
            button_name: Button to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            True if button was pressed, False on timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.button_pressed(button_name):
                return True
            time.sleep(self.DEBOUNCE_TIME)
        return False
    
    def wait_for_any_button(self, timeout: float = 10.0) -> Optional[str]:
        """
        Wait for any button press.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Name of pressed button, or None on timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            for button_name in self.button_pins:
                if self.button_pressed(button_name):
                    return button_name
            time.sleep(self.DEBOUNCE_TIME)
        return None
    
    def simulate_button_press(self, button_name: str):
        """
        Simulate button press for testing.
        
        Args:
            button_name: Button to simulate
        """
        if button_name in self.button_pins:
            self.button_states[button_name] = True
            if self.button_callback:
                self.button_callback(button_name, True)
            # Auto-release after short delay
            time.sleep(0.1)
            self.button_states[button_name] = False
    
    # ---------------------- Terminal Input Simulation ---------------------- #
    
    def start_terminal_listener(self):
        """Start terminal input listener thread for button simulation"""
        if self._terminal_listener_thread is not None:
            return
        
        self._stop_terminal_listener.clear()
        self._terminal_listener_thread = threading.Thread(
            target=self._terminal_input_listener,
            daemon=True
        )
        self._terminal_listener_thread.start()
    
    def _terminal_input_listener(self):
        """Thread function to listen for terminal input"""
        while not self._stop_terminal_listener.is_set():
            try:
                user_input = input()
                self._terminal_input = user_input.strip().lower()
            except EOFError:
                break
            except Exception:
                pass
    
    # ---------------------- Hardware Info ---------------------- #
    
    def get_available_leds(self) -> list:
        """Get list of available LED names"""
        return list(self.led_pins.keys())
    
    def get_available_buttons(self) -> list:
        """Get list of available button names"""
        return list(self.button_pins.keys())
    
    def get_hardware_status(self) -> Dict[str, any]:
        """Get hardware status summary"""
        return {
            "gpio_available": HAS_GPIO,
            "initialized": self._initialized,
            "hardware_generation": self.hardware_generation.value,
            "led_pins": self.led_pins,
            "button_pins": self.button_pins,
            "led_states": self.led_states.copy(),
            "led_brightness": self.led_brightness.copy(),
            "button_states": self.button_states.copy(),
        }


# Convenience functions for backwards compatibility
def turn_on_led(color: str):
    """Backwards compatible function"""
    global _default_gpio
    if _default_gpio is None:
        _default_gpio = LepmonGPIO()
    _default_gpio.turn_on_led(color)


def turn_off_led(color: str):
    """Backwards compatible function"""
    global _default_gpio
    if _default_gpio is None:
        _default_gpio = LepmonGPIO()
    _default_gpio.turn_off_led(color)


def dim_led(color: str, brightness: int):
    """Backwards compatible function"""
    global _default_gpio
    if _default_gpio is None:
        _default_gpio = LepmonGPIO()
    _default_gpio.dim_led(color, brightness)


def button_pressed(button_name: str) -> bool:
    """Backwards compatible function"""
    global _default_gpio
    if _default_gpio is None:
        _default_gpio = LepmonGPIO()
    return _default_gpio.button_pressed(button_name)


# Default GPIO instance (created on first use)
_default_gpio: Optional[LepmonGPIO] = None
