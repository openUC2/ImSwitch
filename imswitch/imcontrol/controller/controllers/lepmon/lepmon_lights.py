"""
LepmonOS LED Control Module

This module handles the specialized LED control for the Lepmon moth trap:
- LepiLED (UV LED) - attracts moths, PWM controlled
- Visible LED (Flash) - illumination for image capture, PWM dimmed
- Status LEDs - handled by lepmon_gpio module

Mirrors Lights.py from LepmonOS_update.

Hardware Configuration:
- LepiLED Pin: GPIO 26
- Visible LED (Dimmer) Pin: GPIO 13 (Gen2/3) or GPIO 6 (Gen1)
- PWM Frequency: 350 Hz
"""

import time
from typing import Callable, Optional
from enum import Enum


# Try to import GPIO library
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("RPi.GPIO not available - running in LED simulation mode")


class LEDState(Enum):
    """LED state enumeration"""
    OFF = 0
    DIMMING_UP = 1
    ON = 2
    DIMMING_DOWN = 3


# Pin configurations
LEPILED_PIN = 26  # UV LED
PWM_FREQUENCY = 350  # Hz for LED PWM

# Dimmer pin varies by hardware generation
DIMMER_PINS = {
    "Pro_Gen_1": 6,
    "Pro_Gen_2": 13,
    "Pro_Gen_3": 13,
}


class LepmonLights:
    """
    LED controller for Lepmon moth trap lights.
    
    Controls:
    - LepiLED (UV LED) - for attracting moths
    - Visible LED (Flash/Dimmer) - for illumination during image capture
    
    Both LEDs support smooth PWM dimming for gradual on/off transitions.
    This mirrors Lights.py from LepmonOS_update.
    """
    
    def __init__(self, 
                 hardware_generation: str = "Pro_Gen_2",
                 flash_duration: float = 0.25,
                 state_callback: Optional[Callable[[str, LEDState], None]] = None):
        """
        Initialize LED controller.
        
        Args:
            hardware_generation: Hardware version ("Pro_Gen_1", "Pro_Gen_2", "Pro_Gen_3")
            flash_duration: Duration in seconds for dimming transitions
            state_callback: Optional callback(led_name, state) for state changes
        """
        self.hardware_generation = hardware_generation
        self.flash_duration = flash_duration
        self.state_callback = state_callback
        
        # Get dimmer pin for hardware generation
        self.lepiled_pin = LEPILED_PIN
        self.dimmer_pin = DIMMER_PINS.get(hardware_generation, DIMMER_PINS["Pro_Gen_2"])
        
        # State tracking
        self.lepiled_state = LEDState.OFF
        self.visible_led_state = LEDState.OFF
        self.lepiled_brightness = 0  # 0-100
        self.visible_led_brightness = 0  # 0-100
        
        # PWM instances
        self.lepiled_pwm = None
        self.dimmer_pwm = None
        
        # Initialize hardware
        self._initialized = False
        self.initialize()
    
    def initialize(self):
        """Initialize LED hardware"""
        if HAS_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                
                # Setup LepiLED pin
                GPIO.setup(self.lepiled_pin, GPIO.OUT)
                self.lepiled_pwm = GPIO.PWM(self.lepiled_pin, PWM_FREQUENCY)
                self.lepiled_pwm.start(0)
                
                # Setup Dimmer (Visible LED) pin
                GPIO.setup(self.dimmer_pin, GPIO.OUT)
                self.dimmer_pwm = GPIO.PWM(self.dimmer_pin, PWM_FREQUENCY)
                self.dimmer_pwm.start(0)
                
                self._initialized = True
                print(f"LED control initialized (LepiLED: GPIO{self.lepiled_pin}, Dimmer: GPIO{self.dimmer_pin})")
                
            except Exception as e:
                print(f"LED initialization failed: {e}")
                self._initialized = False
        else:
            print("Running in LED simulation mode")
            self._initialized = True
    
    def cleanup(self):
        """Cleanup LED resources"""
        if HAS_GPIO and self._initialized:
            try:
                # Turn off LEDs
                if self.lepiled_pwm:
                    self.lepiled_pwm.stop()
                if self.dimmer_pwm:
                    self.dimmer_pwm.stop()
                print("LED control cleaned up")
            except Exception as e:
                print(f"LED cleanup failed: {e}")
        
        self._initialized = False
    
    def __del__(self):
        """Destructor - cleanup LEDs"""
        self.cleanup()
    
    def _notify_state_change(self, led_name: str, state: LEDState):
        """Notify callback of state change"""
        if self.state_callback:
            self.state_callback(led_name, state)
    
    # ---------------------- LepiLED (UV LED) Control ---------------------- #
    
    def lepiled_start(self):
        """
        Start LepiLED (UV LED) with smooth dimming up.
        
        Equivalent to LepmonOS LepiLED_start().
        """
        self.lepiled_state = LEDState.DIMMING_UP
        self._notify_state_change("UV_LED", LEDState.DIMMING_UP)
        
        if HAS_GPIO and self.lepiled_pwm:
            self.lepiled_pwm.start(0)
            step_delay = self.flash_duration / 100
            
            for duty_cycle in range(0, 100, 1):
                self.lepiled_pwm.ChangeDutyCycle(duty_cycle)
                self.lepiled_brightness = duty_cycle
                time.sleep(step_delay)
            
            self.lepiled_pwm.ChangeDutyCycle(100)
        
        self.lepiled_brightness = 100
        self.lepiled_state = LEDState.ON
        self._notify_state_change("UV_LED", LEDState.ON)
        print("LepiLED (UV) started")
    
    def lepiled_ende(self):
        """
        Stop LepiLED (UV LED) with smooth dimming down.
        
        Equivalent to LepmonOS LepiLED_ende().
        """
        self.lepiled_state = LEDState.DIMMING_DOWN
        self._notify_state_change("UV_LED", LEDState.DIMMING_DOWN)
        print("Dimming UV LED down")
        
        if HAS_GPIO and self.lepiled_pwm:
            self.lepiled_pwm.start(100)
            step_delay = self.flash_duration / 100
            
            for duty_cycle in range(99, 0, -1):
                self.lepiled_pwm.ChangeDutyCycle(duty_cycle)
                self.lepiled_brightness = duty_cycle
                time.sleep(step_delay)
            
            self.lepiled_pwm.ChangeDutyCycle(0)
            self.lepiled_pwm.start(0)
        
        self.lepiled_brightness = 0
        self.lepiled_state = LEDState.OFF
        self._notify_state_change("UV_LED", LEDState.OFF)
        print("LepiLED (UV) stopped")
    
    def lepiled_set_brightness(self, brightness: int):
        """
        Set LepiLED brightness directly without dimming animation.
        
        Args:
            brightness: Brightness level 0-100
        """
        brightness = max(0, min(100, brightness))
        
        if HAS_GPIO and self.lepiled_pwm:
            self.lepiled_pwm.ChangeDutyCycle(brightness)
        
        self.lepiled_brightness = brightness
        self.lepiled_state = LEDState.ON if brightness > 0 else LEDState.OFF
    
    # ---------------------- Visible LED (Dimmer/Flash) Control ---------------------- #
    
    def dim_up(self):
        """
        Turn on visible LED (flash) with smooth dimming up.
        
        Equivalent to LepmonOS dim_up().
        Used during image capture for illumination.
        """
        self.visible_led_state = LEDState.DIMMING_UP
        self._notify_state_change("Visible_LED", LEDState.DIMMING_UP)
        
        if HAS_GPIO and self.dimmer_pwm:
            self.dimmer_pwm.start(0)
            step_delay = self.flash_duration / 100
            
            for duty_cycle in range(0, 99, 1):
                self.dimmer_pwm.ChangeDutyCycle(duty_cycle)
                self.visible_led_brightness = duty_cycle
                time.sleep(step_delay)
            
            self.dimmer_pwm.start(100)
        
        self.visible_led_brightness = 100
        self.visible_led_state = LEDState.ON
        self._notify_state_change("Visible_LED", LEDState.ON)
    
    def dim_down(self):
        """
        Turn off visible LED (flash) with smooth dimming down.
        
        Equivalent to LepmonOS dim_down().
        """
        self.visible_led_state = LEDState.DIMMING_DOWN
        self._notify_state_change("Visible_LED", LEDState.DIMMING_DOWN)
        
        if HAS_GPIO and self.dimmer_pwm:
            self.dimmer_pwm.start(100)
            step_delay = self.flash_duration / 100
            
            for duty_cycle in range(99, 0, -1):
                self.dimmer_pwm.ChangeDutyCycle(duty_cycle)
                self.visible_led_brightness = duty_cycle
                time.sleep(step_delay)
            
            self.dimmer_pwm.start(0)
        
        self.visible_led_brightness = 0
        self.visible_led_state = LEDState.OFF
        self._notify_state_change("Visible_LED", LEDState.OFF)
    
    def visible_led_set_brightness(self, brightness: int):
        """
        Set visible LED brightness directly without dimming animation.
        
        Args:
            brightness: Brightness level 0-100
        """
        brightness = max(0, min(100, brightness))
        
        if HAS_GPIO and self.dimmer_pwm:
            self.dimmer_pwm.ChangeDutyCycle(brightness)
        
        self.visible_led_brightness = brightness
        self.visible_led_state = LEDState.ON if brightness > 0 else LEDState.OFF
    
    # ---------------------- Flash Sequence (for image capture) ---------------------- #
    
    def flash_for_capture(self) -> float:
        """
        Perform flash sequence for image capture.
        
        Returns:
            Total time taken for flash sequence
        """
        start_time = time.time()
        
        self.dim_up()
        # Capture happens here (caller should capture image)
        # Then call dim_down() after capture
        
        return time.time() - start_time
    
    def end_flash(self):
        """End flash sequence after capture"""
        self.dim_down()
    
    # ---------------------- Combined Control ---------------------- #
    
    def all_off(self):
        """Turn off all LEDs immediately"""
        if HAS_GPIO:
            if self.lepiled_pwm:
                self.lepiled_pwm.ChangeDutyCycle(0)
            if self.dimmer_pwm:
                self.dimmer_pwm.ChangeDutyCycle(0)
        
        self.lepiled_brightness = 0
        self.visible_led_brightness = 0
        self.lepiled_state = LEDState.OFF
        self.visible_led_state = LEDState.OFF
        
        self._notify_state_change("UV_LED", LEDState.OFF)
        self._notify_state_change("Visible_LED", LEDState.OFF)
    
    def set_flash_duration(self, duration: float):
        """
        Set the dimming transition duration.
        
        Args:
            duration: Duration in seconds
        """
        self.flash_duration = max(0.1, min(5.0, duration))
    
    # ---------------------- State Queries ---------------------- #
    
    def is_lepiled_active(self) -> bool:
        """Check if UV LED is active"""
        return self.lepiled_state == LEDState.ON
    
    def is_visible_led_active(self) -> bool:
        """Check if visible LED is active"""
        return self.visible_led_state == LEDState.ON
    
    def get_lepiled_brightness(self) -> int:
        """Get UV LED brightness level"""
        return self.lepiled_brightness
    
    def get_visible_led_brightness(self) -> int:
        """Get visible LED brightness level"""
        return self.visible_led_brightness
    
    def get_status(self) -> dict:
        """Get complete LED status"""
        return {
            "UV_LED": {
                "state": self.lepiled_state.name,
                "brightness": self.lepiled_brightness,
                "active": self.is_lepiled_active(),
            },
            "Visible_LED": {
                "state": self.visible_led_state.name,
                "brightness": self.visible_led_brightness,
                "active": self.is_visible_led_active(),
            },
            "flash_duration": self.flash_duration,
            "initialized": self._initialized,
        }


# Convenience functions for backwards compatibility
_default_lights: Optional[LepmonLights] = None


def LepiLED_start():
    """Backwards compatible function"""
    global _default_lights
    if _default_lights is None:
        _default_lights = LepmonLights()
    _default_lights.lepiled_start()


def LepiLED_ende():
    """Backwards compatible function"""
    global _default_lights
    if _default_lights is None:
        _default_lights = LepmonLights()
    _default_lights.lepiled_ende()


def dim_up():
    """Backwards compatible function"""
    global _default_lights
    if _default_lights is None:
        _default_lights = LepmonLights()
    _default_lights.dim_up()


def dim_down():
    """Backwards compatible function"""
    global _default_lights
    if _default_lights is None:
        _default_lights = LepmonLights()
    _default_lights.dim_down()
