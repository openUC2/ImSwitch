"""
LepmonOS Modular Backend Components

This package provides modular components for the Lepmon moth trap controller,
mirroring the functionality from LepmonOS_update but integrated into ImSwitch.

Modules:
    - lepmon_config: Configuration management
    - lepmon_gpio: GPIO and button handling (RPi4)
    - lepmon_oled: OLED display control
    - lepmon_lights: LED control (UV, Visible, status)
    - lepmon_sensors: Sensor reading (temperature, humidity, light)
    - lepmon_capturing: Image capture workflow
    - lepmon_times: Sun/moon time calculations
"""

from .lepmon_config import LepmonConfig, DEFAULT_CONFIG
from .lepmon_gpio import LepmonGPIO, LED_PINS, BUTTON_PINS, HAS_GPIO
from .lepmon_oled import LepmonOLED, HAS_OLED
from .lepmon_lights import LepmonLights
from .lepmon_sensors import LepmonSensors, HAS_I2C
from .lepmon_capturing import LepmonCapturing, CaptureState
from .lepmon_times import LepmonTimes

__all__ = [
    'LepmonConfig',
    'DEFAULT_CONFIG',
    'LepmonGPIO',
    'LED_PINS',
    'BUTTON_PINS',
    'HAS_GPIO',
    'LepmonOLED',
    'HAS_OLED',
    'LepmonLights',
    'LepmonSensors',
    'HAS_I2C',
    'LepmonCapturing',
    'CaptureState',
    'LepmonTimes',
]
