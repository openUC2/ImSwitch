"""
OpenUC2 ESP32 Serial Controller Package

A Python interface for controlling OpenUC2 ESP32 microscope systems via USB serial.
"""

from .uc2_serial_controller import UC2SerialController, find_esp32_port, MotorPosition, LEDState, ObjectiveSlot, SamplePosition

__version__ = "1.0.0"
__author__ = "OpenUC2"
__email__ = "info@openuc2.com"

__all__ = [
    "UC2SerialController",
    "find_esp32_port", 
    "MotorPosition",
    "LEDState",
    "ObjectiveSlot",
    "SamplePosition"
]