"""A rotary or pinch valve turned by a hobby servo on the UC2 board."""
from .ValveManager import ValveManager


class ServoValveManager(ValveManager):
    """managerProperties: ``rs232device`` (default ``"ESP32"``), ``channel``
    (servo channel, default 1) and ``positions``: name -> servo value
    (the firmware's ``LASERval`` with ``servo: 1``, typically degrees)."""

    def __init__(self, valveInfo, name, **lowLevelManagers):
        super().__init__(valveInfo, name)
        board = lowLevelManagers["rs232sManager"][self.properties.get("rs232device", "ESP32")]
        self._laser = board._esp32.laser
        self._channel = int(self.properties.get("channel", 1))

    def _apply(self, setting):
        self._laser.set_servo(channel=self._channel, value=int(setting), is_blocking=True)
