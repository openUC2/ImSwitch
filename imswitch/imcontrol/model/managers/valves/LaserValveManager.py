"""A valve switched by a PWM value on a laser channel of the setup."""
from .ValveManager import ValveManager


class LaserValveManager(ValveManager):
    """managerProperties: ``laser`` (the ``lasers`` entry wired to the valve)
    and ``positions``: name -> laser value, e.g. ``{"closed": 0, "open": 1023}``.
    A value of 0 switches the channel off."""

    def __init__(self, valveInfo, name, lasersManager=None, **_lowLevelManagers):
        super().__init__(valveInfo, name)
        self.laserName = self.properties["laser"]
        self._laser = lasersManager[self.laserName]

    def _apply(self, setting):
        value = float(setting)
        self._laser.setValue(value)
        self._laser.setEnabled(value > 0)
