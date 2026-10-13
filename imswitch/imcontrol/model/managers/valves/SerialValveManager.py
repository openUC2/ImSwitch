"""A valve with its own serial controller (e.g. a selector valve)."""
from .ValveManager import ValveManager


class SerialValveManager(ValveManager):
    """managerProperties: ``rs232device`` (an ``RS232Manager`` entry of
    ``rs232devices``) and ``positions``: name -> command string, sent with
    ``query`` (the reply is logged by the RS232 driver)."""

    def __init__(self, valveInfo, name, **lowLevelManagers):
        super().__init__(valveInfo, name)
        self._port = lowLevelManagers["rs232sManager"][self.properties["rs232device"]]

    def _apply(self, setting):
        self._port.query(str(setting))
