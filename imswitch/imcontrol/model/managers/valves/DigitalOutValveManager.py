"""A solenoid or pinch valve on the UC2 board's digital outputs."""
from .ValveManager import ValveManager


class DigitalOutValveManager(ValveManager):
    """managerProperties:

    - ``rs232device``: the UC2 board (default ``"ESP32"``)
    - ``positions``: name -> output levels, either ``{"1": 1, "2": 0}`` (output
      id -> level) or a single level with ``line``: ``{"open": 1, "closed": 0}``
    - ``line``: the output for single-level positions
    """

    def __init__(self, valveInfo, name, **lowLevelManagers):
        super().__init__(valveInfo, name)
        board = lowLevelManagers["rs232sManager"][self.properties.get("rs232device", "ESP32")]
        self._out = board._esp32.digitalout
        self._line = self.properties.get("line")

    def _apply(self, setting):
        levels = setting if isinstance(setting, dict) else {self._line: setting}
        for line, level in levels.items():
            if line is None:
                raise ValueError(f"valve {self.name}: set managerProperties.line")
            self._out.set_digitalout(digitaloutid=int(line), digitaloutval=int(level))
