from typing import List, Union

from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController


class LEDController(ImConWidgetController):
    """Simple LEDs (the ``LEDs`` block of the setup): on/off and value."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

    def _led(self, ledName: str):
        return self._master.LEDsManager[ledName]

    @APIExport()
    def getLEDNames(self) -> List[str]:
        """Device names of all LEDs, for the other LED functions."""
        return self._master.LEDsManager.getAllDeviceNames()

    @APIExport()
    def setLEDActive(self, ledName: str, active: bool) -> None:
        """Power the LED on or off."""
        self._led(ledName).setEnabled(active)

    @APIExport()
    def setLEDValue(self, ledName: str, value: Union[int, float]) -> None:
        """Set the LED value in the LED's own units (see getLEDValueRange)."""
        self._led(ledName).setValue(value)

    @APIExport()
    def getLEDValueRange(self, ledName: str) -> dict:
        """Value range and units of the LED."""
        led = self._led(ledName)
        return {"min": led.valueRangeMin, "max": led.valueRangeMax,
                "step": led.valueRangeStep, "units": led.valueUnits}


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
