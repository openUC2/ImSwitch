from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController


class TemperatureController(ImConWidgetController):
    """Heater PID on the UC2 board (``rs232sManager["ESP32"]``)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)
        self.Kp, self.Ki, self.Kd = 100, 0.1, 0.5
        self.controlTarget = 0.0
        self.temperatureController = self._master.rs232sManager["ESP32"]._esp32.temperature

    @APIExport(runOnUIThread=False)
    def set_temperature(self, active, Kp=None, Ki=None, Kd=None, target=None, sensorOffset=3):
        """Switch the heater PID on/off; the board regulates to target + sensorOffset (°C)."""
        self.Kp = self.Kp if Kp is None else Kp
        self.Ki = self.Ki if Ki is None else Ki
        self.Kd = self.Kd if Kd is None else Kd
        if target is not None:
            self.controlTarget = target
        self.temperatureController.set_temperature(
            active=active, Kp=self.Kp, Ki=self.Ki, Kd=self.Kd,
            target=self.controlTarget + sensorOffset)

    @APIExport(runOnUIThread=False)
    def getTemperature(self) -> dict:
        """Current heater sensor reading in °C, the setpoint and the PID gains.

        target_c is the value last passed to set_temperature (without its
        sensorOffset). Non-blocking: the reading is the board's last report.
        """
        value = self.temperatureController.get_temperature()
        return {"temperature_c": None if value is None or value <= -200 else float(value),
                "target_c": float(self.controlTarget),
                "Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd}


# Copyright (C) 2020-2021 ImSwitch developers
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

