from imswitch.imcommon.model import initLogger
from .LaserManager import LaserManager


class UC2CANOpenLaserManager(LaserManager):
    """ LaserManager for a laser/LED channel driven over **CANopen**.

    Mirrors :class:`ESP32LEDLaserManager` but sends the PWM value over CAN via
    :class:`uc2canopen.UC2Client` (``laser.set_value``) instead of uc2rest.
    Each manager instance controls one channel on the laser/illumination node.

    Manager properties:

    - ``rs232device`` -- name of the ``UC2CANOpenManager`` connection to use.
    - ``channel_index`` -- laser channel index (0-based) on the node.
    - ``laserNodeId`` -- optional override of the laser CAN node id (default 21).
    """

    def __init__(self, laserInfo, name, **lowLevelManagers):
        super().__init__(laserInfo, name, isBinary=False, valueUnits='mW', valueDecimals=0)
        self._rs232manager = lowLevelManagers['rs232sManager'][
            laserInfo.managerProperties['rs232device']
        ]
        if type(laserInfo.managerProperties['channel_index']) != int:
            raise ValueError(
                "channel_index must be an integer (0-based channel on the laser node).")

        self.__logger = initLogger(self, instanceName=name)

        # Try to get commChannel if available (for emitting signals)
        self._commChannel = lowLevelManagers.get('commChannel', None)
        if self._commChannel is None:
            self.__logger.warning(
                "Communication channel not available - laser status updates won't emit signals")

        self._client = self._rs232manager.client
        self.channel_index = laserInfo.managerProperties['channel_index']
        # Node hosting this laser channel (configurable; default NODE.LASER).
        self._node = laserInfo.managerProperties.get(
            'laserNodeId', self._rs232manager.node_for_axis("LASER"))

        # despeckle is accepted for API parity but has no CANopen primitive
        self.laser_despeckle_amplitude = laserInfo.managerProperties.get('laser_despeckle_amplitude', 0.)
        self.laser_despeckle_period = laserInfo.managerProperties.get('laser_despeckle_period', 10)

        self.power = 0
        self.enabled = False
        self.setEnabled(self.enabled)

    def _setDeviceValue(self, value, getReturn=False):
        try:
            self._client.laser.set_value(channel=self.channel_index, pwm=int(value), node_id=self._node)
        except Exception as e:
            self.__logger.error(f"Could not set laser value on node {self._node}: {e}")

    def _emitStatus(self):
        if self._commChannel is not None:
            try:
                laserDict = {self.name: {"power": self.power, "enabled": bool(self.enabled)}}
                self._commChannel.sigUpdateLaserPower.emit(laserDict)
            except Exception as e:
                self.__logger.error(f"Could not emit laser status: {e}")

    def setEnabled(self, enabled, getReturn=False):
        """ Turn laser emission on/off (emits stored power when enabled). """
        self.enabled = enabled
        self._setDeviceValue(int(self.power * self.enabled), getReturn=getReturn)
        self._emitStatus()

    def setValue(self, power, getReturn=False):
        """ Set output power; only emitted to the device when enabled. """
        self.power = power
        if self.enabled:
            self._setDeviceValue(int(self.power), getReturn=getReturn)
        self._emitStatus()

    def sendTrigger(self, triggerId):
        self.__logger.warning("sendTrigger(): not supported over CANopen (v1).")

    def setGalvo(self, channel=1, frequency=1, offset=0, amplitude=1, clk_div=0,
                 phase=0, invert=1, timeout=1):
        self.__logger.warning("setGalvo(): not supported over CANopen (v1).")

    def finalize(self):
        # The CAN bus connection is owned by the rs232 UC2CANOpenManager.
        pass


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
