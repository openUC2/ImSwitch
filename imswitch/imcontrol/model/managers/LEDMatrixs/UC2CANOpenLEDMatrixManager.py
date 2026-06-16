from imswitch.imcommon.model import initLogger
from .LEDMatrixManager import LEDMatrixManager

try:
    from uc2canopen import OD
except Exception:  # pragma: no cover - dep guarded by the rs232 manager
    OD = None


class UC2CANOpenLEDMatrixManager(LEDMatrixManager):
    """ LEDMatrixManager for an RGB LED matrix driven over **CANopen**.

    Mirrors :class:`ESP32LEDMatrixManager` but talks to the LED node via
    :class:`uc2canopen.UC2Client` (``led.fill`` / ``led.off`` /
    ``led.set_brightness``) plus SDO writes for pattern selection.

    Manager properties:

    - ``rs232device`` -- name of the ``UC2CANOpenManager`` connection to use.
    - ``Nx`` / ``Ny`` -- matrix dimensions.
    - ``ledNodeId`` -- optional override of the LED CAN node id (default 20).

    Limitation: per-pixel / region helpers (single pixel, individual pattern,
    rings, circles, halves) require segmented/DOMAIN SDO transfers
    (``0x2210``/``0x2211``) that the expedited-only client can't issue, so they
    are logged no-ops. Uniform fill, brightness and pattern selection work.
    """

    def __init__(self, LEDMatrixInfo, name, **lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.power = 0
        self.I_max = 255
        self.intensity = 0
        self.enabled = False
        self.channel_index = "led"
        self._patternId = 0

        self.Nx = LEDMatrixInfo.managerProperties.get('Nx', 4)
        self.Ny = LEDMatrixInfo.managerProperties.get('Ny', 4)
        self.SpecialPattern1 = LEDMatrixInfo.managerProperties.get('SpecialPattern1', 0)
        self.SpecialPattern2 = LEDMatrixInfo.managerProperties.get('SpecialPattern2', 0)
        self.NLeds = self.Nx * self.Ny

        self._rs232manager = lowLevelManagers['rs232sManager'][
            LEDMatrixInfo.managerProperties['rs232device']
        ]
        self._client = self._rs232manager.client
        self._node = LEDMatrixInfo.managerProperties.get(
            'ledNodeId', self._rs232manager.node_for_axis("LED"))

        super().__init__(LEDMatrixInfo, name, isBinary=False, valueUnits='mW', valueDecimals=0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _toRGB(self, state, intensity):
        """ Resolve a single uniform (r, g, b) from the loose ESP32-style args. """
        if isinstance(state, (tuple, list)) and len(state) >= 3:
            return int(state[0]) & 0xFF, int(state[1]) & 0xFF, int(state[2]) & 0xFF
        if isinstance(intensity, (tuple, list)) and len(intensity) >= 3:
            return int(intensity[0]) & 0xFF, int(intensity[1]) & 0xFF, int(intensity[2]) & 0xFF
        v = 255 if bool(state) else 0
        return v, v, v

    # ------------------------------------------------------------------
    # supported operations (uniform fill / brightness / pattern)
    # ------------------------------------------------------------------
    def setAll(self, state=(0, 0, 0), intensity=None, getReturn=True):
        r, g, b = self._toRGB(state, intensity)
        if isinstance(intensity, (int, float)):
            try:
                self._client.led.set_brightness(int(intensity) & 0xFF, node_id=self._node)
            except Exception as e:
                self.__logger.error(f"set_brightness failed: {e}")
        try:
            if (r, g, b) == (0, 0, 0) and not bool(state):
                self._client.led.off(node_id=self._node)
            else:
                self._client.led.fill(r=r, g=g, b=b, node_id=self._node)
        except Exception as e:
            self.__logger.error(f"setAll fill/off failed: {e}")

    def setValue(self, intensity, getReturn=False):
        """ Uniform white at the given intensity. """
        self.intensity = intensity
        self.setAll(1, (intensity, intensity, intensity))

    def setLEDIntensity(self, intensity=(0, 0, 0)):
        if isinstance(intensity, (tuple, list)):
            self.setAll(tuple(intensity))
        else:
            try:
                self._client.led.set_brightness(int(intensity) & 0xFF, node_id=self._node)
            except Exception as e:
                self.__logger.error(f"setLEDIntensity failed: {e}")

    def setEnabled(self, enabled):
        """ Turn the matrix on/off. """
        self.enabled = enabled
        if not enabled:
            try:
                self._client.led.off(node_id=self._node)
            except Exception as e:
                self.__logger.error(f"led.off failed: {e}")

    def setPattern(self, pattern):
        """ Select a firmware pattern by id (LED_PATTERN_ID). """
        self._patternId = pattern
        if OD is None:
            return
        try:
            self._rs232manager.sdo_write(self._node, OD.LED_PATTERN_ID, 0, int(pattern), "u8")
        except Exception as e:
            self.__logger.error(f"setPattern failed: {e}")

    def getPattern(self):
        if OD is None:
            return self._patternId
        try:
            self._patternId = self._rs232manager.sdo_read(self._node, OD.LED_PATTERN_ID, 0, "u8")
        except Exception as e:
            self.__logger.error(f"getPattern failed: {e}")
        return self._patternId

    def setStatus(self, status: str = "idle"):
        # No firmware status primitive over CAN; map to a no-op (kept for API parity).
        self.__logger.debug(f"setStatus({status}): no CANopen status primitive; ignored.")

    # ------------------------------------------------------------------
    # per-pixel / region helpers — need segmented/DOMAIN SDO (not available)
    # ------------------------------------------------------------------
    def setIndividualPattern(self, pattern, getReturn=False):
        self.__logger.warning(
            "setIndividualPattern(): per-pixel data needs segmented SDO (0x2210), "
            "not supported over CANopen (v1).")
        return None

    def setLEDSingle(self, indexled=0, state=(0, 0, 0)):
        self.__logger.warning(
            "setLEDSingle(): single-pixel write needs 5-byte SDO (0x2211), "
            "not supported over CANopen (v1).")

    def setHalves(self, region="left", intensity=(255, 255, 255), getReturn=True, timeout=1):
        self.__logger.warning("setHalves(): not supported over CANopen (v1).")

    def setRing(self, radius=4, intensity=(255, 255, 255), getReturn=True, timeout=1):
        self.__logger.warning("setRing(): not supported over CANopen (v1).")

    def setCircle(self, radius=4, intensity=(255, 255, 255), getReturn=True, timeout=1):
        self.__logger.warning("setCircle(): not supported over CANopen (v1).")

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
