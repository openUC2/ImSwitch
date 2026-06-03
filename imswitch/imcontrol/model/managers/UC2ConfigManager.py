

from imswitch.imcommon.framework import SignalInterface
from imswitch.imcommon.model import initLogger


class UC2ConfigManager(SignalInterface):

    def __init__(self, Info, lowLevelManagers, setupInfo=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        # Keep a reference so we can persist serial-port / baudrate overrides
        # back to the setup JSON via configfiletools.saveSetupInfo.
        self._setupInfo = setupInfo

        # TODO: HARDCODED!!
        try:
            self.ESP32 = lowLevelManagers["rs232sManager"]["ESP32"]._esp32
        except Exception as e:
            self.__logger.error(f"Could not connect to ESP32 low level manager: {e}")
            return

        # Grab DigitalIn/Out Controller
        self._digitalIn = self.ESP32.digitalin
        self._digitalOut = self.ESP32.digitalout

    '''
    def saveState(self, state_general=None, state_pos=None, state_aber=None):
        if state_general is not None:
            self.state_general = state_general
        if state_pos is not None:
            self.state_pos = state_pos
        if state_aber is not None:
            self.state_aber = state_aber
    def loadDefaultConfig(self):
        return self.ESP32.config.loadDefaultConfig()
    '''
    def closeSerial(self):
        return self.ESP32.closeSerial()

    def isConnected(self):
        try:
            return self.ESP32.serial.is_connected
        except:
            return False

    def ping(self, timeout=0.5):
        """Active health-check (sends /state_get, waits for response)."""
        try:
            return self.ESP32.serial.ping(timeout=timeout)
        except Exception:
            # Older UC2-REST builds may not have .ping yet — fall back to flag.
            return self.isConnected()

    def interruptSerialCommunication(self):
        self.ESP32.serial.interruptCurrentSerialCommunication()

    def initSerial(self, port=None, baudrate=None):
        if not hasattr(self, "ESP32"):
            self.__logger.info("we do not have any esp32 initiliazed")
            return
        try:
            self.ESP32.serial.reconnect(port=port, baudrate=baudrate)
        except TypeError:
            # Older UC2-REST builds: reconnect(baudrate=...) only
            try:
                self.ESP32.serial.reconnect(baudrate=baudrate)
            except Exception:
                self.ESP32.serial.reconnect()
        except Exception:
            # last-ditch fallback to keep behaviour compatible
            self.ESP32.serial.reconnect()

    def setSerialConfig(self, port=None, baudrate=None, persist=True):
        """Apply (and optionally persist) a new serial port / baudrate.

        - Reconnects the running serial link via ``initSerial``.
        - When ``persist`` is True, writes the new values into
          ``setupInfo.rs232devices["ESP32"].managerProperties`` and saves
          the setup JSON so the change survives a restart.
        """
        self.initSerial(port=port, baudrate=baudrate)
        connected = self.isConnected()

        if persist and self._setupInfo is not None:
            try:
                rs232 = getattr(self._setupInfo, "rs232devices", None)
                esp_info = rs232.get("ESP32") if rs232 else None
                if esp_info is None:
                    self.__logger.warning(
                        "Cannot persist serial config: no rs232devices['ESP32'] in setupInfo"
                    )
                else:
                    props = dict(esp_info.managerProperties or {})
                    if port is not None:
                        props["serialport"] = port
                    if baudrate is not None:
                        props["baudrate"] = int(baudrate)
                    esp_info.managerProperties = props

                    from imswitch.imcontrol.model import configfiletools
                    options, _ = configfiletools.loadOptions()
                    configfiletools.saveSetupInfo(options, self._setupInfo)
                    self.__logger.info(
                        f"ESP32 serial config saved (port={port}, baudrate={baudrate})"
                    )
            except Exception as e:
                self.__logger.error(f"Failed to persist serial config: {e}", exc_info=True)

        return {
            "connected": connected,
            "port": port,
            "baudrate": baudrate,
        }

    def pairBT(self):
        self.ESP32.state.pairBT()

    def setDebug(self, debug):
        self.ESP32.serial.DEBUG = debug

    def restartESP(self):
        self.ESP32.state.espRestart()

    def restartCANDevice(self, device_id):
        """
        Restart a CAN device by sending a reboot command to the ESP32.

        0 - Master
        10-19 - Motor
        20-29 - Laser
        30-39 - LED
        Args:
            device_id (_type_): _description_
        """
        self.ESP32.can.reboot_remote(can_address=device_id, isBlocking=True, timeout=1)


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
