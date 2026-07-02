

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
        return self.ESP32.serial.closeSerial()

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

    def getFirmwareInfo(self, timeout=2):
        """Identity of the USB-connected ESP32 master.

        Returns {name, version, date, author, pindef, isMaster, connected,
        serialport}. The build date and pindef are the fields that matter for
        telling boards apart. Best-effort: returns a mostly-empty dict if the
        firmware/serial layer is unavailable.
        """
        info = {}
        try:
            info = self.ESP32.state.get_firmware_info(timeout=timeout) or {}
        except Exception as e:
            self.__logger.debug(f"getFirmwareInfo via state failed: {e}")
        # enrich with what mserial already captured at connect time (no round-trip)
        try:
            cached = getattr(self.ESP32.serial, "firmware_info", None) or {}
            for k, v in cached.items():
                if v and not info.get(k):
                    info[k] = v
        except Exception:
            pass
        try:
            info["connected"] = bool(self.ESP32.serial.is_connected)
            port = getattr(self.ESP32.serial, "serialport", None)
            info["serialport"] = getattr(port, "device", port) if port else None
        except Exception:
            pass
        return info

    def setJoystickDirection(self, axis, inverted=False, timeout=1):
        """Invert (or un-invert) the PS-controller joystick for one motor axis.

        axis: "A"/"X"/"Y"/"Z" (or 0..3). Maps to motor.set_joystick_direction.
        """
        return self.ESP32.motor.set_joystick_direction(axis=axis, inverted=bool(inverted), timeout=timeout)

    def getJoystickDirection(self, axis=None, timeout=1):
        """Read joystick inversion. With axis=None returns a list of
        {axis, inverted} for all axes; otherwise the bool for that axis."""
        return self.ESP32.motor.get_joystick_direction(axis=axis, timeout=timeout)

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

    # ──────────────────────────────────────────────────────────────────────
    # CAN-bus power & emergency-stop (safety)
    # ──────────────────────────────────────────────────────────────────────
    def setBusPower(self, enable=True):
        """Enable (default) / disable the high-current CAN-bus power that feeds
        the slaves. Maps to ESP32.state.set_power."""
        return self.ESP32.state.set_power(int(bool(enable)))

    def getBusPower(self):
        """Current CAN-bus power state: 1=ON, 0=OFF, or None if unavailable."""
        try:
            return int(self.ESP32.state.get_power())
        except Exception:
            return None

    def getEstop(self):
        """E-stop diagnostics dict {estopPolarity, estopRaw, estopActive}."""
        try:
            return self.ESP32.state.get_estop()
        except Exception:
            return {}

    def isEmergencyActive(self):
        """Last known emergency-stop state as reported by the firmware (cached,
        no serial round-trip)."""
        try:
            return bool(self.ESP32.state.is_emergency_active())
        except Exception:
            return False

    def registerEmergencyCallback(self, callbackfct):
        """Register a callback invoked on emergency (E-stop) events. The callback
        receives the firmware "emergency" dict, e.g.
        {"active":1,"reason":"estop","msg":"..."}."""
        try:
            self.ESP32.state.register_emergency_callback(callbackfct)
            return True
        except Exception as e:
            self.__logger.error(f"Could not register emergency callback: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # GPIO slave / collision detector (CAN node, default id 60)
    # ──────────────────────────────────────────────────────────────────────
    def registerCollisionCallback(self, callbackfct):
        """Register a callback invoked on asynchronously pushed GPIO events
        (collision trip/clear, E-stop edge on the GPIO slave). The callback
        receives the event dict, e.g. {"event":1,"trip":1,"filtered":2365,...}."""
        try:
            self.ESP32.gpio.register_collision_callback(callbackfct)
            return True
        except Exception as e:
            self.__logger.error(f"Could not register collision callback: {e}")
            return False

    def getGpioStatus(self, node=None, timeout=1):
        """Poll the collision detector: {mean, filtered, raw, reference,
        threshold, sensitivity, trip, estop}. Triggers SDO reads on the CAN
        bus — the slave never broadcasts sensor values on its own."""
        try:
            return self.ESP32.gpio.get_status(node=node, timeout=timeout)
        except Exception as e:
            self.__logger.error(f"getGpioStatus failed: {e}")
            return {}

    def setCollisionThreshold(self, threshold, node=None):
        return self.ESP32.gpio.set_threshold(threshold, node=node)

    def setCollisionSensitivity(self, sensitivity, node=None):
        return self.ESP32.gpio.set_sensitivity(sensitivity, node=node)

    def setCollisionReference(self, reference, node=None):
        return self.ESP32.gpio.set_reference(reference, node=node)

    def calibrateCollisionReference(self, node=None):
        """Slave takes its current rolling mean as the new reference (NVS)."""
        return self.ESP32.gpio.calibrate(node=node)

    # ──────────────────────────────────────────────────────────────────────
    # Fan & board temperature
    # ──────────────────────────────────────────────────────────────────────
    def getFan(self, blocking=True):
        """Fan state dict {mode,wiper,manual,rpm,stalled,kick,tempC,curve}."""
        try:
            return self.ESP32.fan.get_fan(blocking=blocking)
        except Exception:
            return {}

    def setFanMode(self, mode="auto", wiper=None):
        """Set fan mode 'auto'|'manual'|'off'. wiper 0-127 used for 'manual'."""
        return self.ESP32.fan.set_mode(mode=mode, wiper=wiper)

    def getTemperature(self):
        """Board/air temperatures {pcb,air,esp,pcb_ok,air_ok}."""
        try:
            return self.ESP32.fan.get_temp()
        except Exception:
            return {}


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
