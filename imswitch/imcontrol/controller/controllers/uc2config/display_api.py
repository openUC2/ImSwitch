"""UC2ConfigController endpoints for the status OLED on the CAN-HAT master.

The master firmware drives a 0.91" 128x32 SSD1306 (/display_act, /display_get,
UC2-REST ``ESP32.display``). Mixed into UC2ConfigController, this file adds what
only ImSwitch knows:

* the *device name*: the host's name, which is also the Wi-Fi hotspot SSID of an
  openUC2 OS machine (``openuc2-<machine-name>``). It is pushed to the ESP at
  startup, where it is stored, so the display shows it after every ESP reboot;
* the live *status line*: any controller can write to the display by emitting
  ``commChannel.sigDisplayStatus(text, progress)``; the ExperimentController
  does so for its steps. Bursts are coalesced so the serial link is not flooded;
* a few switches for the frontend (display on/off, rotate, name, status).
"""
import os
import re
import socket
import threading
import time

from imswitch.imcommon.model import APIExport

DEVICE_NAME_ENV = "UC2_DEVICE_NAME"        # explicit override
SSID_MAX_LEN = 32                          # the hotspot SSID is the hostname cut to 32 characters
DISPLAY_MIN_INTERVAL_S = 0.5               # at most 2 status updates per second
NAME_PUSH_RETRIES = 12                     # the ESP may still be booting / not connected yet
NAME_PUSH_RETRY_S = 5
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12}$")   # Docker's default hostname: not a device name


def _dbus_hostname():
    """The host's hostname via systemd-hostnamed on the host's system bus.

    Inside the openUC2 OS container the bus socket is mounted for the pallet
    upgrade (see startPalletUpgrade); socket.gethostname() would only return the
    container id there.
    """
    from jeepney import DBusAddress, new_method_call
    from jeepney.io.blocking import open_dbus_connection
    from jeepney.wrappers import unwrap_msg
    address = DBusAddress("/org/freedesktop/hostname1", bus_name="org.freedesktop.hostname1",
                          interface="org.freedesktop.DBus.Properties")
    with open_dbus_connection(bus="SYSTEM") as conn:
        ((_, hostname),) = unwrap_msg(conn.send_and_get_reply(
            new_method_call(address, "Get", "ss", ("org.freedesktop.hostname1", "Hostname")), timeout=3))
    return hostname


def resolve_device_name(environ=None, dbus_hostname=_dbus_hostname, gethostname=socket.gethostname):
    """Name of the machine ImSwitch runs on, as it appears in the Wi-Fi list (or None).

    Order: $UC2_DEVICE_NAME, the host's hostname over D-Bus, the local hostname.
    Docker's random container id and "localhost" are no names and are skipped.
    """
    environ = os.environ if environ is None else environ
    for source in (lambda: environ.get(DEVICE_NAME_ENV), dbus_hostname, gethostname):
        try:
            name = (source() or "").strip()
        except Exception:
            continue
        if name and name != "localhost" and not _CONTAINER_ID.match(name):
            return name[:SSID_MAX_LEN]
    return None


class DisplayApiMixin:

    def _init_display(self):
        self._displayLock = threading.Lock()
        self._displayPending = None        # (status, progress) waiting for the next flush
        self._displayTimer = None
        self._displayLastSent = 0.0
        self._displayAvailable = None      # None: not probed yet, False: firmware has no display module
        self._commChannel.sigDisplayStatus.connect(self._onDisplayStatus)
        self._pushDeviceNameSoon()

    # -- device name ---------------------------------------------------------

    def _pushDeviceNameSoon(self):
        """Send the device name once the ESP answers (also after a reconnect)."""
        threading.Thread(target=self._pushDeviceNameRetry, daemon=True, name="DisplayDeviceName").start()

    def _pushDeviceNameRetry(self):
        manager = self._master.UC2ConfigManager
        for _ in range(NAME_PUSH_RETRIES):
            if manager.isConnected():
                result = self.updateDisplayDeviceName()
                if result.get("status") in ("success", "unsupported"):
                    return
            time.sleep(NAME_PUSH_RETRY_S)

    @APIExport(runOnUIThread=False)
    def getDeviceName(self):
        """Name of this machine = Wi-Fi hotspot SSID of an openUC2 OS device (None if unknown)."""
        return {"name": resolve_device_name()}

    @APIExport(runOnUIThread=False)
    def updateDisplayDeviceName(self):
        """Look the device name up again and send it to the display.

        :return: {"status": "success"|"unsupported"|"error", "name": ...}
        """
        name = resolve_device_name()
        try:
            if not self._master.UC2ConfigManager.getDisplay():
                self._displayAvailable = False
                return {"status": "unsupported", "name": name,
                        "message": "firmware has no display module"}
            self._displayAvailable = True
            if name:
                self._master.UC2ConfigManager.setDisplayName(name)
            return {"status": "success", "name": name}
        except Exception as e:
            self._logger.error(f"updateDisplayDeviceName failed: {e}")
            return {"status": "error", "message": str(e)}

    # -- status line ---------------------------------------------------------

    def _onDisplayStatus(self, status, progress=-1):
        """sigDisplayStatus handler. Only the latest value of a burst is sent."""
        if self._displayAvailable is False:
            return
        with self._displayLock:
            self._displayPending = (status, progress)
            if self._displayTimer is not None:
                return  # a flush is scheduled and will pick this value up
            wait = max(0.0, DISPLAY_MIN_INTERVAL_S - (time.monotonic() - self._displayLastSent))
            self._displayTimer = threading.Timer(wait, self._flushDisplayStatus)
            self._displayTimer.daemon = True
            self._displayTimer.start()

    def _flushDisplayStatus(self):
        with self._displayLock:
            pending, self._displayPending = self._displayPending, None
            self._displayTimer = None
            self._displayLastSent = time.monotonic()
        if pending is None:
            return
        try:
            self._master.UC2ConfigManager.setDisplayStatus(*pending)
        except Exception as e:
            self._logger.debug(f"Display status not sent: {e}")

    @APIExport(runOnUIThread=False)
    def setDisplayStatus(self, status: str = "", progress: int = -1):
        """Write the status line of the display (scrolls if long); progress 0-100 shows a bar, -1 hides it."""
        self._commChannel.sigDisplayStatus.emit(str(status), int(progress))
        return {"status": "success"}

    # -- display state and switches -----------------------------------------

    @APIExport(runOnUIThread=False)
    def getDisplayState(self):
        """{present, on, addr, name, status, progress, fw, flip, contrast, splash} or {} without display module."""
        try:
            return self._master.UC2ConfigManager.getDisplay()
        except Exception as e:
            self._logger.error(f"getDisplayState failed: {e}")
            return {}

    @APIExport(runOnUIThread=False)
    def setDisplayEnabled(self, enabled: bool = True):
        """Switch the display on or off entirely (stored on the ESP, so it survives a reboot)."""
        try:
            self._master.UC2ConfigManager.setDisplayEnabled(enabled)
            return {"status": "success", "enabled": bool(enabled)}
        except Exception as e:
            self._logger.error(f"setDisplayEnabled failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def setDisplayFlip(self, flip: bool = True):
        """Rotate the picture by 180 degrees (stored on the ESP)."""
        try:
            self._master.UC2ConfigManager.setDisplayFlip(flip)
            return {"status": "success", "flip": bool(flip)}
        except Exception as e:
            self._logger.error(f"setDisplayFlip failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def setDisplayName(self, name: str = ""):
        """Title line of the display; "" shows the default again."""
        try:
            self._master.UC2ConfigManager.setDisplayName(name)
            return {"status": "success", "name": name}
        except Exception as e:
            self._logger.error(f"setDisplayName failed: {e}")
            return {"status": "error", "message": str(e)}
