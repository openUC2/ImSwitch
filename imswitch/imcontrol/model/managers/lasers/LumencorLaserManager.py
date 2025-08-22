"""
Lumencor SPECTRA X LaserManager for ImSwitch (RS-232, TTL-level)

Instance per channel: "red", "green", "cyan", "uv", "blue", "teal".
Hex sequences mirror the proven standalone Controller.set_power()/set_enable().
"""
from __future__ import annotations

import time
from imswitch.imcommon.model import initLogger
from .LaserManager import LaserManager


class LumencorLaserManager(LaserManager):
    # Mandatory init (once per power-cycle)
    _INIT_CMDS = (
        b"\x57\x02\xFF\x50",  # GPIO0-3 open-drain
        b"\x57\x03\xAB\x50",  # GPIO4 open-drain, GPIO5-7 push-pull
    )

    # Enable-bit positions (active-low mask, base 0x7F = all disabled)
    _BIT = {
        "red": 0,     # R
        "green": 1,   # GY (Green/Yellow share in Spectra X)
        "cyan": 2,    # C
        "uv": 3,      # V
        "blue": 5,    # B
        "teal": 6,    # TN
    }

    # Exact color select prefixes (same bytes as Controller.color_cmd)
    _PREFIX = {
        "blue":  b"\x53\x1a\x03\x01",
        "teal":  b"\x53\x1a\x03\x02",
        "uv":    b"\x53\x18\x03\x01",
        "cyan":  b"\x53\x18\x03\x02",
        "green": b"\x53\x18\x03\x04",
        "red":   b"\x53\x18\x03\x08",
    }

    def __init__(self, laserInfo, name, **lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)

        # ImSwitch RS232 manager or plain pyserial via fallback in _write/_read
        self._rs232 = lowLevelManagers["rs232sManager"][
            laserInfo.managerProperties["rs232device"]
        ]

        chan = laserInfo.managerProperties.get("channel_index")
        self.__chan = str(chan).lower()
        if self.__chan not in self._PREFIX or self.__chan not in self._BIT:
            raise ValueError(f"Unknown Lumencor channel '{self.__chan}'")

        # All channels disabled initially (0x7F). Active-low lines.
        self.__mask = 0x7F

        # One-time init
        for cmd in self._INIT_CMDS:
            self._write(cmd)
            time.sleep(0.05)

        # Force a response once so we know the unit is alive
        self._force_response()

        super().__init__(
            laserInfo,
            name,
            isBinary=False,
            valueUnits="%",
            valueDecimals=0,
            isModulated=False,
        )

        # Mirror standalone script state (optional)
        self._power_pct = 0.0
        self.enabled = False

    # ---------------- ImSwitch public API ----------------
    def setEnabled(self, enabled: bool) -> None:
        """
        Same logic as Controller.set_enable(), per-channel.
        enable_code computed against base 0x7F mask; drive bit low to enable.
        """
        bit = self._BIT[self.__chan]
        if enabled:
            self.__mask &= ~(1 << bit)  # enable this channel
        else:
            self.__mask |= (1 << bit)   # disable this channel
        cmd = b"\x4f" + bytes((self.__mask,)) + b"\x50"
        self._write(cmd)
        self._force_response()
        self.enabled = bool(enabled)

    def setValue(self, value: float) -> None:
        """
        Same packet as Controller.set_power():
          prefix = _PREFIX[channel]
          power_cmd = (((4095 - power_8bit) << 12) + 0x50).to_bytes(3, 'big')
        """
        v = max(0.0, min(100.0, float(value)))
        power_8bit = int(255 * (v / 100.0))                 # 0..255
        power_cmd = (((4095 - power_8bit) << 12) + 0x50).to_bytes(3, "big")
        packet = self._PREFIX[self.__chan] + power_cmd
        self.__logger.debug(
            f"{self.__chan} set {v:.1f}% -> power_8bit={power_8bit} | {packet.hex()}"
        )
        self._write(packet)
        self._force_response()
        self._power_pct = (100.0 * power_8bit) / 255.0

    # ---------------- Helpers ----------------
    def _write(self, payload: bytes) -> None:
        self.__logger.debug(f"TX: {payload.hex()}")
        try:
            # ImSwitch v3+ RS232 manager
            self._rs232.write(payload)
        except AttributeError:
            # fallback to plain pyserial
            self._rs232.serial.write(payload)
        time.sleep(0.003)

    def _read(self, nbytes: int, timeout_s: float = 0.25) -> bytes:
        try:
            # If ImSwitch RS232 exposes read with timeout kw
            if hasattr(self._rs232, "read"):
                try:
                    return self._rs232.read(nbytes, timeout=timeout_s)
                except TypeError:
                    # some ImSwitch builds expose read(n) only
                    return self._rs232.read(nbytes)
            # Fallback to pyserial
            return self._rs232.serial.read(nbytes)
        except Exception as e:
            self.__logger.debug(f"RX error: {e}")
            return b""

    def _force_response(self) -> None:
        """Use temperature query to block, mirroring Controller._force_response()."""
        try:
            self._write(b"\x53\x91\x02\x50")
            _ = self._read(2)
        except Exception:
            pass

    # Optional utility, mirrors Controller.get_temperature()
    def getTemperatureC(self) -> float | None:
        self._write(b"\x53\x91\x02\x50")
        resp = self._read(2)
        if len(resp) == 2:
            t_c = 0.125 * (int.from_bytes(resp, "big") >> 5)
            self.__logger.debug(f"Temp = {t_c:.2f} °C")
            return t_c
        self.__logger.debug("Temp read failed")
        return None
