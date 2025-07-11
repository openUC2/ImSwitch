"""Lumencor SPECTRA X LaserManager for ImSwitch (RS‑232, TTL‑level)

Each instance of this manager controls one color channel (Red, Green, Cyan,
UV, Blue or Teal).  The command format follows Lumencor document 57‑10035A.

Manager properties (see device.yml):
- rs232device : name/key of low‑level RS232 channel in lowLevelManagers
- channel     : one of "red", "green", "cyan", "uv", "blue", "teal"
"""
from __future__ import annotations

import time
import struct
from imswitch.imcommon.model import initLogger
from .LaserManager import LaserManager


class LumencorLaserManager(LaserManager):
    _INIT_CMDS: tuple[bytes, bytes] = (
        b"\x57\x02\xFF\x50",  # GPIO0‑3 open‑drain
        b"\x57\x03\xAB\x50",  # GPIO4 open‑drain, GPIO5‑7 push‑pull
    )

    _CHANNEL_INFO = {
        "red":  {"bit": 0, "iic": 0x18, "sel": 0x08},
        "green": {"bit": 1, "iic": 0x18, "sel": 0x04},
        "cyan": {"bit": 2, "iic": 0x18, "sel": 0x02},
        "uv":   {"bit": 3, "iic": 0x18, "sel": 0x01},
        "blue": {"bit": 5, "iic": 0x1A, "sel": 0x01},  # addr 1A, bit0
        "teal": {"bit": 6, "iic": 0x1A, "sel": 0x02},  # addr 1A, bit1
    }

    def __init__(self, laserInfo, name, **lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)

        self._rs232 = lowLevelManagers["rs232sManager"][
            laserInfo.managerProperties["rs232device"]
        ]

        self.__chan = laserInfo.managerProperties["channel_index"]
        if self.__chan not in self._CHANNEL_INFO:
            raise ValueError(f"Unknown Lumencor channel '{self.__chan}'")

        # All channels disabled at start‑up (0x7F). Will be updated as we go.
        self.__mask = 0x7F

        # Send mandatory GPIO initialisation once per power‑cycle.
        for cmd in self._INIT_CMDS:
            self._write(cmd)
            time.sleep(0.05)

        super().__init__(
            laserInfo,
            name,
            isBinary=False,
            valueUnits="%",
            valueDecimals=0,
            isModulated=False,
        )

    # ------------------------------------------------------------------
    # Public API expected by ImSwitch
    # ------------------------------------------------------------------
    def setEnabled(self, enabled: bool) -> None:
        bit = self._CHANNEL_INFO[self.__chan]["bit"]
        if enabled:
            self.__mask &= ~(1 << bit)  # drive bit low == enable
        else:
            self.__mask |= 1 << bit     # drive bit high == disable
        cmd = bytes((0x4F, self.__mask, 0x50))
        self._write(cmd)

    def setValue(self, value: float) -> None:
        """Set channel intensity in percent (0–100).

        The DAC is 8‑bit inverted: 0x00 = max, 0xFF = off.
        """
        value = max(0.0, min(100.0, value))
        raw = int(round(255 * (100.0 - value) / 100.0))
        high_nib = (raw >> 4) & 0x0F
        low_nib = raw & 0x0F

        info = self._CHANNEL_INFO[self.__chan]
        packet = struct.pack(
            "BBBBBBB",
            0x53,            # header
            info["iic"],     # IIC address (0x18 or 0x1A)
            0x03,            # constant per spec
            info["sel"],     # DAC select bits
            0xF0 | low_nib,  # high nibble in bits 3‑0, 0xF in bits 7‑4
            high_nib << 4,   # low nibble in bits 7‑4
            0x50,            # terminator
        )
        self._write(packet)
        self.__logger.debug(
            f"{self.__chan} intensity→{value:.0f}% (raw 0x{raw:02X}) | {packet.hex()}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _write(self, payload: bytes) -> None:
        """Handle differences between bare pyserial and ImSwitch RS232 manager."""
        try:
            self.__logger.debug(f"Writing to RS232: {payload.hex()}")
            # ImSwitch v3+ low‑level RS232 manager
            self._rs232.write(payload)
        except AttributeError:
            # fall‑back to plain pyserial.Serial
            self._rs232.serial.write(payload)
        time.sleep(0.003)
