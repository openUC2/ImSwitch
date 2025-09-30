from __future__ import annotations

import time
from typing import Optional, Union

import serial
from imswitch.imcommon.model import initLogger


class SerialManager:
    """
    PySerial-only RS232 manager with the same public API as RS232Manager.

    Manager properties (keys in rs232Info.managerProperties):
      - port
      - encoding
      - recv_termination
      - send_termination
      - baudrate
      - bytesize
      - parity
      - stopbits
      - rtscts
      - dsrdtr
      - xonxoff
      - timeout (optional, seconds; default 0.25)
      - write_timeout (optional, seconds)

    Public methods:
      - write(data: Union[str, bytes]) -> None
      - read(nbytes: int, timeout: Optional[float] = None) -> bytes
      - query(cmd: str) -> str
      - finalize() -> None

    Notes:
      - write() accepts bytes (sent raw) or str (encoded + send_termination).
      - query() sends a string command (with send_termination) and returns a
        decoded string read until recv_termination (if set) or until timeout.
      - .serial exposes the underlying pyserial.Serial for compatibility.
    """

    def __init__(self, rs232Info, name: str, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self._name = name
        self._settings = dict(rs232Info.managerProperties)

        # Defaults
        self._encoding = self._settings.get("encoding", "utf-8")
        self._send_term_bytes = self._parse_termination(
            self._settings.get("send_termination")
        )
        self._recv_term_bytes = self._parse_termination(
            self._settings.get("recv_termination")
        )
        timeout = self._settings.get("timeout", 0.25)
        write_timeout = self._settings.get("write_timeout", None)

        # Open pyserial.Serial with mapped parameters
        port = self._settings["port"]
        baudrate = int(self._settings.get("baudrate", 9600))
        bytesize = self._map_bytesize(self._settings.get("bytesize", 8))
        parity = self._map_parity(self._settings.get("parity", "N"))
        stopbits = self._map_stopbits(self._settings.get("stopbits", 1))
        rtscts = bool(self._settings.get("rtscts", False))
        dsrdtr = bool(self._settings.get("dsrdtr", False))
        xonxoff = bool(self._settings.get("xonxoff", False))

        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                rtscts=rtscts,
                dsrdtr=dsrdtr,
                xonxoff=xonxoff,
                timeout=timeout,
                write_timeout=write_timeout,
            )
            # Small settle time on some adapters
            time.sleep(0.05)
            self.__logger.info(f"{name}: opened {port} @ {baudrate} bps")
        except Exception as e:
            self.__logger.error(f"{name}: failed to open {port}: {e}")
            raise

    # ---------------- Public API ----------------

    def query(self, arg: str) -> str:
        """Send string command and read reply as string until recv_termination or timeout."""
        if not isinstance(arg, str):
            raise TypeError("query() expects a str")
        self.write(arg)
        data = self._read_until_termination()
        return self._decode(data)

    def write(self, arg: Union[str, bytes]) -> None:
        """If str: encode + append send_termination. If bytes: write raw."""
        if isinstance(arg, str):
            payload = self._encode(arg)
            if self._send_term_bytes is not None:
                payload += self._send_term_bytes
        elif isinstance(arg, (bytes, bytearray, memoryview)):
            payload = bytes(arg)
        else:
            raise TypeError("write() expects str or bytes")
        self.__logger.debug(f"TX ({len(payload)} B): {payload!r}")
        self._ser.write(payload)
        self._ser.flush()

    def read(self, nbytes: int, timeout: Optional[float] = None) -> bytes:
        """Binary read of exactly up to nbytes (or less on timeout)."""
        if timeout is None:
            return self._ser.read(nbytes)
        # Temporarily override timeout
        prev = self._ser.timeout
        try:
            self._ser.timeout = timeout
            return self._ser.read(nbytes)
        finally:
            self._ser.timeout = prev

    def finalize(self) -> None:
        self.close()

    def close(self) -> None:
        try:
            if getattr(self, "_ser", None) and self._ser.is_open:
                self._ser.close()
                self.__logger.info(f"{self._name}: port closed")
        except Exception as e:
            self.__logger.warning(f"{self._name}: error during close: {e}")

    # For compatibility with code that expects .serial
    @property
    def serial(self) -> serial.Serial:
        return self._ser

    # ---------------- Internals ----------------

    def _read_until_termination(self) -> bytes:
        """Read until recv_termination (if set) or until timeout returns."""
        if self._recv_term_bytes:
            data = self._ser.read_until(expected=self._recv_term_bytes)
            self.__logger.debug(f"RX ({len(data)} B, until term): {data!r}")
            # If device echoes terminator, strip it for query() string return
            if data.endswith(self._recv_term_bytes):
                return data[: -len(self._recv_term_bytes)]
            return data
        # No termination configured: read whatever is available within timeout
        chunks = []
        # One blocking read, then drain input buffer
        first = self._ser.read(1)
        if first:
            chunks.append(first)
            time.sleep(0.01)
            while self._ser.in_waiting:
                chunks.append(self._ser.read(self._ser.in_waiting))
        data = b"".join(chunks)
        self.__logger.debug(f"RX ({len(data)} B, no term): {data!r}")
        return data

    def _encode(self, s: str) -> bytes:
        return s.encode(self._encoding, errors="replace")

    def _decode(self, b: bytes) -> str:
        return b.decode(self._encoding, errors="replace")

    @staticmethod
    def _parse_termination(term: Optional[Union[str, bytes]]) -> Optional[bytes]:
        if term is None:
            return None
        if isinstance(term, bytes):
            return term
        s = str(term).strip()
        if s == "" or s.lower() == "none":
            return None
        lookup = {
            "cr": b"\r",
            "\\r": b"\r",
            "lf": b"\n",
            "\\n": b"\n",
            "crlf": b"\r\n",
            "lfcr": b"\n\r",
        }
        key = s.lower()
        if key in lookup:
            return lookup[key]
        # Allow hex escape sequences like "\x50" or "0x50"
        if key.startswith("\\x") and len(key) == 4:
            try:
                return bytes([int(key[2:], 16)])
            except Exception:
                pass
        if key.startswith("0x"):
            try:
                return bytes([int(key, 16)])
            except Exception:
                pass
        return s.encode("utf-8", errors="replace")

    @staticmethod
    def _map_bytesize(val) -> int:
        try:
            n = int(val)
        except Exception:
            n = 8
        mapping = {
            5: serial.FIVEBITS,
            6: serial.SIXBITS,
            7: serial.SEVENBITS,
            8: serial.EIGHTBITS,
        }
        return mapping.get(n, serial.EIGHTBITS)

    @staticmethod
    def _map_parity(val) -> str:
        v = str(val).upper()[:1]
        mapping = {
            "N": serial.PARITY_NONE,
            "E": serial.PARITY_EVEN,
            "O": serial.PARITY_ODD,
            "M": serial.PARITY_MARK,
            "S": serial.PARITY_SPACE,
        }
        return mapping.get(v, serial.PARITY_NONE)

    @staticmethod
    def _map_stopbits(val) -> float:
        try:
            v = float(val)
        except Exception:
            v = 1.0
        if v == 1:
            return serial.STOPBITS_ONE
        if v == 1.5:
            return serial.STOPBITS_ONE_POINT_FIVE
        if v == 2:
            return serial.STOPBITS_TWO
        return serial.STOPBITS_ONE
