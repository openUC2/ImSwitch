import os
import csv
import time
import threading
from collections import deque
from datetime import datetime

from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcommon.framework import Signal


class I2CSensorController(ImConWidgetController):
    """Headless controller that polls I2C environmental sensors (temperature,
    humidity, light) attached to the UC2 CAN GPIO slave via the generic I2C
    passthrough (uc2rest ``ESP32.i2c``), logs each reading to a CSV file and
    pushes it to the React frontend over the Socket.IO signal bus.

    Sensors (both optional / auto-skipped if absent):
        * SHT45   (0x44) -> temperature_c, humidity_pct
        * TSL2591 (0x29) -> lux, ch0_full, ch1_ir

    The polling cadence is user-configurable (default 10 s). The frontend
    renders a rolling window of the last ``bufferSize`` (default 50) samples;
    the same window is available over the API for initial load.
    """

    # Emitted once per poll with the latest reading dict. Auto-broadcast to
    # the frontend as {"name":"sigI2CSensorUpdate","args":{...}} (see noqt.py).
    sigI2CSensorUpdate = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # ── Polling configuration ───────────────────────────────────────
        self.pollPeriod = 10.0          # seconds between reads (user-tunable)
        self.bufferSize = 50            # rolling window length
        self.node = None               # GPIO-slave CAN node (None = default)
        self.enableSHT45 = True
        self.enableTSL2591 = True

        # ── Runtime state ───────────────────────────────────────────────
        self._buffer = deque(maxlen=self.bufferSize)
        self._running = False
        self._thread = None
        self._stopEvent = threading.Event()
        self._bufLock = threading.Lock()    # guards _buffer
        self._devLock = threading.Lock()    # serialises ESP32 serial access

        # ── ESP32 I2C handle (uc2rest) ─────────────────────────────────
        self._i2c = None
        try:
            self._i2c = self._master.rs232sManager["ESP32"]._esp32.i2c
        except Exception as e:
            self._logger.warning(
                f"I2CSensorController: ESP32 I2C interface unavailable ({e}). "
                f"Sensor reads will return empty until a device is connected.")

        # ── CSV logging (ImSwitch user data dir) ───────────────────────
        self.saveDir = os.path.join(dirtools.UserFileDirs.Root, 'i2cSensorController')
        os.makedirs(self.saveDir, exist_ok=True)
        self.csvPath = os.path.join(self.saveDir, 'i2c_sensor_log.csv')
        self._ensureCsvHeader()

        self._logger.info(
            f"I2CSensorController ready (period={self.pollPeriod}s, "
            f"csv={self.csvPath})")

    # ────────────────────────────────────────────────────────────────────
    # CSV helpers
    # ────────────────────────────────────────────────────────────────────
    _CSV_COLUMNS = ['datetime', 'timestamp', 'temperature_c', 'humidity_pct',
                    'lux', 'ch0_full', 'ch1_ir']

    def _ensureCsvHeader(self):
        if not os.path.exists(self.csvPath):
            try:
                with open(self.csvPath, 'w', newline='') as f:
                    csv.writer(f).writerow(self._CSV_COLUMNS)
            except Exception as e:
                self._logger.error(f"Could not create CSV header: {e}")

    def _appendCsv(self, r):
        try:
            with open(self.csvPath, 'a', newline='') as f:
                csv.writer(f).writerow([r.get(c) for c in self._CSV_COLUMNS])
        except Exception as e:
            self._logger.error(f"CSV append failed: {e}")

    # ────────────────────────────────────────────────────────────────────
    # Device read
    # ────────────────────────────────────────────────────────────────────
    def _readOnce(self):
        """Read all enabled sensors once. Never raises — failures leave the
        corresponding fields as None and set ok=False."""
        now = time.time()
        r = {'timestamp': now,
             'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             'temperature_c': None, 'humidity_pct': None,
             'lux': None, 'ch0_full': None, 'ch1_ir': None,
             'ok': False}
        if self._i2c is None:
            return r
        with self._devLock:
            try:
                if self.enableSHT45:
                    th = self._i2c.read_sht45(node=self.node)
                    if th:
                        r['temperature_c'] = round(float(th['temperature_c']), 3)
                        r['humidity_pct'] = round(float(th['humidity_pct']), 3)
                if self.enableTSL2591:
                    lt = self._i2c.read_tsl2591(node=self.node)
                    if lt:
                        r['lux'] = None if lt.get('lux') is None else round(float(lt['lux']), 3)
                        r['ch0_full'] = lt.get('ch0_full')
                        r['ch1_ir'] = lt.get('ch1_ir')
                r['ok'] = True
            except Exception as e:
                self._logger.error(f"I2C sensor read failed: {e}")
        return r

    # ────────────────────────────────────────────────────────────────────
    # Polling loop
    # ────────────────────────────────────────────────────────────────────
    def _pollLoop(self):
        self._logger.debug("I2C sensor poll loop started")
        while not self._stopEvent.is_set():
            r = self._readOnce()
            with self._bufLock:
                self._buffer.append(r)
            self._appendCsv(r)
            try:
                self.sigI2CSensorUpdate.emit(r)
            except Exception as e:
                self._logger.error(f"sigI2CSensorUpdate emit failed: {e}")
            # Sleep the poll period but wake immediately on stop.
            self._stopEvent.wait(self.pollPeriod)
        self._logger.debug("I2C sensor poll loop stopped")

    # ────────────────────────────────────────────────────────────────────
    # API
    # ────────────────────────────────────────────────────────────────────
    @APIExport(runOnUIThread=False)
    def startI2CSensorPolling(self, period: float = None):
        """Start (or restart) continuous polling. Optionally set the period
        (seconds) in the same call."""
        if period is not None:
            self.pollPeriod = max(0.2, float(period))
        if self._running:
            return {'running': True, 'period': self.pollPeriod}
        self._stopEvent.clear()
        self._running = True
        self._thread = threading.Thread(target=self._pollLoop,
                                        name="I2CSensorPoll", daemon=True)
        self._thread.start()
        self._logger.info(f"I2C sensor polling started (period={self.pollPeriod}s)")
        return {'running': True, 'period': self.pollPeriod}

    @APIExport(runOnUIThread=False)
    def stopI2CSensorPolling(self):
        """Stop continuous polling."""
        self._stopEvent.set()
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=max(2.0, self.pollPeriod + 0.5))
            self._thread = None
        self._logger.info("I2C sensor polling stopped")
        return {'running': False}

    @APIExport(runOnUIThread=False)
    def setI2CSensorPollPeriod(self, period: float):
        """Change the poll period (seconds). Takes effect after the current
        sleep; minimum 0.2 s."""
        self.pollPeriod = max(0.2, float(period))
        return {'period': self.pollPeriod}

    @APIExport(runOnUIThread=False)
    def setI2CSensorEnabled(self, sht45: bool = None, tsl2591: bool = None):
        """Enable/disable individual sensors (skips them on read)."""
        if sht45 is not None:
            self.enableSHT45 = bool(sht45)
        if tsl2591 is not None:
            self.enableTSL2591 = bool(tsl2591)
        return {'enableSHT45': self.enableSHT45, 'enableTSL2591': self.enableTSL2591}

    @APIExport(runOnUIThread=False)
    def getI2CSensorStatus(self):
        """Current polling state + configuration."""
        return {'running': self._running,
                'period': self.pollPeriod,
                'bufferSize': self.bufferSize,
                'available': self._i2c is not None,
                'enableSHT45': self.enableSHT45,
                'enableTSL2591': self.enableTSL2591,
                'csvPath': self.csvPath}

    @APIExport(runOnUIThread=False)
    def getI2CSensorBuffer(self):
        """Return the rolling window (up to bufferSize samples) — used by the
        frontend to seed the chart on load."""
        with self._bufLock:
            return {'buffer': list(self._buffer)}

    @APIExport(runOnUIThread=False)
    def getLatestI2CSensorValues(self):
        """One-shot read of all enabled sensors (works whether or not the
        polling loop is running). Does NOT append to the buffer/CSV."""
        return self._readOnce()

    def __del__(self):
        try:
            self._stopEvent.set()
        except Exception:
            pass
        if hasattr(super(), '__del__'):
            super().__del__()


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
