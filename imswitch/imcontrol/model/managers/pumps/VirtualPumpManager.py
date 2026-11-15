"""A pump without hardware: it takes the time the volume needs (divided by
``speedup``) and counts what it moved, so jobs and tests can check it."""
import time

from .PumpManager import PumpManager


class VirtualPumpManager(PumpManager):
    """managerProperties: ``maxRateUlMin``, ``speedup`` (default 1; 60 runs a
    minute of pumping in a second)."""

    def __init__(self, pumpInfo, name, **_lowLevelManagers):
        super().__init__(pumpInfo, name)
        self.speedup = float(self.properties.get("speedup", 1.0))
        self.pumpedUl = 0.0
        self._rate = 0.0
        self._since = None

    def _pump_volume(self, volume_ul, rate_ul_min):
        duration_s = abs(volume_ul) / rate_ul_min * 60.0 / self.speedup
        start = time.monotonic()
        if self._stopped.wait(duration_s):        # stopped early: count what was moved
            fraction = (time.monotonic() - start) / duration_s
            self.pumpedUl += volume_ul * min(1.0, fraction)
        else:
            self.pumpedUl += volume_ul

    def _run(self, rate_ul_min):
        self._settle()
        self._rate, self._since = rate_ul_min, time.monotonic()

    def _stop(self):
        self._settle()
        self._rate, self._since = 0.0, None

    def _settle(self):
        if self._since is not None:
            minutes = (time.monotonic() - self._since) * self.speedup / 60.0
            self.pumpedUl += self._rate * minutes
