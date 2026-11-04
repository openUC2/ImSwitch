"""Pumps: volumes in µL, rates in µL/min; negative values pump backwards."""
import threading
from abc import ABC, abstractmethod


class PumpManager(ABC):
    """Base for one pump of the ``pumps`` setup block.

    managerProperties: ``maxRateUlMin`` (default 1000) bounds every rate.
    ``pump_volume`` blocks until the volume is moved or ``stop`` is called
    from another thread; ``run`` returns at once and pumps until ``stop``.
    """

    def __init__(self, pumpInfo, name: str):
        self.name = name
        self.properties = dict(pumpInfo.managerProperties or {})
        self.maxRateUlMin = float(self.properties.get("maxRateUlMin", 1000.0))
        self.running = False
        self._stopped = threading.Event()

    def _check_rate(self, rate_ul_min: float) -> float:
        rate = float(rate_ul_min)
        if rate == 0 or abs(rate) > self.maxRateUlMin:
            raise ValueError(f"pump {self.name}: rate {rate} µL/min outside "
                             f"(0, {self.maxRateUlMin}]")
        return rate

    def pump_volume(self, volume_ul: float, rate_ul_min: float) -> None:
        """Move ``volume_ul`` at ``abs(rate_ul_min)``; the volume's sign sets the direction."""
        rate = abs(self._check_rate(rate_ul_min))
        if volume_ul == 0:
            return
        self._stopped.clear()
        self.running = True
        try:
            self._pump_volume(float(volume_ul), rate)
        finally:
            self.running = False

    def run(self, rate_ul_min: float) -> None:
        self._stopped.clear()
        self._run(self._check_rate(rate_ul_min))
        self.running = True

    def stop(self) -> None:
        self._stopped.set()
        self._stop()
        self.running = False

    def finalize(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    @abstractmethod
    def _pump_volume(self, volume_ul: float, rate_ul_min: float) -> None: ...

    @abstractmethod
    def _run(self, rate_ul_min: float) -> None: ...

    @abstractmethod
    def _stop(self) -> None: ...
