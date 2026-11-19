"""A peristaltic or syringe pump on a stepper axis of a positioner."""
from .PumpManager import PumpManager


class PositionerPumpManager(PumpManager):
    """managerProperties:

    - ``positioner``: the ``positioners`` entry (default the first one)
    - ``axis``: its axis that drives the pump, e.g. ``"A"``; it must not move
      the stage, and its soft limits must be off (the position only grows)
    - ``stepsPerUl``: motor steps per µL (calibrate by pumping into a balance)
    - ``maxRateUlMin``: highest rate accepted

    Positioners move in their own units (µm = steps x stepSize of that axis)
    and take speeds in steps/s, so both are derived from ``stepsPerUl``.
    ``pump_volume`` blocks until the positioner reports the move done;
    ``stop()`` from another thread stops the axis.
    """

    def __init__(self, pumpInfo, name, positionersManager=None, **_lowLevelManagers):
        super().__init__(pumpInfo, name)
        names = positionersManager.getAllDeviceNames()
        self.positionerName = self.properties.get("positioner") or names[0]
        self._stage = positionersManager[self.positionerName]
        self.axis = str(self.properties.get("axis", "A"))
        self.stepsPerUl = float(self.properties["stepsPerUl"])
        if self.axis not in self._stage.axes:
            raise ValueError(f"pump {name}: {self.positionerName} has no axis {self.axis}")

    def _step_size(self) -> float:
        return float(getattr(self._stage, "stepSizes", {}).get(self.axis, 1) or 1)

    def _steps_per_s(self, rate_ul_min: float) -> float:
        return rate_ul_min / 60.0 * self.stepsPerUl

    def _pump_volume(self, volume_ul, rate_ul_min):
        duration_s = abs(volume_ul) / rate_ul_min * 60.0
        self._stage.move(value=volume_ul * self.stepsPerUl * self._step_size(), axis=self.axis,
                         is_absolute=False, is_blocking=True,
                         speed=self._steps_per_s(rate_ul_min), timeout=duration_s * 1.5 + 5)

    def _run(self, rate_ul_min):
        self._stage.moveForever(speed={self.axis: self._steps_per_s(rate_ul_min)})

    def _stop(self):
        self._stage.forceStop(self.axis)
