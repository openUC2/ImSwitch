"""A peristaltic or syringe pump on a UC2 stepper axis."""
from imswitch.imcommon.model import initLogger

from .PumpManager import PumpManager


class StepperPumpManager(PumpManager):
    """managerProperties:

    - ``rs232device``: the UC2 board (default ``"ESP32"``)
    - ``axis``: its stepper axis, e.g. ``"A"`` or ``"B"``; it must not drive the stage
    - ``stepsPerUl``: motor steps per µL (calibrate by pumping into a balance)
    - ``maxRateUlMin``: highest rate accepted
    - ``acceleration``: steps/s², default the board's

    The firmware moves in steps, uc2rest in the axis' physical units
    (steps x stepSize), so volumes are converted with that axis' stepSize.
    """

    def __init__(self, pumpInfo, name, **lowLevelManagers):
        super().__init__(pumpInfo, name)
        self.__logger = initLogger(self, instanceName=name)
        board = lowLevelManagers["rs232sManager"][self.properties.get("rs232device", "ESP32")]
        self._motor = board._esp32.motor
        self.axis = str(self.properties.get("axis", "A"))
        self.stepsPerUl = float(self.properties["stepsPerUl"])
        self.acceleration = self.properties.get("acceleration")

    def _steps_per_s(self, rate_ul_min: float) -> float:
        return rate_ul_min / 60.0 * self.stepsPerUl

    def _pump_volume(self, volume_ul, rate_ul_min):
        steps = volume_ul * self.stepsPerUl
        stepSize = float(self._motor.stepSize[self._motor.axis_id(self.axis)])
        speed = self._steps_per_s(rate_ul_min)
        duration_s = abs(steps) / speed
        self.__logger.debug(f"{volume_ul} µL = {steps:.0f} steps at {speed:.0f} steps/s")
        # Blocking on the board's completion; stop() from another thread breaks it.
        self._motor.move_axis_by_name(axis=self.axis, steps=steps * stepSize, speed=speed,
                                      acceleration=self.acceleration, is_blocking=True,
                                      is_absolute=False, timeout=duration_s * 1.5 + 5)

    def _run(self, rate_ul_min):
        self._motor.move_forever(speed={self.axis: self._steps_per_s(rate_ul_min)})

    def _stop(self):
        self._motor.stop(self.axis)
