"""A pump whose speed is a PWM value on a laser channel of the setup."""
from .PumpManager import PumpManager


class LaserPumpManager(PumpManager):
    """managerProperties:

    - ``laser``: the ``lasers`` entry whose value drives the pump (e.g. an
      ESP32LEDLaserManager channel wired to the pump driver)
    - ``maxRateUlMin``: flow at ``valueAtMaxRate``; the flow is taken as
      proportional to the value (calibrate by pumping into a balance)
    - ``valueAtMaxRate``: laser value for that flow, default the laser's maximum

    PWM runs one way: negative volumes and rates are refused. The pump runs
    for volume / rate, timed here, so the volume is as good as that
    calibration.
    """

    def __init__(self, pumpInfo, name, lasersManager=None, **_lowLevelManagers):
        super().__init__(pumpInfo, name)
        self.laserName = self.properties["laser"]
        self._laser = lasersManager[self.laserName]
        self.valueAtMaxRate = float(self.properties.get("valueAtMaxRate",
                                                        self._laser.valueRangeMax))

    def _value(self, rate_ul_min: float) -> float:
        if rate_ul_min < 0:
            raise ValueError(f"pump {self.name} runs one way only (PWM)")
        return rate_ul_min / self.maxRateUlMin * self.valueAtMaxRate

    def _pump_volume(self, volume_ul, rate_ul_min):
        if volume_ul < 0:
            raise ValueError(f"pump {self.name} runs one way only (PWM)")
        self._run(rate_ul_min)
        try:
            self._stopped.wait(volume_ul / rate_ul_min * 60.0)
        finally:
            self._stop()

    def _run(self, rate_ul_min):
        self._laser.setValue(self._value(rate_ul_min))
        self._laser.setEnabled(True)

    def _stop(self):
        self._laser.setEnabled(False)
        self._laser.setValue(0)
