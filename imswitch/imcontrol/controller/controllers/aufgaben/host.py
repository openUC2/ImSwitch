"""ImSwitch as an Aufgaben host: the ``Microscope`` protocol over its controllers.

Stage, light, camera and autofocus go through ExperimentController's verbs
(``move_stage_xy/z``, ``grabCameraFrame``, ``set_exposure_time_gain``,
``autofocus`` ...), so the same code drives an experiment and a job. Digital
I/O, temperature and sensors use the UC2 board client, the Temperature and
I2CSensor controllers when they are loaded. Units: µm, ms, s, °C.
"""
import time
from typing import Dict, List, Set, Tuple

import numpy as np

DIGITAL_LINES = ("1", "2", "3")


class ImSwitchMicroscope:
    def __init__(self, master, logger):
        self._master = master
        self._logger = logger

    # -- helpers --------------------------------------------------------------
    def _get(self, name):
        return self._master.getController(name)

    @property
    def _exp(self):
        exp = self._get("Experiment")
        if exp is None:
            raise RuntimeError("Aufgaben needs the Experiment controller in this setup")
        return exp

    def _esp32(self):
        return getattr(getattr(self._master, "UC2ConfigManager", None), "ESP32", None)

    def _laser(self, source: str):
        if source not in self._master.lasersManager.getAllDeviceNames():
            raise ValueError(f"no light source {source!r}")
        return self._master.lasersManager[source]

    def begin(self) -> None:
        """Deterministic, post-move frames for the duration of a run."""
        self._exp._beginTriggeredAcquisition()

    def end(self) -> None:
        self._exp._endTriggeredAcquisition()

    # -- what is there ----------------------------------------------------------
    def capabilities(self) -> Set[str]:
        caps: Set[str] = set()
        exp = self._get("Experiment")
        if exp is not None:
            caps |= {"camera", "autofocus"}
            if exp.mStage is not None:
                caps.add("stage")
            if exp.allIlluNames:
                caps.add("illumination")
            if exp._ledMatrix is not None:
                caps.add("led_matrix")
        if self._get("Objective") is not None:
            caps.add("objective")
        esp = self._esp32()
        if esp is not None:
            caps.add("digital_io")
            if getattr(esp, "temperature", None) is not None:
                caps.add("temperature")
        if self._get("I2CSensor") is not None:
            caps.add("sensors")
        return caps

    def devices(self, kind: str) -> List[str]:
        if kind == "illumination":
            return list(self._master.lasersManager.getAllDeviceNames())
        if kind == "detector":
            return list(self._master.detectorsManager.getAllDeviceNames())
        if kind in ("digital_output", "digital_input"):
            return list(DIGITAL_LINES) if self._esp32() is not None else []
        if kind == "temperature":
            return ["heater"] if "temperature" in self.capabilities() else []
        return []

    # -- stage ----------------------------------------------------------------
    def get_position(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self._exp.mStage.getPosition().items()}

    def move(self, target, relative=False, speed_um_s=None, wait=True) -> Dict[str, float]:
        exp, rest = self._exp, dict(target)
        if "X" in rest and "Y" in rest:
            exp.move_stage_xy(rest.pop("X"), rest.pop("Y"), relative=relative)
        if "Z" in rest:
            exp.move_stage_z(rest.pop("Z"), relative=relative)
        for axis, value in rest.items():
            exp.mStage.move(value=value, axis=axis, is_absolute=not relative, is_blocking=True,
                            **({"speed": speed_um_s} if speed_um_s else {}))
        return self.get_position()

    def stop_stage(self) -> None:
        self._exp.mStage.stopAll()

    def home(self, axis: str) -> None:
        self._exp.mStage.doHome(axis=axis, isBlocking=True)

    def move_to_named_position(self, name: str) -> None:
        if name != "loading":
            raise NotImplementedError(f"named position {name!r}")
        self._get("Positioner").moveToSampleLoadingPosition(is_blocking=True)

    # -- light and camera ------------------------------------------------------
    def set_illumination(self, source, intensity=None, on=True) -> None:
        laser = self._laser(source)
        if intensity is not None:
            laser.setValue(intensity, getReturn=True)
        laser.setEnabled(bool(on), getReturn=True)

    def illumination_off(self) -> None:
        self._exp._switch_off_all_illumination()

    def illumination_range(self, source) -> Dict[str, float]:
        laser = self._laser(source)
        return {"min": float(laser.valueRangeMin), "max": float(laser.valueRangeMax)}

    def set_led_pattern(self, kind, **params) -> None:
        self._exp.set_led_matrix_pattern(kind=kind, **params)

    def set_camera(self, exposure_ms=None, gain=None, detector=None) -> None:
        self._exp.set_exposure_time_gain(exposure_ms, gain, None, None)

    def snap(self, detector=None) -> np.ndarray:
        return np.asarray(self._exp.grabCameraFrame())

    def pixel_size_um(self, detector=None) -> float:
        return float(self._exp.mDetector.pixelSizeUm[-1])

    def fov_um(self, detector=None) -> Tuple[float, float]:
        width, height = self._exp.mDetector.shape[:2]
        px = self.pixel_size_um()
        return width * px, height * px

    # -- focus and objective -----------------------------------------------------
    def autofocus(self, mode, range_um, step_um, algorithm, settle_s, max_attempts):
        # The job switches the light; ExperimentController must not touch it here.
        return self._exp.autofocus(mode=mode, af_range=range_um, af_resolution=step_um,
                                   illuminationChannel="", af_algorithm=algorithm,
                                   af_settle_time=settle_s, max_attempts=max_attempts)

    def set_objective(self, slot: int) -> None:
        self._get("Objective").moveToObjective(slot)

    # -- peripherals -------------------------------------------------------------
    def _board(self):
        esp = self._esp32()
        if esp is None:
            raise RuntimeError("no UC2 board connected")
        return esp

    def set_digital_output(self, name, value) -> None:
        self._board().digitalout.set_digitalout(digitaloutid=int(name), digitaloutval=int(value))

    def pulse_digital_output(self, name, duration_ms) -> None:
        # The firmware pulse has its own fixed length; longer pulses are timed here.
        out = self._board().digitalout
        out.set_digitalout(digitaloutid=int(name), digitaloutval=1)
        time.sleep(duration_ms / 1000.0)
        out.set_digitalout(digitaloutid=int(name), digitaloutval=0)

    def read_digital_input(self, name) -> bool:
        reply = self._board().digitalin.get_digitalin(digitalinid=int(name))
        reply = reply[0] if isinstance(reply, list) else reply
        return bool(reply["digitalin"]["digitalinval"])

    def wait_for_digital_input(self, name, edge, timeout_s) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        want = edge in ("rising", "high")
        if edge in ("rising", "falling"):   # an edge needs the opposite level first
            while self.read_digital_input(name) == want:
                if deadline is not None and time.monotonic() > deadline:
                    return False
                time.sleep(0.02)
        while self.read_digital_input(name) != want:
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.02)
        return True

    def pump_volume(self, pump, volume_ul, rate_ul_min) -> None:
        raise NotImplementedError("pumps arrive with the pump managers (roadmap Phase 4)")

    pump_run = pump_stop = set_valve = pump_volume

    def set_temperature(self, target_c, sensor=None) -> None:
        controller = self._get("Temperature")
        if controller is not None:
            controller.set_temperature(active=target_c is not None, target=target_c or 0.0,
                                       sensorOffset=0)
        else:
            self._board().temperature.set_temperature(active=int(target_c is not None),
                                                      target=target_c or 0.0)

    def read_temperature(self, sensor=None) -> float:
        value = self._board().temperature.get_temperature()
        if value is None or value <= -200:
            raise RuntimeError("no valid temperature reading")
        return float(value)

    def read_sensors(self) -> Dict[str, float]:
        values = self._get("I2CSensor").getLatestI2CSensorValues()
        return {k: v for k, v in values.items() if isinstance(v, (int, float))}
