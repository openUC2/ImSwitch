"""ImSwitch as an Aufgaben host: the ``Microscope`` protocol over its controllers.

Stage, light, camera and autofocus go through ExperimentController's verbs
(``move_stage_xy/z``, ``grabCameraFrame``, ``set_exposure_time_gain``,
``autofocus`` ...), so the same code drives an experiment and a job. The fast
stage scan is ExperimentController's, the prescan the StageMap's. Pumps and
valves are the setup's ``pumps`` and ``valves`` managers. Digital I/O,
temperature and sensors use the UC2 board client, the Temperature and
I2CSensor controllers when they are loaded. Units: µm, ms, s, °C, µL, µL/min.
"""
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import useq

from imswitch.imcontrol.model.aufgaben.hardware import DerivedImage

LED_KINDS = {"ring": "ring", "dpc": "halves", "off": "off"}   # Aufgaben kind -> LED matrix call


def useq_plate(lab) -> Optional[useq.WellPlate]:
    """A labware definition (µm) as a useq plate, when its wells form a regular grid.

    Both use the same convention: A1 first, columns towards +X, rows towards -Y.
    """
    first = lab.wells[f"{lab.rows[0]}{lab.columns[0]}"]
    dx = abs(lab.wells[f"{lab.rows[0]}{lab.columns[1]}"].x - first.x) if len(lab.columns) > 1 else 0
    dy = abs(lab.wells[f"{lab.rows[1]}{lab.columns[0]}"].y - first.y) if len(lab.rows) > 1 else 0
    for r, row in enumerate(lab.rows):
        for c, col in enumerate(lab.columns):
            well = lab.wells.get(f"{row}{col}")
            if well is None or abs(well.x - (first.x + c * dx)) > 1 or \
                    abs(well.y - (first.y - r * dy)) > 1:
                return None
    geometry = first.geometry
    circle = geometry.shape == "circle"
    size = (2 * geometry.radius,) * 2 if circle else (geometry.width, geometry.height)
    return useq.WellPlate(rows=len(lab.rows), columns=len(lab.columns),
                          well_spacing=(dx / 1000 or 1.0, dy / 1000 or 1.0),
                          well_size=(size[0] / 1000, size[1] / 1000), circular_wells=circle,
                          name=lab.display_name or lab.load_name)


def register_labware(manager) -> List[str]:
    """Make ImSwitch labware usable as useq plate names in jobs; returns the names."""
    plates = {}
    for summary in manager.list_summaries():
        name = summary["load_name"]
        plate = useq_plate(manager.get(name))
        if plate is not None:
            plates[name] = plate
    useq.register_well_plates(plates)
    return sorted(plates)


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

    @property
    def _dio(self):
        return getattr(self._master, "digitalIOManager", None)

    def _has(self, group: str) -> bool:
        manager = getattr(self._master, group, None)
        return manager is not None and manager.hasDevices()

    def _group(self, kind: str):
        return getattr(self._master, f"{kind}sManager")

    def _device(self, kind: str, name: str):
        if not self._has(f"{kind}sManager") or name not in self.devices(kind):
            raise ValueError(f"no {kind} {name!r}; this setup has {self.devices(kind)}")
        return self._group(kind)[name]

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
        exp, esp = self._get("Experiment"), self._esp32()
        stage = getattr(exp, "mStage", None)
        checks = {
            "camera": exp is not None, "autofocus": exp is not None,
            "stage": stage is not None,
            "stage_scan": hasattr(stage, "start_stage_scanning")
            and getattr(stage, "scanTriggersFrames", True),
            "illumination": bool(getattr(exp, "allIlluNames", None)),
            "led_matrix": getattr(exp, "_ledMatrix", None) is not None,
            "objective": self._get("Objective") is not None,
            "digital_io": self._dio is not None and self._dio.available,
            "temperature": getattr(esp, "temperature", None) is not None,
            "sensors": self._get("I2CSensor") is not None,
            "stage_map": self._get("StageMap") is not None,
            "pump": self._has("pumpsManager"),
            "valve": self._has("valvesManager"),
        }
        return {cap for cap, present in checks.items() if present}

    def devices(self, kind: str) -> List[str]:
        if kind == "illumination":
            return list(self._master.lasersManager.getAllDeviceNames())
        if kind == "detector":
            return list(self._master.detectorsManager.getAllDeviceNames())
        if kind == "digital_output":
            return self._dio.outputs if self._dio is not None else []
        if kind == "digital_input":
            return self._dio.inputs if self._dio is not None else []
        if kind == "temperature":
            return ["heater"] if "temperature" in self.capabilities() else []
        if kind in ("pump", "valve"):
            return self._group(kind).getAllDeviceNames() if self._has(f"{kind}sManager") else []
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
        self._exp.set_led_matrix_pattern(kind=LED_KINDS[kind], **params)

    def set_camera(self, exposure_ms=None, gain=None, detector=None) -> None:
        self._exp.set_exposure_time_gain(exposure_ms, gain, None, None)

    def snap(self, detector=None) -> np.ndarray:
        return np.asarray(self._exp.grabCameraFrame())

    def pixel_size_um(self, detector=None) -> float:
        return float(self._exp.mDetector.pixelSizeUm[-1])

    def bit_depth(self, detector=None) -> Optional[int]:
        bits = getattr(self._exp.mDetector, "bitDepth", None)   # managers that know it say so
        return int(bits) if bits else None

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

    # -- scans -------------------------------------------------------------------------
    def fast_stage_scan(self, x0_um, y0_um, nx, ny, step_x_um, step_y_um, sources, intensities,
                        exposure_ms, settle_ms, timeout_s) -> str:
        exp = self._exp
        lasers = exp.performance_mode.prepare_illumination_parameters(list(intensities),
                                                                     list(sources))
        path = exp.startFastStageScanAcquisition(
            xstart=x0_um, xstep=step_x_um, nx=nx, ystart=y0_um, ystep=step_y_um, ny=ny,
            tsettle=settle_ms, tExposure=exposure_ms, illumination=lasers, led=0)
        writer = exp._writer_thread_ome
        try:
            if writer is not None:
                writer.join(timeout_s)   # the writer ends after the last expected frame
                if writer.is_alive():
                    raise TimeoutError(f"fast stage scan: frames still missing after {timeout_s} s")
        finally:
            exp.stopFastStageScanAcquisition()   # resets the firmware scan and the run flag
        return path

    def stage_prescan(self, x_min_um, x_max_um, y_min_um, y_max_um, line_step_um, speed_um_s,
                      timeout_s) -> DerivedImage:
        stage_map = self._get("StageMap")
        started = stage_map.runPrescan(minX=x_min_um, maxX=x_max_um, minY=y_min_um,
                                       maxY=y_max_um, dy=line_step_um, speedX=speed_um_s)
        if not started.get("success"):
            raise RuntimeError(f"prescan refused: {started.get('error')}")
        deadline = time.monotonic() + timeout_s
        while stage_map._prescanThread is not None:
            if time.monotonic() > deadline:
                stage_map.stopPrescan()
                raise TimeoutError(f"prescan still running after {timeout_s} s")
            time.sleep(0.2)
        overview = stage_map.getStageMapOverview(maxWidthPx=4096)
        if not overview.get("success"):
            raise RuntimeError(f"no overview: {overview.get('error')}")
        bounds = overview["bounds"]
        return DerivedImage("overview", stage_map._decodePreview(overview["image"]),
                            overview["umPerPixel"], (bounds["minX"], bounds["minY"]))

    # -- peripherals -------------------------------------------------------------
    def _board(self):
        esp = self._esp32()
        if esp is None:
            raise RuntimeError("no UC2 board connected")
        return esp

    def _digital(self):
        if self._dio is None:
            raise RuntimeError("no digital I/O in this setup")
        return self._dio

    def set_digital_output(self, name, value) -> None:
        self._digital().set_output(name, value)

    def pulse_digital_output(self, name, duration_ms) -> None:
        self._digital().pulse_output(name, duration_ms)

    def read_digital_input(self, name) -> bool:
        return self._digital().read_input(name)

    def wait_for_digital_input(self, name, edge, timeout_s) -> bool:
        return self._digital().wait_for_input(name, edge, timeout_s)

    def pump_volume(self, pump, volume_ul, rate_ul_min) -> None:
        self._device("pump", pump).pump_volume(volume_ul, rate_ul_min)

    def pump_run(self, pump, rate_ul_min) -> None:
        self._device("pump", pump).run(rate_ul_min)

    def pump_stop(self, pump) -> None:
        self._device("pump", pump).stop()

    def stop_pumps(self) -> None:
        if self._has("pumpsManager"):
            self._group("pump").execOnAll(lambda pump: pump.stop())

    def set_valve(self, valve, position) -> None:
        self._device("valve", valve).set_position(position)

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
