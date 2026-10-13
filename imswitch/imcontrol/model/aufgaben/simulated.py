"""A ``Microscope`` without hardware, for tests and dry runs.

It records every call in ``calls``. Frames are small images whose value encodes
the light and the stage position, so tests can check what was acquired.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .hardware import CAPABILITIES, DerivedImage


class SimulatedMicroscope:
    """``surface(x, y) -> z``: where autofocus finds focus (default: the current Z)."""

    def __init__(self, capabilities: Optional[Set[str]] = None, frame_shape=(32, 48),
                 pixel_size_um: float = 1.0, sources=("LED", "Laser488"),
                 surface: Optional[Callable[[float, float], float]] = None):
        self.surface = surface
        self.caps = set(CAPABILITIES) if capabilities is None else set(capabilities)
        self.frame_shape, self.px = frame_shape, pixel_size_um
        self.position = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.lights: Dict[str, float] = {s: 0.0 for s in sources}
        self.on: Dict[str, bool] = {s: False for s in sources}
        self.camera = {"exposure_ms": 10.0, "gain": 0.0}
        self.outputs: Dict[str, bool] = {"1": False, "2": False, "3": False}
        self.inputs: Dict[str, bool] = {"1": False, "2": False, "3": False}
        self.valves: Dict[str, str] = {"inlet": "waste"}
        self.pumped_ul: Dict[str, float] = {"peristaltic": 0.0}
        self.temperature_c, self.target_c = 22.0, None
        self.calls: List[tuple] = []

    def _call(self, *args) -> None:
        self.calls.append(args)

    # -- what is there ----------------------------------------------------------
    def capabilities(self) -> Set[str]:
        return set(self.caps)

    def devices(self, kind: str) -> List[str]:
        return {"illumination": list(self.lights), "detector": ["camera"],
                "digital_output": list(self.outputs), "digital_input": list(self.inputs),
                "pump": list(self.pumped_ul), "valve": list(self.valves),
                "temperature": ["heater"]}.get(kind, [])

    # -- stage ----------------------------------------------------------------
    def get_position(self) -> Dict[str, float]:
        return dict(self.position)

    def move(self, target, relative=False, speed_um_s=None, wait=True):
        self._call("move", dict(target), relative)
        for axis, value in target.items():
            self.position[axis] = self.position.get(axis, 0.0) + value if relative else value
        return dict(self.position)

    def stop_stage(self) -> None:
        self._call("stop_stage")

    def home(self, axis: str) -> None:
        self._call("home", axis)
        self.position[axis] = 0.0

    def move_to_named_position(self, name: str) -> None:
        self._call("named_position", name)

    # -- light and camera ------------------------------------------------------
    def set_illumination(self, source, intensity=None, on=True) -> None:
        self._call("light", source, intensity, on)
        if source not in self.lights:
            raise ValueError(f"no light source {source!r}")
        if intensity is not None:
            self.lights[source] = float(intensity)
        self.on[source] = bool(on)

    def illumination_off(self) -> None:
        self._call("all_off")
        self.on = {s: False for s in self.on}

    def illumination_range(self, source):
        return {"min": 0.0, "max": 1000.0}

    def set_led_pattern(self, kind, **params) -> None:
        self._call("led_pattern", kind, params)

    def set_camera(self, exposure_ms=None, gain=None, detector=None) -> None:
        self._call("camera", exposure_ms, gain)
        if exposure_ms is not None:
            self.camera["exposure_ms"] = exposure_ms
        if gain is not None:
            self.camera["gain"] = gain

    def snap(self, detector=None) -> np.ndarray:
        light = sum(self.lights[s] for s in self.lights if self.on[s])
        lit = tuple(s for s in self.on if self.on[s])
        self._call("snap", light, dict(self.position), self.camera["exposure_ms"], lit)
        value = min(65535, int(light + abs(self.position["Z"])))
        return np.full(self.frame_shape, value, dtype=np.uint16)

    def pixel_size_um(self, detector=None) -> float:
        return self.px

    def fov_um(self, detector=None) -> Tuple[float, float]:
        return self.frame_shape[1] * self.px, self.frame_shape[0] * self.px

    def bit_depth(self, detector=None) -> Optional[int]:
        return 12

    # -- focus and objective -----------------------------------------------------
    def autofocus(self, mode, range_um, step_um, algorithm, settle_s, max_attempts):
        self._call("autofocus", mode, range_um)
        if self.surface is not None:
            self.position["Z"] = float(self.surface(self.position["X"], self.position["Y"]))
        return self.position["Z"]

    # -- scans ----------------------------------------------------------------------
    def fast_stage_scan(self, x0_um, y0_um, nx, ny, step_x_um, step_y_um, sources, intensities,
                        exposure_ms, settle_ms, timeout_s) -> str:
        self._call("fast_stage_scan", x0_um, y0_um, nx, ny, step_x_um, step_y_um, list(sources))
        self.position.update(X=x0_um + (nx - 1) * step_x_um, Y=y0_um + (ny - 1) * step_y_um)
        return "/ExperimentController/simulated_FastStageScan.ome.zarr"

    def stage_prescan(self, x_min_um, x_max_um, y_min_um, y_max_um, line_step_um, speed_um_s,
                      timeout_s) -> DerivedImage:
        self._call("stage_prescan", x_min_um, x_max_um, y_min_um, y_max_um)
        um_per_px = 10.0
        shape = (max(1, int((y_max_um - y_min_um) / um_per_px)),
                 max(1, int((x_max_um - x_min_um) / um_per_px)))
        data = np.arange(shape[0] * shape[1], dtype=np.uint8).reshape(shape)
        return DerivedImage("overview", data, um_per_px, (x_min_um, y_min_um))

    def set_objective(self, slot: int) -> None:
        self._call("objective", slot)

    # -- peripherals -------------------------------------------------------------
    def set_digital_output(self, name, value) -> None:
        self._call("do", name, value)
        self.outputs[name] = bool(value)

    def pulse_digital_output(self, name, duration_ms) -> None:
        self._call("pulse", name, duration_ms)

    def read_digital_input(self, name) -> bool:
        return self.inputs[name]

    def wait_for_digital_input(self, name, edge, timeout_s) -> bool:
        self._call("wait_input", name, edge)
        want = edge in ("rising", "high")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while self.inputs[name] != want:
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def pump_volume(self, pump, volume_ul, rate_ul_min) -> None:
        self._call("pump", pump, volume_ul, rate_ul_min)
        self.pumped_ul[pump] += volume_ul

    def pump_run(self, pump, rate_ul_min) -> None:
        self._call("pump_run", pump, rate_ul_min)

    def pump_stop(self, pump) -> None:
        self._call("pump_stop", pump)

    def set_valve(self, valve, position) -> None:
        self._call("valve", valve, position)
        self.valves[valve] = position

    def set_temperature(self, target_c, sensor=None) -> None:
        self._call("heater", target_c)
        self.target_c = target_c

    def read_temperature(self, sensor=None) -> float:
        if self.target_c is not None:   # reaches the target in one read
            self.temperature_c = self.target_c
        return self.temperature_c

    def read_sensors(self) -> Dict[str, float]:
        return {"temperature_c": self.temperature_c}
