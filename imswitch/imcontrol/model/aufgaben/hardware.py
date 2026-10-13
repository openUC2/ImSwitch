"""What a host application provides to the engine: the ``Microscope`` protocol.

Tasks talk to hardware only through this protocol. ImSwitch implements it in an
adapter over its managers (Phase 1); newswitch can implement the same protocol
over its own managers, and the job documents and tasks move unchanged.

Units: µm for positions, µm/s for speeds, ms for exposure, s for time, °C, µL.
Positions are in the user frame (device frame plus stage offset). Moves block
until the stage reports arrival unless ``wait=False``.

Peripheral methods are optional: a host lists what it has in ``capabilities()``
and raises ``NotImplementedError`` for the rest. The catalogue hides tasks whose
capability is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

CAPABILITIES = (
    "stage", "camera", "illumination", "led_matrix", "autofocus", "objective",
    "digital_io", "pump", "valve", "temperature", "sensors", "stage_scan", "stage_map",
)


@dataclass
class DerivedImage:
    """A 2-D image a task computed (overview, map); the engine saves it with the run.

    ``origin_um`` is the stage position (x, y) of the top-left pixel; rows run
    towards +Y, columns towards +X, like the camera tiles.
    """

    name: str
    data: np.ndarray
    pixel_size_um: float
    origin_um: Tuple[float, float]


class Microscope(Protocol):
    # -- what is there -------------------------------------------------------
    def capabilities(self) -> Set[str]: ...
    def devices(self, kind: str) -> List[str]:
        """Names of devices of one kind, e.g. 'illumination', 'detector', 'pump'."""
        ...

    # -- stage ---------------------------------------------------------------
    def get_position(self) -> Dict[str, float]: ...
    def move(self, target: Mapping[str, float], relative: bool = False,
             speed_um_s: Optional[float] = None, wait: bool = True) -> Dict[str, float]:
        """Move the given axes; returns the position afterwards."""
        ...
    def stop_stage(self) -> None: ...
    def home(self, axis: str) -> None: ...
    def move_to_named_position(self, name: Literal["loading", "transport"]) -> None: ...

    # -- light and camera ----------------------------------------------------
    def set_illumination(self, source: str, intensity: Optional[float] = None,
                         on: bool = True) -> None: ...
    def illumination_off(self) -> None: ...
    def illumination_range(self, source: str) -> Dict[str, float]: ...
    def set_led_pattern(self, kind: Literal["ring", "dpc", "off"], **params: Any) -> None:
        """LED-matrix pattern; params as ``document.LedPattern`` (radius, direction,
        intensity_r/g/b, settle_s). ``off`` switches the matrix dark."""
        ...
    def set_camera(self, exposure_ms: Optional[float] = None, gain: Optional[float] = None,
                   detector: Optional[str] = None) -> None: ...
    def snap(self, detector: Optional[str] = None) -> np.ndarray:
        """One frame taken after the call (never a stale buffer frame)."""
        ...
    def pixel_size_um(self, detector: Optional[str] = None) -> float: ...
    def bit_depth(self, detector: Optional[str] = None) -> Optional[int]:
        """Significant bits per pixel of the camera; None when the camera does not say."""
        ...
    def fov_um(self, detector: Optional[str] = None) -> Tuple[float, float]:
        """Field of view (width, height) of the detector at the current objective."""
        ...

    # -- focus and objective -------------------------------------------------
    def autofocus(self, mode: str, range_um: float, step_um: float, algorithm: str,
                  settle_s: float, max_attempts: int) -> Optional[float]:
        """Focus at the current XY with the light as it is; returns Z in µm or None."""
        ...
    def set_objective(self, slot: int) -> None: ...

    # -- scans the firmware or the stage map run (optional) -------------------
    def fast_stage_scan(self, x0_um: float, y0_um: float, nx: int, ny: int, step_x_um: float,
                        step_y_um: float, sources: Sequence[str], intensities: Sequence[float],
                        exposure_ms: float, settle_ms: float, timeout_s: float) -> str:
        """Firmware-timed raster (stage triggers camera and light); returns the data path."""
        ...
    def stage_prescan(self, x_min_um: float, x_max_um: float, y_min_um: float, y_max_um: float,
                      line_step_um: float, speed_um_s: float, timeout_s: float) -> DerivedImage:
        """Continuous sweep that builds a low-resolution overview of the rectangle."""
        ...

    # -- peripherals (optional) ----------------------------------------------
    def set_digital_output(self, name: str, value: bool) -> None: ...
    def pulse_digital_output(self, name: str, duration_ms: float) -> None: ...
    def read_digital_input(self, name: str) -> bool: ...
    def wait_for_digital_input(self, name: str, edge: Literal["rising", "falling", "high", "low"],
                               timeout_s: Optional[float]) -> bool: ...
    def pump_volume(self, pump: str, volume_ul: float, rate_ul_min: float) -> None: ...
    def pump_run(self, pump: str, rate_ul_min: float) -> None: ...
    def pump_stop(self, pump: str) -> None: ...
    def set_valve(self, valve: str, position: str) -> None: ...
    def set_temperature(self, target_c: Optional[float], sensor: Optional[str] = None) -> None: ...
    def read_temperature(self, sensor: Optional[str] = None) -> float: ...
    def read_sensors(self) -> Dict[str, float]: ...
