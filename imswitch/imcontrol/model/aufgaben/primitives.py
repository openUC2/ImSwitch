"""The v1 catalogue of hardware tasks, written against ``Microscope``.

Each function is a complete task: typed parameters, units in the names,
descriptions in ``Args:``. ``@task`` only records metadata, so these can also be
registered as Arkitekt actions unchanged. Loops, captures and flow steps are
control nodes (``control.py``), not tasks.
"""
from __future__ import annotations

import json
import math
import urllib.request
from typing import Annotated, List, Literal, Optional

from annotated_types import Ge, Gt, Lt

from .hardware import DerivedImage, Microscope
from .registry import TaskContext, TaskFlags, task

STAGE_LOCK = ("stage",)


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
@task(category="Stage", locks=STAGE_LOCK, effects="IRREVERSIBLE",
      flags=TaskFlags(moves_xy=True, moves_z=True, needs=("stage",)))
def move_to_position(x_um: Optional[float] = None, y_um: Optional[float] = None,
                     z_um: Optional[float] = None,
                     speed_um_s: Optional[Annotated[float, Gt(0)]] = None,
                     *, hw: Microscope) -> None:
    """Move to position

    Moves to an absolute position in the user frame and waits until it arrives.
    Axes left empty stay where they are.

    Args:
        x_um: Target X.
        y_um: Target Y.
        z_um: Target Z. Moving Z towards the sample can damage the objective.
        speed_um_s: Speed; empty uses the stage default.
    """
    target = {a: v for a, v in (("X", x_um), ("Y", y_um), ("Z", z_um)) if v is not None}
    if not target:
        raise ValueError("move_to_position: give at least one of x_um, y_um, z_um")
    hw.move(target, relative=False, speed_um_s=speed_um_s)


@task(category="Stage", locks=STAGE_LOCK, effects="IRREVERSIBLE",
      flags=TaskFlags(moves_xy=True, moves_z=True, needs=("stage",)))
def move_relative(dx_um: float = 0.0, dy_um: float = 0.0, dz_um: float = 0.0,
                  speed_um_s: Optional[Annotated[float, Gt(0)]] = None,
                  *, hw: Microscope) -> None:
    """Move by

    Moves by the given distances and waits until the stage arrives.

    Args:
        dx_um: Distance in X.
        dy_um: Distance in Y.
        dz_um: Distance in Z. Positive Z direction depends on the stand.
        speed_um_s: Speed; empty uses the stage default.
    """
    target = {a: v for a, v in (("X", dx_um), ("Y", dy_um), ("Z", dz_um)) if v}
    if target:
        hw.move(target, relative=True, speed_um_s=speed_um_s)


@task(category="Stage", locks=STAGE_LOCK, effects="IRREVERSIBLE",
      flags=TaskFlags(moves_xy=True, moves_z=True, homes=True, needs=("stage",)))
def home_axis(axis: Literal["X", "Y", "Z", "A"], *, hw: Microscope) -> None:
    """Home axis

    Drives the axis to its endstop and sets the reference.

    Args:
        axis: Axis to home.
    """
    hw.home(axis)


@task(category="Stage", locks=STAGE_LOCK, effects="IRREVERSIBLE",
      flags=TaskFlags(moves_xy=True, moves_z=True, confirm=True, needs=("stage",)))
def move_to_loading_position(*, hw: Microscope) -> None:
    """Move to loading position

    Lifts Z and drives XY to the sample loading position.
    """
    hw.move_to_named_position("loading")


# --------------------------------------------------------------------------- #
# Light, camera, focus, objective
# --------------------------------------------------------------------------- #
@task(category="Illumination", effects="REPEATABLE", choices={"source": "illumination"},
      flags=TaskFlags(emits_light=True, needs=("illumination",)))
def set_illumination(source: str, intensity: Optional[float] = None, on: bool = True,
                     *, hw: Microscope) -> None:
    """Set illumination

    Args:
        source: Light source as named in the setup.
        intensity: Value in the source's own range; empty keeps the current value.
        on: Switch the source on or off.
    """
    hw.set_illumination(source, intensity=intensity, on=on)


@task(category="Illumination", effects="REPEATABLE", flags=TaskFlags(needs=("illumination",)))
def illumination_off(*, hw: Microscope) -> None:
    """All light off

    Switches every light source off, including LED-matrix patterns.
    """
    hw.illumination_off()


@task(category="Camera", effects="REPEATABLE", choices={"detector": "detector"},
      flags=TaskFlags(needs=("camera",)))
def set_camera(exposure_ms: Optional[Annotated[float, Gt(0)]] = None,
               gain: Optional[float] = None, detector: Optional[str] = None,
               *, hw: Microscope) -> None:
    """Set camera

    Args:
        exposure_ms: Exposure time; empty keeps it.
        gain: Gain; empty keeps it.
        detector: Camera; empty uses the acquisition camera.
    """
    hw.set_camera(exposure_ms=exposure_ms, gain=gain, detector=detector)


@task(category="Focus", locks=STAGE_LOCK, effects="IRREVERSIBLE",
      choices={"channel": "illumination"},
      flags=TaskFlags(moves_z=True, emits_light=True, needs=("autofocus",)))
def autofocus(mode: Literal["software", "hardware"] = "software",
              range_um: Annotated[float, Gt(0)] = 50.0, step_um: Annotated[float, Gt(0)] = 5.0,
              channel: Optional[str] = None, intensity: Optional[float] = None,
              algorithm: Literal["LAPE", "GLVA", "JPEG"] = "LAPE",
              settle_s: float = 0.1, max_attempts: Annotated[int, Gt(0)] = 2,
              *, hw: Microscope, ctx: TaskContext) -> Optional[float]:
    """Autofocus

    Focuses at the current XY and returns the focused Z, or nothing if it failed.

    Args:
        mode: software sweeps Z and scores images; hardware uses the focus lock.
        range_um: Half range around the current Z (software mode).
        step_um: Z step (software mode).
        channel: Light switched on while focusing, and off afterwards; empty keeps the
            light as it is.
        intensity: Intensity of that light; empty keeps its current value.
        algorithm: Focus metric (software mode).
        settle_s: Wait after each Z step.
        max_attempts: Attempts before giving up.
    """
    if channel:
        hw.set_illumination(channel, intensity=intensity, on=True)
    try:
        z = hw.autofocus(mode, range_um, step_um, algorithm, settle_s, max_attempts)
    finally:
        if channel:
            hw.set_illumination(channel, on=False)
    ctx.log(f"autofocus: z = {z} µm" if z is not None else "autofocus failed")
    return z


@task(category="Objective", locks=STAGE_LOCK, effects="IRREVERSIBLE",
      flags=TaskFlags(moves_z=True, confirm=True, needs=("objective",)))
def set_objective(slot: Annotated[int, Gt(-1)], *, hw: Microscope) -> None:
    """Change objective

    Moves the turret; the objective sweeps under the sample.

    Args:
        slot: Turret slot, starting at 0.
    """
    hw.set_objective(slot)


# --------------------------------------------------------------------------- #
# Scans run by the firmware or the stage map
# --------------------------------------------------------------------------- #
def _tiles(lo: float, hi: float, step: float) -> int:
    """Tile count whose centres, step apart from ``lo``, reach ``hi``."""
    return max(1, math.ceil((hi - lo) / step - 1e-9) + 1)


@task(category="Stage scan", locks=STAGE_LOCK, effects="IRREVERSIBLE",
      choices={"sources": "illumination"},
      flags=TaskFlags(moves_xy=True, emits_light=True, needs=("stage_scan",)))
def fast_stage_scan(x_min_um: float, x_max_um: float, y_min_um: float, y_max_um: float,
                    sources: List[str], intensities: List[float],
                    exposure_ms: Annotated[float, Gt(0)] = 50.0,
                    settle_ms: Annotated[float, Ge(0)] = 90.0,
                    overlap: Annotated[float, Ge(0), Lt(1)] = 0.1,
                    timeout_s: Annotated[float, Gt(0)] = 600.0,
                    *, hw: Microscope) -> str:
    """Fast stage scan

    Firmware-timed raster over a rectangle: the board moves the stage and triggers
    camera and light at every tile, with no software round trip per tile.
    ImSwitch writes the tiles to its own OME-Zarr mosaic; the result is its path.

    Args:
        x_min_um: Left edge of the rectangle of tile centres.
        x_max_um: Right edge.
        y_min_um: Lower edge.
        y_max_um: Upper edge.
        sources: Lasers to fire at each tile, as named in the setup.
        intensities: One intensity per source.
        exposure_ms: Exposure and light-on time per tile.
        settle_ms: Wait after each step before the trigger.
        overlap: Tile overlap as a fraction of the field of view.
        timeout_s: Give up when the frames have not all arrived by then.
    """
    if len(sources) != len(intensities) or not sources:
        raise ValueError("fast_stage_scan: give one intensity per source")
    width, height = hw.fov_um()
    step_x, step_y = width * (1 - overlap), height * (1 - overlap)
    return hw.fast_stage_scan(x_min_um, y_min_um, _tiles(x_min_um, x_max_um, step_x),
                              _tiles(y_min_um, y_max_um, step_y), step_x, step_y, sources,
                              intensities, exposure_ms, settle_ms, timeout_s)


@task(category="Stage scan", locks=STAGE_LOCK, effects="IRREVERSIBLE",
      choices={"source": "illumination"},
      flags=TaskFlags(moves_xy=True, emits_light=True, needs=("stage_map",)))
def stage_prescan(x_min_um: float, x_max_um: float, y_min_um: float, y_max_um: float,
                  line_step_um: Annotated[float, Gt(0)] = 500.0,
                  speed_um_s: Annotated[float, Gt(0)] = 10000.0,
                  source: Optional[str] = None, intensity: Optional[float] = None,
                  timeout_s: Annotated[float, Gt(0)] = 600.0,
                  *, hw: Microscope) -> DerivedImage:
    """Prescan overview

    Sweeps the stage line by line over a rectangle while the camera streams and
    builds a low-resolution overview (the stage map's prescan). The overview is
    saved with the run as an OME image in stage coordinates, e.g. to find regions.

    Args:
        x_min_um: Left edge of the rectangle.
        x_max_um: Right edge.
        y_min_um: Lower edge.
        y_max_um: Upper edge.
        line_step_um: Distance between sweep lines.
        speed_um_s: Sweep speed along X.
        source: Light switched on during the sweep and off afterwards; empty keeps it.
        intensity: Its intensity; empty keeps the current value.
        timeout_s: Give up when the sweep has not finished by then.
    """
    if source:
        hw.set_illumination(source, intensity=intensity, on=True)
    try:
        return hw.stage_prescan(x_min_um, x_max_um, y_min_um, y_max_um, line_step_um,
                                speed_um_s, timeout_s)
    finally:
        if source:
            hw.set_illumination(source, on=False)


# --------------------------------------------------------------------------- #
# Digital I/O, fluidics, temperature
# --------------------------------------------------------------------------- #
@task(category="Digital I/O", effects="REPEATABLE", choices={"output": "digital_output"},
      flags=TaskFlags(needs=("digital_io",)))
def set_digital_output(output: str, high: bool, *, hw: Microscope) -> None:
    """Set digital output

    Args:
        output: Output as named in the setup.
        high: True drives the output high, false low.
    """
    hw.set_digital_output(output, high)


@task(category="Digital I/O", effects="REPEATABLE", choices={"output": "digital_output"},
      flags=TaskFlags(needs=("digital_io",)))
def pulse_digital_output(output: str, duration_ms: Annotated[float, Gt(0)] = 10.0,
                         *, hw: Microscope) -> None:
    """Pulse digital output

    Args:
        output: Output as named in the setup.
        duration_ms: Pulse length.
    """
    hw.pulse_digital_output(output, duration_ms)


@task(category="Digital I/O", effects="NONE", choices={"line": "digital_input"},
      flags=TaskFlags(needs=("digital_io",)))
def wait_for_input(line: str, edge: Literal["rising", "falling", "high", "low"] = "rising",
                   timeout_s: Optional[Annotated[float, Gt(0)]] = None,
                   *, hw: Microscope) -> bool:
    """Wait for input

    Waits until a digital input shows the edge or level. Returns false on timeout.

    Args:
        line: Input as named in the setup.
        edge: What to wait for.
        timeout_s: Give up after this long; empty waits until stopped.
    """
    return hw.wait_for_digital_input(line, edge, timeout_s)


@task(category="Fluidics", locks=("fluidics",), effects="IRREVERSIBLE",
      choices={"pump": "pump"}, flags=TaskFlags(needs=("pump",)))
def pump_volume(pump: str, volume_ul: float,
                rate_ul_min: Annotated[float, Gt(0)] = 100.0, *, hw: Microscope) -> None:
    """Pump volume

    Pumps a volume and waits until it is done. Negative volumes pump backwards.

    Args:
        pump: Pump as named in the setup.
        volume_ul: Volume to move.
        rate_ul_min: Flow rate.
    """
    hw.pump_volume(pump, volume_ul, rate_ul_min)


@task(category="Fluidics", locks=("fluidics",), effects="IRREVERSIBLE",
      choices={"pump": "pump"}, flags=TaskFlags(needs=("pump",)))
def pump_run(pump: str, rate_ul_min: float, *, hw: Microscope) -> None:
    """Start pump

    Runs the pump continuously until a stop pump task; negative rates run backwards.

    Args:
        pump: Pump as named in the setup.
        rate_ul_min: Flow rate.
    """
    hw.pump_run(pump, rate_ul_min)


@task(category="Fluidics", effects="REPEATABLE", choices={"pump": "pump"},
      flags=TaskFlags(needs=("pump",)))
def pump_stop(pump: str, *, hw: Microscope) -> None:
    """Stop pump

    Args:
        pump: Pump as named in the setup.
    """
    hw.pump_stop(pump)


@task(category="Fluidics", locks=("fluidics",), effects="REPEATABLE",
      choices={"valve": "valve"}, flags=TaskFlags(needs=("valve",)))
def set_valve(valve: str, position: str, *, hw: Microscope) -> None:
    """Switch valve

    Args:
        valve: Valve as named in the setup.
        position: Position name as defined for that valve.
    """
    hw.set_valve(valve, position)


@task(category="Environment", effects="REPEATABLE", choices={"sensor": "temperature"},
      flags=TaskFlags(needs=("temperature",)))
def set_temperature(target_c: Optional[float] = None, sensor: Optional[str] = None,
                    *, hw: Microscope) -> None:
    """Set temperature

    Sets the heater target; empty switches the heater off. Does not wait;
    combine with wait until for that.

    Args:
        target_c: Target temperature.
        sensor: Heater/sensor pair; empty uses the default.
    """
    hw.set_temperature(target_c, sensor=sensor)


@task(category="Environment", effects="NONE", choices={"sensor": "temperature"},
      flags=TaskFlags(needs=("temperature",)))
def read_temperature(sensor: Optional[str] = None, *, hw: Microscope) -> float:
    """Read temperature

    Args:
        sensor: Sensor; empty uses the default.
    """
    return hw.read_temperature(sensor)


# --------------------------------------------------------------------------- #
# Outside world
# --------------------------------------------------------------------------- #
@task(category="External", effects="UNKNOWN", flags=TaskFlags(external=True))
def http_request(url: str, method: Literal["GET", "POST", "PUT"] = "GET",
                 body_json: Optional[str] = None,
                 timeout_s: Annotated[float, Gt(0)] = 10.0) -> str:
    """HTTP request

    Calls a web service, e.g. a webhook or another instrument's API, and
    returns the response body as text.

    Args:
        url: http or https URL.
        method: HTTP method.
        body_json: JSON text sent as the request body.
        timeout_s: Give up after this long.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("http_request: only http and https URLs")
    data = None
    headers = {}
    if body_json is not None:
        json.loads(body_json)  # fail early on invalid JSON
        data = body_json.encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (scheme checked)
        return resp.read().decode(errors="replace")
