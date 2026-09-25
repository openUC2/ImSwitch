"""Pure planning for a wellplate run: request (ParameterValue) -> what to acquire.

No hardware, no logging, no ``self``. ``ExperimentController._start_wellplate_experiment``
calls these, applies the side effects (LED, detector, focus map, folder) and hands
the result to an execution mode.  Keeping this side-effect free is what makes it
testable: see ``_test/unit/test_scan_plan.py``.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .models import NeighborPoint, ParameterValue


@dataclass
class ChannelPlan:
    """Parallel per-channel arrays (same length, same order) plus resolved flags."""
    sources: List[str]
    kinds: List[str]                  # "default" | "ring" | "dpc"
    intensities: List[float]
    params: Dict[str, Dict[str, Any]]  # LED-matrix radius/RGB, keyed by source name
    gains: List[float]
    exposures: List[float]
    passthrough: bool                 # no channel active: acquire without touching the device
    keep_illumination_on: bool
    performance_mode: bool            # request flag, forced off when LED-matrix channels are in
    notes: List[str] = field(default_factory=list)  # human-readable decisions, for the log


def resolve_channels(p: ParameterValue) -> ChannelPlan:
    """Merge conventional and synthetic (LED-matrix) channels into aligned arrays."""
    sources = list(p.illumination or [])
    intensities = list(p.illuIntensities or [])
    kinds = ["default"] * len(sources)
    params: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []

    gains = list(p.gains or [])
    exposures = list(p.exposureTimes or [])
    if len(gains) != len(sources):
        gains = [-1] * len(sources)
    if len(exposures) != len(sources):
        exposures = [exposures[0] if exposures else 0] * len(sources)
    intensities = (intensities + [0] * len(sources))[: len(sources)]

    for sc in (p.syntheticChannels or []):
        if not sc.enabled or sc.rgb_max <= 0:
            continue
        sources.append(sc.name)
        kinds.append(sc.kind)
        intensities.append(sc.rgb_max)
        params[sc.name] = {
            "radius": int(sc.radius) if sc.radius is not None else 0,
            "intensityR": int(sc.intensityR or 0),
            "intensityG": int(sc.intensityG or 0),
            "intensityB": int(sc.intensityB or 0),
        }
        gains.append(float(sc.gain) if sc.gain is not None else -1)
        exposures.append(float(sc.exposure) if sc.exposure is not None
                         else (exposures[0] if exposures else 0))

    n_active = sum(1 for v in intensities if v and v > 0)
    passthrough = n_active == 0
    keep_on = {"on": True, "off": False}.get(p.keepIlluminationOn, n_active == 1)

    performance_mode = bool(p.performanceMode)
    if performance_mode and any(k in ("ring", "dpc") for k in kinds):
        performance_mode = False
        notes.append("LED-matrix channel selected: performance mode off (patterns need per-frame software control)")

    if passthrough:
        # One sentinel channel so the loop runs once per position; gain<0 and
        # exposure<=0 make set_exposure_time_gain a no-op, keep_on suppresses
        # set_laser_power.
        sources = [sources[0]] if sources else ["default"]
        kinds, intensities, gains, exposures, keep_on = ["default"], [1], [-1], [0], True
        notes.append(f"No illumination set: passthrough via '{sources[0]}', device settings untouched")
    notes.append(f"Illumination: setting={p.keepIlluminationOn}, activeChannels={n_active}, keepOn={keep_on}")

    return ChannelPlan(sources, kinds, intensities, params, gains, exposures,
                       passthrough, keep_on, performance_mode, notes)


def z_offsets(p: ParameterValue) -> List[float]:
    """Relative Z offsets (µm) each tile adds to its own base Z; [0.0] without a stack."""
    if not p.zStack:
        return [0.0]
    if getattr(p, "zStackMode", "range") == "individual" and p.zStackOffsets:
        return sorted(float(v) for v in p.zStackOffsets)
    return [float(v) for v in np.arange(p.zStackMin, p.zStackMax + p.zStackStepSize, p.zStackStepSize)]


def writer_flags(p: ParameterValue, snake_tiles: List[List[Dict[str, Any]]]) -> Dict[str, bool]:
    """Which outputs to write; single-position scans get one TIFF per position, never a stitch."""
    single = all(len(t) == 1 for t in snake_tiles)
    return {
        "write_tiff": bool(p.ome_write_tiff),
        "write_zarr": bool(p.ome_write_zarr),
        "write_stitched_tiff": bool(p.ome_write_stitched_tiff) and not single,
        "write_single_tiff": single,
        "write_individual_tiffs": bool(p.ome_write_individual_tiffs),
    }



# ── Regions ─────────────────────────────────────────────────────────────────

def build_region_meta(area_name, area_type, well_id=None, well_row=None,
                      well_column=None, labware_load_name=None, condition_label=None):
    """Per-region metadata, carried once per region (not on every FOV)."""
    return {
        "areaName": area_name,
        "areaType": area_type,
        "wellId": well_id,
        "wellRow": well_row,
        "wellColumn": well_column,
        "labwareLoadName": labware_load_name,
        "conditionLabel": condition_label,
    }


def region_fovs(region_id, raw_fovs):
    """Validate and normalise one region's FOV list before anything is stored.

    Two rules: a bad coordinate raises here naming itself rather than
    failing deep in the workflow, and Z is homogeneous — a region carries a
    Z on every FOV or on none.
    """
    raw_fovs = list(raw_fovs)
    if not raw_fovs:
        raise ValueError(f"Region {region_id!r} has no positions")

    has_z = all(fov.get("z") is not None for fov in raw_fovs)
    fovs = []
    for index, fov in enumerate(raw_fovs):
        try:
            x = float(fov["x"])
            y = float(fov["y"])
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(
                f"Position {index} of region {region_id!r} has no usable "
                f"x/y coordinate: {fov!r}"
            ) from err
        entry = {
            "iterator": fov.get("iterator", index),
            "region_id": region_id,
            "iX": int(fov.get("iX", 0)),
            "iY": int(fov.get("iY", 0)),
            "x": x,
            "y": y,
            "z": float(fov["z"]) if has_z else None,
        }
        fovs.append(entry)
    return fovs


def regions_to_areas(regions, region_meta):
    """Region id + name + XY bounds, derived from the regions themselves so
    the focus-map phase and the acquisition loop agree on what a region is.
    """
    areas = []
    for fovs in regions:
        region_id = fovs[0]["region_id"]
        xs = [fov["x"] for fov in fovs]
        ys = [fov["y"] for fov in fovs]
        areas.append({
            "areaId": region_id,
            "areaName": region_meta.get(region_id, {}).get("areaName") or region_id,
            "bounds": {
                "minX": min(xs), "maxX": max(xs),
                "minY": min(ys), "maxY": max(ys),
            },
        })
    return areas


def build_scan_regions(mExperiment, current_position, log):
    """Convert the experiment's coordinates into per-region FOV lists.

    ``current_position`` is a callable (the stage's getPosition), used only
    when the request carries no coordinates at all.

    Returns ``(regions, region_meta)``: one ordered FOV list per region,
    plus that region's well/labware/condition metadata.

    Nothing is generated or reordered here — traversal order is decided in
    one place, the frontend's ``CoordinateCalculator``.

    ``region_id`` is a string, always, and is a region's only identifier.
    """
    regions = []
    region_meta = {}

    # Preferred: pre-calculated coordinates from the frontend's scanAreas.
    if mExperiment.scanAreas:
        for area in mExperiment.scanAreas:
            region_id = str(area.areaId)
            fovs = region_fovs(region_id, [
                {
                    "iterator": pos.index,
                    "iX": pos.iX,
                    "iY": pos.iY,
                    "x": pos.x,
                    "y": pos.y,
                    "z": pos.z if pos.z is not None else area.centerPosition.z,
                }
                for pos in area.positions
            ])
            regions.append(fovs)
            region_meta[region_id] = build_region_meta(
                area.areaName, area.areaType, area.wellId, area.wellRow,
                area.wellColumn, area.labwareLoadName, area.conditionLabel,
            )
            log.info(
                f"Scan region [{region_id}] '{area.areaName}': {len(fovs)} position(s)"
                f"{' with per-position Z' if fovs[0]['z'] is not None else ''}"
            )
        return regions, region_meta

    # Fallback: pointList with a pre-ordered neighbourPointList.
    if mExperiment.pointList:
        for index, center in enumerate(mExperiment.pointList):
            region_id = f"area_{index}"
            neighbours = center.neighborPointList or [
                NeighborPoint(x=center.x, y=center.y, z=center.z, iX=0, iY=0)
            ]
            fovs = region_fovs(region_id, [
                {
                    "iterator": i,
                    "iX": n.iX,
                    "iY": n.iY,
                    "x": n.x,
                    "y": n.y,
                    "z": n.z if n.z is not None else center.z,
                }
                for i, n in enumerate(neighbours)
            ])
            regions.append(fovs)
            region_meta[region_id] = build_region_meta(
                center.name, center.areaType or "free_scan", center.wellId,
                center.wellRow, center.wellColumn, center.labwareLoadName,
                center.conditionLabel,
            )
            log.info(
                f"Scan region [{region_id}] '{center.name}': {len(fovs)} position(s)"
            )
        return regions, region_meta

    # Nothing supplied: image wherever the stage currently is.
    log.warning("No scan coordinates provided. Using current stage position.")
    position = current_position()
    region_id = "current"
    regions.append(region_fovs(region_id, [{
        "iterator": 0, "iX": 0, "iY": 0,
        "x": position.get("X", 0), "y": position.get("Y", 0), "z": None,
    }]))
    region_meta[region_id] = build_region_meta("Current Position", "free_scan")
    return regions, region_meta


# ---------------------------------------------------------------------------
# Hardware-triggered stage scan: the frame order the firmware produces.
#
# The ESP32 ``stagescan`` (uc2-ESP main/src/motor/StageScan.cpp, grid mode)
# fires one camera trigger per light channel at every position, walking the
# grid as
#
#     for iy in range(ny):                 # rows
#         for ix in range(nx):             # columns, reversed on odd rows (snake)
#             for iz in range(nz):         # Z planes
#                 for channel in <lasers 0..4 with intensity > 0, then led>:
#                     trigger
#
# and exactly one trigger per (ix, iy, iz) when no light channel is set. The
# writer maps camera frame-id -> row of this table, so both sides MUST agree
# on that order; the firmware side is pinned by uc2-ESP/test/native.
# ---------------------------------------------------------------------------

STAGE_SCAN_LASER_CHANNELS = 5      # illumination[0..4] on the ESP32 side
STAGE_SCAN_LED_CHANNEL = "led"     # the LED array, fired after the lasers


def stage_scan_channel_sequence(illumination, led) -> List[Dict[str, Any]]:
    """Active light channels in firmware order: lasers 0..4 ascending, then LED.

    Returns ``[{"name": "illumination2", "value": 50}, {"name": "led", "value": 255}]``
    style entries; empty when nothing is switched on (the firmware then still
    triggers once per position, see :func:`stage_scan_frame_table`).
    """
    lasers = list(illumination or [])[:STAGE_SCAN_LASER_CHANNELS]
    seq = [{"name": f"illumination{i}", "value": v}
           for i, v in enumerate(lasers) if v is not None and v > 0]
    if led is not None and led > 0:
        seq.append({"name": STAGE_SCAN_LED_CHANNEL, "value": led})
    return seq


def stage_scan_frame_table(nx: int, ny: int, nz: int,
                           xstart: float, ystart: float, zstart: float,
                           xstep: float, ystep: float, zstep: float,
                           illumination=None, led=None,
                           snake: bool = True) -> List[Dict[str, Any]]:
    """One metadata row per camera frame, in the order the firmware fires them.

    Row ``k`` describes camera frame-id ``k``. Each row carries the stage
    position, the light channel and the indices the OME writer places the
    frame by (``z_index``, ``channel_index``); ``runningNumber`` is 1-based
    like the historical protocol files.
    """
    channels = stage_scan_channel_sequence(illumination, led)
    if not channels:
        channels = [{"name": "default", "value": -1}]
    rows: List[Dict[str, Any]] = []
    for iy in range(int(ny)):
        for ix in range(int(nx)):
            jx = (nx - 1 - ix) if (snake and iy % 2 == 1) else ix
            x = xstart + jx * xstep
            y = ystart + iy * ystep
            for iz in range(int(nz)):
                z = zstart + iz * zstep
                for ci, ch in enumerate(channels):
                    rows.append({
                        "x": x, "y": y, "z": z,
                        "ix": jx, "iy": iy, "z_index": iz,
                        "channel_index": ci,
                        "illuminationChannel": ch["name"],
                        "illuminationValue": ch["value"],
                        "time_index": 0,
                        "runningNumber": len(rows) + 1,
                    })
    return rows


def stage_scan_frame_count(nx: int, ny: int, nz: int, illumination=None, led=None) -> int:
    """Frames a stage scan produces: positions x planes x max(active channels, 1)."""
    return int(nx) * int(ny) * int(nz) * max(len(stage_scan_channel_sequence(illumination, led)), 1)
