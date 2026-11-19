"""ImSwitch's side of the dvOPM twin: stage axes <-> model axes, frame positions, deskew.

Model axes (lab frame of ``kinematics``, mm and degrees):
    sample.X, sample.Y, sample.Z, launch.x, launch.y, launch.z, launch.phi_deg, launch.focus
A model axis is linked to a positioner axis ("X", "-X" for the opposite direction,
"LaunchStage.Z" for another positioner) or set by hand. ImSwitch positions are µm.
A reference pairs one set of stage positions with the model values they stand for
("we are here"); everything else is a straight line through it.
"""
import math
from typing import Dict, Optional, Tuple

import numpy as np

from .kinematics import Detection

TWIN_AXES = ("sample.X", "sample.Y", "sample.Z", "launch.x", "launch.y", "launch.z",
             "launch.phi_deg", "launch.focus")

Link = Tuple[str, str, float]          # positioner name, axis, sign


def parse_axis_map(axis_map: Dict[str, str], default_positioner: str) -> Dict[str, Link]:
    """{"sample.X": "X", "launch.z": "-Launch.Z", "launch.x": ""} -> {model axis: link}."""
    links = {}
    for twin, spec in axis_map.items():
        if twin not in TWIN_AXES:
            raise ValueError(f"Unknown OPM axis '{twin}', expected one of {', '.join(TWIN_AXES)}")
        spec = (spec or "").strip()
        if not spec:
            continue
        name, _, axis = spec.lstrip("+-").rpartition(".")
        links[twin] = (name or default_positioner, axis, -1.0 if spec.startswith("-") else 1.0)
    return links


def unit(twin: str) -> float:
    """Model units per ImSwitch unit: mm per µm, or degrees per degree for the tilt."""
    return 1.0 if twin.endswith("_deg") else 1e-3


def to_model(twin: str, raw: float, sign: float, ref_raw: float, ref_model: float) -> float:
    return ref_model + sign * (raw - ref_raw) * unit(twin)


def to_raw(twin: str, value: float, sign: float, ref_raw: float, ref_model: float) -> float:
    return ref_raw + (value - ref_model) / (sign * unit(twin))


# ---------------------------------------------------------------- stage scan
def frame_positions(stamps, t0: float, t1: float, xa: float, xb: float) -> np.ndarray:
    """Sample X (mm) of frames grabbed at ``stamps`` while the stage moved xa -> xb.

    The move was sent at t0 and the blocking move returned at t1, so the speed that
    counts is the one measured, (xb - xa) / (t1 - t0), not the one asked for.
    """
    f = np.clip((np.asarray(stamps, float) - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
    return xa + (xb - xa) * f


def frame_times(stamps, frame_numbers) -> np.ndarray:
    """Arrival times -> times on the camera's own clock.

    Polling sees only the newest frame and stamps it when it arrives, a few ms late and
    unevenly. The camera counts its frames, so a straight line through (frame number,
    arrival time) gives every frame an evenly spaced time at the camera's real rate.
    """
    t, n = np.asarray(stamps, float), np.asarray(frame_numbers, float)
    if len(t) < 3 or np.ptp(n) == 0:
        return t
    b, a = np.polyfit(n, t, 1)
    return a + b * n


def scan_summary(stamps, xs_mm, frame_numbers=None) -> Dict[str, Optional[float]]:
    """Measured frame rate, speed and step per frame of the kept frames. camera_fps
    counts the frames the camera made, fps those that were kept (dropped ones missing)."""
    t, x = np.asarray(stamps, float), np.asarray(xs_mm, float)
    out = {"frames": int(len(t)), "fps": None, "camera_fps": None, "velocity_mm_s": None,
           "step_um": None, "dropped_frames": None}
    if len(t) > 1 and t[-1] > t[0]:
        out["fps"] = (len(t) - 1) / (t[-1] - t[0])
        out["velocity_mm_s"] = (x[-1] - x[0]) / (t[-1] - t[0])
        out["step_um"] = float(np.median(np.diff(x))) * 1e3
    if frame_numbers is not None and len(frame_numbers) > 1:
        d = np.diff(np.asarray(frame_numbers))
        out["dropped_frames"] = int(np.sum(np.clip(d - 1, 0, None)))
        if t[-1] > t[0]:
            out["camera_fps"] = float(frame_numbers[-1] - frame_numbers[0]) / (t[-1] - t[0])
    return out


def deskew(stack: np.ndarray, xs_mm, n: float, det: Detection, flip: bool = False,
           max_bytes: float = 2e9):
    """Stage-scan stack (frame, tilted row, flat column) -> sample volume (z, y, x).

    Frame i was taken with the sample at xs_mm[i] (increasing). Row r images the
    detection plane at lab x = -kx (r - r0), z = kz (r - r0) (``kinematics.deskew_affine``),
    so the sample point it shows sits at x = -kx (r - r0) - xs[i]. The volume samples x
    on the flat-axis pitch p / M, linear between frames; z keeps the row pitch kz.
    ``flip`` reverses the tilted axis (camera mounted the other way round).
    Returns (volume, {"dx_um", "dz_um", "x0_mm"}); x0_mm is the sample X of the first column.
    """
    st = stack[:, ::-1] if flip else stack
    N, R, C = st.shape
    if N < 2:
        raise ValueError("Deskew needs at least two frames.")
    p, a = det.pixel_um * 1e-3, det.alpha
    kx, kz, dx = p * math.cos(a) / det.M, p * n * math.sin(a) / det.M ** 2, p / det.M
    xs = np.asarray(xs_mm, float)
    off = kx * (np.arange(R) - (R - 1) / 2)
    lo, hi = -off.max() - xs.max(), -off.min() - xs.min()
    X = lo + dx * np.arange(int(math.floor((hi - lo) / dx)) + 1)
    if R * C * len(X) * st.dtype.itemsize > max_bytes:
        raise ValueError(f"Deskewed volume would be {R}x{C}x{len(X)} voxels, "
                         f"over {max_bytes / 1e9:.1f} GB; bin the camera or scan less.")
    out = np.zeros((R, C, len(X)), dtype=st.dtype)
    idx = np.arange(N, dtype=float)
    for r in range(R):
        fi = np.interp(-X - off[r], xs, idx, left=-1.0, right=-1.0)
        ok = fi >= 0
        i0 = np.minimum(fi[ok].astype(int), N - 2)
        w = (fi[ok] - i0)[:, None]
        out[r][:, ok] = ((1 - w) * st[i0, r] + w * st[i0 + 1, r]).T
    return out, {"dx_um": dx * 1e3, "dz_um": kz * 1e3, "x0_mm": float(lo)}
