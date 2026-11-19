"""
ctdvopm_kinematics.py - geometry, inverse kinematics and calibration helpers for a
direct-view oblique plane microscope (DvOPM / CtDvOPM, Lamb et al., bioRxiv 2026)
built from openUC2 cubes.

Pure Python + NumPy, no ImSwitch dependency, so it can be unit-tested and reused
in an ImSwitch controller / plugin, a notebook, or the browser simulator.

LAB FRAME (side view, the plane of the drawing)
    x : horizontal, positive towards the light-sheet launch side
    z : vertical, up = away from the detection lens, towards the sample
    y : out of plane (untilted sensor axis, sheet width direction)
    origin: nominal *air* focal point of the telecentric detection lens.
    stage_z > 0 lifts the dish.

MODEL (paraxial, first order - verify against Supplementary Note B of the paper)
    Detection: bi-telecentric, lateral magnification M, air-to-air axial
    magnification M^2. Sensor normal tilted alpha from the optical axis.
    Imaged into a medium of index n the conjugate (detection) plane is tilted
        tan(theta_s) = n * tan(alpha) / M          (theta_s from horizontal)
    Light sheet from below through a glass bottom (n_g, t_g). Snell:
        sin(phi) = n * cos(theta_s) = 1 / sqrt(1/n^2 + tan^2(alpha)/M^2)
    (phi = launch angle in air, measured from the vertical / dish normal).
    Waist position through the tilted interfaces uses the tangential Coddington
    equation for plane surfaces (the thin sheet axis lies in the plane of
    incidence).

All lengths in mm, angles in radians unless the name says _deg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class Detection:
    M: float = 1.45                # lateral magnification sample -> sensor (0.69x telecentric, reversed)
    alpha_deg: float = 50.0        # sensor normal vs detection optical axis
    pixel_um: float = 2.2          # sensor pixel pitch (AR0521, Alvium bare board)
    sensor_tilted_mm: float = 4.28 # sensor extent along the tilted axis (1944 px)
    sensor_flat_mm: float = 5.70   # sensor extent along the untilted axis (2592 px)

    @property
    def alpha(self) -> float:
        return math.radians(self.alpha_deg)


@dataclass
class Dish:
    n_glass: float = 1.52
    t_glass_mm: float = 0.17
    # air-equivalent distance from the INNER dish surface up to the nominal
    # detection focal plane when stage_z == 0 (i.e. how far "into" the dish
    # you focus). Must stay > 0.
    G0_mm: float = 2.0


@dataclass
class Launch:
    """Rigid launch arm on an X/Z carriage with a tilt axis and a focus axis.

    Air focus point  P = R + (d0 + focus) * u(phi),   u = (-sin phi, cos phi)
    Pivot            R = pivot_home + (X_L, Z_L)
    """
    pivot_home: Tuple[float, float] = (0.0, 0.0)
    d0_mm: float = 0.0
    working_distance_mm: float = 17.0   # launch objective WD (drawing only)
    beam_D_mm: float = 1.61             # collimated beam 1/e^2 diameter (f = 8 collimator, NA 0.10)
    f_obj_mm: float = 50.0              # launch objective focal length (4x/200)
    f_cyl_mm: float = 9.9               # cylindrical lens focal length (LJ1878L1 at 488 nm)
    wavelength_nm: float = 488.0


@dataclass
class MotorPose:
    x: float      # carriage X  [mm]
    z: float      # carriage Z  [mm]
    phi: float    # launch tilt from vertical [rad]
    focus: float  # focus axis along the beam [mm]

    def as_dict(self) -> Dict[str, float]:
        return {"x": self.x, "z": self.z, "phi_deg": math.degrees(self.phi),
                "focus": self.focus}


# --------------------------------------------------------------------------
# closed-form geometry
# --------------------------------------------------------------------------
def sheet_angle(n: float, det: Detection) -> float:
    """Tilt of the detection (and required sheet) plane in the medium, from horizontal."""
    return math.atan(n * math.tan(det.alpha) / det.M)


def launch_angle(n: float, det: Detection) -> float:
    """Required launch angle in air, measured from the dish normal."""
    s = 1.0 / math.sqrt(1.0 / n**2 + (math.tan(det.alpha) / det.M) ** 2)
    if s >= 1.0:
        raise ValueError("no real launch angle: n too high for this M/alpha")
    return math.asin(s)


def index_from_launch_angle(phi: float, det: Detection) -> float:
    """Invert launch_angle: the medium index implied by a coplanar launch angle."""
    t = math.tan(det.alpha) / det.M
    return 1.0 / math.sqrt(1.0 / math.sin(phi) ** 2 - t**2)


def max_index(det: Detection) -> float:
    t = math.tan(det.alpha) / det.M
    return float("inf") if t >= 1 else 1.0 / math.sqrt(1 - t**2)


def _u(phi: float) -> np.ndarray:
    return np.array([-math.sin(phi), math.cos(phi)])


def _angles(n: float, n_g: float, phi: float):
    s = math.sin(phi)
    if s / n >= 1 or s / n_g >= 1:
        raise ValueError("total internal reflection / no transmission")
    return math.asin(s / n_g), math.asin(s / n)   # theta_glass, theta_medium (from normal)


def coddington_air_to_medium(s_air: float, n: float, n_g: float, t_g: float, phi: float) -> float:
    """Tangential focus: distance from glass bottom along air ray to virtual air focus
    -> distance from inner glass surface along medium ray to the sheet waist."""
    tg, tm = _angles(n, n_g, phi)
    t1p = s_air * n_g * math.cos(tg) ** 2 / math.cos(phi) ** 2
    t2 = t1p - t_g / math.cos(tg)
    return t2 * n * math.cos(tm) ** 2 / (n_g * math.cos(tg) ** 2)


def coddington_medium_to_air(s_med: float, n: float, n_g: float, t_g: float, phi: float) -> float:
    tg, tm = _angles(n, n_g, phi)
    t2 = s_med * n_g * math.cos(tg) ** 2 / (n * math.cos(tm) ** 2)
    t1p = t2 + t_g / math.cos(tg)
    return t1p * math.cos(phi) ** 2 / (n_g * math.cos(tg) ** 2)


@dataclass
class Target:
    """Where the sheet has to be for a given medium and stage height."""
    n: float
    stage_z: float
    theta_s: float
    phi: float
    H: float                 # physical focus height above inner dish surface
    F: np.ndarray            # detection focal point (lab)
    E: np.ndarray            # sheet entry point on inner surface (lab)
    B: np.ndarray            # air/glass entry point (lab)
    s_med: float             # E -> waist along sheet
    s_air: float             # B -> virtual air focus along air ray
    P: np.ndarray            # required virtual air focus point (lab)


def target(n: float, stage_z: float, det: Detection, dish: Dish,
           waist_offset_mm: float = 0.0) -> Target:
    """waist_offset_mm: move the waist along the sheet (+ = deeper, up-left)."""
    ng, tg_ = dish.n_glass, dish.t_glass_mm
    G = dish.G0_mm - stage_z
    h_i = -G
    H = n * (G + tg_ * (1 - 1 / ng))
    if H <= 0:
        raise ValueError("detection focus is below the dish bottom - lower the stage")
    th = sheet_angle(n, det)
    phi = launch_angle(n, det)
    th_g, _ = _angles(n, ng, phi)
    F = np.array([0.0, h_i + H])
    E = np.array([H / math.tan(th), h_i])        # == (G + t_g(1-1/n_g)) * M/tan(alpha): n-independent!
    B = np.array([E[0] + tg_ * math.tan(th_g), h_i - tg_])
    s_med = H / math.sin(th) + waist_offset_mm
    s_air = coddington_medium_to_air(s_med, n, ng, tg_, phi)
    P = B + s_air * _u(phi)
    return Target(n, stage_z, th, phi, H, F, E, B, s_med, s_air, P)


# --------------------------------------------------------------------------
# inverse / forward kinematics
# --------------------------------------------------------------------------
MODES = ("carriage_xz", "x_plus_focus")
# carriage_xz  : focus axis parked, solve X_L and Z_L (no dedicated focus motor)
# x_plus_focus : carriage Z parked, solve X_L and the focus axis


def inverse_kinematics(n: float, stage_z: float, det: Detection, dish: Dish,
                       launch: Launch, mode: str = "carriage_xz",
                       parked: float = 0.0, waist_offset_mm: float = 0.0) -> MotorPose:
    """Motor pose that puts the sheet on the detection plane with its waist at the
    detection focus. `parked` is the value of the axis that is held fixed
    (focus for carriage_xz, carriage Z for x_plus_focus)."""
    T = target(n, stage_z, det, dish, waist_offset_mm)
    u = _u(T.phi)
    px, pz = launch.pivot_home
    if mode == "carriage_xz":
        R = T.P - (launch.d0_mm + parked) * u
        return MotorPose(R[0] - px, R[1] - pz, T.phi, parked)
    if mode == "x_plus_focus":
        Rz = pz + parked
        k = (Rz - T.B[1]) / u[1]                # R = B + k u
        R = T.B + k * u
        focus = T.s_air - launch.d0_mm - k
        return MotorPose(R[0] - px, parked, T.phi, focus)
    raise ValueError(f"mode must be one of {MODES}")


@dataclass
class SheetState:
    phi: float
    theta_sheet: float        # actual sheet angle in medium (from horizontal)
    B: np.ndarray
    E: np.ndarray
    waist: np.ndarray
    angle_error: float        # rad, sheet - detection plane
    offset_error_mm: float    # signed distance of F from the sheet line
    waist_error_mm: float     # waist position along sheet relative to F's foot point


def forward_kinematics(pose: MotorPose, n: float, stage_z: float, det: Detection,
                       dish: Dish, launch: Launch) -> SheetState:
    ng, tg_ = dish.n_glass, dish.t_glass_mm
    G = dish.G0_mm - stage_z
    h_i, z_b = -G, -G - tg_
    phi = pose.phi
    u = _u(phi)
    R = np.array(launch.pivot_home) + np.array([pose.x, pose.z])
    P = R + (launch.d0_mm + pose.focus) * u
    s_air = (P[1] - z_b) / u[1]
    B = P - s_air * u
    th_g, th_m = _angles(n, ng, phi)
    E = B + tg_ / math.cos(th_g) * np.array([-math.sin(th_g), math.cos(th_g)])
    um = np.array([-math.sin(th_m), math.cos(th_m)])
    s_med = coddington_air_to_medium(s_air, n, ng, tg_, phi)
    waist = E + s_med * um
    H = n * (G + tg_ * (1 - 1 / ng))
    F = np.array([0.0, h_i + H])
    normal = np.array([um[1], -um[0]])          # points to +x side / below the sheet
    d = F - E
    offset = float(d @ normal)
    along_F = float(d @ um)
    theta_sheet = math.pi / 2 - th_m
    return SheetState(phi, theta_sheet, B, E, waist,
                      theta_sheet - sheet_angle(n, det), offset, s_med - along_F)


def tracking_jacobian(n: float, stage_z: float, det: Detection, dish: Dish,
                      launch: Launch, mode: str = "carriage_xz", parked: float = 0.0,
                      h: float = 1e-3) -> Dict[str, float]:
    """d(motor)/d(stage_z) - how each launch axis must follow a sample Z move."""
    a = inverse_kinematics(n, stage_z - h, det, dish, launch, mode, parked)
    b = inverse_kinematics(n, stage_z + h, det, dish, launch, mode, parked)
    return {"dx": (b.x - a.x) / (2 * h), "dz": (b.z - a.z) / (2 * h),
            "dphi": (b.phi - a.phi) / (2 * h), "dfocus": (b.focus - a.focus) / (2 * h)}


def sensitivities(n: float, det: Detection) -> Dict[str, float]:
    """Small-signal gains from launch motion (in air) to sheet motion (in medium),
    ignoring the thin coverslip."""
    th = sheet_angle(n, det)
    phi = launch_angle(n, det)
    return {
        # 1 mm launch shift perpendicular to its beam -> sheet shift perpendicular to itself
        "perp_gain": math.sin(th) / math.cos(phi),
        # 1 rad launch tilt -> sheet tilt
        "tilt_gain": math.cos(phi) / (n * math.sin(th)),
        # 1 mm along-beam focus move -> waist move along the sheet
        "focus_gain": n * math.sin(th) ** 2 / math.cos(phi) ** 2,
        # 1 mm stage-Z -> needed air-line intercept shift (closed form)
        "intercept_per_stage_z": math.tan(phi) - det.M / math.tan(det.alpha),
    }


# --------------------------------------------------------------------------
# sheet optics, field of view, scanning, deskew
# --------------------------------------------------------------------------
def sheet_optics(n: float, det: Detection, launch: Launch) -> Dict[str, float]:
    th = sheet_angle(n, det)
    phi = launch_angle(n, det)
    lam = launch.wavelength_nm * 1e-6
    beta_air = launch.beam_D_mm / (2 * launch.f_obj_mm)          # focusing half-angle in air
    beta_med = beta_air * math.cos(phi) / (n * math.sin(th))    # compressed by refraction
    w0 = lam / (math.pi * n * beta_med)                          # 1/e^2 radius
    zR = w0 / beta_med
    fov_along = det.sensor_tilted_mm * math.hypot(math.cos(det.alpha) / det.M,
                                                  n * math.sin(det.alpha) / det.M**2)
    return {"NA_air": beta_air, "NA_eff_medium": n * beta_med,
            "w0_um": w0 * 1e3, "fwhm_um": w0 * math.sqrt(2 * math.log(2)) * 1e3,
            "rayleigh_mm": zR, "confocal_mm": 2 * zR,
            "sheet_height_mm": launch.beam_D_mm * launch.f_obj_mm / launch.f_cyl_mm,
            "fov_along_sheet_mm": fov_along,
            "fov_depth_mm": fov_along * math.sin(th),
            "fov_width_mm": det.sensor_flat_mm / det.M}


def ideal_beam_NA_for_fov(n: float, det: Detection, lam_nm: float = 488.0) -> float:
    """Launch NA (air) whose confocal parameter equals the FOV along the sheet."""
    th = sheet_angle(n, det)
    phi = launch_angle(n, det)
    L = sheet_optics(n, det, Launch())["fov_along_sheet_mm"]
    lam = lam_nm * 1e-6
    # 2 zR = 2 lam / (pi n beta_m^2) = L
    beta_m = math.sqrt(2 * lam / (math.pi * n * L))
    return beta_m * n * math.sin(th) / math.cos(phi)


def plan_scan(n: float, det: Detection, launch: Launch, scan_length_mm: float,
              fps: float, oversample: float = 2.0,
              step_mm: Optional[float] = None) -> Dict[str, float]:
    """Continuous stage-X scan through the static sheet."""
    th = sheet_angle(n, det)
    fwhm_mm = sheet_optics(n, det, launch)["fwhm_um"] * 1e-3
    step = step_mm if step_mm is not None else fwhm_mm / (oversample * math.sin(th))
    v = step * fps
    frames = int(math.ceil(scan_length_mm / step))
    return {"step_um": step * 1e3, "slice_spacing_um": step * math.sin(th) * 1e3,
            "velocity_mm_s": v, "frames": frames, "duration_s": frames / fps}


def deskew_affine(n: float, det: Detection, step_mm: float,
                  r0: float, c0: float) -> np.ndarray:
    """4x4 affine: (frame i, tilted-axis pixel r, flat-axis pixel c, 1) -> sample (x, y, z, 1) in mm,
    relative to the detection focus at frame 0. Sample moves +x per frame.
    Sign of r depends on camera mounting - verify once with beads."""
    p = det.pixel_um * 1e-3
    a = det.alpha
    A = np.eye(4)
    A[0, :] = [-step_mm, -p * math.cos(a) / det.M, 0, r0 * p * math.cos(a) / det.M]
    A[1, :] = [0, 0, p / det.M, -c0 * p / det.M]
    A[2, :] = [0, p * n * math.sin(a) / det.M**2, 0, -r0 * p * n * math.sin(a) / det.M**2]
    return A


# --------------------------------------------------------------------------
# calibration helpers (image-based)
# --------------------------------------------------------------------------
def band_sharpness(img: np.ndarray, n_bands: int = 16, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Split the frame into bands along the TILTED sensor axis and return
    (band centre pixel, focus metric). Metric: energy of a discrete Laplacian,
    normalised by mean intensity (works on beads or textured samples)."""
    im = np.asarray(img, dtype=np.float32)
    if axis == 1:
        im = im.T
    lap = (-4 * im[1:-1, 1:-1] + im[:-2, 1:-1] + im[2:, 1:-1] + im[1:-1, :-2] + im[1:-1, 2:])
    edges = np.linspace(0, lap.shape[0], n_bands + 1).astype(int)
    centres, metric = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        L = lap[a:b]
        m = im[a + 1:b + 1].mean() + 1e-6
        centres.append(0.5 * (a + b) + 1)
        metric.append(float((L**2).mean()) / m**2)
    return np.array(centres), np.array(metric)


def fit_coplanarity(perp_offsets_mm: Sequence[float], metric: np.ndarray,
                    band_centres_px: np.ndarray, r0: float,
                    n: float, det: Detection) -> Dict[str, float]:
    """Coplanarity sweep: the launch was stepped perpendicular to its beam
    (perp_offsets_mm, in AIR at the launch) and for each step a band_sharpness
    profile was recorded -> metric[k, band].

    Returns the air-side correction: delta_phi (rad) and delta_perp (mm) to apply."""
    offs = np.asarray(perp_offsets_mm, float)
    best = []
    for j in range(metric.shape[1]):
        k = int(np.argmax(metric[:, j]))
        k = min(max(k, 1), len(offs) - 2)
        y0, y1, y2 = metric[k - 1:k + 2, j]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        best.append(offs[k] + frac * (offs[1] - offs[0]))
    best = np.array(best)
    g = sensitivities(n, det)
    best_med = best * g["perp_gain"]                           # sheet offset in medium
    k_along = det.pixel_um * 1e-3 * math.hypot(math.cos(det.alpha) / det.M,
                                               n * math.sin(det.alpha) / det.M**2)
    s = (band_centres_px - r0) * k_along                       # position along the sheet
    slope, icpt = np.polyfit(s, best_med, 1)
    # best offset(s) = gap(s) = gap(0) - s*sin(eps), eps = sheet tilt error
    # -> the sheet must rotate by +atan(slope); a larger phi lowers the sheet angle
    dtheta_sheet = math.atan(slope)
    return {"delta_phi": -dtheta_sheet / g["tilt_gain"],
            "delta_perp_mm": icpt / g["perp_gain"],
            "residual_um": float(np.std(best_med - (slope * s + icpt)) * 1e3),
            # for plotting: best sheet offset per band (medium, um) vs position along the sheet (mm)
            "s_mm": s.tolist(), "best_um": (best_med * 1e3).tolist(),
            "slope": float(slope), "icpt_um": float(icpt * 1e3)}


# --------------------------------------------------------------------------
def paper_like_launch(det: Detection, dish: Dish, n_ref: float = 1.45, **kw) -> Launch:
    """Launch whose tilt pivot sits on the air/glass entry point at n_ref,
    as in the paper (rotation about the point where the sheet meets the coverslip)."""
    T = target(n_ref, 0.0, det, dish)
    return Launch(pivot_home=(float(T.B[0]), float(T.B[1])), d0_mm=T.s_air, **kw)


if __name__ == "__main__":
    det, dish = Detection(), Dish()
    L = paper_like_launch(det, dish)
    print(f"M={det.M}, alpha={det.alpha_deg} deg, glass {dish.t_glass_mm} mm, "
          f"focus {dish.G0_mm} mm (air-eq.) into dish; n_max={max_index(det):.3f}\n")
    hdr = ("n", "theta_s", "phi_air", "E_x", "H", "d(icpt)/dZ",
           "XZ: dX", "XZ: dZ", "X+F: dX", "X+F: dF", "focus gain", "perp gain", "tilt gain",
           "FWHM um", "2zR mm", "FOV_s mm")
    print(" ".join(f"{h:>10}" for h in hdr))
    for n in (1.33, 1.38, 1.45, 1.49, 1.52, 1.56):
        T = target(n, 0, det, dish)
        j1 = tracking_jacobian(n, 0, det, dish, L, "carriage_xz")
        j2 = tracking_jacobian(n, 0, det, dish, L, "x_plus_focus")
        g = sensitivities(n, det)
        o = sheet_optics(n, det, L)
        row = (n, math.degrees(T.theta_s), math.degrees(T.phi), T.E[0], T.H,
               g["intercept_per_stage_z"], j1["dx"], j1["dz"], j2["dx"], j2["dfocus"],
               g["focus_gain"], g["perp_gain"], g["tilt_gain"],
               o["fwhm_um"], o["confocal_mm"], o["fov_along_sheet_mm"])
        print(" ".join(f"{v:>10.3f}" for v in row))
    # round-trip test
    for n in (1.33, 1.56):
        for mode in MODES:
            p = inverse_kinematics(n, 0.7, det, dish, L, mode)
            s = forward_kinematics(p, n, 0.7, det, dish, L)
            assert abs(s.angle_error) < 1e-9 and abs(s.offset_error_mm) < 1e-9 \
                and abs(s.waist_error_mm) < 1e-9, (n, mode, s)
        assert abs(index_from_launch_angle(launch_angle(n, det), det) - n) < 1e-12
    print("\nIK/FK round trip OK")
    print("ideal launch NA (air) for full FOV at n=1.45:",
          round(ideal_beam_NA_for_fov(1.45, det), 4))
    print("scan plan n=1.45, 10 mm @ 20 fps:", plan_scan(1.45, det, L, 10, 20))
