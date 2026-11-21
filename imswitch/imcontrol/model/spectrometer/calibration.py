"""Pixel -> wavelength calibration: grating model, polynomial, unit conversions.

Pixels are always full-sensor column indices (not ROI-relative), so a calibration
survives changes of the software region of interest.

Grating geometry (same convention as openuc2-openraman/spectro_block.py): the camera
arm is fixed at ``deviationDeg`` from the reversed incoming beam; ``turn`` rotates the
grating from that position. alpha = deviation + turn is the angle of incidence,
beta_axis = -turn the diffraction angle of the camera axis, both from the grating
normal, beta positive towards the specular side. A point x on the sensor (mm from the
camera axis, + towards the specular side) sees

    lambda(x) = d/m * (sin(alpha) - sin(beta_axis + atan(x / f)))

so wavelength falls with +x. ``wavelengthDecreasesWithPixel`` says whether +x runs
with the column index (shorter wavelengths to the right in the image).
"""
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------- units
def nm_to_raman_shift(nm, laser_nm: float):
    """Raman shift in cm^-1 of light at nm for excitation at laser_nm (Stokes positive)."""
    return 1e7 / float(laser_nm) - 1e7 / np.asarray(nm, dtype=float)


def raman_shift_to_nm(shift_cm1, laser_nm: float):
    return 1e7 / (1e7 / float(laser_nm) - np.asarray(shift_cm1, dtype=float))


def nm_to_wavenumber(nm):
    """Wavenumber 1/lambda in cm^-1."""
    return 1e7 / np.asarray(nm, dtype=float)


def nm_to_k(nm):
    """Angular wavenumber k = 2 pi / lambda in rad/um (OCT)."""
    return 2.0 * np.pi / (np.asarray(nm, dtype=float) / 1000.0)


def resample_linear_k(nm_axis, values, n: Optional[int] = None):
    """Resample a spectrum onto an evenly spaced k axis (OCT k-linearisation).

    Returns (k in rad/um, ascending; values). Before an FFT per A-line.
    """
    k = nm_to_k(nm_axis)
    v = np.asarray(values, dtype=float)
    order = np.argsort(k)
    k, v = k[order], v[order]
    k_lin = np.linspace(k[0], k[-1], n or k.size)
    return k_lin, np.interp(k_lin, k, v)


# ----------------------------------------------------------------------------- design
@dataclass
class GratingDesign:
    """Nominal optics of a grating spectrometer (defaults: openUC2 openRAMAN block)."""
    grooveDensityPerMm: float = 1200.0
    diffractionOrder: int = 1
    deviationDeg: float = 47.5
    cameraFocalLengthMm: float = 100.0
    collimatorFocalLengthMm: float = 35.0
    pixelPitchUm: float = 2.4
    sensorWidthPx: int = 3072
    slitWidthUm: float = 50.0
    wavelengthDecreasesWithPixel: bool = True
    centerWavelengthNm: float = 592.4
    leverArmMm: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "GratingDesign":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    @property
    def d_nm(self) -> float:
        return 1e6 / self.grooveDensityPerMm

    @property
    def sign(self) -> float:
        return 1.0 if self.wavelengthDecreasesWithPixel else -1.0

    @property
    def axisPx(self) -> float:
        return (self.sensorWidthPx - 1) / 2.0

    def turn_for_axis(self, lam_nm: float) -> float:
        """Grating turn (deg) that puts lam_nm on the camera axis."""
        k2 = math.radians(self.deviationDeg / 2.0)
        s = self.diffractionOrder * lam_nm / (2.0 * self.d_nm * math.cos(k2))
        return math.degrees(math.asin(s)) - self.deviationDeg / 2.0

    def lam_nm(self, x_mm, turn_deg: float, f_mm: Optional[float] = None):
        f = f_mm or self.cameraFocalLengthMm
        a = math.radians(self.deviationDeg + turn_deg)
        b = math.radians(-turn_deg) + np.arctan(np.asarray(x_mm, dtype=float) / f)
        return self.d_nm / self.diffractionOrder * (math.sin(a) - np.sin(b))

    def x_mm(self, lam_nm, turn_deg: float, f_mm: Optional[float] = None):
        """Sensor position (mm from the camera axis) of lam_nm; NaN if not diffracted."""
        f = f_mm or self.cameraFocalLengthMm
        a = math.radians(self.deviationDeg + turn_deg)
        s = math.sin(a) - self.diffractionOrder * np.asarray(lam_nm, dtype=float) / self.d_nm
        with np.errstate(invalid="ignore"):
            beta = np.arcsin(np.where(np.abs(s) <= 1, s, np.nan))
        return f * np.tan(beta - math.radians(-turn_deg))

    def slit_image_px(self, lam_nm: float, turn_deg: float) -> float:
        """Geometric width of the slit image (px) incl. the anamorphic factor."""
        a = math.radians(self.deviationDeg + turn_deg)
        s = math.sin(a) - self.diffractionOrder * lam_nm / self.d_nm
        b = math.asin(max(-1.0, min(1.0, s)))
        mag = self.cameraFocalLengthMm / self.collimatorFocalLengthMm * math.cos(a) / math.cos(b)
        return self.slitWidthUm * mag / self.pixelPitchUm


# ----------------------------------------------------------------------------- models
def grating_px_to_nm(px, design: GratingDesign, turn_deg: float, axis_px: float, f_mm: float):
    x = design.sign * (np.asarray(px, dtype=float) - axis_px) * design.pixelPitchUm / 1000.0
    return design.lam_nm(x, turn_deg, f_mm)


def grating_nm_to_px(nm, design: GratingDesign, turn_deg: float, axis_px: float, f_mm: float):
    x = design.x_mm(nm, turn_deg, f_mm)
    return axis_px + design.sign * x * 1000.0 / design.pixelPitchUm


def poly_eval(px, poly: Dict):
    u = (np.asarray(px, dtype=float) - poly["center"]) / poly["scale"]
    return np.polynomial.polynomial.polyval(u, poly["coeffs"])


@dataclass
class Calibration:
    """Pixel -> nm mapping plus what it was made from.

    model: "design" (nominal grating model, not measured), "grating" (fitted grating
    model) or "polynomial". ``points`` keep the lines it was fitted to.
    """
    model: str = "design"
    design: GratingDesign = field(default_factory=GratingDesign)
    grating: Dict = field(default_factory=dict)       # turnDeg, axisPx, focalLengthMm
    polynomial: Dict = field(default_factory=dict)    # center, scale, coeffs (ascending)
    points: List[Dict] = field(default_factory=list)
    rmsNm: Optional[float] = None
    maxAbsResidualNm: Optional[float] = None
    laserNm: Optional[float] = None
    smile: Dict = field(default_factory=dict)
    frameShape: Optional[List[int]] = None
    createdAt: str = ""
    detector: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.grating:
            self.grating = {"turnDeg": self.design.turn_for_axis(self.design.centerWavelengthNm),
                            "axisPx": self.design.axisPx,
                            "focalLengthMm": self.design.cameraFocalLengthMm}

    @property
    def calibrated(self) -> bool:
        return self.model in ("grating", "polynomial")

    def px_to_nm(self, px):
        if self.model == "polynomial" and self.polynomial:
            return poly_eval(px, self.polynomial)
        g = self.grating
        return grating_px_to_nm(px, self.design, g["turnDeg"], g["axisPx"], g["focalLengthMm"])

    def nm_to_px(self, nm):
        if self.model == "polynomial" and self.polynomial:
            w = self.design.sensorWidthPx
            grid = np.linspace(-0.1 * w, 1.1 * w, 4 * w + 1)
            lam = poly_eval(grid, self.polynomial)
            # keep the monotonic stretch around the sensor centre (a polynomial may turn
            # round outside the fitted range)
            sgn = np.sign(np.diff(lam))
            mid = sgn.size // 2
            lo, hi = mid, mid
            while lo > 0 and sgn[lo - 1] == sgn[mid]:
                lo -= 1
            while hi < sgn.size - 1 and sgn[hi + 1] == sgn[mid]:
                hi += 1
            grid, lam = grid[lo:hi + 2], lam[lo:hi + 2]
            if lam[-1] < lam[0]:
                grid, lam = grid[::-1], lam[::-1]
            return np.interp(nm, lam, grid, left=np.nan, right=np.nan)
        g = self.grating
        return grating_nm_to_px(nm, self.design, g["turnDeg"], g["axisPx"], g["focalLengthMm"])

    def nm_per_px(self, px):
        px = np.asarray(px, dtype=float)
        return np.abs(self.px_to_nm(px + 0.5) - self.px_to_nm(px - 0.5))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["calibrated"] = self.calibrated
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "Calibration":
        d = dict(d or {})
        design = GratingDesign.from_dict(d.pop("design", None))
        names = {f.name for f in fields(cls)} - {"design"}
        return cls(design=design, **{k: v for k, v in d.items() if k in names})

    def summary(self, width_px: Optional[int] = None) -> Dict:
        w = width_px or self.design.sensorWidthPx
        ends = self.px_to_nm(np.array([0.0, w - 1.0]))
        return {"model": self.model, "calibrated": self.calibrated,
                "nmAtFirstPx": float(ends[0]), "nmAtLastPx": float(ends[1]),
                "nmPerPxAtCenter": float(self.nm_per_px((w - 1) / 2.0)),
                "rmsNm": self.rmsNm, "nPoints": len(self.points), "laserNm": self.laserNm,
                "smileEnabled": bool(self.smile.get("coeffs")), "createdAt": self.createdAt}


# ----------------------------------------------------------------------------- fitting
def _lm(residual_fn, p0: np.ndarray, n_iter: int = 60):
    """Small Levenberg-Marquardt with a numeric Jacobian. Returns (p, J at p)."""
    p, lam = np.asarray(p0, dtype=float), 1e-3
    r = residual_fn(p)
    cost = float(r @ r)
    jac = None
    for _ in range(n_iter):
        steps = 1e-6 * np.maximum(np.abs(p), 1.0)
        jac = np.column_stack([(residual_fn(p + np.eye(p.size)[i] * steps[i]) - r) / steps[i]
                               for i in range(p.size)])
        jtj, g = jac.T @ jac, jac.T @ r
        while True:
            dp = -np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), g)
            r_new = residual_fn(p + dp)
            c_new = float(r_new @ r_new)
            if np.isfinite(c_new) and c_new < cost:
                p, r, cost, lam = p + dp, r_new, c_new, max(lam / 10, 1e-12)
                break
            lam *= 10
            if lam > 1e12:
                return p, jac
        if np.max(np.abs(dp) / np.maximum(np.abs(p), 1e-9)) < 1e-12:
            break
    return p, jac


def _point_arrays(points: Sequence[Dict]):
    px = np.array([float(p["px"]) for p in points])
    nm = np.array([float(p["nm"]) for p in points])
    sig = np.array([max(float(p.get("sigmaNm") or 1.0), 1e-4) for p in points])
    return px, nm, sig


def _free_parameters(n_points: int, span_frac: float) -> List[str]:
    """What a set of lines can tell: one line the turn (offset), two lines far enough
    apart also the dispersion (focal length), three lines also the camera-axis pixel
    (the curvature of the axis)."""
    if n_points >= 3:
        return ["turnDeg", "focalLengthMm", "axisPx"]
    if n_points == 2 and span_frac >= 0.1:
        return ["turnDeg", "focalLengthMm"]
    return ["turnDeg"]


def fit_grating(points: Sequence[Dict], design: GratingDesign, start: Optional[Dict] = None,
                free: Optional[Sequence[str]] = None) -> Dict:
    """Fit the grating model to (px, nm) points.

    Free parameters by default (see _free_parameters): the turn from one line, plus the
    focal length (dispersion) from two, plus the camera-axis pixel from three lines.
    Weak priors keep the focal length (sigma 5 %) and the axis pixel (sigma 10 % of the
    sensor) near the design when the lines cannot pin them. Returns the parameters, their
    1-sigma uncertainties and the residuals (nm).
    """
    px, nm, sig = _point_arrays(points)
    if px.size == 0:
        raise ValueError("no calibration points")
    start = start or {}
    base = {"turnDeg": float(start.get("turnDeg", design.turn_for_axis(float(np.mean(nm))))),
            "focalLengthMm": float(design.cameraFocalLengthMm), "axisPx": float(design.axisPx)}
    prior = {"focalLengthMm": (base["focalLengthMm"], 0.05 * base["focalLengthMm"]),
             "axisPx": (base["axisPx"], 0.1 * design.sensorWidthPx)}
    span = float(np.ptp(px)) / design.sensorWidthPx if px.size > 1 else 0.0
    free = list(free or _free_parameters(px.size, span))

    def params(p):
        q = dict(base)
        q.update(zip(free, p))
        return q

    def residuals(p):
        q = params(p)
        model = grating_px_to_nm(px, design, q["turnDeg"], q["axisPx"], q["focalLengthMm"])
        r = list((model - nm) / sig)
        r += [(q[k] - prior[k][0]) / prior[k][1] for k in free if k in prior]
        return np.nan_to_num(np.array(r), nan=1e6)

    p, jac = _lm(residuals, np.array([base[k] for k in free]))
    q = params(p)
    cov = np.linalg.pinv(jac.T @ jac) if jac is not None else np.zeros((p.size, p.size))
    err = dict(zip(free, np.sqrt(np.clip(np.diag(cov), 0, None))))
    model = grating_px_to_nm(px, design, q["turnDeg"], q["axisPx"], q["focalLengthMm"])
    return {**{k: float(v) for k, v in q.items()},
            "sigmaTurnDeg": float(err.get("turnDeg", 0.0)),
            "sigmaAxisPx": float(err["axisPx"]) if "axisPx" in err else None,
            "sigmaFocalLengthMm": (float(err["focalLengthMm"]) if "focalLengthMm" in err
                                   else None),
            "free": free, "residualsNm": (model - nm).tolist()}


def fit_polynomial(points: Sequence[Dict], order: int, width_px: int) -> Dict:
    """Weighted polynomial px -> nm. The order drops to (number of points - 1)."""
    px, nm, sig = _point_arrays(points)
    if px.size < 2:
        raise ValueError("a polynomial needs at least two lines")
    order = int(max(1, min(order, px.size - 1)))
    center, scale = (width_px - 1) / 2.0, (width_px - 1) / 2.0
    u = (px - center) / scale
    coeffs = np.polynomial.polynomial.polyfit(u, nm, order, w=1.0 / sig)
    poly = {"center": center, "scale": scale, "coeffs": coeffs.tolist(), "order": order}
    poly["residualsNm"] = (poly_eval(px, poly) - nm).tolist()
    poly["dof"] = int(px.size - order - 1)
    return poly


def make_calibration(points: Sequence[Dict], design: GratingDesign, model: str = "auto",
                     order: int = 2, start: Optional[Dict] = None, **meta) -> Calibration:
    """Fit a calibration. model "auto": grating model below 5 lines, else polynomial."""
    pts = [dict(p) for p in points]
    if model == "auto":
        model = "grating" if len(pts) < 5 else "polynomial"
    cal = Calibration(model=model, design=design, points=pts,
                      createdAt=datetime.now().isoformat(timespec="seconds"), **meta)
    g = fit_grating(pts, design, start=start)
    cal.grating = {k: g[k] for k in ("turnDeg", "axisPx", "focalLengthMm", "sigmaTurnDeg",
                                     "sigmaAxisPx", "sigmaFocalLengthMm", "free")}
    res = g["residualsNm"]
    if model == "polynomial":
        poly = fit_polynomial(pts, order, design.sensorWidthPx)
        res = poly.pop("residualsNm")
        cal.polynomial = poly
    for p, r in zip(cal.points, res):
        p["residualNm"] = float(r)
    r = np.asarray(res, dtype=float)
    cal.rmsNm = float(np.sqrt(np.mean(r ** 2))) if r.size else None
    cal.maxAbsResidualNm = float(np.max(np.abs(r))) if r.size else None
    return cal


# ----------------------------------------------------------------------------- alignment
def turn_from_line(design: GratingDesign, line_nm: float, px: float,
                   axis_px: Optional[float] = None, f_mm: Optional[float] = None) -> float:
    """Grating turn that puts line_nm at column px (axis and focal length as given)."""
    a = design.axisPx if axis_px is None else axis_px
    f = f_mm or design.cameraFocalLengthMm
    lo, hi = -45.0, 45.0
    g = lambda t: float(grating_px_to_nm(px, design, t, a, f)) - line_nm  # noqa: E731
    glo, ghi = g(lo), g(hi)
    if not (np.isfinite(glo) and np.isfinite(ghi)) or glo * ghi > 0:
        raise ValueError(f"{line_nm} nm cannot reach column {px:.0f} in this design")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if glo * gm <= 0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
    return 0.5 * (lo + hi)


def alignment_plan(design: GratingDesign, line_nm: float, target_center_nm: float,
                   current: Optional[Calibration] = None, measured_px: Optional[float] = None,
                   width_px: Optional[int] = None) -> Dict:
    """How to turn the grating from where a known line sits now to a target window.

    The current turn comes from the measured column of the line (axis and focal length
    from ``current`` if it is calibrated, else the design) or, without a measurement,
    from the current calibration. The target puts target_center_nm on the camera axis.
    """
    w = width_px or design.sensorWidthPx
    # every fitted calibration carries a grating fit (also next to a polynomial)
    g = (current.grating if current is not None and current.calibrated else
         {"axisPx": design.axisPx, "focalLengthMm": design.cameraFocalLengthMm,
          "turnDeg": (current.grating.get("turnDeg") if current is not None else None)})
    axis, f = g.get("axisPx", design.axisPx), g.get("focalLengthMm", design.cameraFocalLengthMm)
    turn_now = (turn_from_line(design, line_nm, measured_px, axis, f) if measured_px is not None
                else g.get("turnDeg"))
    turn_target = design.turn_for_axis(target_center_nm)
    px_target = float(grating_nm_to_px(line_nm, design, turn_target, axis, f))
    window = lambda t: [float(grating_px_to_nm(c, design, t, axis, f)) for c in (0, w - 1)]  # noqa
    plan = {"lineNm": line_nm, "targetCenterNm": target_center_nm, "measuredPx": measured_px,
            "turnNowDeg": turn_now, "turnTargetDeg": turn_target,
            "linePxAtTarget": px_target,
            "lineOnSensorAtTarget": bool(np.isfinite(px_target) and 0 <= px_target <= w - 1),
            "windowAtTargetNm": sorted(window(turn_target))}
    if turn_now is not None:
        dt = turn_target - turn_now
        plan.update(deltaTurnDeg=dt, windowNowNm=sorted(window(turn_now)))
        if design.leverArmMm > 0:
            plan["leverTravelMm"] = design.leverArmMm * math.tan(math.radians(dt))
    if not plan["lineOnSensorAtTarget"]:
        # park the line on the axis first: a known, visible reference turn
        plan["parkPx"] = axis
        plan["turnAtParkDeg"] = turn_from_line(design, line_nm, axis, axis, f)
        plan["deltaTurnFromParkDeg"] = turn_target - plan["turnAtParkDeg"]
        if design.leverArmMm > 0:
            plan["leverTravelFromParkMm"] = design.leverArmMm * math.tan(
                math.radians(plan["deltaTurnFromParkDeg"]))
    return plan
