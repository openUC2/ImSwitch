"""
opm_optics.py - ray-optics model of the light-sheet launch and Optiland export.

Path: fibre tip -> collimator (ideal lens) -> cylindrical lens (power only across the
sheet width) -> launch objective (ideal lens) -> air -> glass dish bottom -> medium.

* Tangential plane (the drawing plane, the thin-sheet axis): exact vector-Snell trace of
  a ray fan through the tilted interfaces, real-ray focus vs the Coddington estimate,
  Fresnel transmission for s and p polarisation.
* Sagittal plane (sheet width): the cylinder focuses into the objective back focal
  plane, so the width is collimated at D * f_obj / f_cyl.
* optiland_script(...) writes a stand-alone Optiland (>= 0.5) Python file with the same
  optics in the launch's local frame, including the tilted dish surfaces.

CLI:
    python opm_optics.py --n 1.52 --out launch_n152_optiland.py
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from .kinematics import (Detection, Dish, Launch, MotorPose, forward_kinematics,
                                inverse_kinematics, paper_like_launch, sheet_angle,
                                sheet_optics)


@dataclass
class LaunchOptics:
    """The real launch: Taobao f = 8 collimator, Thorlabs LJ1878L1 cylinder (f = 10, R 5.17,
    CT 5.88), infinity 4x objective f = 50 (SOPTOP, BFP 5 mm behind the thread)."""
    fiber_na: float = 0.10          # 1/e^2 mode-field NA of the fibre
    f_col_mm: float = 8.0           # collimator focal length -> beam D = 2 f NA
    d_col_cyl_mm: float = 3.0       # collimator front -> cylindrical lens (near vertex)
    f_cyl_mm: float = 10.0          # used when cyl_R_mm is None: R = (n - 1) f
    cyl_thickness_mm: float = 5.88
    cyl_glass: str = "N-BK7"
    cyl_R_mm: Optional[float] = 5.17    # the real surface radius; overrides f_cyl_mm
    f_obj_mm: float = 50.0          # launch objective (4x with a 200 mm tube-lens standard)
    cyl_to_bfp_mm: Optional[float] = None   # None: cylinder focus lands in the objective BFP
    wavelength_nm: float = 488.0
    fan_rays: int = 15
    fill: float = 1.0               # fraction of the beam aperture traced in the fan


def nbk7(lam_um: float) -> float:
    """Schott N-BK7 Sellmeier."""
    l2 = lam_um ** 2
    B = (1.03961212, 0.231792344, 1.01046945)
    C = (0.00600069867, 0.0200179144, 103.560653)
    return math.sqrt(1 + sum(b * l2 / (l2 - c) for b, c in zip(B, C)))


def beam_diameter(o: LaunchOptics) -> float:
    return 2 * o.f_col_mm * math.tan(math.asin(o.fiber_na))


def cyl_focal_length(o: LaunchOptics) -> float:
    """Effective focal length of the cylinder at the working wavelength."""
    if o.cyl_R_mm is None:
        return o.f_cyl_mm
    return o.cyl_R_mm / (nbk7(o.wavelength_nm * 1e-3) - 1)


def cyl_prescription(o: LaunchOptics) -> Dict[str, float]:
    n = nbk7(o.wavelength_nm * 1e-3)
    f = cyl_focal_length(o)
    R = (n - 1) * f                                  # plano-convex, curved side towards the collimator
    bfd = f - o.cyl_thickness_mm / n                 # flat side -> line focus
    gap = (bfd if o.cyl_to_bfp_mm is None else o.cyl_to_bfp_mm) + o.f_obj_mm
    return {"n": n, "R_mm": R, "f_mm": f, "bfd_mm": bfd, "gap_to_objective_mm": gap}


def _refract(d, nrm, n1, n2):
    """Vector Snell. nrm faces the incoming ray. Returns None on TIR."""
    eta = n1 / n2
    ci = -float(np.dot(nrm, d))
    k = 1 - eta ** 2 * (1 - ci ** 2)
    if k < 0:
        return None
    t = eta * d + (eta * ci - math.sqrt(k)) * nrm
    return t / np.linalg.norm(t)


def fresnel_T(n1, n2, ti):
    s = n1 / n2 * math.sin(ti)
    if s >= 1:
        return 0.0, 0.0
    tt = math.asin(s)
    ci, ct = math.cos(ti), math.cos(tt)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return 1 - rs ** 2, 1 - rp ** 2


def _line_meet(p1, d1, p2, d2):
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-14:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def trace_launch(n: float, stage_z: float, det: Detection, dish: Dish, launch: Launch,
                 optics: LaunchOptics, pose: Optional[MotorPose] = None,
                 mode: str = "carriage_xz") -> Dict[str, object]:
    """Full launch geometry in the lab frame (x towards the launch, z up), in mm."""
    if pose is None:
        pose = inverse_kinematics(n, stage_z, det, dish, launch, mode)
    phi = pose.phi
    u = np.array([-math.sin(phi), math.cos(phi)])     # beam direction
    w = np.array([math.cos(phi), math.sin(phi)])      # in-plane transverse direction
    R = np.array(launch.pivot_home) + np.array([pose.x, pose.z])
    P = R + (launch.d0_mm + pose.focus) * u           # ideal air focus of the objective
    cyl = cyl_prescription(optics)
    D = beam_diameter(optics)
    obj = P - optics.f_obj_mm * u
    cyl_back = obj - cyl["gap_to_objective_mm"] * u
    cyl_front = cyl_back - optics.cyl_thickness_mm * u
    col = cyl_front - optics.d_col_cyl_mm * u
    fib = col - optics.f_col_mm * u

    G = dish.G0_mm - stage_z
    hi, zb = -G, -G - dish.t_glass_mm
    H = n * (G + dish.t_glass_mm * (1 - 1 / dish.n_glass))
    F = np.array([0.0, hi + H])
    top = hi + 5.5
    th_s = sheet_angle(n, det)
    up = np.array([0.0, 1.0])
    down = -up

    rays, ends = [], []
    tir = False
    hs = np.linspace(-1, 1, max(3, optics.fan_rays)) * D / 2 * optics.fill
    for h in hs:
        a = col + h * w
        b = obj + h * w
        d = (P - b) / np.linalg.norm(P - b)
        t1 = (zb - b[1]) / d[1]
        Bi = b + t1 * d
        dg = _refract(d, down, 1.0, dish.n_glass)
        if dg is None:
            tir = True; continue
        Ei = Bi + (hi - Bi[1]) / dg[1] * dg
        dm = _refract(dg, down, dish.n_glass, n)
        if dm is None:
            tir = True; rays.append([fib.tolist(), a.tolist(), b.tolist(), Bi.tolist(), Ei.tolist()]); continue
        end = Ei + (top - Ei[1]) / dm[1] * dm
        rays.append([fib.tolist(), a.tolist(), b.tolist(), Bi.tolist(), Ei.tolist(), end.tolist()])
        ends.append((float(h), Ei, dm))

    out: Dict[str, object] = {"tir": tir}
    fk = forward_kinematics(pose, n, stage_z, det, dish, launch) if not tir else None
    summary = {
        "beam_D_mm": D, "NA_air": D / (2 * optics.f_obj_mm),
        "sheet_width_mm": D * optics.f_obj_mm / cyl["f_mm"],
        "phi_deg": math.degrees(phi), "cyl_R_mm": cyl["R_mm"], "cyl_n": cyl["n"],
        "cyl_to_objective_mm": cyl["gap_to_objective_mm"], "cyl_bfd_mm": cyl["bfd_mm"],
    }
    if ends and not tir:
        chief = min(ends, key=lambda e: abs(e[0]))
        Ec, dc = chief[1], chief[2]
        # least-squares best focus of the fan
        A = np.zeros((2, 2)); bb = np.zeros(2)
        for _, p, dd in ends:
            M = np.eye(2) - np.outer(dd, dd)
            A += M; bb += M @ p
        best = np.linalg.solve(A, bb)
        s_best = float((best - Ec) @ dc)
        # zone foci: where each ray crosses the chief ray
        zones = []
        for h, p, dd in ends:
            if abs(h) < 1e-12:
                continue
            m = _line_meet(Ec, dc, p, dd)
            if m is not None:
                zones.append((h, float((m - Ec) @ dc)))
        zones.sort(key=lambda z: abs(z[0]))
        s_parax = float(np.mean([z[1] for z in zones[:2]])) if zones else s_best
        s_marg = [z[1] for z in zones if abs(abs(z[0]) - np.max(np.abs(hs))) < 1e-9]
        opt = sheet_optics(n, det, Launch(beam_D_mm=D, f_obj_mm=optics.f_obj_mm,
                                           f_cyl_mm=cyl["f_mm"],
                                           wavelength_nm=optics.wavelength_nm))
        # rms spot at best focus, perpendicular to the chief ray
        nrm = np.array([dc[1], -dc[0]])
        spot = [float(((p + ((best - p) @ dd) * dd) - best) @ nrm) for _, p, dd in ends]
        env = []
        for s in np.linspace(0, float((top - Ec[1]) / dc[1]), 60):
            wz = opt["w0_um"] * 1e-3 * math.sqrt(1 + ((s - s_parax) / opt["rayleigh_mm"]) ** 2)
            c = Ec + s * dc
            env.append([c.tolist(), (c + wz * nrm).tolist(), (c - wz * nrm).tolist()])
        tg = math.asin(math.sin(phi) / dish.n_glass)
        tm = math.asin(math.sin(phi) / n)
        Ts1, Tp1 = fresnel_T(1.0, dish.n_glass, phi)
        Ts2, Tp2 = fresnel_T(dish.n_glass, n, tg)
        summary.update({
            "NA_medium": opt["NA_eff_medium"], "w0_um": opt["w0_um"], "fwhm_um": opt["fwhm_um"],
            "rayleigh_mm": opt["rayleigh_mm"], "fov_along_sheet_mm": opt["fov_along_sheet_mm"],
            "theta_sheet_deg": 90 - math.degrees(tm),
            "coddington_s_mm": fk.waist_error_mm + float((F - fk.E) @ dc) if fk else None,
            "paraxial_s_mm": s_parax, "best_s_mm": s_best,
            "marginal_s_mm": s_marg, "zone_foci": zones,
            "focus_spread_um": (max(z[1] for z in zones) - min(z[1] for z in zones)) * 1e3 if zones else 0.0,
            "rms_spot_um": float(np.sqrt(np.mean(np.square(spot)))) * 1e3,
            "T_s": Ts1 * Ts2, "T_p": Tp1 * Tp2,
            "angle_err_deg": math.degrees(fk.angle_error), "offset_err_um": fk.offset_error_mm * 1e3,
            "waist_err_mm": fk.waist_error_mm,
        })
        out["envelope"] = env
        out["focus_best"] = best.tolist()
        out["focus_parax"] = (Ec + s_parax * dc).tolist()
        out["chief"] = {"E": Ec.tolist(), "d": dc.tolist()}
    dp = np.array([-math.cos(th_s), math.sin(th_s)])
    L = sheet_optics(n, det, launch)["fov_along_sheet_mm"]
    out.update({
        "lab": {"fiber": fib.tolist(), "collimator": col.tolist(), "cyl_front": cyl_front.tolist(),
                "cyl_back": cyl_back.tolist(), "objective": obj.tolist(), "P": P.tolist(),
                "pivot": R.tolist(), "u": u.tolist(), "w": w.tolist(), "F": F.tolist(),
                "zb": zb, "hi": hi, "top": top,
                "det_plane": [(F - dp * L / 2).tolist(), (F + dp * L / 2).tolist()]},
        "rays": rays,
        "summary": summary,
        "unfolded": _unfolded(optics, cyl, D, P, obj, zb, u),
        "inputs": {"n": n, "stage_z": stage_z, "pose": asdict(pose), "optics": asdict(optics)},
    })
    return out


def _unfolded(o: LaunchOptics, cyl, D, P, obj, zb, u) -> Dict[str, object]:
    """Axial layout along the launch axis (s = 0 at the fibre), tangential and sagittal marginal rays."""
    s_col = o.f_col_mm
    s_cf = s_col + o.d_col_cyl_mm
    s_cb = s_cf + o.cyl_thickness_mm
    s_obj = s_cb + cyl["gap_to_objective_mm"]
    s_P = s_obj + o.f_obj_mm
    s_B = s_obj + (zb - obj[1]) / u[1]
    s_line = s_cb + cyl["bfd_mm"]
    h = D / 2
    wy = h * o.f_obj_mm / cyl["f_mm"]
    tang, sag = [], []
    for k in (-1, -0.5, 0, 0.5, 1):
        tang.append([[0, 0], [s_col, k * h], [s_obj, k * h], [s_P, 0]])
        y_obj = -k * h * (s_obj - s_line) / (s_line - s_cb)
        sag.append([[0, 0], [s_col, k * h], [s_cb, k * h], [s_line, 0], [s_obj, y_obj], [s_B, y_obj]])
    return {"elements": [
        {"name": "fibre", "s": 0.0, "kind": "source"},
        {"name": "collimator", "s": s_col, "kind": "lens", "h": h * 1.4},
        {"name": "cylinder lens", "s": s_cf, "s2": s_cb, "kind": "cyl", "h": h * 1.6},
        {"name": "objective", "s": s_obj, "kind": "lens", "h": max(h, wy) * 1.4},
        {"name": "dish bottom", "s": s_B, "kind": "dish"}],
        "tang": tang, "sag": sag, "s_B": s_B, "s_P": s_P, "s_line": s_line,
        "sheet_halfwidth_mm": wy}


# ----------------------------------------------------------------------------- Optiland
def optiland_script(n: float, stage_z: float, det: Detection, dish: Dish, launch: Launch,
                    optics: LaunchOptics, pose: Optional[MotorPose] = None,
                    mode: str = "carriage_xz", title: str = "dvOPM light-sheet launch") -> str:
    """Stand-alone Optiland script. Local frame: z along the launch axis, y = in-plane
    transverse direction (sheet thickness), x = sheet width (cylinder power)."""
    tr = trace_launch(n, stage_z, det, dish, launch, optics, pose, mode)
    if tr["tir"]:
        raise ValueError("Total internal reflection for this launch angle - nothing to export.")
    lab, s = tr["lab"], tr["summary"]
    u, w = np.array(lab["u"]), np.array(lab["w"])
    C = np.array(lab["collimator"])
    loc = lambda q: (float((np.array(q) - C) @ w), float((np.array(q) - C) @ u))
    phi = math.radians(s["phi_deg"])
    E, dc = np.array(tr["chief"]["E"]), np.array(tr["chief"]["d"])
    B = np.array(tr["rays"][len(tr["rays"]) // 2][3])
    img = np.array(tr["focus_parax"])
    yB, zB = loc(B)
    yE, zE = loc(E)
    yI, zI = loc(img)
    # Optiland rotates a surface normal (0,0,1) about x by rx -> (0, -sin rx, cos rx).
    # Dish normal in the local frame is (0, sin phi, cos phi)  -> rx = -phi.
    rx_dish = -phi
    rx_img = -math.atan2(float(dc @ w), float(dc @ u))
    cyl = cyl_prescription(optics)
    lam = optics.wavelength_nm * 1e-3
    return f'''"""
{title} - exported from the openUC2 dvOPM launch model.

Medium n = {n:.4f}, stage Z = {stage_z:.3f} mm, launch tilt phi = {s["phi_deg"]:.3f} deg from the dish normal.
Local frame: z = launch axis, y = in-plane transverse (sheet THICKNESS, the YZ view),
x = sheet WIDTH (the cylinder's power axis, the XZ view).
Ideal (paraxial) collimator and objective stand in for the real parts - swap in vendor
prescriptions if you have them. Dish surfaces are tilted and placed at absolute
coordinates, so the paraxial solver does not apply past the objective; use real rays.

Model predictions to compare with the real-ray result printed below:
    beam diameter           {s["beam_D_mm"]:.3f} mm      (2 f_col tan asin NA)
    sheet NA in air         {s["NA_air"]:.4f}
    sheet width (x)         {s["sheet_width_mm"]:.3f} mm
    Coddington waist        {s["coddington_s_mm"]:.4f} mm along the refracted chief ray from the inner dish surface
    real-ray paraxial focus {s["paraxial_s_mm"]:.4f} mm   (fan focus spread {s["focus_spread_um"]:.1f} um)
    transmission s / p      {s["T_s"]:.3f} / {s["T_p"]:.3f}
"""
import numpy as np
import matplotlib.pyplot as plt
from optiland import optic
from optiland.materials import IdealMaterial

WAVELENGTH_UM = {lam:.4f}
lens = optic.Optic()
_add = lens.surfaces.add if hasattr(lens, "surfaces") and hasattr(lens.surfaces, "add") else lens.add_surface

# 0  fibre tip (point source) one collimator focal length in front of the collimator
_add(index=0, thickness={optics.f_col_mm:.6f}, comment="fibre tip")
# 1  collimator (ideal), aperture stop
_add(index=1, surface_type="paraxial", f={optics.f_col_mm:.6f}, thickness={optics.d_col_cyl_mm:.6f},
     is_stop=True, comment="collimator f={optics.f_col_mm:g}")
# 2-3  plano-convex cylinder lens, power in x only (sheet width), R = (n-1) f
_add(index=2, surface_type="toroidal", radius_x={cyl["R_mm"]:.6f}, radius_y=np.inf,
     thickness={optics.cyl_thickness_mm:.6f}, material="{optics.cyl_glass}", comment="cyl f={cyl["f_mm"]:.2f} (R {cyl["R_mm"]:.2f})")
_add(index=3, thickness={cyl["gap_to_objective_mm"]:.6f}, comment="line focus in the objective BFP")
# 4  launch objective (ideal)
_add(index=4, surface_type="paraxial", f={optics.f_obj_mm:.6f}, thickness=0.0, comment="objective f={optics.f_obj_mm:g}")
# 5-6  dish bottom, tilted by the launch angle (absolute placement)
_add(index=5, x=0.0, y={yB:.6f}, z={zB:.6f}, rx={rx_dish:.8f},
     material=IdealMaterial(n={dish.n_glass:.5f}), comment="dish bottom, outer")
_add(index=6, x=0.0, y={yE:.6f}, z={zE:.6f}, rx={rx_dish:.8f},
     material=IdealMaterial(n={n:.5f}), comment="dish bottom, inner -> medium")
# 7  image plane at the sheet focus, perpendicular to the refracted chief ray
_add(index=7, x=0.0, y={yI:.6f}, z={zI:.6f}, rx={rx_img:.8f}, comment="sheet focus")

lens.set_aperture(aperture_type="objectNA", value={optics.fiber_na:.5f})
(lens.fields.set_type if hasattr(lens, "fields") and hasattr(lens.fields, "set_type") else lens.set_field_type)("angle")
(lens.fields.add if hasattr(lens, "fields") and hasattr(lens.fields, "add") else lens.add_field)(y=0.0)
(lens.wavelengths.add if hasattr(lens, "wavelengths") and hasattr(lens.wavelengths, "add") else lens.add_wavelength)(
    value=WAVELENGTH_UM, is_primary=True)


def tangential_focus(num=21):
    """Real-ray tangential (sheet-thickness) focus in the medium, measured from the
    chief ray's hit on the inner dish surface, plus the rms line width there."""
    py = np.linspace(-1, 1, num)
    lens.trace_generic(0.0, 0.0, np.zeros(num), py, WAVELENGTH_UM)
    sg = lens.surfaces if hasattr(type(lens.surfaces), "y") else lens.surface_group
    k = 6
    p = np.stack([np.asarray(sg.y[k]), np.asarray(sg.z[k])], 1)
    d = np.stack([np.asarray(sg.M[k]), np.asarray(sg.N[k])], 1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A = np.zeros((2, 2)); b = np.zeros(2)
    for pi, di in zip(p, d):
        M = np.eye(2) - np.outer(di, di); A += M; b += M @ pi
    f = np.linalg.solve(A, b)
    c = num // 2
    s = float((f - p[c]) @ d[c])
    nrm = np.array([d[c][1], -d[c][0]])
    spot = [float(((pi + ((f - pi) @ di) * di) - f) @ nrm) for pi, di in zip(p, d)]
    return s, float(np.sqrt(np.mean(np.square(spot)))) * 1e3


if __name__ == "__main__":
    s_best, rms_um = tangential_focus()
    print(f"real-ray best focus: {{s_best:.4f}} mm along the chief ray (model: {s["best_s_mm"]:.4f} mm), "
          f"geometric rms line width {{rms_um:.2f}} um")
    for proj, name in (("YZ", "thickness"), ("XZ", "width")):
        res = lens.draw(num_rays=7, projection=proj)
        fig = res[0] if isinstance(res, tuple) else plt.gcf()
        fig.suptitle(f"{title}: sheet {{name}} plane ({{proj}}), n = {n:.3f}")
        fig.savefig(f"launch_{{proj}}.png", dpi=150, bbox_inches="tight")
    # close-up of the refraction at the dish (unequal axes so the tilt is readable)
    fig, ax = lens.draw(num_rays=9, projection="YZ")[:2]
    ax.set_aspect("auto")
    ax.set_xlim({min(zB, zE, zI) - 3:.4f}, {max(zB, zE, zI) + 1.5:.4f})
    ax.set_ylim({min(yB, yE, yI) - 1.5:.4f}, {max(yB, yE, yI) + 1.5:.4f})
    fig.suptitle(f"{title}: refraction into the dish, n = {n:.3f}, phi = {s["phi_deg"]:.2f} deg")
    fig.savefig("launch_YZ_dish.png", dpi=150, bbox_inches="tight")
    print("saved launch_YZ.png, launch_XZ.png and launch_YZ_dish.png")
    plt.show()
'''


def main():
    ap = argparse.ArgumentParser(description="Export the dvOPM light-sheet launch as an Optiland script")
    ap.add_argument("--n", type=float, default=1.45)
    ap.add_argument("--stage-z", type=float, default=0.0)
    ap.add_argument("--fiber-na", type=float, default=0.10)
    ap.add_argument("--f-col", type=float, default=8.0)
    ap.add_argument("--f-cyl", type=float, default=None, help="cylinder focal length; default: the LJ1878L1 radius")
    ap.add_argument("--f-obj", type=float, default=50.0)
    ap.add_argument("--wavelength", type=float, default=488.0)
    ap.add_argument("--out", default="launch_optiland.py")
    a = ap.parse_args()
    det, dish = Detection(), Dish()
    launch = paper_like_launch(det, dish)
    o = LaunchOptics(fiber_na=a.fiber_na, f_col_mm=a.f_col, f_obj_mm=a.f_obj, wavelength_nm=a.wavelength)
    if a.f_cyl is not None:
        o.f_cyl_mm, o.cyl_R_mm = a.f_cyl, None
    with open(a.out, "w") as fh:
        fh.write(optiland_script(a.n, a.stage_z, det, dish, launch, o))
    s = trace_launch(a.n, a.stage_z, det, dish, launch, o)["summary"]
    print(f"wrote {a.out}")
    for k in ("beam_D_mm", "NA_air", "sheet_width_mm", "phi_deg", "coddington_s_mm", "paraxial_s_mm",
              "best_s_mm", "focus_spread_um", "rms_spot_um", "fwhm_um", "T_s", "T_p"):
        v = s[k]
        print(f"  {k:18s} {v:.4f}" if isinstance(v, float) else f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
