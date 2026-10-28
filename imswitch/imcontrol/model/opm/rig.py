"""
opm_rig.py - mechanical model of the dvOPM rig for the setup diagram and move checks.

Coordinates are the same lab frame as ctdvopm_kinematics (mm):
    x  horizontal, positive towards the launch side
    y  horizontal, out of the side-view page (sheet width / untilted sensor axis)
    z  up, towards the sample; origin = nominal air focus of the detection lens

Parts and what moves them
    detection  telecentric lens below the dish looking up, camera below it with the
               sensor tilted by alpha. Fixed to the table.
    sample     XYZ stage (left) -> riser -> holder plate (cantilevered, detection hole)
               -> dish. Sample X/Y/Z move all of it rigidly.
    launch     XYZ stage (right) -> circular rail centred on the rotation axis R ->
               rail carriage -> focus slide -> arm with objective, cylinder lens,
               collimator and fibre, pointing radially at R. Launch X/Y/Z move R and
               everything on it; tilt phi moves the carriage around the rail; focus
               slides the optics along the arm.

The default dimensions are placeholders for typical parts (4x objective, 50 mm
telecentric lens, 35 mm dish). Put your real numbers in opm_rig_config.json - every
field of RigConfig can be set there.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kinematics import (Detection, Dish, Launch, MotorPose, forward_kinematics,
                                sheet_angle, sheet_optics)
from .optics import LaunchOptics, beam_diameter, cyl_prescription

Pt = Tuple[float, float]


@dataclass
class RigConfig:
    # detection, fixed to the table
    det_wd_mm: float = 65.0            # lens front -> nominal air focus
    det_diameter_mm: float = 50.0
    det_length_mm: float = 110.0
    cam_gap_mm: float = 12.0           # lens back -> sensor centre
    cam_width_mm: float = 60.0
    cam_depth_mm: float = 40.0
    # sample carrier
    dish_diameter_mm: float = 35.0
    dish_wall_mm: float = 1.0
    dish_height_mm: float = 10.0
    medium_height_mm: float = 5.5
    holder_thickness_mm: float = 3.0
    holder_right_mm: float = 14.0      # plate edge, launch side, from dish centre
    holder_hole_mm: float = 12.0       # detection hole radius
    holder_half_width_mm: float = 30.0 # plate half width along y
    sample_stage_x_mm: float = -90.0   # stage centre relative to the dish centre
    sample_riser_mm: float = 45.0      # plate underside -> stage carriage top
    # launch arm
    obj_wd_mm: float = 17.0
    obj_parfocal_mm: float = 45.0
    obj_diameter_mm: float = 20.0
    mount_size_mm: float = 25.0        # cylinder-lens mount
    collimator_diameter_mm: float = 12.0
    collimator_length_mm: float = 25.0
    rail_radius_mm: float = 110.0      # carriage distance from the rotation axis
    # stage bodies
    stage_width_mm: float = 50.0
    stage_height_mm: float = 22.0
    stage_base_height_mm: float = 16.0
    # travel limits
    sample_travel_mm: Tuple[float, float, float] = (12.5, 12.5, 2.0)   # +- X, Y, Z
    launch_travel_mm: Tuple[float, float, float] = (12.5, 12.5, 12.5)  # +- X, Y, Z
    phi_range_deg: Tuple[float, float] = (40.0, 88.0)
    focus_travel_mm: float = 10.0

    @classmethod
    def load(cls, path: Path) -> "RigConfig":
        cfg = cls()
        if path.exists():
            cfg.update(json.loads(path.read_text()))
        return cfg

    def update(self, d: Dict) -> List[str]:
        names = {f.name: f for f in fields(self)}
        bad = [k for k in d if k not in names]
        for k, v in d.items():
            if k in names:
                cur = getattr(self, k)
                setattr(self, k, tuple(float(x) for x in v) if isinstance(cur, tuple) else float(v))
        return bad

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


# ------------------------------------------------------------------------ geometry helpers
def _rect_along(p0: np.ndarray, u: np.ndarray, s0: float, s1: float, h0: float, h1: Optional[float] = None) -> List[Pt]:
    """Quad along -u from p0: from distance s0 to s1, half-widths h0 (at s0) and h1 (at s1)."""
    h1 = h0 if h1 is None else h1
    w = np.array([u[1], -u[0]])
    a, b = p0 - s0 * u, p0 - s1 * u
    return [tuple(a + h0 * w), tuple(b + h1 * w), tuple(b - h1 * w), tuple(a - h0 * w)]


def _box(cx: float, cy: float, hw: float, hh: float) -> List[Pt]:
    return [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]


def _pt_seg(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _seg_dist(p, q, a, b) -> float:
    if _seg_intersect(p, q, a, b):
        return 0.0
    return min(_pt_seg(p[0], p[1], a[0], a[1], b[0], b[1]), _pt_seg(q[0], q[1], a[0], a[1], b[0], b[1]),
               _pt_seg(a[0], a[1], p[0], p[1], q[0], q[1]), _pt_seg(b[0], b[1], p[0], p[1], q[0], q[1]))


def _seg_intersect(p, q, a, b) -> bool:
    def cross(o, s, t):
        return (s[0] - o[0]) * (t[1] - o[1]) - (s[1] - o[1]) * (t[0] - o[0])
    d1, d2 = cross(a, b, p), cross(a, b, q)
    d3, d4 = cross(p, q, a), cross(p, q, b)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _inside(pt, poly) -> bool:
    x, y = pt
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def polys_overlap(A: Sequence[Pt], B: Sequence[Pt]) -> bool:
    if _inside(A[0], B) or _inside(B[0], A):
        return True
    return any(_seg_intersect(A[i], A[(i + 1) % len(A)], B[j], B[(j + 1) % len(B)])
               for i in range(len(A)) for j in range(len(B)))


def poly_distance(A: Sequence[Pt], B: Sequence[Pt]) -> float:
    """0 if the polygons overlap, otherwise the smallest edge-to-edge distance."""
    if _inside(A[0], B) or _inside(B[0], A):
        return 0.0
    best = math.inf
    for i in range(len(A)):
        for j in range(len(B)):
            best = min(best, _seg_dist(A[i], A[(i + 1) % len(A)], B[j], B[(j + 1) % len(B)]))
            if best == 0.0:
                return 0.0
    return best


def seg_hits_poly(p: Pt, q: Pt, poly: Sequence[Pt]) -> bool:
    if _inside(p, poly) or _inside(q, poly):
        return True
    return any(_seg_intersect(p, q, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


# ------------------------------------------------------------------------ the rig
@dataclass
class RigState:
    sample: Dict[str, float]           # X, Y, Z
    launch: MotorPose                  # x, z, phi, focus
    launch_y: float = 0.0


class Rig:
    def __init__(self, cfg: RigConfig, det: Detection, dish: Dish, launch: Launch,
                 optics: Optional[LaunchOptics] = None):
        self.cfg, self.det, self.dish, self.launch = cfg, det, dish, launch
        self.optics = optics or LaunchOptics()

    # ---------------------------------------------------------------- limits
    def limits(self) -> Dict[str, Tuple[float, float]]:
        c = self.cfg
        sx, sy, sz = c.sample_travel_mm
        lx, ly, lz = c.launch_travel_mm
        return {"sample.X": (-sx, sx), "sample.Y": (-sy, sy), "sample.Z": (-sz, sz),
                "launch.x": (-lx, lx), "launch.y": (-ly, ly), "launch.z": (-lz, lz),
                "launch.phi_deg": c.phi_range_deg, "launch.focus": (-c.focus_travel_mm, c.focus_travel_mm)}

    def check_limits(self, st: RigState) -> List[Dict]:
        vals = {"sample.X": st.sample["X"], "sample.Y": st.sample["Y"], "sample.Z": st.sample["Z"],
                "launch.x": st.launch.x, "launch.y": st.launch_y, "launch.z": st.launch.z,
                "launch.phi_deg": math.degrees(st.launch.phi), "launch.focus": st.launch.focus}
        out = []
        for k, (lo, hi) in self.limits().items():
            v = vals[k]
            out.append({"axis": k, "value": v, "min": lo, "max": hi, "ok": lo - 1e-9 <= v <= hi + 1e-9})
        return out

    # ---------------------------------------------------------------- parts, side view (x, z)
    def _side_parts(self, st: RigState, n: float) -> Dict[str, Dict]:
        c, d = self.cfg, self.dish
        sx, sz = st.sample["X"], st.sample["Z"]
        G = d.G0_mm - sz
        hi = -G                          # inner dish bottom
        zb = hi - d.t_glass_mm           # outer dish bottom = holder plate top
        r = c.dish_diameter_mm / 2
        parts: Dict[str, Dict] = {}

        def put(pid, pts, group, label=None, solid=True):
            parts[pid] = {"id": pid, "kind": "poly", "pts": [list(map(float, p)) for p in pts],
                          "group": group, "label": label, "solid": solid}

        # detection
        put("det_lens", _box(0, -c.det_wd_mm - c.det_length_mm / 2, c.det_diameter_mm / 2, c.det_length_mm / 2),
            "detection", "Telecentric lens")
        # sensor centred on the detection axis at the image plane, normal tilted by alpha;
        # the camera body sits behind the sensor
        sens_a = np.array([0.0, -c.det_wd_mm - c.det_length_mm - c.cam_gap_mm])
        a = self.det.alpha
        ua = np.array([math.sin(a), math.cos(a)])          # sensor normal, tilted by alpha from the axis
        put("camera", _rect_along(sens_a, ua, 0, c.cam_depth_mm, c.cam_width_mm / 2),
            "detection", f"Camera, sensor at {self.det.alpha_deg:g}°")
        w_a = np.array([ua[1], -ua[0]])
        parts["sensor"] = {"id": "sensor", "kind": "line", "group": "detection_sensor",
                           "pts": [list(sens_a + w_a * c.cam_width_mm * 0.38), list(sens_a - w_a * c.cam_width_mm * 0.38)]}

        # sample carrier (moves with sample X, Z)
        put("dish_bottom", _box(sx, zb + d.t_glass_mm / 2, r, d.t_glass_mm / 2), "dish")
        put("dish_wall_l", _box(sx - r + c.dish_wall_mm / 2, zb + c.dish_height_mm / 2, c.dish_wall_mm / 2, c.dish_height_mm / 2), "dish")
        put("dish_wall_r", _box(sx + r - c.dish_wall_mm / 2, zb + c.dish_height_mm / 2, c.dish_wall_mm / 2, c.dish_height_mm / 2), "dish")
        put("medium", _box(sx, hi + c.medium_height_mm / 2, r - c.dish_wall_mm, c.medium_height_mm / 2), "medium", solid=False)
        t = c.holder_thickness_mm
        stage_x = sx + c.sample_stage_x_mm
        left_end = stage_x - c.stage_width_mm / 2
        put("holder_l", [(left_end, zb - t), (sx - c.holder_hole_mm, zb - t), (sx - c.holder_hole_mm, zb), (left_end, zb)],
            "sample", "Holder plate")
        put("holder_r", [(sx + c.holder_hole_mm, zb - t), (sx + c.holder_right_mm, zb - t),
                         (sx + c.holder_right_mm, zb), (sx + c.holder_hole_mm, zb)], "sample")
        car_top = zb - t - c.sample_riser_mm
        put("sample_riser", _box(stage_x, (car_top + zb - t) / 2, 6, c.sample_riser_mm / 2), "sample")
        put("sample_carriage", _box(stage_x, car_top - c.stage_height_mm / 2, c.stage_width_mm / 2, c.stage_height_mm / 2),
            "sample_stage", "Sample XYZ stage")
        base_top = car_top - c.stage_height_mm - sz     # the base does not move with Z
        put("sample_base", _box(stage_x - sx + 0.0, base_top - c.stage_base_height_mm / 2 - 2,
                                c.stage_width_mm / 2 + 6, c.stage_base_height_mm / 2), "stage_base")

        # launch (moves with launch X, Z; tilt; focus)
        pose = st.launch
        u = np.array([-math.sin(pose.phi), math.cos(pose.phi)])          # beam direction
        R = np.array(self.launch.pivot_home) + np.array([pose.x, pose.z])
        P = R + (self.launch.d0_mm + pose.focus) * u                    # objective's air focus
        o = self.optics
        cyl = cyl_prescription(o)
        s_cyl_back = o.f_obj_mm + cyl["gap_to_objective_mm"]
        s_cyl_front = s_cyl_back + o.cyl_thickness_mm
        s_col = s_cyl_front + o.d_col_cyl_mm
        ro = c.obj_diameter_mm / 2
        # objective: tapered nose (front face at the working distance) + body up to the shoulder
        wv = np.array([u[1], -u[0]])
        f0, f1, sh = P - c.obj_wd_mm * u, P - (c.obj_wd_mm + 6) * u, P - c.obj_parfocal_mm * u
        put("objective", [f0 + 0.55 * ro * wv, f1 + ro * wv, sh + ro * wv,
                          sh - ro * wv, f1 - ro * wv, f0 - 0.55 * ro * wv], "launch", "4x objective")
        put("arm_tube", _rect_along(P, u, c.obj_parfocal_mm, s_cyl_back - c.mount_size_mm / 2 + o.cyl_thickness_mm / 2, 6),
            "launch_rail")
        s_mount = (s_cyl_back + s_cyl_front) / 2
        put("cyl_mount", _rect_along(P, u, s_mount - c.mount_size_mm / 2, s_mount + c.mount_size_mm / 2, c.mount_size_mm / 2),
            "launch", f"Cylinder lens f={cyl['f_mm']:.1f}")
        put("arm_tube2", _rect_along(P, u, s_mount + c.mount_size_mm / 2, s_col, 6), "launch_rail")
        put("collimator", _rect_along(P, u, s_col, s_col + c.collimator_length_mm, c.collimator_diameter_mm / 2),
            "launch", "Collimator")
        tail = P - (s_col + c.collimator_length_mm) * u
        fib = [tail, tail - 18 * u + 10 * np.array([0, -1.0]), tail - 22 * u + 28 * np.array([0, -1.0])]
        parts["fibre"] = {"id": "fibre", "kind": "line", "group": "fibre", "pts": [list(p) for p in fib]}

        # circular rail about R, rail carriage and focus slide
        rr = c.rail_radius_mm
        lo, hi_deg = [math.radians(v) for v in c.phi_range_deg]
        arc = [list(R + rr * np.array([math.sin(p), -math.cos(p)])) for p in np.linspace(lo, hi_deg, 40)]
        arc_out = [list(R + (rr + 6) * np.array([math.sin(p), -math.cos(p)])) for p in np.linspace(hi_deg, lo, 40)]
        parts["rail"] = {"id": "rail", "kind": "poly", "group": "rail", "pts": arc + arc_out,
                         "label": "Circular rail (rotation about R)", "solid": True}
        car = R - rr * u
        put("rail_carriage", _rect_along(R, -u, -rr + 3, -rr - 9, 14), "rail")   # sits on the rail, radially outside
        put("focus_slide", _rect_along(R, -u, -(rr - 12), -(rr + 12), 5), "rail")
        # rail bracket down to the launch XYZ stage (bracket fixed to the rail, not to phi)
        mid = 0.5 * (lo + hi_deg)
        rail_mid = R + (rr + 6) * np.array([math.sin(mid), -math.cos(mid)])
        stage_c = np.array(self.launch.pivot_home) + np.array([pose.x, pose.z]) + \
            np.array([rr * 0.95 + 20, -rr * 0.95 - 10])
        # bracket from rail midpoint to stage top
        put("rail_bracket", [tuple(rail_mid + [-4, 0]), tuple(rail_mid + [4, 0]),
                             (stage_c[0] + 4, stage_c[1] + c.stage_height_mm / 2),
                             (stage_c[0] - 4, stage_c[1] + c.stage_height_mm / 2)], "rail")
        put("launch_carriage", _box(stage_c[0], stage_c[1], c.stage_width_mm / 2, c.stage_height_mm / 2),
            "launch_stage", "Launch XYZ stage")
        base_c = stage_c - np.array([pose.x, pose.z])
        put("launch_base", _box(base_c[0], base_c[1] - c.stage_height_mm / 2 - c.stage_base_height_mm / 2 - 2,
                                c.stage_width_mm / 2 + 6, c.stage_base_height_mm / 2), "stage_base")
        parts["_pts"] = {"R": R, "P": P, "u": u, "hi": hi, "zb": zb, "car": car,
                         "obj_front": f0, "stage_c": stage_c}
        return parts

    # ---------------------------------------------------------------- full layout + checks
    def layout(self, st: RigState, n: float) -> Dict:
        c = self.cfg
        parts = self._side_parts(st, n)
        pts = parts.pop("_pts")
        R, P, u, hi, zb = pts["R"], pts["P"], pts["u"], pts["hi"], pts["zb"]
        sx, sy, sz = st.sample["X"], st.sample["Y"], st.sample["Z"]
        items = list(parts.values())

        # optics: beam (air), sheet in the medium, detection plane and field
        det, dish = self.det, self.dish
        th = sheet_angle(n, det)
        dp = np.array([-math.cos(th), math.sin(th)])
        G = dish.G0_mm - sz
        H = n * (G + dish.t_glass_mm * (1 - 1 / dish.n_glass))
        F = np.array([0.0, hi + H])
        opt = sheet_optics(n, det, self.launch)
        Lf = opt["fov_along_sheet_mm"]
        fov_a, fov_b = F - dp * Lf / 2, F + dp * Lf / 2
        checks: Dict = {"collisions": [], "clearances": [], "notes": []}
        try:
            fk = forward_kinematics(st.launch, n, sz, det, dish, self.launch)
            um = np.array([-math.cos(fk.theta_sheet), math.sin(fk.theta_sheet)])
            top = hi + c.medium_height_mm
            s_top = (top - fk.E[1]) / um[1]
            beam_air = [pts["obj_front"], fk.B]
            items.append({"id": "beam_air", "kind": "line", "group": "beam", "pts": [list(p) for p in beam_air]})
            items.append({"id": "beam_glass", "kind": "line", "group": "beam", "pts": [list(fk.B), list(fk.E)]})
            items.append({"id": "sheet", "kind": "line", "group": "sheet",
                          "pts": [list(fk.E), list(fk.E + s_top * um)]})
            items.append({"id": "waist", "kind": "point", "group": "sheet", "pts": [list(fk.waist)], "label": "waist"})
            sheet_ok = True
        except ValueError as e:
            checks["notes"].append(str(e))
            fk, sheet_ok = None, False
        items.append({"id": "det_plane", "kind": "line", "group": "det_plane",
                      "pts": [list(fov_a - dp * 2), list(fov_b + dp * 2)]})
        items.append({"id": "fov", "kind": "line", "group": "fov", "pts": [list(fov_a), list(fov_b)],
                      "label": "field of view"})
        # telecentric chief rays: straight down from the field ends to the lens front
        for k, e in (("a", fov_a), ("b", fov_b)):
            items.append({"id": f"chief_{k}", "kind": "line", "group": "chief",
                          "pts": [list(e), [float(e[0]), -c.det_wd_mm]]})
        items.append({"id": "F", "kind": "point", "group": "focus", "pts": [list(F)], "label": "detection focus"})
        items.append({"id": "R", "kind": "point", "group": "pivot", "pts": [list(R)], "label": "rotation axis R"})
        items.append({"id": "rot_circle", "kind": "circle", "group": "annot",
                      "pts": [list(R)], "r": c.rail_radius_mm})

        # -------- checks: collisions / clearances between moving launch parts and everything else
        launch_ids = ["objective", "cyl_mount", "collimator", "arm_tube", "arm_tube2", "rail_carriage"]
        other_ids = ["holder_l", "holder_r", "dish_bottom", "dish_wall_l", "dish_wall_r", "det_lens", "camera",
                     "sample_carriage", "sample_riser"]
        for a in launch_ids:
            for b in other_ids:
                dmin = poly_distance(parts[a]["pts"], parts[b]["pts"])
                if dmin <= 0:
                    checks["collisions"].append({"a": a, "b": b})
                    parts[a]["hit"] = parts[b]["hit"] = True
                else:
                    checks["clearances"].append({"a": a, "b": b, "mm": dmin})
        checks["clearances"].sort(key=lambda x: x["mm"])
        checks["clearances"] = checks["clearances"][:4]
        # sample carrier vs detection lens (sample Z too low)
        dz = zb - c.holder_thickness_mm - (-c.det_wd_mm)
        checks["holder_to_detection_mm"] = dz
        if dz <= 0:
            checks["collisions"].append({"a": "holder_l", "b": "det_lens"})
        # beam and field clipping by the holder plate / dish wall
        blockers = ["holder_l", "holder_r", "dish_wall_l", "dish_wall_r"]
        checks["beam_blocked_by"] = [b for b in blockers if fk is not None and
                                     seg_hits_poly(tuple(pts["obj_front"]), tuple(fk.B), parts[b]["pts"])]
        field_block = []
        for e in (fov_a, fov_b):
            for b in ("holder_l", "holder_r"):
                if seg_hits_poly((float(e[0]), float(e[1])), (float(e[0]), -c.det_wd_mm), parts[b]["pts"]):
                    field_block.append(b)
        checks["field_blocked_by"] = sorted(set(field_block))
        for it in items:
            if it["id"] == "beam_air" and checks["beam_blocked_by"]:
                it["hit"] = True
            if it["id"].startswith("chief_") and field_block:
                it["hit"] = True
        checks["limits"] = self.check_limits(st)
        checks["scan_range_mm"] = self._scan_range(st, n, fov_a, fov_b, pts, fk)

        # sheet width vs field width (top view)
        D = beam_diameter(self.optics)
        sheet_h = D * self.optics.f_obj_mm / cyl_prescription(self.optics)["f_mm"]
        fov_w = det.sensor_flat_mm / det.M
        cover = max(0.0, min(st.launch_y + sheet_h / 2, fov_w / 2) - max(st.launch_y - sheet_h / 2, -fov_w / 2))
        checks["sheet_height_mm"] = sheet_h
        checks["fov_width_mm"] = fov_w
        checks["sheet_covers_fov_frac"] = cover / fov_w

        side_bbox = self._bbox(items)
        dish_bbox = [min(fov_a[0], fov_b[0], F[0], (fk.E[0] if fk else 0)) - 4 + 0, zb - 6,
                     max(fov_a[0], fov_b[0], (fk.E[0] if fk else 0)) + 10, hi + c.medium_height_mm + 3]
        return {"side": {"items": items, "bbox": side_bbox, "dish_bbox": [float(v) for v in dish_bbox]},
                "top": self._top(st, n, fov_a, fov_b, fk, pts, sheet_h, fov_w),
                "checks": checks,
                "state": {"sample": st.sample, "launch": {"x": st.launch.x, "y": st.launch_y, "z": st.launch.z,
                                                          "phi_deg": math.degrees(st.launch.phi),
                                                          "focus": st.launch.focus}},
                "limits": {k: list(v) for k, v in self.limits().items()},
                "config": asdict(c)}

    def _scan_range(self, st, n, fov_a, fov_b, pts, fk) -> Optional[List[float]]:
        """Contiguous sample-X range around the current position in which the field and the
        beam stay clear of the holder, the field stays inside the dish, and the launch
        parts don't collide with the carrier. Sample X translates the carrier rigidly, so
        its polygons are built once and shifted."""
        c = self.cfg
        lo_t, hi_t = self.limits()["sample.X"]
        xs = np.round(np.arange(lo_t, hi_t + 1e-9, 0.25), 3)
        s0 = RigState({"X": 0.0, "Y": st.sample["Y"], "Z": st.sample["Z"]}, st.launch, st.launch_y)
        base = self._side_parts(s0, n)
        base.pop("_pts")
        carrier = {k: np.array(base[k]["pts"]) for k in ("holder_l", "holder_r", "dish_bottom", "dish_wall_l", "dish_wall_r")}
        movers = {k: base[k]["pts"] for k in ("objective", "cyl_mount", "collimator")}
        r_in = c.dish_diameter_mm / 2 - c.dish_wall_mm
        beam = (tuple(pts["obj_front"]), tuple(fk.B)) if fk is not None else None

        def ok(x):
            for e in (fov_a, fov_b):
                if abs(e[0] - x) > min(c.holder_hole_mm, r_in):
                    return False
            shifted = {k: [tuple(p) for p in v + [x, 0.0]] for k, v in carrier.items()}
            if beam is not None and any(seg_hits_poly(beam[0], beam[1], shifted[b])
                                        for b in ("holder_l", "holder_r", "dish_wall_l", "dish_wall_r")):
                return False
            return not any(polys_overlap(movers[a], shifted[b]) for a in movers for b in shifted)
        flags = [ok(x) for x in xs]
        i0 = int(np.argmin(np.abs(xs - st.sample["X"])))
        if not flags[i0]:
            return None
        a = i0
        while a > 0 and flags[a - 1]:
            a -= 1
        b = i0
        while b < len(xs) - 1 and flags[b + 1]:
            b += 1
        return [float(xs[a]), float(xs[b])]

    # ---------------------------------------------------------------- top view (x, y)
    def _top(self, st, n, fov_a, fov_b, fk, pts, sheet_h, fov_w) -> Dict:
        c = self.cfg
        sx, sy = st.sample["X"], st.sample["Y"]
        ly = st.launch_y
        r = c.dish_diameter_mm / 2
        items = []

        def put(pid, pts_, group, label=None, kind="poly", **kw):
            items.append({"id": pid, "kind": kind, "pts": [list(map(float, p)) for p in pts_], "group": group,
                          "label": label, **kw})
        stage_x = sx + c.sample_stage_x_mm
        put("sample_base", _box(stage_x - sx, 0, c.stage_width_mm / 2 + 6, c.stage_width_mm / 2 + 6), "stage_base")
        put("sample_carriage", _box(stage_x, sy, c.stage_width_mm / 2, c.stage_width_mm / 2), "sample_stage",
            "Sample XYZ stage")
        left_end = stage_x - c.stage_width_mm / 2
        put("holder", [(left_end, sy - c.holder_half_width_mm), (sx + c.holder_right_mm, sy - c.holder_half_width_mm),
                       (sx + c.holder_right_mm, sy + c.holder_half_width_mm), (left_end, sy + c.holder_half_width_mm)],
            "sample", "Holder plate")
        put("holder_hole", [(sx, sy)], "hole", kind="circle", r=c.holder_hole_mm)
        put("det_lens", [(0, 0)], "detection_top", "Telecentric lens (below)", kind="circle", r=c.det_diameter_mm / 2)
        put("dish", [(sx, sy)], "dish", "Dish", kind="circle", r=r)
        x0, x1 = sorted([float(fov_a[0]), float(fov_b[0])])
        put("fov", _box((x0 + x1) / 2, 0, (x1 - x0) / 2, fov_w / 2), "fov", "field of view")
        # launch arm projected on x-y: along x, at y = launch_y
        P, u = pts["P"], pts["u"]
        o = self.optics
        cyl = cyl_prescription(o)
        s_cyl_back = o.f_obj_mm + cyl["gap_to_objective_mm"]
        s_mount = s_cyl_back + o.cyl_thickness_mm / 2
        s_col = s_cyl_back + o.cyl_thickness_mm + o.d_col_cyl_mm

        def span(s0, s1, hw, pid, group, label=None):
            xa, xb = P[0] - s0 * u[0], P[0] - s1 * u[0]
            put(pid, _box((xa + xb) / 2, ly, abs(xb - xa) / 2, hw), group, label)
        span(c.obj_wd_mm, c.obj_parfocal_mm, c.obj_diameter_mm / 2, "objective", "launch", "4x objective")
        span(c.obj_parfocal_mm, s_col, 6, "arm", "launch_rail")
        span(s_mount - c.mount_size_mm / 2, s_mount + c.mount_size_mm / 2, c.mount_size_mm / 2, "cyl_mount", "launch")
        span(s_col, s_col + c.collimator_length_mm, c.collimator_diameter_mm / 2, "collimator", "launch")
        sc = pts["stage_c"]
        put("launch_base", _box(sc[0] - st.launch.x, 0, c.stage_width_mm / 2 + 6, c.stage_width_mm / 2 + 6), "stage_base")
        put("launch_carriage", _box(sc[0], ly, c.stage_width_mm / 2, c.stage_width_mm / 2), "launch_stage", "Launch XYZ stage")
        if fk is not None:
            xs0 = float(pts["obj_front"][0])
            put("beam_air", _box((xs0 + fk.E[0]) / 2, ly, abs(xs0 - fk.E[0]) / 2, sheet_h / 2), "beam_top")
            um_x = -math.cos(fk.theta_sheet)
            x_end = fk.E[0] + um_x * (c.medium_height_mm / math.sin(fk.theta_sheet))
            put("sheet", _box((fk.E[0] + x_end) / 2, ly, abs(fk.E[0] - x_end) / 2, sheet_h / 2), "sheet_top", "light sheet")
        put("F", [(0.0, 0.0)], "focus", "detection axis", kind="point")
        return {"items": items, "bbox": self._bbox(items)}

    @staticmethod
    def _bbox(items) -> List[float]:
        xs, ys = [], []
        for it in items:
            if it["kind"] == "circle":
                (cx, cy), r = it["pts"][0], it["r"]
                xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
            else:
                for p in it["pts"]:
                    xs.append(p[0]); ys.append(p[1])
        return [min(xs), min(ys), max(xs), max(ys)]


# ------------------------------------------------------------------------ self-test
if __name__ == "__main__":
    from .kinematics import inverse_kinematics, paper_like_launch
    det, dish = Detection(), Dish()
    L = paper_like_launch(det, dish)
    rig = Rig(RigConfig(), det, dish, L)
    pose = inverse_kinematics(1.45, 0.0, det, dish, L)
    st = RigState({"X": 0.0, "Y": 0.0, "Z": 0.0}, pose)
    lay = rig.layout(st, 1.45)
    ck = lay["checks"]
    print("collisions:", ck["collisions"])
    print("closest:", [(c["a"], c["b"], round(c["mm"], 2)) for c in ck["clearances"]])
    print("beam blocked by:", ck["beam_blocked_by"], " field blocked by:", ck["field_blocked_by"])
    print("scan range X:", ck["scan_range_mm"], " sheet covers", round(ck["sheet_covers_fov_frac"] * 100, 1), "% of field width")
    print("holder -> detection lens:", round(ck["holder_to_detection_mm"], 2), "mm")
    # invariants
    it = {i["id"]: i for i in lay["side"]["items"]}
    R = np.array(it["R"]["pts"][0])
    for phi_deg in (50, 70, 85):
        p2 = MotorPose(pose.x, pose.z, math.radians(phi_deg), pose.focus)
        l2 = rig.layout(RigState(st.sample, p2), 1.45)
        i2 = {i["id"]: i for i in l2["side"]["items"]}
        # rotation leaves R fixed and keeps the carriage on the rail
        assert np.allclose(i2["R"]["pts"][0], R)
        car = np.mean(np.array(i2["rail_carriage"]["pts"]), 0)
        assert abs(np.linalg.norm(car - R) - (rig.cfg.rail_radius_mm + 3)) < 1e-6, np.linalg.norm(car - R)
    # sample X moves the dish and holder, not the detection lens
    l3 = rig.layout(RigState({"X": 3.0, "Y": 0, "Z": 0}, pose), 1.45)
    i3 = {i["id"]: i for i in l3["side"]["items"]}
    assert np.allclose(np.array(i3["dish_bottom"]["pts"]) - np.array(it["dish_bottom"]["pts"]), [[3, 0]] * 4)
    assert np.allclose(i3["det_lens"]["pts"], it["det_lens"]["pts"])
    print("invariants OK")
