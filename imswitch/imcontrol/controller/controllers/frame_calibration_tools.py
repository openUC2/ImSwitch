"""Hardware-free helpers for FrameCalibrationController: plate geometry,
detectors and fits (numpy + OpenCV only).

Conventions
- Stage/user frame = plate frame, top view with the well-label side up:
  origin = A1-notch corner (stage 0,0), +x along 127.76 mm, +y towards row P, µm.
  KiCad front-side coordinates are mirrored: x_user = 127760 - x_kicad.
- Image pixels (x = column, y = row). ``J`` (2x2) = content displacement in px per
  µm of stage move. A plate point at offset d from the point under the image
  centre c appears at c - J @ d, so pixel p shows plate point u - inv(J) @ (p - c).
- Overview camera (fixed, plate moves): plate point P with the stage at u is seen
  at pixel o = H [P - u, 1] (homogeneous).

Self-check: python -m imswitch.imcontrol.controller.controllers.frame_calibration_tools
"""
import math

import cv2
import numpy as np

# FRAME calibration plate, KiCad "calibration-plate" Rev E (SBS footprint)
PLATE = {
    "key": "plate",
    "name": "FRAME calibration plate Rev E",
    "size": [127760, 85480],
    # both top corners are chamfered 10 mm; bottom corners 3.18 mm fillets (ignored)
    "outline": [[10000, 0], [117760, 0], [127760, 10000], [127760, 85480], [0, 85480], [0, 10000]],
    "via": [63910, 42025],            # centre via: drill 300, copper annulus 600
    "viaDiameter": 300,
    "roundHole": [21260, 60000],      # ⌀6000 routed
    "roundHoleDiameter": 6000,
    "rectHole": [21260, 51500],       # 6000 x 7000 routed
    "rectHoleSize": [6000, 7000],
    "viaGrid": {"first": [16885, 16225], "pitch": 1000, "n": 7},
    "focusTexture": True,             # the sunburst rays: focus anywhere around the via
}

# Alternative target: the Heidstar HDS-SH-80120-IS4 4-slide insert (slides along y, 30 mm
# pitch) with a KiCad "pinhole-slide" Rev A in slots 1 and 4. The slide is 76 x 26 mm
# with three vias on its centre line, 19 mm apart: drills 0.2 / 0.3 / 0.4 mm, the 0.3 mm
# one at the slide centre (the same drill as the plate's via). Which end has the 0.2 mm
# one depends on how the slide went in; the positions do not. Slot centres: ImSwitch's
# "Heidstar 4x Histosample" layout (frontend WellSelectorUtils.js), in the same plate
# frame as the SBS plate. The black PCB is opaque: no focus texture, no overview landmark.
_SLOT_X, _SLOT_Y = [18400 + 30000 * k for k in range(4)], 40600
SLIDES = {
    "key": "slides",
    "name": "4-slide insert with pinhole slides in slots 1 and 4",
    "size": [127760, 85480],
    "outline": [[0, 0], [127760, 0], [127760, 85480], [0, 85480]],
    "slides": [[x, _SLOT_Y] for x in _SLOT_X],      # slot centres
    "slideSize": [26000, 76000],
    "loadedSlots": [1, 4],
    "via": [_SLOT_X[0], _SLOT_Y],                    # the 0.3 mm pinhole of the slide in slot 1
    "viaDiameter": 300,
    "pinholes": [[x, _SLOT_Y + dy] for x in (_SLOT_X[0], _SLOT_X[3]) for dy in (-19000, 0, 19000)],
    "pinholeDiameters": [200, 300, 400],
    # placement: slot 4's centre pinhole, 90 mm away (insert rotation and the stage scale over
    # a long baseline); each slide's outer pinholes (its rotation in the slot, and from their
    # sizes which end the 0.2 mm one is at)
    "checkPoints": {"slot 4, centre": [_SLOT_X[3], _SLOT_Y],
                    **{f"slot {k}, {d:+d} mm": [_SLOT_X[k - 1], _SLOT_Y + 1000 * d] for k in (1, 4) for d in (-19, 19)}},
    "focusTexture": False,
}
TARGETS = {"plate": PLATE, "slides": SLIDES}

# where the rect hole lands if the plate is inserted wrongly (diagnosis only)
WRONG_ORIENTATIONS = {
    "rotated 180°": [127760 - 21260, 85480 - 51500],
    "upside down (sunburst side up)": [127760 - 21260, 51500],
    "upside down and rotated 180°": [21260, 85480 - 51500],
}


def gray(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img.astype(np.float32)


def fit_circle(pts):
    """Algebraic least-squares circle. Returns (cx, cy, r, rms_residual)."""
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = D / 2, E / 2
    r = math.sqrt(max(F + cx * cx + cy * cy, 0.0))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), float(r), rms


def _border_free_contour(mask, margin=3):
    """Outer contour points of a binary mask, minus points touching the image border."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2))
    pts = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    h, w = mask.shape
    keep = (pts[:, 0] > margin) & (pts[:, 0] < w - 1 - margin) & (pts[:, 1] > margin) & (pts[:, 1] < h - 1 - margin)
    return pts[keep]


def find_bright_disc(img, diameter_px, min_contrast=8.0, edge_margin=0.4):
    """The centre via in transmission: a bright disc of roughly ``diameter_px``, at the
    short via exposure ~100x brighter than the plate (the sunburst rays: ~4x).

    The circle is fitted to the blob outline without the points on the image
    border, so a disc cut by the frame edge (20x) still gives its true centre.
    A fitted centre less than ``edge_margin`` radii inside the frame is rejected: the
    halo of a via beyond the edge, or a large bright region cut by it (rig 2026-11-02:
    a "disc" on the last row that never came to the centre), cannot be centred.
    Returns dict(x, y, r, contrast, rms) in full-resolution px, or None.
    """
    g = gray(img)
    b = max(1, int(diameter_px // 120))          # work at ~120 px disc diameter
    if b > 1:   # crop to a multiple of b so the scale factor is exactly b
        g = g[:g.shape[0] // b * b, :g.shape[1] // b * b]
        g = cv2.resize(g, (g.shape[1] // b, g.shape[0] // b), interpolation=cv2.INTER_AREA)
    d = diameter_px / b
    k = max(3, int(d / 15) | 1)
    g = cv2.GaussianBlur(g, (k, k), 0)
    bg = float(np.percentile(g, 20))             # not the median: at 20x the via fills half the frame
    hi = float(np.percentile(g, 99.5))
    if hi < min_contrast * max(bg, 1e-6):
        return None
    mask = (g > (bg + hi) / 2).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.05 * math.pi * (d / 2) ** 2:
            continue
        pts = _border_free_contour(lab == i)
        if len(pts) < 20:
            continue
        cx, cy, r, rms = fit_circle(pts)
        if rms > 0.08 * r:      # something attached (flare): refit to the outline near the inscribed circle
            dist = cv2.distanceTransform((lab == i).astype(np.uint8), cv2.DIST_L2, 5)
            cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            r = float(dist.max())
            for _ in range(3):
                near = pts[np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < 0.15 * r]
                if len(near) < 20:
                    break
                cx, cy, r, rms = fit_circle(near)
        if not (0.5 < 2 * r / d < 2.0) or rms > 0.08 * r:
            continue
        m = edge_margin * r
        if not (m <= cx < g.shape[1] - m and m <= cy < g.shape[0] - m):
            continue                                 # cut by the frame edge: halo or a large bright region
        inside = g[lab == i].mean()
        contrast = float(inside / max(bg, 1e-6))
        if contrast < min_contrast:
            continue
        score = area * contrast
        if best is None or score > best[0]:
            best = (score, cx, cy, r, contrast, rms)
    if best is None:
        return None
    _, cx, cy, r, contrast, rms = best
    # pixel centres: binned px i covers full-res [i*b, (i+1)*b)
    return {"x": (cx + 0.5) * b - 0.5, "y": (cy + 0.5) * b - 0.5, "r": r * b,
            "contrast": contrast, "rms": rms * b}


def light_spot(img, min_ratio=4.0):
    """Where the light is in an otherwise dark frame: [x, y] (full-resolution px) of the
    pixels above half the brightest level, or None when nothing stands out from the
    black level. Size-agnostic: a far-defocused pinhole on an opaque slide is a large dim
    blur that find_bright_disc rejects. A fully lit frame (an open window) gives None."""
    g = gray(img)
    b = max(1, g.shape[1] // 384)
    if b > 1:
        g = cv2.resize(g[:g.shape[0] // b * b, :g.shape[1] // b * b], (g.shape[1] // b, g.shape[0] // b),
                       interpolation=cv2.INTER_AREA)
    lo, hi = float(np.percentile(g, 5)), float(np.percentile(g, 99.9))
    if hi < min_ratio * max(lo, 1.0):
        return None
    ys, xs = np.nonzero(g > (lo + hi) / 2)
    return [float((xs.mean() + 0.5) * b - 0.5), float((ys.mean() + 0.5) * b - 0.5)]


def image_shift(a, b, width=768):
    """Content displacement (full-resolution px) of image b relative to image a by phase
    correlation at ~``width`` px, and the peak response (0..1; below ~0.05 no reliable match)."""
    a, b = gray(a), gray(b)
    f = max(1, a.shape[1] // width)
    if f > 1:   # crop to a multiple of f so the scale factor is exactly f
        h, w = a.shape[0] // f * f, a.shape[1] // f * f
        a = cv2.resize(a[:h, :w], (w // f, h // f), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b[:h, :w], (w // f, h // f), interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), r = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), win)
    return np.array([dx, dy]) * f, float(r)


def focus_metric(img):
    """Normalised Laplacian energy at ~768 px width (the sensor noise averages out)."""
    g = gray(img)
    b = max(1, g.shape[1] // 768)
    if b > 1:
        g = cv2.resize(g, (g.shape[1] // b, g.shape[0] // b), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F)
    return float(lap.var() / max(float(g.mean()), 1e-6) ** 2)


def focus_peaked(vals, rise=1.5, fall=0.6):
    """A Z sweep has passed a clear focus peak: the best value stands out from the
    lowest one and the curve has fallen again after it."""
    v = np.asarray(vals, float)
    if len(v) < 3:
        return False
    i = int(np.argmax(v))
    return bool(v[i] > rise * v.min() and i < len(v) - 1 and v[i + 1:].min() < fall * v[i])


def boundary_points(img, margin=3):
    """Pixels on the bright/dark boundary (Otsu), without the image border."""
    g = cv2.GaussianBlur(gray(img), (5, 5), 0)
    g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(g8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2))
    pts = np.vstack([c[:, 0, :] for c in cnts]).astype(float)
    h, w = mask.shape
    keep = (pts[:, 0] > margin) & (pts[:, 0] < w - 1 - margin) & (pts[:, 1] > margin) & (pts[:, 1] < h - 1 - margin)
    return pts[keep]


def px_to_plate(p, J, u, shape):
    """Plate coordinates (µm) of pixel(s) p, stage at u, image shape (h, w)."""
    c = np.array([(shape[1] - 1) / 2, (shape[0] - 1) / 2])
    return np.asarray(u, float)[:2] - (np.linalg.inv(J) @ (np.atleast_2d(p) - c).T).T


def fit_J(stage_um, pixels):
    """Least-squares J (px per µm) from stage positions and the tracked feature's pixel
    positions. Returns (J, rms_px)."""
    s = np.asarray(stage_um, float)
    p = np.asarray(pixels, float)
    S = np.column_stack([s - s.mean(axis=0), np.ones(len(s))])
    M, *_ = np.linalg.lstsq(S, p, rcond=None)
    J = M[:2].T
    rms = float(np.sqrt(np.mean(np.sum((S @ M - p) ** 2, axis=1))))
    return J, rms


def J_metrics(J):
    """µm/px scales, rotation and shear of the camera for the PixelCalibration store."""
    A = np.linalg.inv(J)                      # µm per px of content displacement
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt                                # polar factor: unique even for isotropic scale (U alone is not)
    return {
        "scale_x_um_per_pixel": float(np.linalg.norm(A[:, 0])),
        "scale_y_um_per_pixel": float(np.linalg.norm(A[:, 1])),
        "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "shear_deg": float(90 - np.degrees(np.arccos(np.clip(
            abs(A[:, 0] @ A[:, 1]) / (np.linalg.norm(A[:, 0]) * np.linalg.norm(A[:, 1])), 0, 1)))),
        "condition_number": float(s[0] / s[1]),
        "determinant": float(np.linalg.det(A)),
    }


def affine_for_store(J):
    """2x3 matrix in PixelCalibration's convention (maps the registering shift = minus the
    content displacement to stage µm, see affine_stage_calibration.compute_affine_matrix)."""
    A = -np.linalg.inv(J)
    return [[float(A[0, 0]), float(A[0, 1]), 0.0], [float(A[1, 0]), float(A[1, 1]), 0.0]]


def axes_check(shifts, step_um, min_px=3.0, min_angle_deg=45.0):
    """Robust motion-direction check from ``shifts = {"+X": (dx, dy), "-X": ..., "+Y": ..., "-Y": ...}``
    (image displacement in px for a +/- step on each axis, measured from the same reference).

    Returns (J, problems). Checks: each move is visible, +/- moves are antisymmetric (no
    stale frame, no slipping axis), the two axes are not parallel (no swapped/duplicated
    axis; an oblique overview camera shows them ~70° apart). ``J`` columns are px per µm
    for X and Y. Feed it landmark positions, not global image correlation: a fixed
    glare or reflection pins the correlation peak at zero shift."""
    problems = []
    cols = []
    for ax in ("X", "Y"):
        p, m = np.asarray(shifts["+" + ax], float), np.asarray(shifts["-" + ax], float)
        if np.linalg.norm(p) < min_px or np.linalg.norm(m) < min_px:
            problems.append(f"{ax}: image did not move (|+|={np.linalg.norm(p):.1f}px, |-|={np.linalg.norm(m):.1f}px)")
        elif np.linalg.norm(p + m) > 0.3 * max(np.linalg.norm(p), np.linalg.norm(m)):
            problems.append(f"{ax}: +/- moves not symmetric ({p.round(1).tolist()} vs {m.round(1).tolist()}) - "
                            "stale frame, slipping motor or backlash larger than the step")
        cols.append((p - m) / (2 * step_um))
    J = np.column_stack(cols)
    nx, ny = np.linalg.norm(J[:, 0]), np.linalg.norm(J[:, 1])
    if nx > 0 and ny > 0:
        ang = np.degrees(np.arccos(np.clip(abs(J[:, 0] @ J[:, 1]) / (nx * ny), 0, 1)))
        if ang < min_angle_deg:
            problems.append(f"X and Y move the image in nearly the same direction ({ang:.0f}° apart) - axes swapped/miswired?")
    return J, problems


def direction_change_deg(J_now, J_ref):
    """Largest angle (deg) between the per-axis image directions of two J's."""
    out = 0.0
    for i in range(2):
        a, b = np.asarray(J_now)[:, i], np.asarray(J_ref)[:, i]
        out = max(out, float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))))
    return out


def glare_center(img):
    """Centroid [x, y] of the largest saturated blob: the LED glare marks the optical
    axis in the overview image (camera and LED are fixed to the frame)."""
    g = cv2.GaussianBlur(gray(img), (5, 5), 0)
    n, lab, stats, cent = cv2.connectedComponentsWithStats((g > 0.9 * g.max()).astype(np.uint8))
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return [float(cent[i][0]), float(cent[i][1])] if stats[i, cv2.CC_STAT_AREA] > 1e-3 * g.size else None


def find_plate_holes(img, center, radius):
    """Overview camera: {"round": [x, y], "rect": [x, y]} of the two through-holes
    within ``radius`` px of ``center`` (the optical-axis pixel), or None.

    The holes are dark blobs enclosed by the lit plate (Otsu split inside the region).
    Not simply the darkest blobs: in a dark enclosure only the LED's light cone lights
    the plate, and the unlit surroundings are darker than the holes (Pi rig
    2026-11-02); those touch the region border and drop out. A saturated cone (camera
    auto exposure) washes the holes out and gives no pair: take the frame at a short
    exposure, or by room light with the LED off. In the oblique view the 6x7 mm rect
    looks as elliptical as the ⌀6 round hole, so they are told apart as a pair: two
    hole-sized neighbours, the rect 1.2-2.6x the area of the round one. Silkscreen
    letters (an "O" has a dark ellipse inside) are far smaller."""
    blobs = _dark_blobs(img, center, radius)
    best = None
    for a, ca in blobs:
        for b, cb in blobs:
            if 1.2 < b / a < 2.6 and np.linalg.norm(ca - cb) < 3 * math.sqrt(b) and (best is None or a + b > best[0]):
                best = (a + b, ca, cb)
    if best is None:
        return None
    return {"round": [float(best[1][0]), float(best[1][1])], "rect": [float(best[2][0]), float(best[2][1])]}


def track_round_hole(img, center, radius, near, max_px):
    """The round hole followed from a known pixel position: the hole-sized dark blob
    nearest to ``near`` within ``max_px`` ([x, y]), or None. Only the round hole is
    tracked: through the rect hole the camera sees the objective turret below the plate,
    whose reflections cut into its dark area differently at every stage position (Pi rig
    2026-11-02), so the pair test fails or swaps the two."""
    blobs = _dark_blobs(img, center, radius)
    if not blobs:
        return None
    d = [float(np.linalg.norm(np.asarray(c) - np.asarray(near, float))) for _, c in blobs]
    i = int(np.argmin(d))
    return [float(blobs[i][1][0]), float(blobs[i][1][1])] if d[i] <= max_px else None


def _dark_blobs(img, center, radius):
    """[(area, centroid)] of hole-sized, compact dark blobs enclosed by the lit plate
    within ``radius`` px of ``center`` (see find_plate_holes)."""
    g = cv2.GaussianBlur(gray(img), (7, 7), 0)
    full = 255.0 if np.asarray(img).dtype == np.uint8 else max(float(g.max()), 1.0)
    roi = np.zeros(g.shape, np.uint8)
    cv2.circle(roi, (int(center[0]), int(center[1])), int(radius), 1, -1)
    v8 = np.clip(g[roi > 0] / full * 255, 0, 255).astype(np.uint8).reshape(-1, 1)
    t8, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lit = (g > t8 / 255 * full) & (roi > 0)
    n, lab, stats, cent = cv2.connectedComponentsWithStats((~lit).astype(np.uint8))
    outside = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if i in outside or not 2e-4 * g.size < area < 1e-2 * g.size:
            continue
        cnt, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), (rw, rh), _ = cv2.minAreaRect(max(cnt, key=cv2.contourArea))
        if 0.5 < area / max(rw * rh, 1) < 0.95 and max(rw, rh) / max(min(rw, rh), 1) < 2.2:
            blobs.append((area, cent[i]))
    return blobs


def fit_overview_homography(samples):
    """samples: [(u_xy, plate_xy, pixel_xy), ...] -> (H, rms_px, n_inliers) or (None, None, 0)."""
    if len(samples) < 4:
        return None, None, 0
    src = np.array([[P[0] - u[0], P[1] - u[1]] for u, P, _ in samples], dtype=np.float64)
    dst = np.array([o for _, _, o in samples], dtype=np.float64)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, None, 0
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - dst[inl]) ** 2, axis=1))))
    return H, rms, int(inl.sum())


def plate_to_overview_px(H, P, u):
    v = np.array([[[P[0] - u[0], P[1] - u[1]]]], dtype=np.float64)
    return cv2.perspectiveTransform(v, np.asarray(H, float)).reshape(2).tolist()


def overview_px_to_plate(H, o, u):
    v = np.array([[o]], dtype=np.float64)
    d = cv2.perspectiveTransform(v, np.linalg.inv(np.asarray(H, float))).reshape(2)
    return [float(d[0] + u[0]), float(d[1] + u[1])]


def placement(a_measured, b_measured, b_nominal=PLATE["rectHole"], a_nominal=PLATE["via"]):
    """Pose from two landmarks, a (the via, the offset anchor) and b (default: the plate's
    rect-hole centre): rotation of a->b against its nominal direction, and the measured /
    nominal length of a->b (the stage scale over that baseline)."""
    v = np.subtract(b_nominal, a_nominal)
    m = np.subtract(b_measured, a_measured)
    theta = math.atan2(m[1], m[0]) - math.atan2(v[1], v[0])
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    scale = float(np.linalg.norm(m) / np.linalg.norm(v))
    return {"rotationDeg": math.degrees(theta), "baselineScale": scale,
            "rectCenter": [float(x) for x in b_measured]}


def spiral(radius, step):
    """Square spiral of (dx, dy) offsets from (0, 0) out to ``radius``."""
    out = [(0.0, 0.0)]
    x = y = 0
    dx, dy = 1, 0
    n = int(math.ceil(radius / step))
    seg, k = 1, 0
    while True:
        for _ in range(2):
            for _ in range(seg):
                x, y = x + dx, y + dy
                if max(abs(x), abs(y)) > n:
                    return out
                out.append((x * step, y * step))
            dx, dy = -dy, dx
        seg += 1
        k += 1


def home_status(reply):
    """'done' / 'timeout' / error text from a blocking uc2rest Home.home() reply."""
    if isinstance(reply, list):
        for frame in reversed(reply):
            h = frame.get("home") if isinstance(frame, dict) else None
            if isinstance(h, dict) and "status" in h:
                return str(h["status"])
            if isinstance(h, dict) and "steppers" in h:     # legacy firmware
                st = h["steppers"]
                st = st[0] if isinstance(st, list) and st else st
                if isinstance(st, dict) and "isDone" in st:
                    return "done" if st["isDone"] else "timeout"
        return "no home-done frame received"
    if reply == "done":
        return "done"
    return f"error: {reply!r}"


def _demo():
    rng = np.random.default_rng(0)
    # disc detection incl. a disc cut by the frame edge
    for (cx, cy, d) in [(700.3, 410.7, 480), (1500.0, 1000.0, 1900)]:
        yy, xx = np.mgrid[0:2048, 0:3072]
        img = 200 + 40 * rng.standard_normal((2048, 3072))
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < (d / 2) ** 2] = 3000
        r = find_bright_disc(img, d)
        assert r and abs(r["x"] - cx) < 1.0 and abs(r["y"] - cy) < 1.0, (r, cx, cy)
    assert find_bright_disc(200 + 40 * rng.standard_normal((512, 512)), 100) is None
    # the halo of a via beyond the frame corner (rig 2026-11-02: black plate, glow up to
    # ~500 counts in the corner, fitted centre outside the frame) is not a via; the same
    # disc with its centre just inside the frame is
    yy, xx = np.mgrid[0:2048, 0:3072]
    halo = 8 + 500 * np.exp(-((xx - 3300) ** 2 + (yy - 2300) ** 2) / (2 * 400.0 ** 2))
    assert find_bright_disc(halo, 486) is None
    edge = 8 + 40 * rng.standard_normal((2048, 3072))
    edge[(xx - 2950) ** 2 + (yy - 1900) ** 2 < 243 ** 2] = 3000
    r = find_bright_disc(edge, 486)
    assert r and abs(r["x"] - 2950) < 1.0 and abs(r["y"] - 1900) < 1.0, r
    edge[:] = 8 + 40 * rng.standard_normal((2048, 3072))   # a large bright region cut by the last row
    edge[(xx - 1474) ** 2 + (yy - 2041) ** 2 < 311 ** 2] = 3000
    assert find_bright_disc(edge, 486) is None
    # image shift by phase correlation (pixel-size measurement from stage moves)
    tex = cv2.GaussianBlur(rng.random((1024, 1536)).astype(np.float32), (0, 0), 6)
    sh, resp = image_shift(tex, np.roll(tex, (-37, 52), axis=(0, 1)))
    assert np.allclose(sh, (52, -37), atol=1.5) and resp > 0.2, (sh, resp)
    # J round trip + px_to_plate
    J = np.array([[-1.6, 0.05], [-0.04, -1.62]])
    s = np.array([[x, y] for x in (-300, 0, 300) for y in (-300, 0, 300)], float)
    p = np.array([1000, 800]) + (J @ s.T).T + 0.1 * rng.standard_normal((9, 2))
    Jf, rms = fit_J(s, p)
    assert np.allclose(Jf, J, atol=2e-3) and rms < 0.3
    u = np.array([5000.0, 7000.0])
    P = u + np.array([100.0, -50.0])
    pix = np.array([(3072 - 1) / 2, (2048 - 1) / 2]) - J @ (P - u)
    assert np.allclose(px_to_plate(pix, J, u, (2048, 3072))[0], P)
    assert abs(J_metrics(J)["scale_x_um_per_pixel"] - 1 / 1.6) < 0.02
    assert abs(abs(J_metrics(J)["rotation_deg"]) - 180) < 2 and abs(abs(J_metrics(-1.67 * np.eye(2))["rotation_deg"]) - 180) < 1e-6
    # axes check: good, then swapped/stale
    sh = {"+X": J[:, 0] * 500, "-X": -J[:, 0] * 500, "+Y": J[:, 1] * 500, "-Y": -J[:, 1] * 500}
    Jc, probs = axes_check(sh, 500)
    assert not probs and np.allclose(Jc, J)
    assert axes_check({**sh, "-X": (0, 0)}, 500)[1]
    assert axes_check({**sh, "+Y": sh["+X"], "-Y": sh["-X"]}, 500)[1]
    # homography with stage translation
    Htrue = np.array([[0.0065, 0.0003, 320], [-0.0002, 0.0064, 240], [1e-8, 2e-8, 1]])
    samples = [((x, y), PLATE["roundHole"], plate_to_overview_px(Htrue, PLATE["roundHole"], (x, y)))
               for x in (0, 10000, 20000) for y in (0, 10000, 20000)]
    H, rms, n = fit_overview_homography(samples)
    assert n == 9 and rms < 1e-3
    u = (5000, 15000)       # inside the sampled range
    assert np.allclose(overview_px_to_plate(H, plate_to_overview_px(Htrue, (20000, 58000), u), u), (20000, 58000), atol=5)
    # optical axis from the glare; the two holes near it, a darker screw far away, small letters
    ov = np.full((1080, 1920, 3), 90, np.uint8)
    cv2.circle(ov, (950, 750), 250, (255, 255, 255), -1)
    assert np.allclose(glare_center(ov), (950, 750), atol=1)
    ov[:] = 60
    cv2.ellipse(ov, (975, 790), (34, 28), 0, 0, 360, (5, 5, 5), -1)
    cv2.ellipse(ov, (980, 897), (44, 40), 0, 0, 360, (5, 5, 5), -1)
    for x in range(700, 1200, 60):
        cv2.ellipse(ov, (x, 650), (6, 9), 0, 0, 360, (10, 10, 10), -1)
    cv2.circle(ov, (1600, 850), 50, (0, 0, 0), -1)
    hz = find_plate_holes(ov, (950, 750), 320)
    assert hz and np.allclose(hz["round"], (975, 790), atol=1) and np.allclose(hz["rect"], (980, 897), atol=1), hz
    # Pi rig 2026-11-02 (640x480 webcam in a dark enclosure, LED on, 10 ms): only the LED
    # cone lights the plate; the holes (~20 and ~40 counts, the rect one with a bright
    # reflection inside) sit in it and the unlit surroundings (~0) are darker than they are
    pi = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(pi, (316, 345), 80, (190, 190, 190), -1)
    cv2.circle(pi, (316, 345), 50, (252, 252, 252), -1)    # saturated core: marks the optical axis
    for x in range(250, 390, 22):                      # silkscreen well outlines in the cone
        cv2.line(pi, (x, 270), (x, 420), (240, 240, 240), 2)
    cv2.ellipse(pi, (325, 342), (16, 12), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(pi, (328, 387), (18, 17), 0, 0, 360, (40, 40, 40), -1)
    cv2.circle(pi, (329, 391), 3, (250, 250, 250), -1)
    hp = find_plate_holes(pi, glare_center(pi), 640 / 6)
    assert hp and np.allclose(hp["round"], (325, 342), atol=1.5) and np.allclose(hp["rect"], (328, 387), atol=1.5), hp
    # the turret seen through the rect hole: a bright crescent splits its dark area (run 5 frames
    # 4, 7, 9); the pair test may fail, the round hole is still tracked from its last position
    cv2.ellipse(pi, (322, 390), (16, 15), 0, 60, 300, (230, 230, 230), 6)
    assert np.allclose(track_round_hole(pi, glare_center(pi), 640 / 6, (318, 340), 18), (325, 342), atol=1.5)
    assert track_round_hole(pi, glare_center(pi), 640 / 6, (318, 300), 18) is None
    # same rig, camera auto exposure: the cone saturates over both holes, dark wells
    # remain at its rim - that must not give a hole pair (2026-11-02 run: swapped pairs)
    sat = np.full((480, 640, 3), 30, np.uint8)
    cv2.circle(sat, (318, 360), 75, (255, 255, 255), -1)
    for (x, y) in ((240, 426), (267, 450), (222, 403), (233, 404)):
        cv2.rectangle(sat, (x - 7, y - 7), (x + 7, y + 7), (60, 60, 60), -1)
    assert find_plate_holes(sat, glare_center(sat), 640 / 6) is None
    # placement: rotate the rect centre by 0.5° about the via
    t = math.radians(0.5)
    v = np.subtract(PLATE["rectHole"], PLATE["via"])
    rc = np.add(PLATE["via"], [v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t)])
    assert abs(placement(PLATE["via"], rc)["rotationDeg"] - 0.5) < 1e-6
    # slides: slot 4's centre pinhole 90 mm from slot 1's gives the rotation and the scale
    t, k = math.radians(-0.2), 1.001
    v = np.subtract(SLIDES["checkPoints"]["slot 4, centre"], SLIDES["via"])
    m4 = np.add(SLIDES["via"], k * np.array([v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t)]))
    pl = placement(SLIDES["via"], m4, SLIDES["checkPoints"]["slot 4, centre"], SLIDES["via"])
    assert abs(pl["rotationDeg"] + 0.2) < 1e-6 and abs(pl["baselineScale"] - k) < 1e-9, pl
    assert SLIDES["pinholes"][1] == SLIDES["via"] and len(TARGETS) == 2
    # light spot: a far-defocused pinhole (a dim blur ~3x the FOV height wide, centre off the
    # frame) on a black slide is found; black and fully lit frames give None
    yy, xx = np.mgrid[0:1024, 0:1536]
    blur = 8 + 300 * np.exp(-((xx - 1300) ** 2 + (yy - 200) ** 2) / (2 * 500.0 ** 2)) + 3 * rng.standard_normal((1024, 1536))
    ls = light_spot(blur)
    assert ls and ls[0] > 1000 and ls[1] < 450, ls
    assert light_spot(8 + 3 * rng.standard_normal((1024, 1536))) is None
    assert light_spot(np.full((1024, 1536), 3000.0) + 30 * rng.standard_normal((1024, 1536))) is None
    # via with a flare blob attached (rig 2026-10-30): the disc is still found
    fl = np.full((1024, 1536), 50.0)
    cv2.circle(fl, (760, 300), 120, 4000, -1)
    cv2.circle(fl, (600, 230), 70, 3000, -1)
    hf = find_bright_disc(fl, 240)
    assert hf and abs(hf["x"] - 760) < 3 and abs(hf["y"] - 300) < 3 and abs(hf["r"] - 120) < 4, hf
    # focus: a sharp edge scores higher than the blurred one; sweep stops after the peak (rig 2026-10-30)
    edge = np.full((512, 768), 100.0)
    edge[:, 400:] = 400
    assert focus_metric(edge) > 3 * focus_metric(cv2.GaussianBlur(edge, (0, 0), 8))
    rig = [30, 31, 30, 29, 28, 27, 27, 27, 28, 30, 31, 33, 36, 49, 64, 60, 43]
    assert not focus_peaked(rig) and focus_peaked(rig + [35]) and not focus_peaked([30, 31, 32, 31, 30])
    assert len(spiral(1000, 500)) == 25 and spiral(1000, 500)[0] == (0.0, 0.0)
    assert home_status([{"qid": 3}, {"home": {"stepperid": 1, "status": "done", "pos": 0}, "qid": 3}]) == "done"
    assert home_status([{"home": {"stepperid": 1, "status": "timeout"}}]) == "timeout"
    assert home_status("No response received").startswith("error")
    print("frame_calibration_tools self-check OK")


if __name__ == "__main__":
    _demo()
