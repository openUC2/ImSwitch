"""LED illumination geometry: LED positions -> illumination NA (k-vectors) in camera frame.

Every LED source (geodesic dome, ring board, rectangular matrix) is a list of LED positions in
chain order; DPC and FPM only need the illumination NA of each LED in camera coordinates.

Frames: positions in mm, sample focus at the origin, +z = optical axis towards the LEDs.
NA = transverse direction cosines of the light travelling LED -> sample, -(x, y) / |p|.
Camera frame = (column, row) of the detector image. The calibration maps one to the other:
the LED structure sits at ``offset_mm`` relative to its nominal coordinates (for a planar board
offset_mm[2] is its height above the sample), the camera sees the NA vectors mirrored in x
(``flip``) and then rotated by ``rotation_deg``.
"""
import csv
import os

import cv2
import numpy as np

from imswitch.imcommon.model import dirtools

LAYOUT_DIR = os.path.join(dirtools.DataFileDirs.Root, 'led_layouts')
DEFAULT_CALIBRATION = {'rotation_deg': 0.0, 'flip': False, 'offset_mm': [0.0, 0.0, 0.0]}


def load_layout(spec):
    """(N, 3) LED positions [mm] in chain order.

    spec: name of a bundled layout ("geodome80", "illumination_ring136"), a CSV path with
    columns x, y, z (one row per LED in chain order), or a rectangular matrix
    {"grid": [nx, ny], "pitch_mm": 4.0, "height_mm": 40.0, "serpentine": false} (row-major
    chain). Planar CSV layouts (z = 0, e.g. the ring board) get their height from the
    calibration's offset_mm[2].
    """
    if isinstance(spec, dict):
        nx, ny = spec['grid']
        iy, ix = np.divmod(np.arange(nx * ny), nx)
        if spec.get('serpentine', False):
            ix = np.where(iy % 2 == 1, nx - 1 - ix, ix)
        pitch = float(spec.get('pitch_mm', 4.0))
        return np.stack([(ix - (nx - 1) / 2) * pitch, (iy - (ny - 1) / 2) * pitch,
                         np.full(nx * ny, float(spec.get('height_mm', 40.0)))], axis=1)
    path = spec if os.path.isfile(spec) else os.path.join(LAYOUT_DIR, f'{spec}.csv')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r['x']), float(r['y']), float(r.get('z') or 0.0)] for r in rows])


def illumination_na(positions, calibration=None):
    """(N, 2) illumination NA of every LED in camera frame (column, row)."""
    cal = {**DEFAULT_CALIBRATION, **(calibration or {})}
    p = np.asarray(positions, float) + np.asarray(cal['offset_mm'], float)
    na = -p[:, :2] / np.linalg.norm(p, axis=1, keepdims=True)
    if cal['flip']:
        na[:, 0] *= -1
    a = np.deg2rad(cal['rotation_deg'])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return na @ rot.T


def dpc_patterns(na_cam, na_max, na_min=0.0, dead_band=0.01):
    """Four half-annulus LED masks in camera frame: +x, -x, +y, -y (column / row direction)."""
    r = np.hypot(na_cam[:, 0], na_cam[:, 1])
    ring = (r >= na_min) & (r <= na_max)
    x, y = na_cam[:, 0], na_cam[:, 1]
    return {'+x': ring & (x > dead_band), '-x': ring & (x < -dead_band),
            '+y': ring & (y > dead_band), '-y': ring & (y < -dead_band)}


def brightfield_scores(stack):
    """Mean intensity per image (needs a fixed exposure: auto exposure evens it out)."""
    return np.asarray(stack, np.float32).reshape(len(stack), -1).mean(axis=1)


def split_brightfield(scores):
    """Brightfield mask by Otsu's threshold on the scores."""
    s = np.asarray(scores, float)
    u8 = np.uint8(255 * (s - s.min()) / (np.ptp(s) + 1e-12)).reshape(1, -1)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return u8[0] > thr


def _texture(img):
    """Flat-fielded, zero-mean, Hann-windowed float image for phase correlation."""
    img = np.asarray(img, np.float32)
    img = img / (cv2.GaussianBlur(img, (0, 0), max(img.shape) / 20) + 1e-6) - 1
    return img * cv2.createHanningWindow(img.shape[::-1], cv2.CV_32F)


def image_shift(ref, img, max_size=512):
    """(dx, dy) [px] by which ``img`` is shifted relative to ``ref`` (phase correlation,
    on a copy binned down to ``max_size`` px)."""
    f = max(1, int(np.ceil(max(np.shape(ref)) / max_size)))
    small = lambda a: cv2.resize(np.asarray(a, np.float32), None, fx=1 / f, fy=1 / f,
                                 interpolation=cv2.INTER_AREA)
    (dx, dy), _ = cv2.phaseCorrelate(_texture(small(ref)), _texture(small(img)))
    return dx * f, dy * f


def fit_orientation(na_led, shifts_px):
    """Rotation / mirror between LED frame and camera from defocus image shifts.

    A defocused sample lit by LED i shifts by a * R F tan(theta_i) (px); the reference image
    adds a common offset, so the fit runs on mean-centred data. a > 0 is assumed: the sign of
    the defocus is unknown, so the result is ambiguous by 180 deg (= sign of the phase).
    na_led: (N, 2) NA in LED frame (illumination_na with rotation 0, no flip).
    """
    na_led = np.asarray(na_led, float)
    t = na_led / np.sqrt(1 - np.sum(na_led ** 2, axis=1, keepdims=True))
    y = np.asarray(shifts_px, float)
    y = y - y.mean(axis=0)
    best = None
    for flip in (False, True):
        x = t * ([-1, 1] if flip else [1, 1])
        x = x - x.mean(axis=0)
        dot = np.sum(x * y)
        cross = np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])
        a = np.hypot(dot, cross) / np.sum(x ** 2)
        ang = np.arctan2(cross, dot)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        rms = float(np.sqrt(np.mean(np.sum((y - a * x @ rot.T) ** 2, axis=1))))
        if best is None or rms < best['rms_px']:
            best = {'rotation_deg': float(np.rad2deg(ang)), 'flip': flip,
                    'scale_px': float(a), 'rms_px': rms}
    return best
