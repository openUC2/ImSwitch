"""
Goniometer / Contact Angle Measurement Controller
===================================================
Measures left and right contact angles of a sessile drop on a substrate.

Provides:
- Snap image from camera
- Automated contact angle measurement (Canny + contour + polynomial fit)
- Manual contact angle measurement (3-point tangent)
- Measurement history with CSV/JSON export
- Annotated result image download
"""

import numpy as np
import io
import time
import json
import base64
import traceback
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

try:
    import cv2
    hasCV2 = True
except ImportError:
    hasCV2 = False

from imswitch.imcommon.model import initLogger, APIExport
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex
from ..basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS


# ──────────────────────────────────────────────
#  Default config (mirrors contact_angle.py)
# ──────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    canny_low=30,
    canny_high=100,
    blur_ksize=5,
    bright_row_thresh=60,
    roi_x_margin_frac=0.12,
    roi_y_above_frac=0.3,
    roi_y_below_px=60,
    min_contour_length=100,
    baseline_tolerance=5,
    local_fit_frac=0.08,
    poly_degree=3,
    tangent_delta=1.0,
    angle_range_low=5,
    angle_range_high=175,
)


# ──────────────────────────────────────────────
#  Contact-angle image processing functions
# ──────────────────────────────────────────────

def _find_roi(gray, config):
    h, w = gray.shape
    row_means = np.mean(gray, axis=1)
    bright_rows = np.where(row_means > config['bright_row_thresh'])[0]
    if len(bright_rows) == 0:
        return (h // 4, 3 * h // 4, w // 6, 5 * w // 6)
    substrate_row = bright_rows[-1]
    y1 = max(0, substrate_row - int(h * config['roi_y_above_frac']))
    y2 = min(h, substrate_row + config['roi_y_below_px'])
    x1 = int(w * config['roi_x_margin_frac'])
    x2 = int(w * (1 - config['roi_x_margin_frac']))
    return (y1, y2, x1, x2)


def _find_contours_in_roi(gray_roi, config):
    k = config['blur_ksize']
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(gray_roi, (k, k), 1.5)
    edges = cv2.Canny(blurred, config['canny_low'], config['canny_high'])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def _classify_contours(contours, min_length=100):
    flat_lines = []
    drops = []
    for cnt in contours:
        length = cv2.arcLength(cnt, False)
        if length < min_length:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / (bh + 1e-6)
        if aspect > 10 and bh < 15:
            flat_lines.append((length, cnt))
        elif aspect > 1.5:
            drops.append((length, cnt))
    drops.sort(key=lambda x: x[0], reverse=True)
    baselines = []
    if drops:
        drop_pts = drops[0][1][:, 0, :]
        drop_bottom = drop_pts[:, 1].max()
        drop_height = drop_pts[:, 1].max() - drop_pts[:, 1].min()
        for length, cnt in flat_lines:
            line_y = cnt[:, 0, 1].mean()
            if abs(line_y - drop_bottom) < drop_height * 0.4:
                baselines.append((length, cnt))
    baselines.sort(key=lambda x: x[0], reverse=True)
    return baselines, drops


def _determine_baseline(baselines, drop_contour, tolerance_frac=0.05):
    if baselines:
        pts = baselines[0][1][:, 0, :]
        x = pts[:, 0].astype(float)
        y = pts[:, 1].astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept, np.arctan(slope)
    pts = drop_contour[:, 0, :]
    y_max = pts[:, 1].max()
    height = y_max - pts[:, 1].min()
    threshold_y = y_max - height * tolerance_frac
    bottom_pts = pts[pts[:, 1] >= threshold_y]
    if len(bottom_pts) < 3:
        return 0.0, float(y_max), 0.0
    x = bottom_pts[:, 0].astype(float)
    y = bottom_pts[:, 1].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept, np.arctan(slope)


def _find_contact_points(drop_contour, slope, intercept, tolerance=5, baseline_contour=None):
    if baseline_contour is not None:
        bl_pts = baseline_contour[:, 0, :]
        bl_sorted = bl_pts[bl_pts[:, 0].argsort()]
        left = bl_sorted[0]
        right = bl_sorted[-1]
        left_y = slope * left[0] + intercept
        right_y = slope * right[0] + intercept
        return (int(left[0]), int(round(left_y))), (int(right[0]), int(round(right_y)))
    pts = drop_contour[:, 0, :]
    denom = np.sqrt(1 + slope ** 2)
    distances = np.abs(pts[:, 1] - (slope * pts[:, 0] + intercept)) / denom
    near = pts[distances <= tolerance]
    if len(near) < 2:
        near = pts[distances <= tolerance * 3]
    if len(near) < 2:
        closest_idx = np.argsort(distances)[:20]
        closest_pts = pts[closest_idx]
        if len(closest_pts) >= 2:
            sorted_by_x = closest_pts[closest_pts[:, 0].argsort()]
            near = np.array([sorted_by_x[0], sorted_by_x[-1]])
        else:
            return None, None
    sorted_pts = near[near[:, 0].argsort()]
    left = sorted_pts[0]
    right = sorted_pts[-1]
    left_y = slope * left[0] + intercept
    right_y = slope * right[0] + intercept
    return (int(left[0]), int(round(left_y))), (int(right[0]), int(round(right_y)))


def _extract_upper_contour(contour, slope, intercept, tolerance=5):
    pts = contour[:, 0, :]
    baseline_y_at_x = slope * pts[:, 0] + intercept
    mask = pts[:, 1] < (baseline_y_at_x - tolerance)
    upper = pts[mask]
    if len(upper) < 5:
        return pts[pts[:, 0].argsort()]
    return upper[upper[:, 0].argsort()]


def _local_polynomial_fit(contour_pts, contact_x, radius, degree=3, side='left'):
    if side == 'left':
        mask = (contour_pts[:, 0] >= contact_x) & (contour_pts[:, 0] <= contact_x + radius)
    else:
        mask = (contour_pts[:, 0] <= contact_x) & (contour_pts[:, 0] >= contact_x - radius)
    local_pts = contour_pts[mask]
    if len(local_pts) < degree + 1:
        mask = np.abs(contour_pts[:, 0] - contact_x) <= radius * 2
        local_pts = contour_pts[mask]
    if len(local_pts) < degree + 1:
        return None
    try:
        return np.polyfit(local_pts[:, 0].astype(float), local_pts[:, 1].astype(float), degree)
    except np.linalg.LinAlgError:
        return None


def _compute_tangent_angle(poly_coeffs, x_contact, baseline_slope, delta=1.0):
    y_minus = np.polyval(poly_coeffs, x_contact - delta)
    y_plus = np.polyval(poly_coeffs, x_contact + delta)
    tangent = np.array([2 * delta, y_plus - y_minus])
    baseline_dir = np.array([1.0, baseline_slope])
    cos_angle = np.dot(tangent, baseline_dir) / (
        np.linalg.norm(tangent) * np.linalg.norm(baseline_dir) + 1e-10
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def measure_contact_angles(gray, config=None, return_debug=False):
    """Full pipeline: measure left and right contact angles."""
    if config is None:
        config = DEFAULT_CONFIG

    result = {'left_angle': 9999, 'right_angle': 9999, 'success': False}
    debug = {}

    y1, y2, x1, x2 = _find_roi(gray, config)
    roi = gray[y1:y2, x1:x2]
    debug['roi_bbox'] = (y1, y2, x1, x2)

    contours = _find_contours_in_roi(roi, config)
    if not contours:
        return (result, debug) if return_debug else result

    baselines, drops = _classify_contours(contours, config['min_contour_length'])
    debug['n_baselines'] = len(baselines)
    debug['n_drops'] = len(drops)

    if not drops:
        return (result, debug) if return_debug else result

    drop_contour = drops[0][1]
    debug['contour'] = drop_contour

    slope, intercept, baseline_angle = _determine_baseline(baselines, drop_contour)
    result['baseline_slope'] = slope
    result['baseline_angle_deg'] = np.degrees(baseline_angle)
    debug['baseline_slope'] = slope
    debug['baseline_intercept'] = intercept
    if baselines:
        debug['baseline_contour'] = baselines[0][1]

    tol = config['baseline_tolerance']
    bl_cnt = baselines[0][1] if baselines else None
    left_pt, right_pt = _find_contact_points(drop_contour, slope, intercept, tol, baseline_contour=bl_cnt)
    if left_pt is None or right_pt is None:
        return (result, debug) if return_debug else result

    result['left_contact'] = (float(left_pt[0] + x1), float(left_pt[1] + y1))
    result['right_contact'] = (float(right_pt[0] + x1), float(right_pt[1] + y1))
    if result['left_contact'] is None or result['right_contact'] is None: 
        result['left_contact'] = 99999
    if result['right_contact'] is None:
        result['right_contact'] = 99999
    upper = _extract_upper_contour(drop_contour, slope, intercept, tol)
    debug['upper_contour'] = upper

    drop_width = abs(right_pt[0] - left_pt[0])
    fit_radius = max(20, int(drop_width * config['local_fit_frac']))
    debug['fit_radius'] = fit_radius

    poly_left = _local_polynomial_fit(upper, left_pt[0], fit_radius, config['poly_degree'], 'left')
    poly_right = _local_polynomial_fit(upper, right_pt[0], fit_radius, config['poly_degree'], 'right')
    debug['poly_left'] = poly_left
    debug['poly_right'] = poly_right

    if poly_left is None or poly_right is None:
        return (result, debug) if return_debug else result

    delta = config['tangent_delta']
    left_angle = _compute_tangent_angle(poly_left, left_pt[0], slope, delta)
    right_angle = _compute_tangent_angle(poly_right, right_pt[0], slope, delta)

    lo, hi = config['angle_range_low'], config['angle_range_high']
    if lo <= left_angle <= hi:
        result['left_angle'] = left_angle
    if lo <= right_angle <= hi:
        result['right_angle'] = right_angle
    if result['left_angle'] is None:
        result['left_angle'] = 99999
    if result['right_angle'] is None:
        result['right_angle'] = 99999
    result['success'] = result['left_angle'] != 99999 and result['right_angle'] != 99999
    return (result, debug) if return_debug else result


def draw_result(frame, result, debug, config=None):
    """Draw contact angle overlay on the frame."""
    if config is None:
        config = DEFAULT_CONFIG

    vis = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    y1, y2, x1, x2 = debug.get('roi_bbox', (0, h, 0, w))

    cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 100, 100), 1)

    if 'contour' in debug:
        shifted = debug['contour'].copy()
        shifted[:, 0, 0] += x1
        shifted[:, 0, 1] += y1
        cv2.drawContours(vis, [shifted], -1, (0, 0, 255), 2)

    if 'baseline_contour' in debug:
        bl = debug['baseline_contour'].copy()
        bl[:, 0, 0] += x1
        bl[:, 0, 1] += y1
        cv2.drawContours(vis, [bl], -1, (0, 200, 0), 2)

    if not result.get('success', False):
        cv2.putText(vis, "Detection failed", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis

    slope = debug['baseline_slope']
    intercept = debug['baseline_intercept']
    left_pt = result['left_contact']
    right_pt = result['right_contact']

    bx0, bx1_px = 0, w
    by0 = int(slope * (bx0 - x1) + intercept + y1)
    by1 = int(slope * (bx1_px - x1) + intercept + y1)
    cv2.line(vis, (bx0, by0), (bx1_px, by1), (0, 255, 0), 2)

    cv2.circle(vis, (int(left_pt[0]), int(left_pt[1])), 6, (255, 255, 0), -1)
    cv2.circle(vis, (int(right_pt[0]), int(right_pt[1])), 6, (255, 255, 0), -1)

    fit_radius = debug.get('fit_radius', 60)
    tangent_len = max(80, fit_radius)

    for side, poly, contact, angle in [
        ('left', debug.get('poly_left'), left_pt, result['left_angle']),
        ('right', debug.get('poly_right'), right_pt, result['right_angle']),
    ]:
        if poly is None or angle is None:
            continue

        cx = contact[0] - x1
        if side == 'left':
            xs = np.linspace(cx, cx + fit_radius, 200)
        else:
            xs = np.linspace(cx - fit_radius, cx, 200)
        ys = np.polyval(poly, xs)
        curve_pts = np.column_stack((xs + x1, ys + y1)).astype(np.int32)
        for i in range(len(curve_pts) - 1):
            cv2.line(vis, tuple(curve_pts[i]), tuple(curve_pts[i + 1]), (0, 255, 255), 2)

        delta = config['tangent_delta']
        dy = np.polyval(poly, cx + delta) - np.polyval(poly, cx - delta)
        tangent_vec = np.array([2 * delta, dy])
        tangent_vec = tangent_vec / (np.linalg.norm(tangent_vec) + 1e-10)
        pt = np.array([float(contact[0]), float(contact[1])])
        cv2.line(vis,
                 tuple((pt - tangent_vec * tangent_len).astype(int)),
                 tuple((pt + tangent_vec * tangent_len).astype(int)),
                 (255, 0, 255), 2)

        text = f"{angle:.1f} deg"
        tx = int(contact[0]) - (20 if side == 'left' else 120)
        ty = int(contact[1]) - 40
        cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    tilt = result.get('baseline_angle_deg', 0)
    cv2.putText(vis, f"Baseline tilt: {tilt:.2f} deg", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis


def compute_manual_angle(baseline_pt1, baseline_pt2, tangent_pt):
    """Compute contact angle from 3 manually selected points.

    baseline_pt1, baseline_pt2 define the substrate line.
    tangent_pt and closest baseline point define the tangent direction.
    The contact point is assumed to be the closest baseline endpoint.
    """
    b1 = np.array(baseline_pt1, dtype=float)
    b2 = np.array(baseline_pt2, dtype=float)
    tp = np.array(tangent_pt, dtype=float)

    # Contact point is whichever baseline endpoint is closer to tangent_pt
    d1 = np.linalg.norm(tp - b1)
    d2 = np.linalg.norm(tp - b2)
    contact = b1 if d1 < d2 else b2

    baseline_vec = b2 - b1
    tangent_vec = tp - contact

    cos_angle = np.dot(baseline_vec, tangent_vec) / (
        np.linalg.norm(baseline_vec) * np.linalg.norm(tangent_vec) + 1e-10
    )
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return float(angle)


# ──────────────────────────────────────────────
#  Controller
# ──────────────────────────────────────────────

class GoniometerController(LiveUpdatedController):
    """
    Goniometer – contact angle measurement controller.

    Provides snap, automated measurement, manual measurement,
    measurement history, and result image download.
    """

    sigGoniometerResult = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # Camera
        if hasattr(self._setupInfo, 'goniometer') and self._setupInfo.goniometer is not None:
            self.camera = getattr(self._setupInfo.goniometer, 'camera', None)
        else:
            self.camera = None
        if self.camera is None:
            try:
                all_detectors = self._master.detectorsManager.getAllDeviceNames()
                if all_detectors:
                    self.camera = all_detectors[0]
            except Exception:
                pass

        # Processing config (mutable copy)
        self._config = dict(DEFAULT_CONFIG)

        # Snapped image (numpy, kept in memory)
        self._snapped_image = None  # BGR or gray
        self._snapped_gray = None

        # Last annotated result image (BGR)
        self._result_image = None

        # Measurement history
        self._measurements: List[Dict[str, Any]] = []

        self._logger.info("GoniometerController initialized (camera=%s)", self.camera)

    # ───────── helpers ─────────

    def _get_frame(self):
        """Grab the latest frame from the detector."""
        if self.camera is None:
            raise ValueError("No camera configured")
        detector = self._master.detectorsManager[self.camera]
        frame = detector.getLatestFrame()
        if frame is None:
            raise ValueError("No frame available from camera")
        return frame

    def _encode_image_png(self, image):
        """Encode a numpy image to PNG bytes."""
        ok, buf = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return buf.tobytes()

    def _image_to_base64(self, image):
        """Encode image to base64 data-URI (JPEG)."""
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"

    # ───────── API: configuration ─────────

    @APIExport(runOnUIThread=True)
    def get_config_goniometer(self) -> Dict[str, Any]:
        """Return current processing configuration."""
        return dict(self._config)

    @APIExport(runOnUIThread=True, requestType="POST")
    def set_config_goniometer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update one or more processing parameters."""
        for key, value in params.items():
            if key in self._config:
                self._config[key] = value
        return dict(self._config)

    @APIExport(runOnUIThread=True, requestType="POST")
    def reset_config_goniometer(self) -> Dict[str, Any]:
        """Reset processing parameters to defaults."""
        self._config = dict(DEFAULT_CONFIG)
        return dict(self._config)

    # ───────── API: snap ─────────

    @APIExport(runOnUIThread=True)
    def snap_goniometer(self) -> Dict[str, Any]:
        """Capture the current frame and return it as base64 JPEG."""
        try:
            frame = self._get_frame()
            # Normalise to uint8 BGR for display / processing
            if frame.dtype != np.uint8:
                if frame.max() > 0:
                    frame = (frame.astype(float) / frame.max() * 255).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            if len(frame.shape) == 2:
                self._snapped_gray = frame
                self._snapped_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                self._snapped_image = frame
                self._snapped_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            return {
                "success": True,
                "image": self._image_to_base64(self._snapped_image),
                "shape": list(self._snapped_image.shape),
                "timestamp": time.time(),
            }
        except Exception as e:
            self._logger.error("snap_goniometer failed: %s", traceback.format_exc())
            return {"success": False, "error": str(e)}

    # ───────── API: automated measurement ─────────

    @APIExport(runOnUIThread=True)
    def measure_auto_goniometer(self) -> Dict[str, Any]:
        """Run automated contact-angle measurement on the snapped image."""
        if self._snapped_gray is None:
            return {"success": False, "error": "No image snapped yet"}
        try:
            result, debug = measure_contact_angles(
                self._snapped_gray, self._config, return_debug=True
            )
            self._result_image = draw_result(self._snapped_image, result, debug, self._config)

            out = {
                "success": result.get("success", False),
                "left_angle": float(result.get("left_angle", 0)) if result.get("left_angle") is not None else 99999,
                "right_angle": float(result.get("right_angle", 0)) if result.get("right_angle") is not None else 99999,
                "baseline_angle_deg": float(result.get("baseline_angle_deg", 0)) if result.get("baseline_angle_deg") is not None else 99999,
                "left_contact": result.get("left_contact"),
                "right_contact": result.get("right_contact"),
                "annotated_image": self._image_to_base64(self._result_image),
                "timestamp": time.time(),
            }
            return out
        except Exception as e:
            self._logger.error("measure_auto_goniometer failed: %s", traceback.format_exc())
            return {"success": False, "error": str(e)}

    # ───────── API: manual measurement ─────────

    @APIExport(runOnUIThread=True, requestType="POST")
    def measure_manual_goniometer(self,
                                  baseline_x1: float = 0, baseline_y1: float = 0,
                                  baseline_x2: float = 0, baseline_y2: float = 0,
                                  tangent_x: float = 0, tangent_y: float = 0) -> Dict[str, Any]:
        """Compute contact angle from 3 user-selected points (baseline pair + tangent point)."""
        try:
            angle = compute_manual_angle(
                (baseline_x1, baseline_y1),
                (baseline_x2, baseline_y2),
                (tangent_x, tangent_y),
            )
            # Build annotated image if we have a snap
            annotated_b64 = None
            if self._snapped_image is not None:
                vis = self._snapped_image.copy()
                b1 = (int(baseline_x1), int(baseline_y1))
                b2 = (int(baseline_x2), int(baseline_y2))
                tp = (int(tangent_x), int(tangent_y))
                # Determine contact point
                d1 = np.linalg.norm(np.array(tp) - np.array(b1, dtype=float))
                d2 = np.linalg.norm(np.array(tp) - np.array(b2, dtype=float))
                contact = b1 if d1 < d2 else b2
                cv2.line(vis, b1, b2, (0, 255, 0), 2)
                cv2.line(vis, contact, tp, (255, 0, 255), 2)
                cv2.circle(vis, contact, 6, (255, 255, 0), -1)
                cv2.circle(vis, tp, 6, (0, 255, 255), -1)
                cv2.putText(vis, f"{angle:.1f} deg", (contact[0] + 10, contact[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                self._result_image = vis
                annotated_b64 = self._image_to_base64(vis)

            return {
                "success": True,
                "angle": angle,
                "annotated_image": annotated_b64,
                "timestamp": time.time(),
            }
        except Exception as e:
            self._logger.error("measure_manual_goniometer failed: %s", traceback.format_exc())
            return {"success": False, "error": str(e)}

    # ───────── API: measurement history ─────────

    @APIExport(runOnUIThread=True, requestType="POST")
    def add_measurement_goniometer(self, measurement: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a measurement result to the history list."""
        if measurement is None:
            measurement = {}
        entry = {
            "id": len(self._measurements) + 1,
            "timestamp": measurement.get("timestamp", time.time()),
            "left_angle": measurement.get("left_angle"),
            "right_angle": measurement.get("right_angle"),
            "mode": measurement.get("mode", "auto"),
        }
        self._measurements.append(entry)
        return {"success": True, "measurement": entry, "total": len(self._measurements)}

    @APIExport(runOnUIThread=True)
    def get_measurements_goniometer(self) -> Dict[str, Any]:
        """Return the full measurement history."""
        return {"measurements": self._measurements, "total": len(self._measurements)}

    @APIExport(runOnUIThread=True, requestType="POST")
    def clear_measurements_goniometer(self) -> Dict[str, Any]:
        """Clear all stored measurements."""
        self._measurements = []
        return {"success": True, "total": 0}

    # ───────── API: download result image ─────────

    @APIExport(runOnUIThread=True)
    def download_result_image_goniometer(self, format: str = "png"):
        """Return the last annotated result image as a downloadable response."""
        if self._result_image is None:
            return {"success": False, "error": "No result image available"}
        try:
            from fastapi.responses import Response
            if format == "jpg":
                ok, buf = cv2.imencode('.jpg', self._result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                media = "image/jpeg"
            else:
                ok, buf = cv2.imencode('.png', self._result_image)
                media = "image/png"
            if not ok:
                return {"success": False, "error": "Encoding failed"}
            return Response(content=buf.tobytes(), media_type=media)
        except ImportError:
            # Fallback: return base64
            return {"success": True, "image": self._image_to_base64(self._result_image)}

    @APIExport(runOnUIThread=True)
    def download_snapped_image_goniometer(self, format: str = "png"):
        """Return the snapped (raw) image as a downloadable response."""
        if self._snapped_image is None:
            return {"success": False, "error": "No snapped image available"}
        try:
            from fastapi.responses import Response
            if format == "jpg":
                ok, buf = cv2.imencode('.jpg', self._snapped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                media = "image/jpeg"
            else:
                ok, buf = cv2.imencode('.png', self._snapped_image)
                media = "image/png"
            if not ok:
                return {"success": False, "error": "Encoding failed"}
            return Response(content=buf.tobytes(), media_type=media)
        except ImportError:
            return {"success": True, "image": self._image_to_base64(self._snapped_image)}
