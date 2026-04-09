"""
Goniometer / Contact Angle Measurement Controller
===================================================
Measures left and right contact angles of a sessile drop on a substrate
using the mirror-geometry algorithm (contact_angle_cropped.py).

Assumes the droplet is cropped in view (or a crop ROI is set).
The mirror reflection produces a symmetric lens shape; the widest horizontal
line is the substrate baseline.

Provides:
- Snap image from camera (with optional crop ROI)
- Automated contact angle measurement (Canny + widest contour + polynomial tangent)
- Manual contact angle measurement (3-point tangent)
- Crop ROI management (set_crop / reset_crop)
- Live focus metric endpoint (Laplacian variance)
- Measurement history with CSV/JSON export
- Annotated result image download
"""

import numpy as np
import time
import base64
import traceback
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
#  Default config — mirror-geometry algorithm
# ──────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    # Edge detection
    canny_low=30,
    canny_high=100,
    blur_ksize=5,
    # Polynomial fit near contact points
    fit_frac=0.15,        # fit radius as fraction of drop width
    poly_degree=2,        # 2 = parabola (robust default)
    tangent_delta=1.0,    # finite-difference step for tangent
    # Validation
    angle_range_low=1,
    angle_range_high=179,
)


# ──────────────────────────────────────────────
#  Contact-angle image processing (mirror-geometry algorithm)
# ──────────────────────────────────────────────

def _find_drop_contour(gray, config):
    """Return the widest external contour (drop+reflection lens)."""
    k = config['blur_ksize']
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(gray, (k, k), 1.5)
    edges = cv2.Canny(blurred, config['canny_low'], config['canny_high'])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=lambda c: cv2.boundingRect(c)[2])


def _find_tips(contour):
    """Return (left_tip, right_tip) as the leftmost/rightmost contour points."""
    pts = contour[:, 0, :]
    left_tip = pts[np.argmin(pts[:, 0])]
    right_tip = pts[np.argmax(pts[:, 0])]
    return left_tip, right_tip


def _compute_baseline(left_tip, right_tip):
    """Baseline through the two tips. Returns (slope, intercept, tilt_deg)."""
    dx = right_tip[0] - left_tip[0]
    dy = right_tip[1] - left_tip[1]
    if dx == 0:
        return 0.0, float(left_tip[1]), 0.0
    slope = dy / dx
    intercept = left_tip[1] - slope * left_tip[0]
    return slope, intercept, float(np.degrees(np.arctan(slope)))


def _extract_upper_contour(contour, slope, intercept, tolerance=1):
    """Points above the baseline, sorted by x."""
    pts = contour[:, 0, :]
    baseline_y = slope * pts[:, 0] + intercept
    upper = pts[pts[:, 1] < baseline_y - tolerance]
    if len(upper) < 5:
        return pts[pts[:, 0].argsort()]
    return upper[upper[:, 0].argsort()]


def _fit_local_polynomial(upper_pts, tip_x, radius, degree, side):
    """Fit a polynomial y(x) locally near a contact tip."""
    if side == 'left':
        mask = (upper_pts[:, 0] >= tip_x) & (upper_pts[:, 0] <= tip_x + radius)
    else:
        mask = (upper_pts[:, 0] <= tip_x) & (upper_pts[:, 0] >= tip_x - radius)
    local = upper_pts[mask]
    if len(local) < degree + 1:
        return None
    try:
        return np.polyfit(local[:, 0].astype(float), local[:, 1].astype(float), degree)
    except np.linalg.LinAlgError:
        return None


def _compute_tangent_angle(poly, x_contact, baseline_slope, delta=1.0):
    """Contact angle between polynomial tangent and baseline direction (degrees)."""
    y_minus = np.polyval(poly, x_contact - delta)
    y_plus = np.polyval(poly, x_contact + delta)
    tangent = np.array([2 * delta, y_plus - y_minus])
    baseline_dir = np.array([1.0, baseline_slope])
    cos_angle = np.dot(tangent, baseline_dir) / (
        np.linalg.norm(tangent) * np.linalg.norm(baseline_dir) + 1e-10
    )
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def measure_contact_angles(gray, config=None, return_debug=False):
    """Full pipeline: detect drop contour, find tips, build baseline, fit tangents."""
    if config is None:
        config = DEFAULT_CONFIG

   # resize image if it's too large for faster processing (optional)
    max_dim = 500 # TODO: this is important otherwise the slope fit will fail for no particular reason, the contouring is flipping for some reason
    if max(gray.shape) > max_dim:
        scale = max_dim / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    result = {
        'left_angle': None, 'right_angle': None,
        'baseline_tilt_deg': None, 'success': False,
        'left_contact': None, 'right_contact': None,
    }
    debug = {}

    contour = _find_drop_contour(gray, config)
    if contour is None:
        return (result, debug) if return_debug else result
    debug['contour'] = contour

    left_tip, right_tip = _find_tips(contour)
    result['left_contact'] = (int(left_tip[0]), int(left_tip[1]))
    result['right_contact'] = (int(right_tip[0]), int(right_tip[1]))

    slope, intercept, tilt = _compute_baseline(left_tip, right_tip)
    result['baseline_tilt_deg'] = tilt
    debug['baseline_slope'] = slope
    debug['baseline_intercept'] = intercept

    upper = _extract_upper_contour(contour, slope, intercept)
    debug['upper_contour'] = upper
    if len(upper) < 10:
        return (result, debug) if return_debug else result

    drop_width = right_tip[0] - left_tip[0]
    fit_radius = max(10, int(drop_width * config['fit_frac']))
    debug['fit_radius'] = fit_radius

    poly_left = _fit_local_polynomial(upper, left_tip[0], fit_radius, config['poly_degree'], 'left')
    poly_right = _fit_local_polynomial(upper, right_tip[0], fit_radius, config['poly_degree'], 'right')
    debug['poly_left'] = poly_left
    debug['poly_right'] = poly_right

    if poly_left is None or poly_right is None:
        return (result, debug) if return_debug else result

    delta = config['tangent_delta']
    left_angle = _compute_tangent_angle(poly_left, left_tip[0], slope, delta)
    right_angle = _compute_tangent_angle(poly_right, right_tip[0], slope, delta)

    lo, hi = config['angle_range_low'], config['angle_range_high']
    if lo <= left_angle <= hi:
        result['left_angle'] = left_angle
    if lo <= right_angle <= hi:
        result['right_angle'] = right_angle

    result['success'] = (result['left_angle'] is not None and result['right_angle'] is not None)
    return (result, debug) if return_debug else result


def draw_result(frame, result, debug, config=None, crop_offset=None):
    """Overlay measurement annotations.
    crop_offset = (ox, oy) shifts contour/curve points from cropped into full-frame coords.
    """
    if config is None:
        config = DEFAULT_CONFIG

    vis = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]

    ox, oy = crop_offset if crop_offset else (0, 0)

    if 'contour' in debug:
        shifted = debug['contour'].copy()
        shifted[:, 0, 0] += ox
        shifted[:, 0, 1] += oy
        cv2.drawContours(vis, [shifted], -1, (0, 0, 180), 1)

    if not result.get('success', False):
        cv2.putText(vis, "Detection failed", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return vis

    slope = debug['baseline_slope']
    intercept = debug['baseline_intercept']
    left_pt = result['left_contact']   # already in full-frame coords
    right_pt = result['right_contact']

    # Baseline across full image
    y_left = int(slope * (0 - ox) + intercept + oy)
    y_right = int(slope * (w - ox) + intercept + oy)
    cv2.line(vis, (0, y_left), (w, y_right), (0, 255, 0), 2)

    cv2.circle(vis, left_pt, 6, (0, 255, 255), -1)
    cv2.circle(vis, right_pt, 6, (0, 255, 255), -1)

    fit_radius = debug.get('fit_radius', 60)
    tangent_len = max(80, int(fit_radius * 1.5))

    for side, poly, contact, angle in [
        ('left', debug.get('poly_left'), left_pt, result['left_angle']),
        ('right', debug.get('poly_right'), right_pt, result['right_angle']),
    ]:
        if poly is None or angle is None:
            continue

        cx = contact[0] - ox  # contact x in cropped space
        if side == 'left':
            xs = np.linspace(cx, cx + fit_radius, 200)
        else:
            xs = np.linspace(cx - fit_radius, cx, 200)
        ys = np.polyval(poly, xs)
        curve = np.column_stack((xs + ox, ys + oy)).astype(np.int32)
        for i in range(len(curve) - 1):
            cv2.line(vis, tuple(curve[i]), tuple(curve[i + 1]), (0, 255, 255), 2)

        delta = config['tangent_delta']
        dy = np.polyval(poly, cx + delta) - np.polyval(poly, cx - delta)
        tvec = np.array([2 * delta, dy])
        tvec = tvec / (np.linalg.norm(tvec) + 1e-10)
        pt = np.array([float(contact[0]), float(contact[1])])
        cv2.line(vis,
                 tuple((pt - tvec * tangent_len).astype(int)),
                 tuple((pt + tvec * tangent_len).astype(int)),
                 (255, 0, 255), 2)

        text = f"{angle:.1f} deg"
        if side == 'left':
            tx, ty = contact[0] + 12, contact[1] - 15
        else:
            tx, ty = contact[0] - 120, contact[1] - 15
        cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    tilt = result.get('baseline_tilt_deg', 0) or 0
    cv2.putText(vis, f"Tilt: {tilt:+.2f} deg", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    left_a = result['left_angle'] or 0
    right_a = result['right_angle'] or 0
    cv2.putText(vis, f"Avg: {(left_a + right_a) / 2:.1f} deg", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def compute_manual_angle(baseline_pt1, baseline_pt2, tangent_pt):
    """Compute contact angle from 3 manually selected points.

    baseline_pt1/2 define the substrate line; tangent_pt is on the drop edge.
    The contact point is the closest baseline endpoint to tangent_pt.
    """
    b1 = np.array(baseline_pt1, dtype=float)
    b2 = np.array(baseline_pt2, dtype=float)
    tp = np.array(tangent_pt, dtype=float)
    contact = b1 if np.linalg.norm(tp - b1) < np.linalg.norm(tp - b2) else b2
    baseline_vec = b2 - b1
    tangent_vec = tp - contact
    cos_angle = np.dot(baseline_vec, tangent_vec) / (
        np.linalg.norm(baseline_vec) * np.linalg.norm(tangent_vec) + 1e-10
    )
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


# ──────────────────────────────────────────────
#  Controller
# ──────────────────────────────────────────────

class GoniometerController(LiveUpdatedController):
    """
    Goniometer – contact angle measurement controller (mirror-geometry algorithm).

    Provides snap, crop ROI management, automated measurement,
    manual 3-point measurement, live focus metric, measurement history,
    and annotated result image download.
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

        # Processing config
        self._config = dict(DEFAULT_CONFIG)

        # Snapped image (numpy)
        self._snapped_image = None   # BGR
        self._snapped_gray = None    # grayscale

        # Crop ROI in full-frame camera pixel space: dict(x1, y1, x2, y2) or None
        self._crop_roi: Optional[Dict[str, int]] = None

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

    def _normalise_frame(self, frame):
        """Ensure uint8 BGR with percentile-based contrast stretch.

        Normalises by the 1st–99th percentile so that a single hot pixel or
        a near-uniformly-bright 16-bit frame does not collapse to white.
        """
        if frame.dtype != np.uint8:
            f = frame.astype(np.float32)
            lo = float(np.percentile(f, 1))
            hi = float(np.percentile(f, 99))
            if hi > lo:
                f = np.clip((f - lo) / (hi - lo) * 255.0, 0.0, 255.0)
            else:
                # Fallback: simple bit-shift for 16-bit, zero for others
                f = (frame >> 8).astype(np.float32) if frame.dtype == np.uint16 else np.zeros_like(frame, dtype=np.float32)
            frame = f.astype(np.uint8)
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def _image_to_base64(self, image):
        """Encode image to base64 data-URI (JPEG)."""
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"

    def _apply_crop(self, frame_bgr, frame_gray):
        """Apply _crop_roi to both BGR and gray frames. Returns (bgr, gray, offset)."""
        if self._crop_roi is None:
            return frame_bgr, frame_gray, (0, 0)
        h, w = frame_bgr.shape[:2]
        x1 = max(0, self._crop_roi['x1'])
        y1 = max(0, self._crop_roi['y1'])
        x2 = min(w, self._crop_roi['x2'])
        y2 = min(h, self._crop_roi['y2'])
        if x2 <= x1 or y2 <= y1:
            return frame_bgr, frame_gray, (0, 0)
        return frame_bgr[y1:y2, x1:x2], frame_gray[y1:y2, x1:x2], (x1, y1)

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

    # ───────── API: crop ROI ─────────

    @APIExport(runOnUIThread=True, requestType="POST")
    def set_crop_goniometer(self,
                             x1: int = 0, y1: int = 0,
                             x2: int = 0, y2: int = 0) -> Dict[str, Any]:
        """Set a crop ROI in full sensor-pixel coordinates.

        The snap and processing pipeline will apply this crop.
        x1, y1, x2, y2 are pixel indices in the full camera frame.
        """
        if x2 <= x1 or y2 <= y1:
            return {"success": False, "error": "Invalid crop: x2>x1 and y2>y1 required"}
        self._crop_roi = dict(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
        self._logger.info("Crop ROI set: %s", self._crop_roi)
        return {"success": True, "crop_roi": self._crop_roi}

    @APIExport(runOnUIThread=True, requestType="POST")
    def reset_crop_goniometer(self) -> Dict[str, Any]:
        """Clear the crop ROI so the full frame is used."""
        self._crop_roi = None
        return {"success": True, "crop_roi": None}

    @APIExport(runOnUIThread=True)
    def get_crop_goniometer(self) -> Dict[str, Any]:
        """Return the current crop ROI (or null if none)."""
        return {"crop_roi": self._crop_roi}

    # ───────── API: snap ─────────

    @APIExport(runOnUIThread=True)
    def snap_goniometer(self) -> Dict[str, Any]:
        """Capture the current frame, apply crop if set, and return as base64 JPEG."""
        try:
            raw = self._get_frame()
            bgr = self._normalise_frame(raw)
            if len(bgr.shape) == 3:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = bgr
                bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Apply crop (stores cropped versions; offset used during drawing)
            crop_bgr, crop_gray, offset = self._apply_crop(bgr, gray)

            self._snapped_image = crop_bgr
            self._snapped_gray = crop_gray
            self._result_image = None

            # Encode the cropped image for display
            display_img = self._image_to_base64(self._snapped_image)

            # If a crop is active, also encode a full-frame view with the crop rectangle drawn
            preview_img = display_img
            if self._crop_roi is not None:
                full_vis = bgr.copy()
                r = self._crop_roi
                cv2.rectangle(full_vis, (r['x1'], r['y1']), (r['x2'], r['y2']), (80, 180, 255), 2)
                preview_img = self._image_to_base64(full_vis)

            return {
                "success": True,
                "image": display_img,
                "preview_image": preview_img,
                "shape": list(self._snapped_image.shape),
                "crop_roi": self._crop_roi,
                "timestamp": time.time(),
            }
        except Exception as e:
            self._logger.error("snap_goniometer failed: %s", traceback.format_exc())
            return {"success": False, "error": str(e)}

    # ───────── API: automated measurement ─────────

    @APIExport(runOnUIThread=True)
    def measure_auto_goniometer(self) -> Dict[str, Any]:
        """Run automated contact-angle measurement on the snapped (cropped) image."""
        if self._snapped_gray is None:
            return {"success": False, "error": "No image snapped yet"}
        try:
            result, debug = measure_contact_angles(
                self._snapped_gray, self._config, return_debug=True
            )

            # Shift contact points into full-frame coords if crop was active
            crop_offset = (self._crop_roi['x1'], self._crop_roi['y1']) if self._crop_roi else None
            if crop_offset and result.get('left_contact') is not None:
                result['left_contact'] = (
                    result['left_contact'][0] + crop_offset[0],
                    result['left_contact'][1] + crop_offset[1],
                )
                result['right_contact'] = (
                    result['right_contact'][0] + crop_offset[0],
                    result['right_contact'][1] + crop_offset[1],
                )

            # Draw on snapped image (already cropped so offset=None for contour coords,
            # but we need the full frame for display if no crop)
            self._result_image = draw_result(
                self._snapped_image, result, debug, self._config, crop_offset=None
            )

            return {
                "success": result.get("success", False),
                "left_angle": result.get("left_angle"),
                "right_angle": result.get("right_angle"),
                "baseline_tilt_deg": result.get("baseline_tilt_deg"),
                "left_contact": result.get("left_contact"),
                "right_contact": result.get("right_contact"),
                "annotated_image": self._image_to_base64(self._result_image),
                "timestamp": time.time(),
            }
        except Exception as e:
            self._logger.error("measure_auto_goniometer failed: %s", traceback.format_exc())
            return {"success": False, "error": str(e)}

    # ───────── API: manual measurement ─────────

    @APIExport(runOnUIThread=True, requestType="POST")
    def measure_manual_goniometer(self,
                                  baseline_x1: float = 0, baseline_y1: float = 0,
                                  baseline_x2: float = 0, baseline_y2: float = 0,
                                  tangent_x: float = 0, tangent_y: float = 0) -> Dict[str, Any]:
        """Compute contact angle from 3 user-selected points."""
        try:
            angle = compute_manual_angle(
                (baseline_x1, baseline_y1),
                (baseline_x2, baseline_y2),
                (tangent_x, tangent_y),
            )
            annotated_b64 = None
            if self._snapped_image is not None:
                vis = self._snapped_image.copy()
                b1 = (int(baseline_x1), int(baseline_y1))
                b2 = (int(baseline_x2), int(baseline_y2))
                tp = (int(tangent_x), int(tangent_y))
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

    # ───────── API: focus metric ─────────

    @APIExport(runOnUIThread=True)
    def get_focus_metric_goniometer(self) -> Dict[str, Any]:
        """Return a live focus sharpness metric (Laplacian variance) of the current frame.

        Higher is sharper. Useful for manual focus alignment before measurement.
        The crop ROI is applied if set, so the metric reflects only the droplet region.
        """
        try:
            raw = self._get_frame()
            bgr = self._normalise_frame(raw)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr

            # Apply crop for more localised metric
            _, gray_crop, _ = self._apply_crop(bgr, gray)

            lap = cv2.Laplacian(gray_crop, cv2.CV_64F)
            metric = float(np.var(lap))
            return {
                "success": True,
                "focus_metric": metric,
                "timestamp": time.time(),
            }
        except Exception as e:
            return {"success": False, "focus_metric": 0.0, "error": str(e)}

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
            return {"success": True, "image": self._image_to_base64(self._result_image)}

    @APIExport(runOnUIThread=True)
    def download_snapped_image_goniometer(self, format: str = "png"):
        """Return the snapped (raw/cropped) image as a downloadable response."""
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
