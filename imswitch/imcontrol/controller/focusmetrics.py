"""
Focus metrics module for ImSwitch focus lock functionality.

Extracted from FocusLockController for better modularity and testability.
Provides various focus measurement algorithms including astigmatism-based metrics.
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, center_of_mass
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample



logger = logging.getLogger(__name__)


@dataclass
class FocusConfig:
    """Configuration for focus metric computation."""
    gaussian_sigma: float = 11.0
    background_threshold: int = 20
    crop_radius: int = 300
    enable_gaussian_blur: bool = True
    min_signal_threshold: float = 10.0  # Minimum signal for valid measurement
    max_focus_value: float = 1e6  # Maximum valid focus value
    # peak-specific
    peak_distance: int = 200                 # minimal separation (px) between the two peaks
    peak_height: Optional[float] = 20      # required absolute height in projection units
    max_peaks: int = 2                       # keep at most two strongest peaks

class FocusMetricBase:
    """Base class for focus metrics."""

    def __init__(self, config: Optional[FocusConfig] = None):
        self.config = config or FocusConfig()
        logger.debug(f"Focus metric initialized with config: {self.config}")
        self._rotation_angle = 0.0  # Placeholder for potential future use

    def reset_history(self):
        """Reset any internal history (if applicable)."""
        pass

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for focus computation.

        Args:
            frame: Input image frame

        Returns:
            Preprocessed frame
        """
        # Convert to grayscale if needed
        if frame.ndim == 3:
            im = np.mean(frame, axis=-1).astype(np.uint16)
        else:
            im = frame.astype(np.uint16)

        im = im.astype(float)
        # rotate image if needed
        if self._rotation_angle != 0.0:
            from scipy.ndimage import rotate
            im = rotate(im, self._rotation_angle, reshape=False)

        # Crop around brightest region if crop_radius > 0
        if 0 and self.config.crop_radius > 0:
            im_gauss = gaussian_filter(im, sigma=111)
            max_coord = np.unravel_index(np.argmax(im_gauss), im_gauss.shape)
            h, w = im.shape
            y_min = max(0, max_coord[0] - self.config.crop_radius)
            y_max = min(h, max_coord[0] + self.config.crop_radius)
            x_min = max(0, max_coord[1] - self.config.crop_radius)
            x_max = min(w, max_coord[1] + self.config.crop_radius)
            im = im[y_min:y_max, x_min:x_max]

        # Apply Gaussian blur if enabled
        if self.config.enable_gaussian_blur:
            im = gaussian_filter(im, sigma=self.config.gaussian_sigma)

        # Background subtraction and thresholding
        # TODO: normalize intensity to compensate laser fluctuations?
        if 0:
            im = im / np.max((np.max(im), 0.1)) * 255
            im = im - np.min(im) / 2.0
            im[im < self.config.background_threshold] = 0

        return im

    def compute(self, frame: np.ndarray):
        """
        Compute focus metric.

        Args:
            frame: Input image frame

        Returns:
            Dictionary with focus metric results
        """
        raise NotImplementedError("Subclasses must implement compute method")

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


class AstigmatismFocusMetric(FocusMetricBase):
    """
    Astigmatism-based focus metric.

    Computes focus by fitting Gaussian profiles to X and Y projections
    and calculating the ratio of their widths.
    """

    @staticmethod
    def gaussian_1d(xdata: np.ndarray, i0: float, x0: float, sigma: float, amp: float) -> np.ndarray:
        """1D Gaussian function for curve fitting."""
        x = xdata
        x0 = float(x0)
        return i0 + amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def double_gaussian_1d(xdata: np.ndarray, i0: float, x0: float, sigma: float, amp: float, dist: float) -> np.ndarray:
        """Double 1D Gaussian function for curve fitting."""
        x = xdata
        x0 = float(x0)
        return (
            i0
            + amp * np.exp(-((x - (x0 - dist / 2)) ** 2) / (2 * sigma ** 2))
            + amp * np.exp(-((x - (x0 + dist / 2)) ** 2) / (2 * sigma ** 2))
        )

    def compute_projections(self, im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute X and Y projections of the image."""
        projX = np.mean(im, axis=0)
        projY = np.mean(im, axis=1)
        return projX, projY

    def fit_projections(self, projX: np.ndarray, projY: np.ndarray,
                       isDoubleGaussX: bool = False,
                       isDoubleGaussY: bool = False) -> Tuple[float, float]:
        """
        Fit Gaussian profiles to projections.

        Args:
            projX: X projection
            projY: Y projection
            isDoubleGaussX: Whether to use double Gaussian for X projection

        Returns:
            Tuple of (sigma_x, sigma_y)
        """
        h1, w1 = len(projY), len(projX)
        x = np.arange(w1)
        y = np.arange(h1)

        # Initial parameter estimates
        i0_x = float(np.mean(projX))
        amp_x = float(np.max(projX) - i0_x)
        sigma_x_init = float(np.std(projX))
        i0_y = float(np.mean(projY))
        amp_y = float(np.max(projY) - i0_y)
        sigma_y_init = float(np.std(projY))

        # Set up initial guesses
        if isDoubleGaussX:
            init_guess_x = [i0_x, w1 / 2, sigma_x_init, amp_x, 100.0]
        else:
            init_guess_x = [i0_x, w1 / 2, sigma_x_init, amp_x]
        if isDoubleGaussY:
            init_guess_y = [i0_y, h1 / 2, sigma_y_init, amp_y, 100.0]
        else:
            init_guess_y = [i0_y, h1 / 2, sigma_y_init, amp_y]

        try:
            # Fit X projection
            if isDoubleGaussX:
                popt_x, _ = curve_fit(self.double_gaussian_1d, x, projX,
                                    p0=init_guess_x, maxfev=50000)
                sigma_x = abs(float(popt_x[2]))
            else:
                popt_x, _ = curve_fit(self.gaussian_1d, x, projX,
                                    p0=init_guess_x, maxfev=50000)
                sigma_x = abs(float(popt_x[2]))

            # Fit Y projection
            if isDoubleGaussY:
                popt_y, _ = curve_fit(self.double_gaussian_1d, y, projY,
                                p0=init_guess_y, maxfev=50000)
                sigma_y = abs(float(popt_y[2]))
            else:
                popt_y, _ = curve_fit(self.gaussian_1d, y, projY,
                                    p0=init_guess_y, maxfev=50000)
                sigma_y = abs(float(popt_y[2]))

        except Exception as e:
            logger.warning(f"Gaussian fitting failed, using std: {e}")
            sigma_x = float(np.std(projX))
            sigma_y = float(np.std(projY))

        return sigma_x, sigma_y

    def compute(self, frame: np.ndarray):
        """
        Compute astigmatism-based focus metric.

        Args:
            frame: Input image frame

        Returns:
            Dictionary with focus metric results
        """
        timestamp = time.time()

        try:
            # Preprocess frame
            processed_image = self.preprocess_frame(np.array(frame))

            # Check for minimum signal
            if np.max(processed_image) < self.config.min_signal_threshold:
                logger.warning("Signal below threshold")
                return {"t": timestamp, "focus": self.config.max_focus_value, "error": "low_signal"}, None

            # Compute projections and fit Gaussians
            projX, projY = self.compute_projections(processed_image)
            sigma_x, sigma_y = self.fit_projections(projX, projY)

            # Calculate focus value as ratio
            if 1:
                if sigma_y == 0 or sigma_y < 1e-6:
                    focus_value = self.config.max_focus_value
                else:
                    focus_value = float(sigma_x / sigma_y)
            else:
                focus_value = float(sigma_x)  # Use product for better sensitivity

            # Clamp to reasonable range
            focus_value = min(focus_value, self.config.max_focus_value)

            logger.debug(f"Astigmatism focus: sigma_x={sigma_x:.3f}, sigma_y={sigma_y:.3f}, focus={focus_value:.4f}")

            return {
                "t": timestamp,
                "focus": focus_value,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "signal_max": float(np.max(frame)),
                "signal_mean": float(np.mean(frame)),
            }, processed_image

        except Exception as e:
            logger.error(f"Focus computation failed: {e}")
            return {"t": timestamp, "focus": self.config.max_focus_value, "error": str(e)}, None



class PeakMetric1D(FocusMetricBase):
    """
    X-only peak finder.
    Returns integer peak positions along X and their distance. No PID, no Y-fit.
    """

    @staticmethod
    def _projection_x(im: np.ndarray) -> np.ndarray:
        return np.mean(im, axis=0)

    @staticmethod
    def _smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter1d(x, sigma) if sigma and sigma > 0 else x

    def compute(self, frame: np.ndarray) -> Dict[str, Any]:
        ts = time.time()

        # preprocess (uses FocusMetricBase.preprocess_frame)
        im = self.preprocess_frame(np.asarray(frame))
        if im.size == 0 or np.max(im) < self.config.min_signal_threshold:
            return {"t": ts, "x_peaks": np.array([], dtype=int), "x_peak_distance": None, "error": "low_signal"}

        # X projection and optional 1D smoothing
        projx = self._projection_x(im)
        projx_s = self._smooth_1d(projx, self.config.gaussian_sigma if self.config.enable_gaussian_blur else 0.0)

        # peak detection (keep the strongest two if more found)
        peaks, props = find_peaks(
            projx_s,
            distance=self.config.peak_distance,
            height=self.config.peak_height
        )
        if not type(peaks) == np.ndarray and len(peaks) <2:
            peaks = (1,1)
        focus_value = np.mean(peaks)

        # outputs
        result: Dict[str, Any] = {
            "t": ts,
            "x_peaks": focus_value,
            "proj_x": projx_s.astype(float),
            "signal_max": float(np.max(im)),
            "signal_mean": float(np.mean(im)),
            "focus": focus_value
        }

        return result




class PeakMetric(FocusMetricBase):
    """
    Robust X-only peak finder for 2 laser dots.
    - Uses max-projection by default (better for bright dots)
    - Robust baseline removal + Gaussian smoothing
    - Adaptive peak thresholds via MAD (noise units)
    - Always returns left-most strong peak; right peak optional
    """

    def __init__(self, config: Optional[FocusConfig] = None):
        super().__init__(config)
        self.peak_distances: List[float] = []
        self.max_history = 5
        self.outlier_threshold_mad = 6.0  # z-score in MAD units for distance history

    # -----------------------------
    # 1D helpers
    # -----------------------------
    @staticmethod
    def _projection_x(im: np.ndarray, mode: str = "max") -> np.ndarray:
        if mode == "mean":
            return np.mean(im, axis=0)
        return np.max(im, axis=0)

    @staticmethod
    def _smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter1d(x, sigma) if sigma and sigma > 0 else x

    @staticmethod
    def _robust_baseline(x: np.ndarray, p: float = 20.0) -> np.ndarray:
        base = np.percentile(x, p)
        return np.clip(x - base, 0, None)

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        med = np.median(x)
        return float(np.median(np.abs(x - med)) + 1e-9)

    @staticmethod
    def _parabolic_subpixel(y: np.ndarray, i: int) -> float:
        """3-point quadratic interpolation for subpixel peak position."""
        if i <= 0 or i >= len(y) - 1:
            return float(i)
        y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) < 1e-9:
            return float(i)
        delta = 0.5 * (y0 - y2) / denom
        return float(i) + delta

    # -----------------------------
    # Distance history / outlier
    # -----------------------------
    def _is_outlier(self, distance: float) -> bool:
        if len(self.peak_distances) < 3:
            return False
        hist = np.asarray(self.peak_distances, dtype=float)
        med = float(np.median(hist))
        mad = self._mad(hist)
        z = abs(distance - med) / mad
        return z > self.outlier_threshold_mad

    def _update_history(self, distance: float) -> None:
        self.peak_distances.append(float(distance))
        if len(self.peak_distances) > self.max_history:
            self.peak_distances.pop(0)

    def _get_average_distance(self) -> Optional[float]:
        return float(np.mean(self.peak_distances)) if self.peak_distances else None

    def reset_history(self):
        self.peak_distances = []

    # -----------------------------
    # Main compute
    # -----------------------------
    def compute(self, frame: np.ndarray, n_supersample: int = 1) -> Dict[str, Any]:
        """
        n_supersample kept for API compatibility; not needed because we do subpixel fit.
        """
        ts = time.time()
        t0 = time.time()

        im = np.asarray(frame)

        # 1) projection
        proj_mode = getattr(self.config, "projection_mode", "max")  # "max" default
        projx = self._projection_x(im, mode=proj_mode)

        # 2) baseline remove
        if hasattr(self.config, "background_threshold"):
            bt = float(getattr(self.config, "background_threshold", 0.0))
            projx = np.clip(projx - bt, 0, None)
        else:
            p = float(getattr(self.config, "baseline_percentile", 20.0))
            projx = self._robust_baseline(projx, p=p)

        # 3) smoothing
        enable_blur = bool(getattr(self.config, "enable_gaussian_blur", True))
        sigma = float(getattr(self.config, "gaussian_sigma", 2.0))  # <- explicit sigma default
        projx_s = self._smooth_1d(projx, sigma) if enable_blur else projx

        # 4) adaptive scaling to noise units
        med = float(np.median(projx_s))
        mad = self._mad(projx_s)
        zsig = (projx_s - med) / mad
        zsig = np.clip(zsig, 0, None)

        # 5) peak detection in noise units
        min_dist = int(getattr(self.config, "peak_distance", 200))
        prom_mad = float(getattr(self.config, "auto_peak_prom_mad", 4.0))
        height_mad = float(getattr(self.config, "auto_peak_height_mad", 3.0))

        peaks, props = find_peaks(
            zsig,
            distance=min_dist,
            prominence=prom_mad,
            height=height_mad
        )

        # optional ROI for left dot (xmin, xmax)
        left_roi: Optional[Tuple[int, int]] = getattr(self.config, "left_peak_roi", None)
        if left_roi is not None and len(peaks):
            xmin, xmax = left_roi
            mask = (peaks >= xmin) & (peaks <= xmax)
            peaks = peaks[mask]
            if "prominences" in props:
                props["prominences"] = np.asarray(props["prominences"])[mask]
            if "peak_heights" in props:
                props["peak_heights"] = np.asarray(props["peak_heights"])[mask]

        left_peak = None
        right_peak = None
        x_peak_distance = None

        if len(peaks) >= 1:
            prominences = props.get("prominences", zsig[peaks])
            order = np.argsort(prominences)
            best = peaks[order][-2:] if len(peaks) >= 2 else peaks[order][-1:]
            best = np.sort(best)

            # optional max distance guard
            max_sep = getattr(self.config, "peak_max_distance", None)
            if max_sep is not None and len(best) == 2:
                if (best[1] - best[0]) > int(max_sep):
                    best = best[-1:]  # keep only strongest

            # leftmost always returned
            left_i = int(best[0])
            left_peak = self._parabolic_subpixel(projx_s, left_i)

            if len(best) == 2:
                right_i = int(best[1])
                right_peak = self._parabolic_subpixel(projx_s, right_i)
                x_peak_distance = float(right_peak - left_peak)

                if not self._is_outlier(x_peak_distance):
                    self._update_history(x_peak_distance)

        focus_value = left_peak  # <- what you care about

        if 1 or bool(getattr(self.config, "debug_plot", False)):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(zsig, label="zsig(MAD units)")
            if len(peaks):
                plt.plot(peaks, zsig[peaks], "x")
            plt.title(f"peaks={len(peaks)} left={left_peak} right={right_peak}")
            plt.legend()
            plt.savefig("test.png")

        return {
            "t": ts,
            "focus": focus_value,
            "left_peak_x": left_peak,
            "right_peak_x": right_peak,
            "x_peak_distance": x_peak_distance,
            "proj_x": projx_s.astype(float),
            "avg_peak_distance": self._get_average_distance(),
            "peak_history_length": len(self.peak_distances),
            "compute_ms": (time.time() - t0) * 1000.0,
        }


'''

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class PeakMetric(FocusMetricBase):
    """
    Left-dot-only peak metric (prevents jumps to the right peak).

    Strategy:
    1) Build a 1D x-projection (max or mean).
    2) Robust baseline removal + optional Gaussian smoothing.
    3) Convert to noise units (MAD z-score).
    4) Detect peaks ONLY inside a left-ROI:
         - Prefer config.left_peak_roi if given
         - Else track around last_left_x with a radius (config.left_track_radius)
         - Else (first frame) use full width, pick leftmost of top-2 peaks, then lock-on.
    5) If no peak is detected inside ROI, fallback to argmax inside ROI (still left-only).
    """

    def __init__(self, config: Optional[FocusConfig] = None):
        super().__init__(config)
        self._last_left_x: Optional[float] = None

        # kept for compatibility (can still be useful for monitoring)
        self.peak_distances: List[float] = []
        self.max_history = 5
        self.outlier_threshold_mad = 6.0  # MAD-z for distance history

    # -----------------------------
    # 1D helpers
    # -----------------------------
    @staticmethod
    def _projection_x(im: np.ndarray, mode: str = "max") -> np.ndarray:
        if mode == "mean":
            return np.mean(im, axis=0)
        return np.max(im, axis=0)

    @staticmethod
    def _smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter1d(x, sigma) if sigma and sigma > 0 else x

    @staticmethod
    def _robust_baseline(x: np.ndarray, p: float = 20.0) -> np.ndarray:
        base = np.percentile(x, p)
        return np.clip(x - base, 0, None)

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        med = np.median(x)
        return float(np.median(np.abs(x - med)) + 1e-9)

    @staticmethod
    def _parabolic_subpixel(y: np.ndarray, i: int) -> float:
        """3-point quadratic interpolation for subpixel peak position."""
        if i <= 0 or i >= len(y) - 1:
            return float(i)
        y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) < 1e-12:
            return float(i)
        delta = 0.5 * (y0 - y2) / denom
        return float(i) + delta

    # -----------------------------
    # Distance history / outlier
    # -----------------------------
    def _is_outlier(self, distance: float) -> bool:
        if len(self.peak_distances) < 3:
            return False
        hist = np.asarray(self.peak_distances, dtype=float)
        med = float(np.median(hist))
        mad = self._mad(hist)
        z = abs(distance - med) / mad
        return z > self.outlier_threshold_mad

    def _update_history(self, distance: float) -> None:
        self.peak_distances.append(float(distance))
        if len(self.peak_distances) > self.max_history:
            self.peak_distances.pop(0)

    def _get_average_distance(self) -> Optional[float]:
        return float(np.mean(self.peak_distances)) if self.peak_distances else None

    def reset_history(self):
        self.peak_distances = []
        self._last_left_x = None

    # -----------------------------
    # ROI logic
    # -----------------------------
    def _compute_left_roi(self, width: int) -> Tuple[int, int, str]:
        """
        Returns (xmin, xmax, roi_source)
        roi_source in {"config", "track", "full"}.
        """
        left_roi: Optional[Tuple[int, int]] = getattr(self.config, "left_peak_roi", None)
        if left_roi is not None:
            xmin, xmax = int(left_roi[0]), int(left_roi[1])
            xmin = max(0, min(width - 1, xmin))
            xmax = max(0, min(width, xmax))
            if xmax <= xmin:
                xmin, xmax = 0, width
                return xmin, xmax, "full"
            return xmin, xmax, "config"

        if self._last_left_x is not None:
            r = int(getattr(self.config, "left_track_radius", 180))
            cx = int(round(self._last_left_x))
            xmin = max(0, cx - r)
            xmax = min(width, cx + r + 1)
            if xmax <= xmin:
                xmin, xmax = 0, width
                return xmin, xmax, "full"
            return xmin, xmax, "track"

        return 0, width, "full"

    # -----------------------------
    # Main compute
    # -----------------------------
    def compute(self, frame: np.ndarray, n_supersample: int = 1) -> Dict[str, Any]:
        ts = time.time()
        t0 = time.time()

        im = np.asarray(frame)
        if im.ndim != 2:
            raise ValueError(f"PeakMetric expects a 2D grayscale frame, got shape={im.shape}")

        # 1) projection
        proj_mode = getattr(self.config, "projection_mode", "max")  # "max" default
        projx = self._projection_x(im, mode=proj_mode)

        # 2) baseline remove
        if hasattr(self.config, "background_threshold"):
            bt = float(getattr(self.config, "background_threshold", 0.0))
            projx = np.clip(projx - bt, 0, None)
        else:
            p = float(getattr(self.config, "baseline_percentile", 20.0))
            projx = self._robust_baseline(projx, p=p)

        # 3) smoothing
        enable_blur = bool(getattr(self.config, "enable_gaussian_blur", True))
        sigma = float(getattr(self.config, "gaussian_sigma", 2.0))
        projx_s = self._smooth_1d(projx, sigma) if enable_blur else projx

        # 4) convert to MAD z-score (noise units)
        med = float(np.median(projx_s))
        mad = self._mad(projx_s)
        zsig = (projx_s - med) / mad
        zsig = np.clip(zsig, 0, None)

        # 5) left-only ROI (config or tracking)
        width = int(zsig.shape[0])
        xmin, xmax, roi_source = self._compute_left_roi(width)
        z_roi = zsig[xmin:xmax]

        min_dist = int(getattr(self.config, "peak_distance", 200))
        prom_mad = float(getattr(self.config, "auto_peak_prom_mad", 4.0))
        height_mad = float(getattr(self.config, "auto_peak_height_mad", 3.0))

        # tighten distance inside small ROIs so we can still find a single peak
        if (xmax - xmin) < 2 * min_dist:
            min_dist_eff = max(1, (xmax - xmin) // 3)
        else:
            min_dist_eff = min_dist

        peaks_roi, props_roi = find_peaks(
            z_roi,
            distance=min_dist_eff,
            prominence=prom_mad,
            height=height_mad,
        )
        peaks = (peaks_roi + xmin).astype(int)

        left_peak_x: Optional[float] = None
        left_peak_i: Optional[int] = None
        left_peak_height_mad: Optional[float] = None
        left_peak_prom_mad: Optional[float] = None
        used_fallback_argmax = False

        # helper: pick best peak inside ROI by prominence (then height)
        def _pick_best_peak(peaks_abs: np.ndarray, props: Dict[str, np.ndarray]) -> int:
            prom = props.get("prominences", None)
            h = props.get("peak_heights", None)

            if prom is None or len(prom) != len(peaks_abs):
                prom = zsig[peaks_abs]
            if h is None or len(h) != len(peaks_abs):
                h = zsig[peaks_abs]

            # sort by (prominence, height) descending
            order = np.lexsort((h, prom))  # ascending
            return int(peaks_abs[order][-1])

        if len(peaks) > 0:
            # If we are NOT locked yet (first frame, no ROI), try to initialize using two peaks
            if roi_source == "full" and self._last_left_x is None:
                # pick two most prominent peaks (if available), then choose the leftmost of those
                prom = props_roi.get("prominences", None)
                if prom is None or len(prom) != len(peaks_roi):
                    prom = z_roi[peaks_roi]
                order = np.argsort(prom)
                top = order[-2:] if len(order) >= 2 else order[-1:]
                top_abs = np.sort(peaks[top])
                left_peak_i = int(top_abs[0])
            else:
                # locked / ROI mode: choose best peak inside ROI
                left_peak_i = _pick_best_peak(peaks, props_roi)

            left_peak_x = self._parabolic_subpixel(projx_s, int(left_peak_i))
            left_peak_height_mad = float(zsig[int(left_peak_i)])
            # prominence for the chosen peak (map from ROI index if possible)
            try:
                idx_roi = int(left_peak_i - xmin)
                if "prominences" in props_roi and idx_roi in peaks_roi.tolist():
                    # find matching entry
                    j = int(np.where(peaks_roi == idx_roi)[0][0])
                    left_peak_prom_mad = float(np.asarray(props_roi["prominences"])[j])
            except Exception:
                left_peak_prom_mad = None

        # 6) fallback: if no peak found inside ROI, use argmax inside ROI (still left-only)
        if left_peak_x is None:
            fallback = bool(getattr(self.config, "left_fallback_argmax", True))
            if fallback and (xmax - xmin) >= 3:
                used_fallback_argmax = True
                i0 = int(np.argmax(zsig[xmin:xmax]) + xmin)
                # validate a minimum height in MAD units
                min_h = float(getattr(self.config, "left_min_height_mad", height_mad))
                if float(zsig[i0]) >= min_h:
                    left_peak_i = i0
                    left_peak_x = self._parabolic_subpixel(projx_s, i0)
                    left_peak_height_mad = float(zsig[i0])
                else:
                    left_peak_i = None
                    left_peak_x = None

        # 7) tracking update (only if confident)
        if left_peak_x is not None:
            min_h_track = float(getattr(self.config, "left_min_height_mad", height_mad))
            if left_peak_height_mad is None or left_peak_height_mad >= min_h_track:
                self._last_left_x = float(left_peak_x)

        # ---- optional: still compute right peak distance for monitoring (never used as focus) ----
        right_peak_x: Optional[float] = None
        x_peak_distance: Optional[float] = None
        if bool(getattr(self.config, "report_right_peak", False)):
            # detect on full signal for reporting only
            peaks_full, props_full = find_peaks(
                zsig,
                distance=min_dist,
                prominence=prom_mad,
                height=height_mad,
            )
            if len(peaks_full) >= 2:
                # choose the nearest peak to the right of left_peak_i
                if left_peak_x is not None:
                    li = int(round(left_peak_x))
                    rights = peaks_full[peaks_full > li]
                    if len(rights) > 0:
                        ri = int(rights[np.argmin(rights - li)])
                        right_peak_x = self._parabolic_subpixel(projx_s, ri)
                        x_peak_distance = float(right_peak_x - left_peak_x)
                        if not self._is_outlier(x_peak_distance):
                            self._update_history(x_peak_distance)

        focus_value = left_peak_x  # <- ONLY LEFT DOT

        return {
            "t": ts,
            "focus": focus_value,

            "left_peak_x": left_peak_x,
            "left_peak_i": left_peak_i,
            "left_peak_height_mad": left_peak_height_mad,
            "left_peak_prom_mad": left_peak_prom_mad,

            "right_peak_x": right_peak_x,            # optional (reporting only)
            "x_peak_distance": x_peak_distance,      # optional (reporting only)

            "roi_left": (xmin, xmax),
            "roi_source": roi_source,
            "used_fallback_argmax": used_fallback_argmax,

            "proj_x": projx_s.astype(float),
            "avg_peak_distance": self._get_average_distance(),
            "peak_history_length": len(self.peak_distances),
            "compute_ms": (time.time() - t0) * 1000.0,
        }



class CenterOfMassFocusMetric(FocusMetricBase):
    """
    Center of mass based focus metric.

    Finds the center of mass of the brightest spots and uses it as focus signal.
    """

    def find_peak_centers(self, im: np.ndarray, two_foci: bool = False) -> np.ndarray:
        """
        Find peak centers in the image.

        Args:
            im: Preprocessed image
            two_foci: Whether to detect two foci

        Returns:
            Center coordinates
        """
        if two_foci:
            # Find two brightest peaks
            allmaxcoords = peak_local_max(im, min_distance=60)
            size = allmaxcoords.shape[0]

            if size >= 2:
                maxvals = np.full(2, -np.inf)
                maxvalpos = np.zeros(2, dtype=int)

                for n in range(size):
                    val = im[allmaxcoords[n][0], allmaxcoords[n][1]]
                    if val > maxvals[0]:
                        if val > maxvals[1]:
                            maxvals[0] = maxvals[1]
                            maxvals[1] = val
                            maxvalpos[0] = maxvalpos[1]
                            maxvalpos[1] = n
                        else:
                            maxvals[0] = val
                            maxvalpos[0] = n

                # Use the lower peak (in Y)
                xcenter = allmaxcoords[maxvalpos[0]][0]
                ycenter = allmaxcoords[maxvalpos[0]][1]
                if allmaxcoords[maxvalpos[1]][1] < ycenter:
                    xcenter = allmaxcoords[maxvalpos[1]][0]
                    ycenter = allmaxcoords[maxvalpos[1]][1]

                return np.array([xcenter, ycenter])
            else:
                # Fall back to single peak
                centercoords = np.where(im == np.max(im))
                return np.array([centercoords[0][0], centercoords[1][0]])
        else:
            # Single peak detection
            centercoords = np.where(im == np.max(im))
            return np.array([centercoords[0][0], centercoords[1][0]])

    def compute_center_of_mass(self, im: np.ndarray, center_coords: np.ndarray) -> float:
        """
        Compute center of mass around detected peak.

        Args:
            im: Preprocessed image
            center_coords: Peak center coordinates

        Returns:
            Focus metric value
        """
        subsizey = 50
        subsizex = 50
        h, w = im.shape[:2]

        # Extract subregion around peak
        xlow = max(0, int(center_coords[0] - subsizex))
        xhigh = min(h, int(center_coords[0] + subsizex))
        ylow = max(0, int(center_coords[1] - subsizey))
        yhigh = min(w, int(center_coords[1] + subsizey))

        im_sub = im[xlow:xhigh, ylow:yhigh]

        # Compute center of mass
        mass_center = np.array(center_of_mass(im_sub))

        # Return Y component as focus metric
        return float(mass_center[1] + center_coords[1])

    def compute(self, frame: np.ndarray, two_foci: bool = False) -> Dict[str, Any]:
        """
        Compute center of mass focus metric.

        Args:
            frame: Input image frame
            two_foci: Whether to detect two foci

        Returns:
            Dictionary with focus metric results
        """
        timestamp = time.time()

        try:
            # Preprocess frame (use Gaussian filtering)
            im = gaussian_filter(frame.astype(float), 7)

            # Find peak centers
            center_coords = self.find_peak_centers(im, two_foci)

            # Compute center of mass
            focus_value = self.compute_center_of_mass(im, center_coords)

            logger.debug(f"Center of mass focus: center={center_coords}, focus={focus_value:.4f}")

            return {
                "t": timestamp,
                "focus": focus_value,
                "center_x": float(center_coords[0]),
                "center_y": float(center_coords[1]),
                "signal_max": float(np.max(im)),
            }

        except Exception as e:
            logger.error(f"Center of mass focus computation failed: {e}")
            return {"t": timestamp, "focus": self.config.max_focus_value, "error": str(e)}

'''
class FocusMetricFactory:
    """Factory for creating focus metric instances."""

    _metrics = {
        "astigmatism": AstigmatismFocusMetric,
        "center_of_mass": CenterOfMassFocusMetric,
        "gaussian": AstigmatismFocusMetric,
        "gradient": CenterOfMassFocusMetric,
        "peak": PeakMetric,
    }

    @classmethod
    def create(cls, metric_type: str, config: Optional[FocusConfig] = None) -> FocusMetricBase:
        """
        Create focus metric instance.

        Args:
            metric_type: Type of focus metric
            config: Focus configuration

        Returns:
            Focus metric instance
        """
        if metric_type not in cls._metrics:
            raise ValueError(f"Unknown focus metric type: {metric_type}. Available: {list(cls._metrics.keys())}")

        metric_class = cls._metrics[metric_type]
        return metric_class(config)

    @classmethod
    def available_metrics(cls) -> list:
        """Get list of available focus metrics."""
        return list(cls._metrics.keys())
