"""
Focus Map – Surface fitting and interpolation for Z-focus across XY coordinates.

This module provides the core logic for:
  - Generating measurement grids within arbitrary bounding boxes
  - Fitting a Z-surface from measured focus points (spline, RBF, constant)
  - Interpolating Z for any (x, y) position during acquisition
  - Serialization of raw points, fit parameters, and quality stats

The FocusMap class does NOT interact with hardware directly; the caller
(ExperimentController) is responsible for stage motion and autofocus calls.
"""

import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Attempt imports for surface fitting
try:
    from scipy.interpolate import RBFInterpolator, RectBivariateSpline
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class FitMethod(str, Enum):
    """Available surface-fit methods."""
    SPLINE = "spline"
    RBF = "rbf"
    CONSTANT = "constant"


@dataclass
class FocusPoint:
    """A single measured focus point."""
    x: float
    y: float
    z: float
    group_id: str = ""
    quality_metric: float = 0.0  # e.g. sharpness score from autofocus
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitStats:
    """Quality statistics for a fitted surface.

    The error fields are ``None`` when they cannot be measured. Residuals are
    taken against the very points that were fitted, so with fewer points than
    the fit has degrees of freedom they are trivially zero — a confident
    "MAE 0.000" exactly when the surface is least trustworthy.
    """
    method: str = "constant"
    n_points: int = 0
    mean_abs_error: Optional[float] = None
    std_error: Optional[float] = None
    max_error: Optional[float] = None
    r_squared: Optional[float] = None
    z_offset: float = 0.0
    # True when the errors come from leave-one-out refits rather than from the
    # points the surface was fitted to.
    error_is_cross_validated: bool = False
    bounds_x: Tuple[float, float] = (0.0, 0.0)
    bounds_y: Tuple[float, float] = (0.0, 0.0)
    bounds_z: Tuple[float, float] = (0.0, 0.0)
    fallback_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def fit_quality_text(stats: "FitStats") -> str:
    """One line an operator can read: how good is this surface, and why.

    Below 4 points the residuals are taken against the fitted points
    themselves, so there is no number to report — say that instead of
    printing a zero.
    """
    if stats.mean_abs_error is None:
        detail = "not enough points to validate"
    else:
        kind = "cross-validated" if stats.error_is_cross_validated else "in-sample"
        detail = f"MAE={stats.mean_abs_error:.3f} um ({kind})"
    if stats.fallback_reason:
        detail += f" ({stats.fallback_reason})"
    return detail


@dataclass
class FocusMapResult:
    """Complete result of a focus-map computation for one group."""
    group_id: str
    group_name: str = ""
    points: List[FocusPoint] = field(default_factory=list)
    fit_stats: FitStats = field(default_factory=FitStats)
    preview_grid: Optional[Dict[str, Any]] = None  # {x: [...], y: [...], z: [[...]]}
    status: str = "pending"  # pending | measuring | fitting | ready | error

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "points": [p.to_dict() for p in self.points],
            "fit_stats": self.fit_stats.to_dict(),
            "preview_grid": self.preview_grid,
            "status": self.status,
        }


class FocusMap:
    """
    Manages focus-surface estimation for a single region.

    Workflow:
        1. generate_grid(bounds, rows, cols) → list of (x, y) measurement coords
        2. add_point(x, y, z) for each measured focus position
        3. fit() → builds the surface interpolator
        4. interpolate(x, y) → returns estimated z
    """

    def __init__(self,
                 group_id: str = "default",
                 group_name: str = "",
                 method: str = "spline",
                 smoothing_factor: float = 0.1,
                 z_offset: float = 0.0,
                 clamp_enabled: bool = False,
                 z_min: float = 0.0,
                 z_max: float = 0.0,
                 logger=None):
        self.group_id = group_id
        self.group_name = group_name
        self.method = method
        self.smoothing_factor = smoothing_factor
        self.z_offset = z_offset
        self.clamp_enabled = clamp_enabled
        self.z_min = z_min
        self.z_max = z_max
        self._logger = logger

        self._points: List[FocusPoint] = []
        self._interpolator = None
        self._fit_stats = FitStats()
        self._is_fitted = False
        self._constant_z: Optional[float] = None
        self._plane_coeffs: Optional[np.ndarray] = None  # (a, b, c): z = a*x + b*y + c

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_grid(bounds: Dict[str, float],
                      rows: int = 3,
                      cols: int = 3,
                      add_margin: bool = False,
                      margin_fraction: float = 0.05) -> List[Tuple[float, float]]:
        """
        Generate a uniform measurement grid within the given bounds.

        Args:
            bounds: dict with keys minX, maxX, minY, maxY
            rows: number of grid rows
            cols: number of grid columns
            add_margin: if True, shrink the grid inward by margin_fraction
            margin_fraction: fraction of width/height to use as margin

        Returns:
            List of (x, y) tuples representing measurement positions
        """
        min_x, max_x = bounds["minX"], bounds["maxX"]
        min_y, max_y = bounds["minY"], bounds["maxY"]

        if add_margin:
            dx = (max_x - min_x) * margin_fraction
            dy = (max_y - min_y) * margin_fraction
            min_x += dx
            max_x -= dx
            min_y += dy
            max_y -= dy

        if rows < 1:
            rows = 1
        if cols < 1:
            cols = 1

        # Single row/col → center
        xs = np.linspace(min_x, max_x, cols) if cols > 1 else np.array([(min_x + max_x) / 2.0])
        ys = np.linspace(min_y, max_y, rows) if rows > 1 else np.array([(min_y + max_y) / 2.0])

        grid = []
        for y in ys:
            for x in xs:
                grid.append((float(x), float(y)))

        # Deduplicate grid points (handles degenerate single-FOV bounds where
        # e.g. minX == maxX produces identical columns in the grid)
        seen = set()
        unique_grid = []
        for p in grid:
            key = (round(p[0], 3), round(p[1], 3))
            if key not in seen:
                seen.add(key)
                unique_grid.append(p)
        return unique_grid

    # ------------------------------------------------------------------
    # Point management
    # ------------------------------------------------------------------

    def add_point(self, x: float, y: float, z: float,
                  quality_metric: float = 0.0) -> None:
        """Add a measured focus point."""
        self._points.append(FocusPoint(
            x=x, y=y, z=z,
            group_id=self.group_id,
            quality_metric=quality_metric,
            timestamp=time.time(),
        ))
        self._is_fitted = False  # Invalidate existing fit

    def clear_points(self) -> None:
        """Remove all measured points and reset fit."""
        self._points.clear()
        self._interpolator = None
        self._is_fitted = False
        self._constant_z = None
        self._plane_coeffs = None
        self._fit_stats = FitStats()

    @property
    def points(self) -> List[FocusPoint]:
        return list(self._points)

    @property
    def n_points(self) -> int:
        return len(self._points)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def fit_stats(self) -> FitStats:
        return self._fit_stats

    # ------------------------------------------------------------------
    # Surface fitting
    # ------------------------------------------------------------------

    def fit(self, reject_outliers: bool = True, outlier_sigma: float = 3.0,
            cross_validate: bool = True) -> FitStats:
        """
        Fit a Z-surface through the measured focus points.

        Rules:
          - 0 points: error
          - 1 point: constant surface at that Z
          - 2-3 points: reject with error (need >= 4 for 2D fit) unless
            method == "constant" → use mean
          - >= 4 points: try requested method; fall back if necessary

        Args:
            reject_outliers: If True, remove Z values further than
                             outlier_sigma standard deviations from
                             the median before fitting.
            outlier_sigma:   Number of standard deviations for outlier
                             rejection (default 3.0).

        Returns:
            FitStats with quality metrics
        """
        n = len(self._points)

        if n == 0:
            raise ValueError("No focus points available. Run focus measurement first.")

        xs = np.array([p.x for p in self._points])
        ys = np.array([p.y for p in self._points])
        zs = np.array([p.z for p in self._points])

        # --- Outlier rejection (based on Z values) ---
        n_rejected = 0
        if reject_outliers and n >= 4:
            median_z = np.median(zs)
            mad = np.median(np.abs(zs - median_z))  # median absolute deviation
            # Use MAD-based sigma estimate (robust); fall back to std if MAD==0
            sigma_est = mad * 2*1.4826 if mad > 0 else np.std(zs) # actually twice the stdv
            if sigma_est > 0:
                mask = np.abs(zs - median_z) <= outlier_sigma * sigma_est
                n_rejected = int(np.sum(~mask))
                if n_rejected > 0 and np.sum(mask) >= 1:
                    if self._logger:
                        self._logger.warning(
                            f"FocusMap [{self.group_id}]: rejected {n_rejected}/{n} "
                            f"outlier points (>{outlier_sigma}σ from median Z={median_z:.2f})"
                        )
                    xs, ys, zs = xs[mask], ys[mask], zs[mask]
                    n = len(xs)

        self._fit_stats = FitStats(
            n_points=n,
            z_offset=float(self.z_offset),
            bounds_x=(float(xs.min()), float(xs.max())),
            bounds_y=(float(ys.min()), float(ys.max())),
            bounds_z=(float(zs.min()), float(zs.max())),
        )

        # Reset any representation from a previous fit so re-fitting (e.g. after
        # adding points) can't leave a stale constant/plane shadowing the new one.
        self._constant_z = None
        self._plane_coeffs = None
        self._interpolator = None

        # --- Single point: constant ---
        if n == 1:
            self._fit_constant(zs)
            if self._logger:
                self._logger.info(f"FocusMap [{self.group_id}]: 1 point → constant Z={self._constant_z:.3f}")
            return self._fit_stats

        # --- fewer than 4 points: a plane at best, and only if it is defined ---
        # A tilted plane needs 3 points that are not on one line. Two points, or
        # three collinear ones, leave the tilt undetermined; lstsq returns the
        # minimum-norm answer without complaint, which is how a 2-point map used
        # to present itself as a confident "plane".
        if n < 4:
            if self.method == "constant":
                self._fit_constant(zs)
            elif n < 3:
                self._fit_constant(
                    zs, f"{n} point(s) cannot define a tilted plane - using constant Z")
            elif self._is_collinear(xs, ys):
                self._fit_constant(
                    zs, "the points lie on one line, which does not define a "
                        "plane - using constant Z")
            else:
                self._fit_plane(xs, ys, zs)
                self._fit_stats.method = "plane"
                self._fit_stats.fallback_used = True
                self._fit_stats.fallback_reason = (
                    f"3 points - planar fit (needs 4 for {self.method})")
                self._is_fitted = True
                if self._logger:
                    self._logger.info(
                        f"FocusMap [{self.group_id}]: 3 points -> planar (tilted) fit")
            # Errors are deliberately left as None: with <4 points the residuals
            # against the fitted points are meaningless.
            return self._fit_stats

        # --- >= 4 points: try requested method ---
        # A whole row of points along one edge is a realistic operator mistake,
        # and no 2-D surface is determined by it.
        if self.method != "constant" and self._is_collinear(xs, ys):
            self._fit_constant(
                zs, "the points lie on one line, which does not define a "
                    "surface - using constant Z")
            return self._fit_stats

        if not HAS_SCIPY:
            self._fit_constant(zs, "scipy not available - using constant Z")
            self._compute_fit_stats(xs, ys, zs, "constant", True,
                                    "scipy not available - using constant Z",
                                    cross_validate=cross_validate)
            return self._fit_stats

        method = self.method
        fallback_used = False
        fallback_reason = ""

        # Try spline first, then RBF, then constant
        if method == FitMethod.SPLINE:
            try:
                self._fit_spline(xs, ys, zs)
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"FocusMap [{self.group_id}]: spline fit failed ({e}), trying RBF")
                method = FitMethod.RBF
                fallback_used = True
                fallback_reason = f"spline failed: {e}"

        if method == FitMethod.RBF:
            try:
                self._fit_rbf(xs, ys, zs)
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"FocusMap [{self.group_id}]: RBF fit failed ({e}), using constant")
                method = FitMethod.CONSTANT
                fallback_used = True
                fallback_reason += f"; rbf failed: {e}" if fallback_reason else f"rbf failed: {e}"

        if method == FitMethod.CONSTANT:
            self._constant_z = float(np.mean(zs))
            self._interpolator = None

        # Compute fit statistics
        self._compute_fit_stats(xs, ys, zs, method, fallback_used, fallback_reason,
                                cross_validate=cross_validate)
        self._is_fitted = True

        if self._logger:
            self._logger.info(
                f"FocusMap [{self.group_id}]: fitted with {self._fit_stats.method}, "
                f"MAE={self._fit_stats.mean_abs_error:.4f}, "
                f"std={self._fit_stats.std_error:.4f}, "
                f"n={n}"
            )

        return self._fit_stats

    @staticmethod
    def _is_collinear(xs: np.ndarray, ys: np.ndarray) -> bool:
        """True when the XY points lie on (or vanishingly close to) one line.

        Collinear points do not define a plane: infinitely many planes contain
        the line, and lstsq silently returns the minimum-norm one.
        """
        if len(xs) < 3:
            return True
        centred = np.column_stack([xs - np.mean(xs), ys - np.mean(ys)])
        spread = np.linalg.svd(centred, compute_uv=False)
        return bool(spread[-1] <= 1e-6 * max(spread[0], 1e-12))

    def _fit_constant(self, zs: np.ndarray, reason: str = "") -> None:
        """Flat surface at the mean Z — the honest answer when the points
        cannot define a tilted one."""
        self._constant_z = float(np.mean(zs))
        self._interpolator = None
        self._plane_coeffs = None
        self._fit_stats.method = "constant"
        self._fit_stats.n_points = len(zs)
        self._fit_stats.fallback_used = bool(reason)
        self._fit_stats.fallback_reason = reason
        self._is_fitted = True
        if reason and self._logger:
            self._logger.warning(f"FocusMap [{self.group_id}]: {reason}")

    def _fit_plane(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        """Least-squares tilted plane: z = a*x + b*y + c.

        Used for 2-3 points, where a spline/RBF surface is not defined. For 3
        non-collinear points this is the exact interpolating plane; for 2 points
        or collinear points np.linalg.lstsq returns the minimum-norm solution
        (a sensible ramp / flat surface).
        """
        A = np.column_stack([xs, ys, np.ones_like(xs, dtype=float)])
        coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
        self._plane_coeffs = coeffs
        self._interpolator = None
        self._constant_z = None

    def _fit_spline(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        """Fit using scipy griddata (linear interpolation) as a practical spline substitute."""
        # For scattered data, use RBF-based interpolator with thin-plate spline kernel
        points = np.column_stack([xs, ys])
        self._interpolator = RBFInterpolator(
            points, zs,
            kernel="thin_plate_spline",
            smoothing=self.smoothing_factor,
        )

    def _fit_rbf(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        """Fit using RBF interpolation with thin-plate-spline kernel.
        
        We use 'thin_plate_spline' because it does not require an explicit
        epsilon parameter (unlike 'multiquadric', 'gaussian', etc.) and
        works well for smooth 2-D surface interpolation.
        """
        points = np.column_stack([xs, ys])
        self._interpolator = RBFInterpolator(
            points, zs,
            kernel="thin_plate_spline",
            smoothing=self.smoothing_factor,
        )

    _LOO_MAX_POINTS = 40  # n refits; beyond this the wait stops being worth it

    def _leave_one_out_residuals(self, xs, ys, zs):
        """Predict each point from a surface fitted WITHOUT it.

        Spline and RBF interpolate: they pass exactly through every point they
        were given, so residuals against those same points are ~0 no matter how
        wrong the surface is between them. Refitting without each point in turn
        is the cheapest honest answer. Returns None when it cannot be done.
        """
        n = len(xs)
        if n < 4 or n > self._LOO_MAX_POINTS:
            return None
        residuals = []
        for i in range(n):
            keep = np.arange(n) != i
            fold = FocusMap(group_id=f"{self.group_id}:loo",
                            method=self.method,
                            smoothing_factor=self.smoothing_factor)
            for x, y, z in zip(xs[keep], ys[keep], zs[keep]):
                fold.add_point(float(x), float(y), float(z))
            try:
                fold.fit(reject_outliers=False, cross_validate=False)
                residuals.append(zs[i] - fold._interpolate_raw(xs[i], ys[i]))
            except Exception:
                return None
        return np.array(residuals)

    def _compute_fit_stats(self, xs, ys, zs, method, fallback_used, fallback_reason,
                           cross_validate=True):
        """Residual stats, cross-validated where that is meaningful."""
        residuals = self._leave_one_out_residuals(xs, ys, zs) if cross_validate else None
        if residuals is None:
            residuals = zs - np.array([self._interpolate_raw(x, y) for x, y in zip(xs, ys)])
            self._fit_stats.error_is_cross_validated = False
        else:
            self._fit_stats.error_is_cross_validated = True

        self._fit_stats.method = method
        self._fit_stats.z_offset = float(self.z_offset)
        self._fit_stats.mean_abs_error = float(np.mean(np.abs(residuals)))
        self._fit_stats.std_error = float(np.std(residuals))
        self._fit_stats.max_error = float(np.max(np.abs(residuals)))
        self._fit_stats.fallback_used = fallback_used
        self._fit_stats.fallback_reason = fallback_reason

        # R² calculation
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((zs - np.mean(zs)) ** 2)
        self._fit_stats.r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _interpolate_raw(self, x: float, y: float) -> float:
        """Raw interpolation without offset or clamping."""
        if self._constant_z is not None:
            return self._constant_z

        if self._plane_coeffs is not None:
            a, b, c = self._plane_coeffs
            return float(a * x + b * y + c)

        if self._interpolator is not None:
            point = np.array([[x, y]])
            return float(self._interpolator(point)[0])

        raise RuntimeError(f"FocusMap [{self.group_id}]: not fitted yet")

    def interpolate(self, x: float, y: float) -> float:
        """
        Get the estimated Z for a given (x, y) position.

        Applies z_offset and optional clamping.

        Args:
            x: X position
            y: Y position

        Returns:
            Estimated Z position (with offset and clamping applied)
        """
        if not self._is_fitted:
            raise RuntimeError(f"FocusMap [{self.group_id}]: not fitted yet. Call fit() first.")

        z = self._interpolate_raw(x, y) + self.z_offset

        if self.clamp_enabled:
            z = np.clip(z, self.z_min, self.z_max)

        return float(z)

    # ------------------------------------------------------------------
    # Interpolation to a new region
    # ------------------------------------------------------------------

    def interpolate_to_region(
        self,
        group_id: str,
        group_name: str,
        bounds: Dict[str, float],
        rows: int = 3,
        cols: int = 3,
        logger=None,
    ) -> "FocusMap":
        """
        Create a new FocusMap for a different region by sampling this fitted
        map at a grid of points within the given bounds.

        This is used when the user has a manual/global map and wants to
        reuse it for per-region groups without measuring again.

        Args:
            group_id:   ID for the new map
            group_name: Human-readable name
            bounds:     dict with minX, maxX, minY, maxY
            rows:       number of sample rows
            cols:       number of sample columns
            logger:     optional logger

        Returns:
            A new FocusMap instance with interpolated points, already fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"FocusMap [{self.group_id}]: cannot interpolate – not fitted yet"
            )

        grid = FocusMap.generate_grid(bounds=bounds, rows=rows, cols=cols)

        new_fm = FocusMap(
            group_id=group_id,
            group_name=group_name,
            method=self.method,
            smoothing_factor=self.smoothing_factor,
            z_offset=self.z_offset,
            clamp_enabled=self.clamp_enabled,
            z_min=self.z_min,
            z_max=self.z_max,
            logger=logger or self._logger,
        )

        for gx, gy in grid:
            try:
                gz = self.interpolate(gx, gy)
                new_fm.add_point(gx, gy, gz)
            except Exception:
                # Point outside the fitted domain – skip
                pass

        if new_fm.n_points > 0:
            new_fm.fit()

        return new_fm

    # ------------------------------------------------------------------
    # Preview grid generation
    # ------------------------------------------------------------------

    def generate_preview_grid(self, resolution: int = 20) -> Dict[str, Any]:
        """
        Generate a regular grid of interpolated Z values for visualization.

        Args:
            resolution: number of points along each axis

        Returns:
            dict with keys: x (1D), y (1D), z (2D), points (raw measured)
        """
        if not self._is_fitted:
            raise RuntimeError("FocusMap not fitted")

        if len(self._points) == 0:
            return {"x": [], "y": [], "z": [], "points": []}

        xs = [p.x for p in self._points]
        ys = [p.y for p in self._points]

        x_lin = np.linspace(min(xs), max(xs), resolution)
        y_lin = np.linspace(min(ys), max(ys), resolution)

        z_grid = np.zeros((resolution, resolution))
        for i, yi in enumerate(y_lin):
            for j, xi in enumerate(x_lin):
                try:
                    z_grid[i, j] = self.interpolate(xi, yi)
                except Exception:
                    z_grid[i, j] = float("nan")

        return {
            "x": x_lin.tolist(),
            "y": y_lin.tolist(),
            "z": z_grid.tolist(),
            "points": [p.to_dict() for p in self._points],
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_result(self) -> FocusMapResult:
        """Export as a FocusMapResult for API responses and persistence."""
        preview = self.generate_preview_grid() if self._is_fitted else None
        status = "ready" if self._is_fitted else ("measuring" if len(self._points) > 0 else "pending")
        return FocusMapResult(
            group_id=self.group_id,
            group_name=self.group_name,
            points=list(self._points),
            fit_stats=self._fit_stats,
            preview_grid=preview,
            status=status,
        )

    def save(self, path: str) -> str:
        """
        Save focus map data to a JSON file.

        Args:
            path: directory path to save into

        Returns:
            Full file path of saved JSON
        """
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"focus_map_{self.group_id}.json")
        data = self.to_result().to_dict()
        data["config"] = {
            "method": self.method,
            "smoothing_factor": self.smoothing_factor,
            "z_offset": self.z_offset,
            "clamp_enabled": self.clamp_enabled,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return file_path

    @classmethod
    def load(cls, file_path: str, logger=None) -> "FocusMap":
        """
        Load a focus map from a saved JSON file.

        Args:
            file_path: path to JSON file

        Returns:
            FocusMap instance (not yet re-fitted; call fit() to rebuild interpolator)
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        config = data.get("config", {})
        fm = cls(
            group_id=data.get("group_id", "default"),
            group_name=data.get("group_name", ""),
            method=config.get("method", "spline"),
            smoothing_factor=config.get("smoothing_factor", 0.1),
            z_offset=config.get("z_offset", 0.0),
            clamp_enabled=config.get("clamp_enabled", False),
            z_min=config.get("z_min", 0.0),
            z_max=config.get("z_max", 0.0),
            logger=logger,
        )

        for p_data in data.get("points", []):
            fm.add_point(
                x=p_data["x"],
                y=p_data["y"],
                z=p_data["z"],
                quality_metric=p_data.get("quality_metric", 0.0),
            )

        return fm


class FocusMapManager:
    """
    Manages multiple FocusMap instances (one per group / scan area / well).

    Used by ExperimentController to coordinate focus-map computation
    across all experiment groups.
    """

    def __init__(self, logger=None):
        self._logger = logger
        self._maps: Dict[str, FocusMap] = {}
        self._abort_requested = False

    @property
    def abort_requested(self) -> bool:
        return self._abort_requested

    def request_abort(self):
        """Request cancellation of ongoing focus map computation."""
        self._abort_requested = True

    def clear_abort(self):
        """Reset abort flag before a new computation."""
        self._abort_requested = False

    def get_or_create(self,
                      group_id: str,
                      group_name: str = "",
                      method: str = "spline",
                      smoothing_factor: float = 0.1,
                      z_offset: float = 0.0,
                      clamp_enabled: bool = False,
                      z_min: float = 0.0,
                      z_max: float = 0.0) -> FocusMap:
        """Get existing FocusMap for group, or create a new one.
        
        If the map already exists, its configuration parameters are
        updated so that changes from the frontend (e.g. z_min, z_max,
        method) are reflected without having to clear and recreate.
        """
        if group_id not in self._maps:
            self._maps[group_id] = FocusMap(
                group_id=group_id,
                group_name=group_name,
                method=method,
                smoothing_factor=smoothing_factor,
                z_offset=z_offset,
                clamp_enabled=clamp_enabled,
                z_min=z_min,
                z_max=z_max,
                logger=self._logger,
            )
        else:
            # Update parameters on existing map so frontend changes propagate.
            # Only update config params (method, z_offset, etc.) – do NOT clear
            # existing points or fit. This allows users to adjust parameters
            # like z_offset, clamp_enabled, z_min, z_max on the fly without
            # losing their existing measurements. A new fit() call will use
            # the updated parameters.
            fm = self._maps[group_id]
            fm.group_name = group_name
            fm.method = method
            fm.smoothing_factor = smoothing_factor
            fm.z_offset = z_offset
            fm.clamp_enabled = clamp_enabled
            fm.z_min = z_min
            fm.z_max = z_max
        return self._maps[group_id]

    def get(self, group_id: str) -> Optional[FocusMap]:
        """Get a FocusMap by group ID."""
        return self._maps.get(group_id)

    def get_all(self) -> Dict[str, FocusMap]:
        """Get all focus maps."""
        return dict(self._maps)

    def clear(self, group_id: Optional[str] = None) -> None:
        """Clear one or all focus maps."""
        if group_id is not None:
            if group_id in self._maps:
                del self._maps[group_id]
        else:
            self._maps.clear()

    def has_fitted_map(self, group_id: str) -> bool:
        """Check if a group has a fitted focus map."""
        fm = self._maps.get(group_id)
        return fm is not None and fm.is_fitted

    def interpolate(self, x: float, y: float, group_id: str) -> Optional[float]:
        """
        Interpolate Z for a position, using the specified group's map.

        Returns None if no fitted map exists for the group.
        """
        fm = self._maps.get(group_id)
        if fm is not None and fm.is_fitted:
            return fm.interpolate(x, y)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all maps for API response."""
        return {
            gid: fm.to_result().to_dict()
            for gid, fm in self._maps.items()
        }

    def save_all(self, path: str) -> List[str]:
        """Save all focus maps to a directory."""
        saved = []
        for fm in self._maps.values():
            if fm.n_points > 0:
                saved.append(fm.save(path))
        return saved

    def load_all(self, path: str) -> int:
        """Load all focus maps from a directory, REPLACING what is in memory.

        Merging meant a map from an earlier sample survived a load and could
        still be picked up by a later run.
        """
        loaded = 0
        if not os.path.isdir(path):
            return loaded
        self._maps.clear()
        for fname in os.listdir(path):
            if fname.startswith("focus_map_") and fname.endswith(".json"):
                fpath = os.path.join(path, fname)
                try:
                    fm = FocusMap.load(fpath, logger=self._logger)
                    fm.fit()
                    self._maps[fm.group_id] = fm
                    loaded += 1
                except Exception as e:
                    if self._logger:
                        self._logger.warning(f"Failed to load focus map {fname}: {e}")
        return loaded
