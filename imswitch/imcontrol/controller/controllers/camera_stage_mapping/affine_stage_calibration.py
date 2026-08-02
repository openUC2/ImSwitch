"""
Simple and explicit stage-to-camera affine calibration.

This module performs affine calibration to map camera pixel coordinates to stage 
micron coordinates. It uses a straightforward approach without complex abstractions.

The calibration:
1. Moves stage to known positions in a cross or grid pattern
2. Measures pixel displacement using phase correlation
3. Computes 2×3 affine transformation matrix via least squares
4. Validates and stores the calibration

Copyright 2024, released under GNU GPL v3
"""
import numpy as np
import logging
from typing import Optional, Dict


class TrackingError(RuntimeError):
    """Raised when the cross-correlation peak is too weak to trust the shift."""


# --------------------------------------------------------------------------- #
# High-pass FFT image tracking                                                #
#                                                                             #
# Adapted (and trimmed) from openflexure / camera-stage-mapping               #
# (fft_image_tracking.py, R. Bowman, GNU GPL v3). The key advantage over a    #
# plain phase correlation is the high-pass filter: it suppresses the slowly   #
# varying illumination gradient / vignetting that otherwise dominates the     #
# correlation and biases the measured shift. It also yields an honest         #
# normalised peak height (0..1) we can use as a per-step quality / failure    #
# detector — independent of any external dependency (no skimage required).    #
# --------------------------------------------------------------------------- #

def _prep(image: np.ndarray) -> np.ndarray:
    """Grayscale + float + mean-subtract (so zero-padding adds the mean, not an edge)."""
    image = np.asarray(image)
    if image.ndim == 3:
        image = image.mean(axis=2)
    return image.astype(float) - float(np.mean(image))


def _high_pass_mask(full_shape, sigma: float) -> np.ndarray:
    """``1 - Gaussian`` low-pass on the real-FFT frequency grid.

    ``sigma`` is the Gaussian standard deviation in *pixels* of the suppressed
    structure: larger ``sigma`` removes more of the low (coarse) frequencies.
    """
    fy = np.fft.fftfreq(full_shape[0])[:, np.newaxis]   # rows (full axis)
    fx = np.fft.rfftfreq(full_shape[1])[np.newaxis, :]  # cols (halved by rfft)
    r2 = fy ** 2 + fx ** 2
    return 1.0 - np.exp(-2.0 * (np.pi * sigma) ** 2 * r2)


def _background_subtracted_com(corr: np.ndarray, fractional_threshold: float = 0.1,
                               quadrant_swap: bool = False) -> np.ndarray:
    """Sub-pixel peak as the centre-of-mass of the top fraction of the correlation."""
    corr = corr.astype(float)
    hi, lo = float(np.max(corr)), float(np.min(corr))
    background = hi - fractional_threshold * (hi - lo)
    bs = corr - background
    bs[bs < 0] = 0.0
    rows = np.arange(corr.shape[0], dtype=float)
    cols = np.arange(corr.shape[1], dtype=float)
    if quadrant_swap:  # put DC at the centre so the shift is signed (smallest move)
        rows[corr.shape[0] // 2:] -= corr.shape[0]
        cols[corr.shape[1] // 2:] -= corr.shape[1]
    total = float(np.sum(bs))
    if total <= 0:
        return np.array([0.0, 0.0])
    r = float(np.sum(bs * rows[:, np.newaxis])) / total
    c = float(np.sum(bs * cols[np.newaxis, :])) / total
    return np.array([r, c])  # [row_shift, col_shift]


def high_pass_fft_template(image: np.ndarray, sigma: float = 10.0, pad: bool = True):
    """Build a reusable high-pass FFT template from the reference image.

    Returns ``(template, fft_shape, self_peak)`` where ``self_peak`` is the
    correlation height of the template against itself (the best achievable),
    used to normalise the quality of later measurements.
    """
    img = _prep(image)
    fft_shape = tuple(int(2 * s) for s in img.shape) if pad else tuple(img.shape)
    fft = np.fft.rfft2(img, s=fft_shape)
    template = np.conj(fft) * _high_pass_mask(fft_shape, sigma)
    self_peak = float(np.max(np.fft.irfft2(template * fft, s=fft_shape)))
    return template, fft_shape, self_peak


def displacement_from_template(template, fft_shape, self_peak, image,
                               fractional_threshold: float = 0.1,
                               error_threshold: float = 0.0):
    """Measure the (dx, dy) shift of ``image`` relative to the template's reference.

    ``(dx, dy)`` are in image pixels (x = columns, y = rows), matching the sign
    convention of the previous phase-correlation implementation. The second
    return value is the normalised peak height in [0, 1] (1 == perfect match).
    Raises :class:`TrackingError` if ``error_threshold > 0`` and the peak drops
    below it.
    """
    img = _prep(image)
    corr = np.fft.irfft2(template * np.fft.rfft2(img, s=fft_shape), s=fft_shape)
    peak = float(np.max(corr))
    quality = (peak / self_peak) if self_peak > 0 else 0.0
    if error_threshold > 0 and quality < error_threshold:
        raise TrackingError(f"correlation peak {quality:.3f} < threshold {error_threshold:.3f}")
    com = _background_subtracted_com(corr, fractional_threshold, quadrant_swap=True)
    # com is [row_shift, col_shift] = the displacement of the image content.
    # Negate to match the legacy phase-correlation sign (the shift that registers
    # `image` back onto the reference) so the downstream affine fit and the
    # calibration-owned flip detection are unchanged. Return (dx, dy) = (col, row).
    return (float(-com[1]), float(-com[0])), float(np.clip(quality, 0.0, 1.0))


def measure_pixel_shift(image1: np.ndarray, image2: np.ndarray,
                        sigma: float = 10.0, pad: bool = True) -> tuple:
    """Measure pixel displacement of ``image2`` relative to ``image1``.

    Drop-in replacement for the previous phase-correlation helper (same return
    shape ``((shift_x, shift_y), correlation)``) but illumination-robust and
    dependency-free. For a fixed reference measured against many frames, prefer
    :func:`high_pass_fft_template` + :func:`displacement_from_template` to build
    the template only once.
    """
    template, fft_shape, self_peak = high_pass_fft_template(image1, sigma=sigma, pad=pad)
    return displacement_from_template(template, fft_shape, self_peak, image2)


def compute_affine_matrix(pixel_shifts: np.ndarray, stage_shifts: np.ndarray) -> tuple:
    """
    Compute 2×3 affine transformation matrix from pixel-stage correspondences.
    
    The affine matrix maps pixel coordinates to stage coordinates:
        stage_coords = pixel_coords @ A.T + b
    
    Args:
        pixel_shifts: Nx2 array of measured pixel displacements
        stage_shifts: Nx2 array of commanded stage displacements (in microns)
    
    Returns:
        affine_matrix: 2×3 matrix [[a11, a12, tx], [a21, a22, ty]]
        inliers: Boolean mask of points used (after outlier removal)
        metrics: Dictionary with calibration quality metrics
    """
    n_points = len(pixel_shifts)

    # Center coordinates for numerical stability
    pixel_mean = np.mean(pixel_shifts, axis=0)
    stage_mean = np.mean(stage_shifts, axis=0)
    pixel_centered = pixel_shifts - pixel_mean
    stage_centered = stage_shifts - stage_mean

    # Solve for 2×2 transformation matrix A using least squares:
    # stage_centered = pixel_centered @ A.T
    A, residuals, rank, s = np.linalg.lstsq(pixel_centered, stage_centered, rcond=None)
    A = A.T  # Transpose to get 2×2 matrix

    # Compute residuals for outlier detection
    predicted_stage = pixel_centered @ A.T
    errors = stage_centered - predicted_stage
    error_norms = np.linalg.norm(errors, axis=1)

    # Outlier detection: remove points with error > 3×median
    median_error = np.median(error_norms)
    mad = np.median(np.abs(error_norms - median_error))
    threshold = median_error + 3.0 * 1.4826 * mad
    inliers = error_norms <= threshold

    # Refit with inliers only if we removed any outliers
    n_outliers = np.sum(~inliers)
    if n_outliers > 0 and np.sum(inliers) >= 3:
        A_refined, _, _, _ = np.linalg.lstsq(
            pixel_centered[inliers],
            stage_centered[inliers],
            rcond=None
        )
        A = A_refined.T

        # Recompute errors with refined fit
        predicted_stage = pixel_centered @ A.T
        errors = stage_centered - predicted_stage
        error_norms = np.linalg.norm(errors, axis=1)

    # Compute translation: stage_mean = pixel_mean @ A.T + b => b = stage_mean - pixel_mean @ A.T
    b = stage_mean - pixel_mean @ A.T

    # Build full 2×3 affine matrix [A | b]
    affine_matrix = np.column_stack([A, b])

    # Compute quality metrics
    inlier_errors = error_norms[inliers]
    rmse = np.sqrt(np.mean(inlier_errors**2))

    # Extract rotation and scale from A using SVD
    U, singular_values, Vt = np.linalg.svd(A)
    rotation_deg = np.arctan2(U[1, 0], U[0, 0]) * 180 / np.pi
    scale_x = singular_values[0]
    scale_y = singular_values[1]


    # store all information in metrics to disk


    metrics = {
        "rmse_um": rmse,
        "max_error_um": np.max(inlier_errors),
        "mean_error_um": np.mean(inlier_errors),
        "n_inliers": np.sum(inliers),
        "n_outliers": n_outliers,
        "rotation_deg": rotation_deg,
        "scale_x_um_per_pixel": scale_x,
        "scale_y_um_per_pixel": scale_y,
        "condition_number": np.linalg.cond(A)
    }

    return affine_matrix, inliers, metrics




def apply_affine_transform(affine_matrix: np.ndarray, pixel_coords: np.ndarray) -> np.ndarray:
    """
    Transform pixel coordinates to stage coordinates using affine matrix.
    
    Args:
        affine_matrix: 2×3 matrix [[a11, a12, tx], [a21, a22, ty]]
        pixel_coords: [x, y] or Nx2 array of pixel coordinates
    
    Returns:
        Stage coordinates in microns
    """
    if pixel_coords.ndim == 1:
        pixel_coords = pixel_coords.reshape(1, -1)

    A = affine_matrix[:, :2]  # 2×2 transformation
    b = affine_matrix[:, 2]   # 2×1 translation

    # stage = pixel @ A.T + b
    return pixel_coords @ A.T + b


def validate_calibration(affine_matrix: np.ndarray, metrics: Dict,
                        logger: Optional[logging.Logger] = None) -> tuple:
    """
    Check if calibration quality is acceptable.
    
    Args:
        affine_matrix: 2×3 calibration matrix
        metrics: Quality metrics from calibration
        logger: Optional logger
    
    Returns:
        (is_valid, message): True if acceptable, False otherwise with reason
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Check RMSE
    if metrics["rmse_um"] > 10.0:
        return False, f"RMSE too high: {metrics['rmse_um']:.1f}µm (max 10µm)"

    # Check correlation
    if metrics["mean_correlation"] < 0.2:
        return False, f"Correlation too low: {metrics['mean_correlation']:.3f} (min 0.2)"

    # Check matrix conditioning
    if metrics["condition_number"] > 100:
        return False, f"Matrix poorly conditioned: {metrics['condition_number']:.1f} (max 100)"

    # Check we have enough inliers
    if metrics["n_inliers"] < 4:
        return False, f"Too few valid points: {metrics['n_inliers']} (min 4)"

    logger.info("Calibration validation passed")
    return True, "Calibration passed all validation checks"


# --------------------------------------------------------------------------- #
# Backlash measurement                                                        #
#                                                                             #
# Adapted from openflexure / camera-stage-mapping (GNU GPL v3). Backlash is   #
# the mechanical "dead zone": after a direction reversal the stage does not   #
# move for the first part of the commanded travel while the slack is taken    #
# up. We recover it from a 1-D scan that reverses direction: the value that   #
# best linearises the (commanded stage) -> (observed image) relationship is   #
# the backlash.                                                               #
# --------------------------------------------------------------------------- #

def apply_backlash(x, backlash: float = 0.0, start_unwound: bool = True) -> np.ndarray:
    """Forward model of backlash: the realised position ``y`` lags commanded ``x``.

    ``y`` follows ``x`` but only after ``x`` reverses by at least ``backlash``;
    until then it stays put (the dead zone). With ``start_unwound`` we assume the
    mechanism is slack at the first sample, so the first move must take up the
    full ``backlash`` before anything happens.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if x.size == 0:
        return y
    if start_unwound and x.size > 1:
        y[0] = x[0] + np.sign(x[1] - x[0]) * backlash
    else:
        y[0] = x[0]
    for i in range(1, x.size):
        d = x[i] - y[i - 1]
        if abs(d) >= backlash:
            y[i] = x[i] - np.sign(d) * backlash
        else:
            y[i] = y[i - 1]
    return y


def fit_backlash_1d(stage_pos, image_pos, max_backlash: Optional[float] = None) -> Dict:
    """Estimate backlash (in the units of ``stage_pos``) from a reversing 1-D scan.

    ``stage_pos`` is the commanded position along one axis (e.g. µm) and
    ``image_pos`` the measured image displacement projected on that axis (px).
    A brute-force search picks the backlash that minimises the residual of a
    linear stage->image fit after applying the backlash model.

    Returns ``{"backlash", "scale", "offset", "residual_std", "residual_std_zero"}``
    where ``scale`` is image-px per stage-unit and ``residual_std_zero`` is the
    fit residual with no backlash (for an improvement check).
    """
    stage_pos = np.asarray(stage_pos, dtype=float)
    image_pos = np.asarray(image_pos, dtype=float)

    def fit_motion(backlash):
        xb = apply_backlash(stage_pos, backlash)
        xb = xb - np.mean(xb)
        m, c = np.polyfit(xb, image_pos, 1)
        resid = image_pos - (xb * m + c)
        ddof = 3 if image_pos.size > 3 else 0
        return float(m), float(c), float(np.std(resid, ddof=ddof))

    span = float(np.max(stage_pos) - np.min(stage_pos)) if stage_pos.size else 0.0
    if max_backlash is None:
        max_backlash = span / 2.0
    # Candidate backlash values: 0, then geometrically spaced up to max_backlash.
    candidates = [0.0]
    step = max(1e-6, span / 200.0)
    while step < max_backlash:
        candidates.append(step)
        step *= 1.33

    m0, c0, res0 = fit_motion(0.0)
    best = {"backlash": 0.0, "scale": m0, "offset": c0,
            "residual_std": res0, "residual_std_zero": res0}
    for cand in candidates:
        m, c, res = fit_motion(cand)
        if res < best["residual_std"]:
            best.update(backlash=float(cand), scale=m, offset=c, residual_std=res)
    return best
