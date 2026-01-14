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

try:
    from skimage.registration import phase_cross_correlation
    PHASE_CORRELATION_AVAILABLE = True
except ImportError:
    PHASE_CORRELATION_AVAILABLE = False
    logging.warning("phase_cross_correlation not available")


def measure_pixel_shift(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """
    Measure pixel displacement between two images using phase correlation.
    
    Args:
        image1: Reference image
        image2: Shifted image
    
    Returns:
        (shift_x, shift_y): Pixel displacement in X and Y
        correlation: Correlation quality (0-1)
    """
    if not PHASE_CORRELATION_AVAILABLE:
        raise ImportError("phase_cross_correlation is required for calibration")

    # Compute shift using phase correlation (sub-pixel accuracy)
    shift, error, phase_diff = phase_cross_correlation(
        image1, image2,
        upsample_factor=3  # 0.01 pixel accuracy
    )

    # shift is [dy, dx], convert to [dx, dy]
    shift_x = shift[1]
    shift_y = shift[0]

    # Estimate correlation quality from error
    correlation = 1.0 - min(error, 1.0)

    return (shift_x, shift_y), correlation


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
