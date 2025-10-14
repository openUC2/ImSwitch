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
import time
import logging
from typing import Optional, Callable, Dict

try:
    from skimage.registration import phase_cross_correlation
    PHASE_CORRELATION_AVAILABLE = True
except ImportError:
    PHASE_CORRELATION_AVAILABLE = False
    logging.warning("phase_cross_correlation not available")


def auto_adjust_exposure(
    grab_image: Callable,
    set_exposure: Callable,
    target_intensity: float = 0.75,
    max_iterations: int = 10,
    tolerance: float = 0.05,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Automatically adjust camera exposure to achieve target peak intensity.
    
    Args:
        grab_image: Function that captures and returns an image
        set_exposure: Function that sets exposure time (in ms)
        target_intensity: Target peak intensity as fraction of max (0.7-0.8 recommended)
        max_iterations: Maximum number of adjustment iterations
        tolerance: Acceptable deviation from target (as fraction)
        logger: Optional logger for progress reporting
    
    Returns:
        Final exposure time in milliseconds
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get current exposure
    current_exposure = 50.0  # Start with a reasonable default
    
    for iteration in range(max_iterations):
        image = grab_image()
        
        # Calculate peak intensity (normalized to 0-1)
        if image.dtype == np.uint8:
            max_value = 255.0
        elif image.dtype == np.uint16:
            max_value = 65535.0
        else:
            max_value = float(np.max(image))
        
        peak_intensity = np.max(image) / max_value
        
        logger.info(f"Auto-exposure iteration {iteration + 1}: "
                   f"peak={peak_intensity:.3f}, exposure={current_exposure:.1f}ms")
        
        # Check if we're within tolerance
        if abs(peak_intensity - target_intensity) < tolerance:
            logger.info(f"Target intensity achieved: {peak_intensity:.3f}")
            return current_exposure
        
        # Avoid division by zero or negative values
        if peak_intensity < 0.01:
            # Image too dark, increase exposure significantly
            adjustment = 2.0
        else:
            # Proportional adjustment
            adjustment = target_intensity / peak_intensity
            # Limit adjustment rate for stability
            adjustment = np.clip(adjustment, 0.5, 2.0)
        
        current_exposure *= adjustment
        # Clamp exposure to reasonable bounds
        current_exposure = np.clip(current_exposure, 0.1, 1000.0)
        
        set_exposure(current_exposure)
        time.sleep(0.1)  # Allow camera to stabilize
    
    logger.warning(f"Auto-exposure did not converge after {max_iterations} iterations")
    return current_exposure


def compute_displacement_phase_correlation(
    image1: np.ndarray,
    image2: np.ndarray,
    upsample_factor: int = 100
) -> Tuple[np.ndarray, float]:
    """
    Compute sub-pixel displacement between two images using phase correlation.
    
    Args:
        image1: Reference image
        image2: Shifted image
        upsample_factor: Sub-pixel precision factor (100 = 0.01 pixel accuracy)
    
    Returns:
        Tuple of (shift_vector, correlation_value)
        shift_vector: [dy, dx] displacement in pixels (image2 relative to image1)
        correlation_value: Quality metric of the correlation
    """
    # Convert to grayscale if needed
    if len(image1.shape) == 3:
        image1 = np.mean(image1, axis=2)
    if len(image2.shape) == 3:
        image2 = np.mean(image2, axis=2)
    
    # Normalize images
    image1 = (image1 - np.mean(image1)) / (np.std(image1) + 1e-10)
    image2 = (image2 - np.mean(image2)) / (np.std(image2) + 1e-10)
    
    if PHASE_CORRELATION_AVAILABLE:
        # Use scikit-image phase correlation (more robust)
        shift, error, diffphase = phase_cross_correlation(
            image1, image2, upsample_factor=upsample_factor
        )
        # Note: scikit-image returns (row_shift, col_shift) = (y, x)
        return shift, 1.0 - error  # Convert error to correlation quality
    else:
        # Fallback to basic FFT correlation
        f1 = np.fft.fft2(image1)
        f2 = np.fft.fft2(image2)
        
        # Cross-power spectrum
        cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
        correlation = np.fft.ifft2(cross_power)
        correlation = np.abs(np.fft.fftshift(correlation))
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        shift = np.array(peak_idx) - np.array(image1.shape) / 2.0
        
        # Correlation quality
        correlation_value = np.max(correlation) / np.mean(correlation)
        
        return shift, correlation_value


def robust_affine_from_correspondences(
    pixel_coords: np.ndarray,
    stage_coords: np.ndarray,
    outlier_threshold: float = 2.5,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute affine transformation with outlier rejection using RANSAC-like approach.
    
    Args:
        pixel_coords: Nx2 array of pixel displacements [dx, dy]
        stage_coords: Nx2 array of stage displacements [dx_stage, dy_stage] in microns
        outlier_threshold: Threshold in standard deviations for outlier detection
        logger: Optional logger
    
    Returns:
        Tuple of (affine_matrix, inlier_mask, metrics_dict)
        affine_matrix: 2x3 matrix [A | b] where A is 2x2 rotation/scale, b is 2x1 translation
        inlier_mask: Boolean array indicating which points were used
        metrics_dict: Dictionary with quality metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    n_points = pixel_coords.shape[0]
    if n_points < 3:
        raise ValueError(f"Need at least 3 points for affine calibration, got {n_points}")
    
    # Center the coordinates for numerical stability
    pixel_mean = np.mean(pixel_coords, axis=0)
    stage_mean = np.mean(stage_coords, axis=0)
    
    pixel_centered = pixel_coords - pixel_mean
    stage_centered = stage_coords - stage_mean
    
    # Initial least-squares fit
    # We solve: stage = pixel @ A.T + b
    # In centered coordinates: stage_centered = pixel_centered @ A.T
    A_init, residuals, rank, s = np.linalg.lstsq(pixel_centered, stage_centered, rcond=None)
    
    # A_init is 2x2, transpose to get the form we want
    A = A_init.T
    
    # Compute residuals
    predicted_stage = pixel_centered @ A.T
    residuals = stage_centered - predicted_stage
    residual_norms = np.linalg.norm(residuals, axis=1)
    
    # Outlier detection using median absolute deviation (more robust than std)
    median_residual = np.median(residual_norms)
    mad = np.median(np.abs(residual_norms - median_residual))
    threshold = median_residual + outlier_threshold * 1.4826 * mad  # 1.4826 converts MAD to std
    
    inlier_mask = residual_norms < threshold
    n_outliers = np.sum(~inlier_mask)
    
    if n_outliers > 0:
        logger.warning(f"Detected {n_outliers} outliers out of {n_points} points")
        
        # Refit using only inliers
        if np.sum(inlier_mask) >= 3:
            A_refined, _, _, _ = np.linalg.lstsq(
                pixel_centered[inlier_mask], 
                stage_centered[inlier_mask], 
                rcond=None
            )
            A = A_refined.T
            
            # Recompute residuals with refined fit
            predicted_stage = pixel_centered @ A.T
            residuals = stage_centered - predicted_stage
            residual_norms = np.linalg.norm(residuals, axis=1)
    
    # Compute translation component (using centered coordinates)
    # stage = pixel @ A.T + b
    # stage_mean = pixel_mean @ A.T + b
    # => b = stage_mean - pixel_mean @ A.T
    b = stage_mean - pixel_mean @ A.T
    
    # Build full 2x3 affine matrix [A | b]
    affine_matrix = np.column_stack([A, b])
    
    # Compute metrics
    mean_error = np.mean(residual_norms[inlier_mask])
    max_error = np.max(residual_norms[inlier_mask])
    std_error = np.std(residual_norms[inlier_mask])
    
    # Decompose affine matrix to get rotation and scale
    # A = R @ S where R is rotation and S is scale/shear
    U, singular_values, Vt = np.linalg.svd(A)
    rotation_angle = np.arctan2(U[1, 0], U[0, 0]) * 180 / np.pi
    scale_x = singular_values[0]
    scale_y = singular_values[1]
    
    metrics = {
        "mean_error_um": mean_error,
        "max_error_um": max_error,
        "std_error_um": std_error,
        "rmse_um": np.sqrt(np.mean(residual_norms[inlier_mask]**2)),
        "n_inliers": np.sum(inlier_mask),
        "n_outliers": n_outliers,
        "rotation_deg": rotation_angle,
        "scale_x_um_per_pixel": scale_x,
        "scale_y_um_per_pixel": scale_y,
        "condition_number": np.linalg.cond(A)
    }
    
    logger.info(f"Affine calibration: RMSE={metrics['rmse_um']:.3f}µm, "
               f"rotation={rotation_angle:.2f}°, "
               f"scale=({scale_x:.3f}, {scale_y:.3f})µm/px")
    
    return affine_matrix, inlier_mask, metrics


def calibrate_affine_transform(
    tracker: Tracker,
    move: Callable,
    step_size_um: float = 100.0,
    pattern: str = "cross",
    n_steps: int = 4,
    settle_time: float = 0.2,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Perform robust affine calibration using structured movement pattern.
    
    Args:
        tracker: Initialized Tracker object for image acquisition and tracking
        move: Function to move stage to absolute position [x, y, z]
        step_size_um: Step size in microns (50-200 recommended)
        pattern: Movement pattern - "cross" or "grid"
        n_steps: Number of steps in each direction
        settle_time: Time to wait after each move for mechanical settling
        logger: Optional logger
    
    Returns:
        Dictionary containing:
        - affine_matrix: 2x3 transformation matrix
        - metrics: Quality metrics
        - pixel_displacements: Measured pixel shifts
        - stage_displacements: Commanded stage shifts
        - inlier_mask: Boolean mask of valid measurements
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting affine calibration with {pattern} pattern, step={step_size_um}µm")
    
    # Ensure tracker has a template
    try:
        _ = tracker.template
    except:
        tracker.acquire_template()
    
    # Record starting position
    tracker.reset_history()
    starting_position = tracker.get_position()
    reference_image = tracker.grab_image()
    
    # Generate movement pattern
    if pattern == "cross":
        # Cross pattern: center + 4 cardinal directions
        offsets = [
            (0, 0),  # center (reference)
            (step_size_um, 0),   # +X
            (0, step_size_um),   # +Y
            (-step_size_um, 0),  # -X
            (0, -step_size_um),  # -Y
        ]
        if n_steps > 1:
            # Add diagonal points for better conditioning
            offsets.extend([
                (step_size_um, step_size_um),
                (step_size_um, -step_size_um),
                (-step_size_um, step_size_um),
                (-step_size_um, -step_size_um),
            ])
    elif pattern == "grid":
        # Grid pattern
        half_range = n_steps // 2
        offsets = []
        for i in range(-half_range, half_range + 1):
            for j in range(-half_range, half_range + 1):
                offsets.append((i * step_size_um, j * step_size_um))
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Collect measurements
    pixel_displacements = []
    stage_displacements = []
    correlation_values = []
    
    try:
        for i, (dx, dy) in enumerate(offsets):
            target_pos = starting_position + np.array([dx, dy, 0])
            move(target_pos)
            time.sleep(settle_time)
            
            # Capture image and compute displacement
            current_image = tracker.grab_image()
            
            # Use phase correlation for sub-pixel accuracy
            shift, corr_value = compute_displacement_phase_correlation(
                reference_image, current_image
            )
            
            # shift is [dy, dx] in pixels, convert to [dx, dy]
            pixel_shift = np.array([shift[1], shift[0]])
            pixel_displacements.append(pixel_shift)
            stage_displacements.append([dx, dy])
            correlation_values.append(corr_value)
            
            logger.debug(f"Move {i+1}/{len(offsets)}: stage=({dx:.1f},{dy:.1f})µm, "
                        f"pixel=({pixel_shift[0]:.2f},{pixel_shift[1]:.2f})px, "
                        f"corr={corr_value:.3f}")
    
    finally:
        # Always return to starting position
        move(starting_position)
    
    # Convert to arrays
    pixel_displacements = np.array(pixel_displacements)
    stage_displacements = np.array(stage_displacements)
    correlation_values = np.array(correlation_values)
    
    # Compute affine transformation with outlier rejection
    affine_matrix, inlier_mask, metrics = robust_affine_from_correspondences(
        pixel_displacements,
        stage_displacements,
        logger=logger
    )
    
    # Add correlation metrics
    metrics["mean_correlation"] = np.mean(correlation_values[inlier_mask])
    metrics["min_correlation"] = np.min(correlation_values[inlier_mask])
    
    # Determine calibration quality
    if metrics["rmse_um"] < 1.0 and metrics["mean_correlation"] > 0.5:
        quality = "excellent"
    elif metrics["rmse_um"] < 2.0 and metrics["mean_correlation"] > 0.3:
        quality = "good"
    elif metrics["rmse_um"] < 5.0:
        quality = "acceptable"
    else:
        quality = "poor"
    
    metrics["quality"] = quality
    logger.info(f"Calibration quality: {quality}")
    
    return {
        "affine_matrix": affine_matrix,
        "metrics": metrics,
        "pixel_displacements": pixel_displacements,
        "stage_displacements": stage_displacements,
        "correlation_values": correlation_values,
        "inlier_mask": inlier_mask,
        "starting_position": starting_position,
        "timestamp": time.time()
    }


def apply_affine_transform(
    affine_matrix: np.ndarray,
    pixel_coords: np.ndarray
) -> np.ndarray:
    """
    Apply affine transformation to pixel coordinates.
    
    Args:
        affine_matrix: 2x3 affine matrix [A | b]
        pixel_coords: Nx2 array of pixel coordinates or 1D array of length 2
    
    Returns:
        Transformed coordinates in stage units (microns)
    """
    if pixel_coords.ndim == 1:
        pixel_coords = pixel_coords.reshape(1, -1)
    
    # Extract A and b from affine matrix
    A = affine_matrix[:, :2]
    b = affine_matrix[:, 2]
    
    # Apply transformation: stage = pixel @ A.T + b
    stage_coords = pixel_coords @ A.T + b
    
    return stage_coords


def validate_calibration(
    affine_matrix: np.ndarray,
    metrics: Dict,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str]:
    """
    Validate calibration quality and return pass/fail status.
    
    Args:
        affine_matrix: 2x3 affine matrix
        metrics: Dictionary with calibration metrics
        logger: Optional logger
    
    Returns:
        Tuple of (is_valid, message)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    issues = []
    
    # Check RMSE
    if metrics["rmse_um"] > 5.0:
        issues.append(f"High RMSE: {metrics['rmse_um']:.2f}µm (threshold: 5.0µm)")
    
    # Check correlation quality
    if metrics.get("mean_correlation", 0) < 0.2:
        issues.append(f"Low correlation: {metrics.get('mean_correlation', 0):.3f} (threshold: 0.2)")
    
    # Check condition number (numerical stability)
    if metrics.get("condition_number", 0) > 100:
        issues.append(f"Poor conditioning: {metrics.get('condition_number', 0):.1f} (threshold: 100)")
    
    # Check for too many outliers
    n_total = metrics["n_inliers"] + metrics["n_outliers"]
    outlier_ratio = metrics["n_outliers"] / n_total if n_total > 0 else 0
    if outlier_ratio > 0.3:
        issues.append(f"Too many outliers: {outlier_ratio*100:.1f}% (threshold: 30%)")
    
    # Check scale consistency (x and y should be similar for square pixels)
    scale_ratio = metrics["scale_x_um_per_pixel"] / metrics["scale_y_um_per_pixel"]
    if scale_ratio < 0.8 or scale_ratio > 1.25:
        issues.append(f"Anisotropic scaling: ratio={scale_ratio:.3f} (expected ~1.0)")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        message = "Calibration passed all validation checks"
        logger.info(message)
    else:
        message = "Calibration validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
        logger.warning(message)
    
    return is_valid, message
