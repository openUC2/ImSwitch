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

logger = logging.getLogger(__name__)


@dataclass
class FocusConfig:
    """Configuration for focus metric computation."""
    gaussian_sigma: float = 11.0
    background_threshold: int = 40
    crop_radius: int = 300
    enable_gaussian_blur: bool = True
    min_signal_threshold: float = 10.0  # Minimum signal for valid measurement
    max_focus_value: float = 1e6  # Maximum valid focus value


class FocusMetricBase:
    """Base class for focus metrics."""
    
    def __init__(self, config: Optional[FocusConfig] = None):
        self.config = config or FocusConfig()
        logger.debug(f"Focus metric initialized with config: {self.config}")

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
        im = im - np.mean(im) / 2.0
        im[im < self.config.background_threshold] = 0
        
        return im

    def compute(self, frame: np.ndarray) -> Dict[str, Any]:
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

    def compute(self, frame: np.ndarray) -> Dict[str, Any]:
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
            im = self.preprocess_frame(np.array(frame))
            
            # Check for minimum signal
            if np.max(im) < self.config.min_signal_threshold:
                logger.warning("Signal below threshold")
                return {"t": timestamp, "focus": self.config.max_focus_value, "error": "low_signal"}
            
            # Compute projections and fit Gaussians
            projX, projY = self.compute_projections(im)
            sigma_x, sigma_y = self.fit_projections(projX, projY)
            
            # Calculate focus value as ratio
            if sigma_y == 0 or sigma_y < 1e-6:
                focus_value = self.config.max_focus_value
            else:
                focus_value = float(sigma_x / sigma_y)
                
            # Clamp to reasonable range
            focus_value = min(focus_value, self.config.max_focus_value)
            
            logger.debug(f"Astigmatism focus: sigma_x={sigma_x:.3f}, sigma_y={sigma_y:.3f}, focus={focus_value:.4f}")
            
            return {
                "t": timestamp,
                "focus": focus_value,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "signal_max": float(np.max(im)),
                "signal_mean": float(np.mean(im))
            }
            
        except Exception as e:
            logger.error(f"Focus computation failed: {e}")
            return {"t": timestamp, "focus": self.config.max_focus_value, "error": str(e)}


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


class FocusMetricFactory:
    """Factory for creating focus metric instances."""
    
    _metrics = {
        "astigmatism": AstigmatismFocusMetric,
        "center_of_mass": CenterOfMassFocusMetric,
        "gaussian": AstigmatismFocusMetric,  # Alias
        "gradient": CenterOfMassFocusMetric,  # Alias for now
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


# Convenience function for backward compatibility
def create_focus_metric(metric_type: str, **config_kwargs) -> FocusMetricBase:
    """
    Create focus metric with configuration.
    
    Args:
        metric_type: Type of focus metric
        **config_kwargs: Configuration parameters
        
    Returns:
        Focus metric instance
    """
    config = FocusConfig(**config_kwargs) if config_kwargs else None
    return FocusMetricFactory.create(metric_type, config)