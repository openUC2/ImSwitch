"""
Stage-to-camera calibration using affine transformation.

This module provides robust automated calibration for mapping stage coordinates
to camera pixel coordinates using full 2×3 affine transformations.
"""


from imswitch.imcontrol.controller.controllers.camera_stage_mapping.scan_coords_times import ordered_spiral
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.affine_stage_calibration import (
    measure_pixel_shift, compute_affine_matrix, validate_calibration, apply_affine_transform
)
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.calibration_storage import CalibrationStorage
import logging
import time
import numpy as np

# Stage axis configuration
SIGN_AXES = {"X":1, "Y":1, "Z":1}
STAGE_ORDER = ["X", "Y", "Z"]


class StageMappingCalibration(object):
    """
    Stage-to-camera coordinate mapping with affine calibration.
    
    This class manages the calibration and use of affine transformations
    to map between stage coordinates (microns) and camera pixel coordinates.
    """

    def __init__(self, client=None, calibration_file_path="camera_stage_calibration.json", effPixelsize=1.0, stageStepSize=1.0, IS_CLIENT=False, mDetector=None, mStage=None):
        """
        Initialize stage mapping calibration.
        
        Args:
            client: HTTP client or parent class
            calibration_file_path: Path to JSON calibration file
            effPixelsize: Effective pixel size in microns
            stageStepSize: Stage step size in microns
            IS_CLIENT: Whether running as client
            mDetector: Detector/camera object
            mStage: Stage/positioner object
        """
        self._is_client = IS_CLIENT
        self._client = client
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self.isStop = False

        self._effPixelsize = effPixelsize
        self._stageStepSize = stageStepSize
        self._micronToPixel = self._effPixelsize / self._stageStepSize
        self._stageOrder = STAGE_ORDER
        self._calibration_file_path = calibration_file_path

        # Get hold on detector and stage
        self.microscopeDetector = mDetector
        self.microscopeStage = mStage
        
        # Initialize calibration storage for per-objective data
        self._calibration_storage = CalibrationStorage(calibration_file_path, logger=self._logger)

    def stop(self):
        self.isStop = True

    def getIsStop(self):
        return self.isStop

    def _grab_image(self):
        """Capture an image from the camera."""
        if self._is_client:
            return self._client.recordingManager.snapNumpyToFastAPI()
        else:
            return self.microscopeDetector.getLatestFrame()
    
    def _get_stage_position(self):
        """Get current stage position in microns (X, Y, Z)."""
        if self._is_client:
            positioner_names = self._client.positionersManager.getAllDeviceNames()
            positioner_name = positioner_names[0]
            pos_dict = self._client.positionersManager.getPositionerPositions()[positioner_name]
        else:
            pos_dict = self.microscopeStage.getPosition()
        
        # Convert to microns and apply axis configuration
        x = SIGN_AXES["X"] * pos_dict["X"]
        y = SIGN_AXES["Y"] * pos_dict["Y"]
        z = SIGN_AXES["Z"] * pos_dict["Z"]
        return np.array([x, y, z])
    
    def _move_stage(self, position_um):
        """Move stage to absolute position in microns (X, Y, Z)."""
        x, y, z = position_um[0], position_um[1], position_um[2] if len(position_um) > 2 else 0
        
        if self._is_client:
            positioner_names = self._client.positionersManager.getAllDeviceNames()
            positioner_name = positioner_names[0]
            self._client.positionersManager.movePositioner(
                positioner_name, 
                dist=SIGN_AXES["X"] * x, 
                axis=self._stageOrder[0], 
                is_absolute=True, 
                is_blocking=True
            )
            time.sleep(0.1)
            self._client.positionersManager.movePositioner(
                positioner_name, 
                dist=SIGN_AXES["Y"] * y, 
                axis=self._stageOrder[1], 
                is_absolute=True, 
                is_blocking=True
            )
            if len(position_um) > 2:
                self._client.positionersManager.movePositioner(
                    positioner_name, 
                    dist=SIGN_AXES["Z"] * z, 
                    axis=self._stageOrder[2], 
                    is_absolute=True, 
                    is_blocking=True
                )
        else:
            self.microscopeStage.move(
                value=SIGN_AXES["X"] * x, 
                axis=self._stageOrder[0], 
                is_absolute=True, 
                is_blocking=True
            )
            self.microscopeStage.move(
                value=SIGN_AXES["Y"] * y, 
                axis=self._stageOrder[1], 
                is_absolute=True, 
                is_blocking=True
            )
            if len(position_um) > 2:
                self.microscopeStage.move(
                    value=SIGN_AXES["Z"] * z, 
                    axis=self._stageOrder[2], 
                    is_absolute=True, 
                    is_blocking=True
                )

    def calibrate_affine(
        self,
        objective_id: str = "default",
        step_size_um: float = 100.0,
        pattern: str = "cross",
        n_steps: int = 4,
        settle_time: float = 0.2,
        validate: bool = True
    ):
        """
        Perform robust affine calibration.
        
        Direct calibration procedure:
        1. Capture reference image at starting position
        2. Move stage to multiple positions (cross or grid pattern)
        3. Measure pixel displacement at each position using phase correlation
        4. Compute 2×3 affine matrix from correspondences
        5. Validate and store calibration data
        
        Args:
            objective_id: Identifier for the objective being calibrated
            step_size_um: Step size in microns (50-200 recommended)
            pattern: Movement pattern - "cross" (9 points) or "grid" (n×n points)
            n_steps: For grid pattern, number of steps per axis
            settle_time: Seconds to wait after each move
            validate: Whether to validate the calibration
        
        Returns:
            Dictionary with calibration results including affine_matrix and metrics
        """
        self._logger.info(f"Starting affine calibration for objective '{objective_id}'")
        self._logger.info(f"Pattern: {pattern}, step size: {step_size_um}µm")
        
        # Direct calibration without abstractions
        try:
            # 1. Get starting position and reference image
            start_position = self._get_stage_position()
            ref_image = self._grab_image()
            self._logger.info(f"Starting position: {start_position[:2]} µm")
            
            # 2. Generate calibration positions
            if pattern == "cross":
                # Cross: center + 4 cardinal + 4 diagonal = 9 points
                offsets = [
                    (0, 0),
                    (step_size_um, 0), (0, step_size_um), (-step_size_um, 0), (0, -step_size_um),
                    (step_size_um, step_size_um), (step_size_um, -step_size_um),
                    (-step_size_um, step_size_um), (-step_size_um, -step_size_um)
                ]
            elif pattern == "grid":
                # Grid: n_steps × n_steps points
                half = n_steps // 2
                offsets = [(i * step_size_um, j * step_size_um) 
                           for i in range(-half, half + 1) 
                           for j in range(-half, half + 1)]
            else:
                raise ValueError(f"Unknown pattern '{pattern}'. Use 'cross' or 'grid'")
            
            # 3. Move and measure pixel shifts
            pixel_shifts = []
            stage_shifts = []
            correlations = []
            
            for i, (dx, dy) in enumerate(offsets):
                # Move stage
                target_position = start_position + np.array([dx, dy, 0])
                self._move_stage(target_position)
                time.sleep(settle_time)
                
                # Capture image and measure shift
                image = self._grab_image()
                (shift_x, shift_y), corr = measure_pixel_shift(ref_image, image)
                
                pixel_shifts.append([shift_x, shift_y])
                stage_shifts.append([dx, dy])
                correlations.append(corr)
                
                self._logger.debug(f"Point {i+1}/{len(offsets)}: stage=({dx:+.0f},{dy:+.0f})µm, "
                                  f"pixel=({shift_x:+.2f},{shift_y:+.2f})px, corr={corr:.3f}")
            
            # Return to starting position
            self._move_stage(start_position)
            
            # 4. Compute affine transformation matrix
            pixel_shifts = np.array(pixel_shifts)
            stage_shifts = np.array(stage_shifts)
            correlations = np.array(correlations)
            
            affine_matrix, inliers, metrics = compute_affine_matrix(pixel_shifts, stage_shifts)
            
            # Add correlation metrics
            metrics["mean_correlation"] = np.mean(correlations[inliers])
            metrics["min_correlation"] = np.min(correlations[inliers])
            
            # Classify quality
            rmse = metrics["rmse_um"]
            corr_mean = metrics["mean_correlation"]
            if rmse < 1.0 and corr_mean > 0.5:
                quality = "excellent"
            elif rmse < 2.0 and corr_mean > 0.3:
                quality = "good"
            elif rmse < 5.0:
                quality = "acceptable"
            else:
                quality = "poor"
            
            metrics["quality"] = quality
            
            self._logger.info(f"Calibration quality: {quality}, RMSE={rmse:.3f}µm, "
                            f"rotation={metrics['rotation_deg']:.2f}°")
            
            # 5. Validate calibration if requested
            if validate:
                is_valid, message = validate_calibration(affine_matrix, metrics, logger=self._logger)
                validation_result = {
                    "is_valid": is_valid,
                    "message": message
                }
                if not is_valid:
                    self._logger.warning(f"Validation warning: {message}")
            else:
                validation_result = None
            
            # 6. Store calibration data
            objective_info = {
                "name": objective_id,
                "effective_pixel_size_um": self._effPixelsize,
                "stage_step_size_um": self._stageStepSize
            }
            
            self._calibration_storage.save_calibration(
                objective_id=objective_id,
                affine_matrix=affine_matrix,
                metrics=metrics,
                objective_info=objective_info
            )
            
            result = {
                "affine_matrix": affine_matrix,
                "metrics": metrics,
                "quality": quality,
                "pixel_displacements": pixel_shifts,
                "stage_displacements": stage_shifts,
                "correlation_values": correlations,
                "inlier_mask": inliers,
                "starting_position": start_position,
                "timestamp": time.time()
            }
            
            if validation_result:
                result["validation"] = validation_result
            
            self._logger.info(f"Affine calibration completed for objective '{objective_id}'")
            return result
            
        except Exception as e:
            self._logger.error(f"Affine calibration failed: {e}")
            # Try to return to starting position on error
            try:
                self._move_stage(start_position)
            except:
                pass
            raise

    def get_affine_matrix(self, objective_id: str = "default") -> np.ndarray:
        """
        Get the affine transformation matrix for a specific objective.
        
        Args:
            objective_id: Identifier for the objective
        
        Returns:
            2x3 affine transformation matrix
        
        Raises:
            ValueError: If calibration not found
        """
        matrix = self._calibration_storage.get_affine_matrix(objective_id)
        if matrix is None:
            raise ValueError(f"No calibration found for objective '{objective_id}'")
        return matrix

    def list_calibrated_objectives(self) -> list:
        """
        Get list of all objectives with calibration data.
        
        Returns:
            List of objective identifiers
        """
        return self._calibration_storage.list_objectives()

    def move_in_image_coordinates_affine(
        self,
        displacement_in_pixels: np.ndarray,
        objective_id: str = "default"
    ):
        """
        Move by a given number of pixels using affine transformation.
        
        Args:
            displacement_in_pixels: 2D displacement [dx, dy] in pixels
            objective_id: Identifier for the objective to use
        """
        affine_matrix = self.get_affine_matrix(objective_id)
        
        # Apply affine transformation
        stage_displacement = apply_affine_transform(affine_matrix, displacement_in_pixels)
        
        # Convert from microns to stage steps
        stage_displacement_steps = stage_displacement / self._stageStepSize
        
        # Move stage
        if self._is_client:
            positioner_names = self._client.positionersManager.getAllDeviceNames()
            positioner_name = positioner_names[0]
            self._client.positionersManager.movePositioner(
                positioner_name,
                dist=float(stage_displacement_steps[0]),
                axis=self._stageOrder[0],
                is_absolute=False,
                is_blocking=True
            )
            time.sleep(.1)
            self._client.positionersManager.movePositioner(
                positioner_name,
                dist=float(stage_displacement_steps[1]),
                axis=self._stageOrder[1],
                is_absolute=False,
                is_blocking=True
            )
        else:
            self.microscopeStage.move(
                value=float(stage_displacement_steps[0]),
                axis=self._stageOrder[0],
                is_absolute=False,
                is_blocking=True
            )
            self.microscopeStage.move(
                value=float(stage_displacement_steps[1]),
                axis=self._stageOrder[1],
                is_absolute=False,
                is_blocking=True
            )

    @property
    def image_to_stage_displacement_matrix(self):
        """
        Get 2×2 matrix that converts displacement in image coordinates to stage coordinates.
        
        Returns:
            2×2 matrix extracted from the affine transformation (ignoring translation)
        
        Raises:
            ValueError: If no calibration is found
        """
        objectives = self.list_calibrated_objectives()
        if not objectives:
            raise ValueError("The microscope has not yet been calibrated. Please run calibrate_affine() first.")
        
        # Use first available objective
        affine_matrix = self.get_affine_matrix(objectives[0])
        # Return just the 2×2 part (ignore translation component)
        return affine_matrix[:, :2]

    def move_in_image_coordinates(self, displacement_in_pixels, objective_id: str = "default"):
        """
        Move by a given number of pixels on the camera using affine transformation.
        
        Args:
            displacement_in_pixels: 2D displacement [dx, dy] in pixels
            objective_id: Identifier for the objective to use
        """
        # This is just a wrapper that calls move_in_image_coordinates_affine
        self.move_in_image_coordinates_affine(displacement_in_pixels, objective_id)
