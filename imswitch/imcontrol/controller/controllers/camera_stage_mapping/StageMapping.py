"""
Stage-to-camera calibration using affine transformation.

This module provides robust automated calibration for mapping stage coordinates
to camera pixel coordinates using full 2×3 affine transformations.
"""

from imswitch.imcontrol.controller.controllers.camera_stage_mapping.camera_stage_tracker import Tracker
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.closed_loop_move import closed_loop_move, closed_loop_scan
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.scan_coords_times import ordered_spiral
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.affine_stage_calibration import (
    calibrate_affine_transform, validate_calibration, apply_affine_transform
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

    def camera_stage_functions(self):
        """Return functions that allow us to interface with the microscope"""

        def grabCroppedFrame(crop_size=512):
            if self._is_client:
                marray = self._client.recordingManager.snapNumpyToFastAPI()
            else:
                marray = self.microscopeDetector.getLatestFrame()
            #marray = self._client.detector.getLatestFrame()
            center_x, center_y = marray.shape[1] // 2, marray.shape[0] // 2

            # Calculate the starting and ending indices for cropping
            x_start = center_x - crop_size // 2
            x_end = x_start + crop_size
            y_start = center_y - crop_size // 2
            y_end = y_start + crop_size

            # Crop the center region
            return marray[y_start:y_end, x_start:x_end]

        def getPositionList():
            if self._is_client:
                positioner_names = self._client.positionersManager.getAllDeviceNames()
                positioner_name = positioner_names[0]
                posDict = self._client.positionersManager.getPositionerPositions()[positioner_name]
            else:
                posDict = self.microscopeStage.getPosition()
            return (SIGN_AXES[STAGE_ORDER[0]]*posDict["X"]/self._micronToPixel, SIGN_AXES[STAGE_ORDER[1]]*posDict["Y"]/self._micronToPixel, SIGN_AXES[STAGE_ORDER[2]]*posDict["Z"]/self._micronToPixel)

        def movePosition(posList):

            if self._is_client:
                positioner_names = self._client.positionersManager.getAllDeviceNames()
                positioner_name = positioner_names[0]
                self._client.positionersManager.movePositioner(positioner_name, dist=SIGN_AXES[STAGE_ORDER[0]]*posList[0]*self._micronToPixel, axis=self._stageOrder[0], is_absolute=True, is_blocking=True)
                time.sleep(.1)
                self._client.positionersManager.movePositioner(positioner_name, dist=SIGN_AXES[STAGE_ORDER[1]]*posList[1]*self._micronToPixel, axis=self._stageOrder[1], is_absolute=True, is_blocking=True)
            else:
                self.microscopeStage.move(value=SIGN_AXES[STAGE_ORDER[0]]*posList[0]*self._micronToPixel, axis=self._stageOrder[0], is_absolute=True, is_blocking=True)
                self.microscopeStage.move(value=SIGN_AXES[STAGE_ORDER[1]]*posList[1]*self._micronToPixel, axis=self._stageOrder[1], is_absolute=True, is_blocking=True)

            if len(posList)>2:
                if self._is_client:
                    self._client.positionersManager.movePositioner(positioner_name, dist=SIGN_AXES[STAGE_ORDER[2]]*posList[2]*self._micronToPixel, axis=self._stageOrder[2], is_absolute=True, is_blocking=True)
                else:
                    self.microscopeStage.move(value=SIGN_AXES[STAGE_ORDER[2]]*posList[2]*self._micronToPixel, axis=self._stageOrder[2], is_absolute=True, is_blocking=True)

        def settle(tWait=.1):
            time.sleep(tWait)
        grab_image = grabCroppedFrame
        get_position = getPositionList
        move = movePosition
        wait = settle

        return grab_image, get_position, move, wait

    def calibrate_affine(
        self,
        objective_id: str = "default",
        step_size_um: float = 100.0,
        pattern: str = "cross",
        n_steps: int = 4,
        auto_exposure: bool = True,
        validate: bool = True
    ):
        """
        Perform robust affine calibration.
        
        Uses phase correlation for sub-pixel accuracy, computes full 2×3 affine
        transformation, includes outlier detection and validation, and stores
        per-objective calibration data.
        
        Args:
            objective_id: Identifier for the objective being calibrated
            step_size_um: Step size in microns (50-200 recommended)
            pattern: Movement pattern - "cross" or "grid"
            n_steps: Number of steps in each direction
            auto_exposure: Whether to automatically adjust exposure
            validate: Whether to validate the calibration
        
        Returns:
            Dictionary with calibration results
        """
        self._logger.info(f"Starting affine calibration for objective '{objective_id}'")
        
        # Get camera and stage interface functions
        grab_image, get_position, move, wait = self.camera_stage_functions()
        
        # Auto-adjust exposure if requested
        if auto_exposure:
            self._logger.info("Auto-adjusting exposure...")
            try:
                # Create a simple set_exposure function
                # Note: This is a placeholder - actual implementation depends on camera API
                def set_exposure(exposure_ms):
                    self._logger.info(f"Setting exposure to {exposure_ms:.1f}ms")
                    # TODO: Implement actual exposure setting via camera API
                    pass
                
                # For now, skip auto-exposure if we can't set it
                self._logger.info("Auto-exposure not implemented for this camera, using current settings")
            except Exception as e:
                self._logger.warning(f"Could not adjust exposure: {e}")
        
        # Create tracker
        tracker = Tracker(grab_image, get_position, settle=wait)
        
        # Perform calibration
        try:
            result = calibrate_affine_transform(
                tracker=tracker,
                move=move,
                step_size_um=step_size_um,
                pattern=pattern,
                n_steps=n_steps,
                settle_time=0.2,
                logger=self._logger
            )
            
            # Validate calibration if requested
            if validate:
                is_valid, message = validate_calibration(
                    result["affine_matrix"],
                    result["metrics"],
                    logger=self._logger
                )
                result["validation"] = {
                    "is_valid": is_valid,
                    "message": message
                }
                
                if not is_valid:
                    self._logger.warning("Calibration validation failed but data will still be saved")
            
            # Store calibration data
            objective_info = {
                "name": objective_id,
                "effective_pixel_size_um": self._effPixelsize,
                "stage_step_size_um": self._stageStepSize
            }
            
            self._calibration_storage.save_calibration(
                objective_id=objective_id,
                affine_matrix=result["affine_matrix"],
                metrics=result["metrics"],
                objective_info=objective_info
            )
            
            self._logger.info(f"Affine calibration completed for objective '{objective_id}'")
            return result
            
        except Exception as e:
            self._logger.error(f"Affine calibration failed: {e}")
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

    def closed_loop_move_in_image_coordinates(self, displacement_in_pixels, objective_id: str = "default", **kwargs):
        """Move by a given number of pixels on the camera, using the camera as an encoder."""
        grab_image, get_position, move, wait = self.camera_stage_functions()

        tracker = Tracker(grab_image, get_position, settle=wait)
        tracker.acquire_template()
        
        # Create a move function that uses affine transformation
        def move_func(disp):
            self.move_in_image_coordinates_affine(disp, objective_id)
        
        closed_loop_move(tracker, move_func, displacement_in_pixels, **kwargs)

    def closed_loop_scan(self, scan_path, objective_id: str = "default", **kwargs):
        """
        Perform closed-loop moves to each point defined in scan_path.

        This returns a generator, which will move the stage to each point in
        ``scan_path``, then yield ``i, pos`` where ``i``
        is the index of the scan point, and ``pos`` is the estimated position
        in pixels relative to the starting point.  To use it properly, you
        should iterate over it, for example::

            for i, pos in stage_mapping.closed_loop_scan(scan_path):
                capture_image(f"image_{i}.jpg")

        ``scan_path`` should be an Nx2 numpy array defining
        the points to visit in pixels relative to the current position.

        If an exception occurs during the scan, we automatically return to the
        starting point.  Keyword arguments are passed to
        ``closed_loop_move.closed_loop_scan``.
        
        Args:
            scan_path: Nx2 array of pixel coordinates
            objective_id: Identifier for the objective to use
        """
        grab_image, get_position, move, wait = self.camera_stage_functions()

        tracker = Tracker(grab_image, get_position, settle=wait)
        tracker.acquire_template()
        
        # Create a move function that uses affine transformation
        def move_func(disp):
            self.move_in_image_coordinates_affine(disp, objective_id)

        # Get actual stage move function
        _, _, stage_move, _ = self.camera_stage_functions()
        
        return closed_loop_scan(tracker, move_func, stage_move, np.array(scan_path), **kwargs)

    def test_closed_loop_spiral_scan(self, step_size, N, objective_id: str = "default", **kwargs):
        """
        Test closed-loop spiral scan.
        
        Args:
            step_size: Step size for spiral
            N: Number of spiral steps
            objective_id: Identifier for the objective to use
        """
        scan_path = ordered_spiral(0, 0, N, *step_size)

        for i, pos in self.closed_loop_scan(np.array(scan_path), objective_id=objective_id, **kwargs):
            pass

