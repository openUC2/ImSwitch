import os

import numpy as np
import time
import threading
from imswitch.imcommon.model import initLogger, dirtools, APIExport
from imswitch.imcommon.framework import Signal
import time
from imswitch import IS_HEADLESS
import time
import numpy as np
import NanoImagingPack as nip
import os
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.affine_stage_calibration import (
    measure_pixel_shift, compute_affine_matrix, validate_calibration
)
from ..basecontrollers import LiveUpdatedController

#import NanoImagingPack as nip


class PixelCalibrationController(LiveUpdatedController):
    """Linked to PixelCalibrationWidget."""

    sigImageReceived = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        
        # Get pixel size from setup info or use default
        if hasattr(self._setupInfo, 'PixelCalibration') and self._setupInfo.PixelCalibration:
            # Pixel size might be stored in the calibration info or detector info
            self.pixelSize = 500  # Default, will be updated per objective
        else:
            self.pixelSize = 500  # Default value

        # Get detector - prefer the one used for acquisition
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]


    def snapPreview(self):
        self._logger.info("Snap preview...")
        previewImage = self._master.detectorsManager.execOnCurrent(lambda c: c.getLatestFrame())



    # API Methods for web interface
    
    @APIExport(runOnUIThread=False)  # Run in background thread
    def calibrateStageAffine(self, objectiveId: str = "default", stepSizeUm: float = 100.0, 
                             pattern: str = "cross", nSteps: int = 4, validate: bool = True):
        """
        Perform affine stage-to-camera calibration via API.
        
        This runs in a background thread and can be monitored via signals.
        
        Args:
            objectiveId: Identifier for the objective being calibrated
            stepSizeUm: Step size in microns (50-200 recommended)
            pattern: Movement pattern - "cross" or "grid"
            nSteps: Number of steps in each direction
            validate: Whether to validate the calibration
            
        Returns:
            Dictionary with calibration results including metrics
        """
        try:
            csm_extension = CSMExtension(self)
            
            result = csm_extension.calibrate_affine(
                objective_id=objectiveId,
                step_size_um=stepSizeUm,
                pattern=pattern,
                n_steps=nSteps,
                validate=validate
            )
            
            # Convert all numpy types to Python native types for JSON serialization
            def convert_to_native(obj):
                """Recursively convert numpy types to Python native types."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_to_native(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_to_native(item) for item in obj)
                else:
                    return obj
            
            # Convert result to JSON-serializable format
            result_serializable = convert_to_native(result)
            
            return {
                "success": True,
                "objectiveId": objectiveId,
                "metrics": result_serializable.get("metrics", {}),
                "validation": result_serializable.get("validation", {}),
                "affineMatrix": result_serializable.get("affine_matrix", [])
            }
        except Exception as e:
            self._logger.error(f"API calibration failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}
    
    @APIExport()
    def getCalibrationObjectives(self):
        """
        Get list of all objectives with calibration data.
        
        Returns:
            List of objective identifiers that have been calibrated
        """

        try:
            csm_extension = CSMExtension(self)
            objectives = csm_extension.list_calibrated_objectives()
            return {"success": True, "objectives": objectives}
        except Exception as e:
            self._logger.error(f"Failed to get calibrated objectives: {e}")
            return {"error": str(e), "success": False}
    
    @APIExport()
    def getCalibrationData(self, objectiveId: str = "default"):
        """
        Get calibration data for a specific objective.
        
        Args:
            objectiveId: Identifier for the objective
            
        Returns:
            Dictionary with calibration data including affine matrix and metrics
        """

        try:
            csm_extension = CSMExtension(self)
            
            # Get affine matrix
            affine_matrix = csm_extension.get_affine_matrix(objectiveId)
            if affine_matrix is None:
                return {"error": f"No calibration found for objective '{objectiveId}'", "success": False}
            
            # Get metrics
            metrics = csm_extension.get_metrics(objectiveId)
            
            return {
                "success": True,
                "objectiveId": objectiveId,
                "affineMatrix": affine_matrix.tolist(),
                "metrics": metrics
            }
        except Exception as e:
            self._logger.error(f"Failed to get calibration data: {e}")
            return {"error": str(e), "success": False}
    
    @APIExport()
    def deleteCalibration(self, objectiveId: str):
        """
        Delete calibration data for a specific objective.
        
        Args:
            objectiveId: Identifier for the objective
            
        Returns:
            Success status
        """

        try:
            # Check if calibration exists
            if self._setupInfo.PixelCalibration is None or objectiveId not in self._setupInfo.PixelCalibration.affineCalibrations:
                return {
                    "success": False,
                    "objectiveId": objectiveId,
                    "message": f"No calibration found for '{objectiveId}'"
                }
            
            # Delete from setup info
            del self._setupInfo.PixelCalibration.affineCalibrations[objectiveId]
            
            # Save to disk
            try:
                import imswitch.imcontrol.model.configfiletools as configfiletools
                config_file_path, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(config_file_path, self._setupInfo)
            except Exception as e:
                self._logger.warning(f"Could not save setup configuration: {e}")
            
            return {
                "success": True,
                "objectiveId": objectiveId,
                "message": f"Calibration deleted for '{objectiveId}'"
            }
        except Exception as e:
            self._logger.error(f"Failed to delete calibration: {e}")
            return {"error": str(e), "success": False}







class CSMExtension(object):
    """
    Camera-to-stage mapping calibration.
    
    Performs affine calibration and stores results in the setup configuration.
    The affine matrix is applied by:
    - Camera transformations (rotation/flip) handled in DetectorManager
    - Stage coordinate mapping handled here
    """

    def __init__(self, parent):
        self._parent = parent
        # Use setup info for storage (not separate JSON file)
        # Calibration data is stored in self._parent._setupInfo.PixelCalibration


    def _grab_image(self, crop_size=512):
        """Capture a cropped image from the detector."""
        marray = self._parent.detector.getLatestFrame()
        crop_size = min(crop_size, marray.shape[0], marray.shape[1])
        # Crop the center region
        return np.array(nip.extract(marray, crop_size))

    def _get_stage_position(self):
        """Get current stage position in microns [X, Y, Z]."""
        stage = self._parent._master.positionersManager[self._parent._master.positionersManager.getAllDeviceNames()[0]]
        posDict = stage.getPosition()
        return np.array([posDict["X"], posDict["Y"], posDict["Z"]])

    def _move_stage(self, position_um):
        """Move stage to absolute position in microns [X, Y, Z]."""
        stage = self._parent._master.positionersManager[self._parent._master.positionersManager.getAllDeviceNames()[0]]
        stage.move(value=position_um[0], axis="X", is_absolute=True, is_blocking=True)
        stage.move(value=position_um[1], axis="Y", is_absolute=True, is_blocking=True)
        if len(position_um) > 2:
            stage.move(value=position_um[2], axis="Z", is_absolute=True, is_blocking=True)

    def calibrate_affine(
        self,
        objective_id: str = "default",
        step_size_um: float = 100.0,
        pattern: str = "cross",
        n_steps: int = 4,
        validate: bool = False,
        settle_time: float = 0.2
    ):
        """
        Perform robust affine calibration using direct method calls.
        
        This method provides a straightforward calibration approach:
        - Captures reference image
        - Moves stage to multiple positions
        - Measures pixel shifts using phase correlation
        - Computes full 2x3 affine transformation
        - Validates and stores per-objective calibration data
        
        Args:
            objective_id: Identifier for the objective being calibrated
            step_size_um: Step size in microns (50-200 recommended)
            pattern: Movement pattern - "cross" or "grid"
            n_steps: Number of steps in each direction
            validate: Whether to validate the calibration
            settle_time: Time to wait after stage movement (seconds)
        
        Returns:
            Dictionary with calibration results
        """
        self._parent._logger.info(f"Starting affine calibration for objective '{objective_id}'")
        
        try:
            # 1. Get starting position and capture reference image
            start_position = self._get_stage_position()
            self._parent._logger.info(f"Starting position: {start_position[:2]} µm")
            
            time.sleep(settle_time)
            ref_image = self._grab_image()
            self._parent._logger.info(f"Reference image captured: {ref_image.shape}")
            
            # 2. Generate movement pattern
            if pattern == "cross":
                # Cross pattern: center + 4 cardinal + 4 diagonal = 9 positions
                offsets = [
                    (0, 0),
                    (step_size_um, 0), (0, step_size_um), (-step_size_um, 0), (0, -step_size_um),
                    (step_size_um, step_size_um), (step_size_um, -step_size_um),
                    (-step_size_um, step_size_um), (-step_size_um, -step_size_um)
                ]
            elif pattern == "grid":
                # Grid pattern: n_steps x n_steps
                offsets = []
                half_range = (n_steps - 1) / 2.0
                for i in range(n_steps):
                    for j in range(n_steps):
                        dx = (i - half_range) * step_size_um
                        dy = (j - half_range) * step_size_um
                        offsets.append((dx, dy))
            else:
                raise ValueError(f"Unknown pattern: {pattern}")
            
            self._parent._logger.info(f"Using pattern '{pattern}' with {len(offsets)} positions")
            
            # 3. Move stage and measure pixel shifts
            pixel_shifts = []
            stage_shifts = []
            correlation_values = []
            
            for i, (dx, dy) in enumerate(offsets):
                # Move to target position
                target_position = start_position + np.array([dx, dy, 0])
                self._move_stage(target_position)
                time.sleep(settle_time)
                
                # Capture image
                image = self._grab_image()
                
                # Measure pixel shift using phase correlation
                shift, correlation = measure_pixel_shift(np.array(ref_image), np.array(image))
                
                pixel_shifts.append(shift)
                stage_shifts.append([dx, dy])
                correlation_values.append(correlation)
                
                self._parent._logger.debug(f"Position {i+1}/{len(offsets)}: stage=({dx:.1f}, {dy:.1f}), "
                                          f"pixels=({shift[0]:.2f}, {shift[1]:.2f}), corr={correlation:.3f}")
            
            # 4. Return to starting position
            self._move_stage(start_position)
            self._parent._logger.info("Returned to starting position")
            
            # 5. Compute affine transformation matrix
            pixel_shifts = np.array(pixel_shifts)
            stage_shifts = np.array(stage_shifts)
            correlation_values = np.array(correlation_values)
            
            affine_matrix, inlier_mask, metrics = compute_affine_matrix(
                pixel_shifts, stage_shifts
            )
            
            # Add correlation info to metrics
            metrics["mean_correlation"] = float(np.mean(correlation_values))
            metrics["min_correlation"] = float(np.min(correlation_values))
            
            self._parent._logger.info(f"Calibration quality: {metrics.get('quality', 'unknown')}")
            self._parent._logger.info(f"RMSE: {metrics.get('rmse_um', 0):.3f} µm")
            self._parent._logger.info(f"Rotation: {metrics.get('rotation_deg', 0):.2f}°")
            
            # 6. Validate calibration if requested
            result = {
                "affine_matrix": affine_matrix,
                "metrics": metrics,
                "pixel_displacements": pixel_shifts,
                "stage_displacements": stage_shifts,
                "correlation_values": correlation_values,
                "inlier_mask": inlier_mask,
                "starting_position": start_position
            }
            
            if validate:
                is_valid, message = validate_calibration(
                    affine_matrix, metrics, logger=self._parent._logger
                )
                result["validation"] = {
                    "is_valid": is_valid,
                    "message": message
                }
                
                if not is_valid:
                    self._parent._logger.warning("Calibration validation failed but data will still be saved")
            
            # 7. Store calibration data in setup configuration
            objective_info = {
                "name": objective_id,
                "detector": self._parent.detector._camera.model if hasattr(self._parent.detector._camera, 'model') else "unknown"
            }
            
            calibration_data = {
                "affine_matrix": affine_matrix.tolist(),
                "metrics": {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                           for k, v in metrics.items()},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "objective_info": objective_info
            }
            
            # Save to setup configuration
            self._parent._setupInfo.setAffineCalibration(objective_id, calibration_data)
            
            # Save setup configuration to disk
            try:
                import imswitch.imcontrol.model.configfiletools as configfiletools
                config_file_path, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(config_file_path, self._parent._setupInfo)
                self._parent._logger.info(f"Calibration saved to setup configuration: {config_file_path}")
            except Exception as e:
                self._parent._logger.warning(f"Could not save setup configuration: {e}")
            
            self._parent._logger.info(f"Affine calibration completed for objective '{objective_id}'")
            
            return result
            
        except Exception as e:
            self._parent._logger.error(f"Affine calibration failed: {e}")
            # Try to return to start position
            try:
                self._move_stage(start_position)
            except:
                pass
            raise

    def get_affine_matrix(self, objective_id: str = "default") -> np.ndarray:
        """
        Get the affine transformation matrix for a specific objective.
        Returns default identity matrix if no calibration exists.
        
        Args:
            objective_id: Identifier for the objective
        
        Returns:
            2x3 affine transformation matrix
        """
        return self._parent._setupInfo.getAffineMatrix(objective_id)

    def list_calibrated_objectives(self) -> list:
        """
        Get list of all objectives with calibration data.
        
        Returns:
            List of objective identifiers
        """
        if self._parent._setupInfo.PixelCalibration is None:
            return []
        return list(self._parent._setupInfo.PixelCalibration.affineCalibrations.keys())
    
    def get_metrics(self, objective_id: str = "default"):
        """
        Get calibration metrics for a specific objective.
        
        Args:
            objective_id: Identifier for the objective
            
        Returns:
            Dictionary of metrics or None if not found
        """
        calib = self._parent._setupInfo.getAffineCalibration(objective_id)
        return calib.get("metrics", {}) if calib else {}

    @property
    def image_to_stage_displacement_matrix(self):
        """
        A 2x2 matrix that converts displacement in image coordinates to stage coordinates.
        Returns the default identity matrix if no calibration exists.
        """
        try:
            # Try to get affine calibration
            objectives = self.list_calibrated_objectives()
            if objectives:
                # Use first available objective
                affine_matrix = self.get_affine_matrix(objectives[0])
                # Return just the 2x2 part (ignore translation)
                return affine_matrix[:, :2]
        except Exception as e:
            self._parent._logger.debug(f"Could not get calibrated matrix: {e}")
        
        # Return default identity matrix
        return np.array([[1.0, 0.0], [0.0, 1.0]])
        
    def move_in_image_coordinates(self, displacement_in_pixels):
        """Move by a given number of pixels on the camera"""
        p = np.array(displacement_in_pixels)
        relative_move = np.dot(p, self.image_to_stage_displacement_matrix)
        self.microscope.stage.move_rel([relative_move[0], relative_move[1], 0])





# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
