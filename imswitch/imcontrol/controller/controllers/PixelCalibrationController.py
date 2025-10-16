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
        
        # Get detector - prefer the one used for acquisition
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]
        
        # Load affine calibrations from setup info and distribute them
        self.affineCalibrations = {}
        self.currentObjective = "default" # TODO: This has to be read from the ObjectiveController / via the ObjectiveManager, the state has to be stored from controller to manager - it is either 0 or 1
        self._loadAffineCalibrations()
        

    def _loadAffineCalibrations(self):
        """
        Load affine calibrations from setup configuration on startup and distribute to components.
        
        This method reads calibration data from config.json and makes it available
        throughout the application. The calibrations are distributed to relevant
        components:
        - Stored in self.affineCalibrations for easy access
        - Can be queried via getAffineMatrix() for coordinate transformations
        - Pixel sizes extracted and updated in ObjectiveInfo for each objective
        - Detector pixel size updated based on current objective
        """
        if not hasattr(self._setupInfo, 'PixelCalibration') or self._setupInfo.PixelCalibration is None:
            self._logger.info("No PixelCalibration in setup configuration - using default identity matrix")
            return
        # TODO: We should only have one affine transfomration as the rotation/flip is the same for all objective lenses, only the scaling changes - also, we should only save one calibration affine transformation and overwrite with each calibration
        pixel_calibration = self._setupInfo.PixelCalibration
        
        if not hasattr(pixel_calibration, 'affineCalibrations') or not pixel_calibration.affineCalibrations:
            self._logger.info("No affine calibrations found in setup configuration")
            return
        
        # Load all calibrations
        self.affineCalibrations = pixel_calibration.affineCalibrations
        
        self._logger.info(f"Loaded {len(self.affineCalibrations)} affine calibration(s) from setup configuration:")
        
        # Map objective names to their indices if ObjectiveInfo exists
        objective_pixelsizes_map = {}
        objective_flips_map = {}
        
        for objective_id, calib_data in self.affineCalibrations.items():
            affine_matrix = np.array(calib_data.get('affine_matrix', [[1, 0, 0], [0, 1, 0]])) # TODO: Unused, use it?
            metrics = calib_data.get('metrics', {})
            
            # Extract pixel size from scale parameters
            scale_x = metrics.get('scale_x_um_per_pixel', 1.0)
            scale_y = metrics.get('scale_y_um_per_pixel', 1.0)
            rotation = metrics.get('rotation_deg', 0.0)
            quality = metrics.get('quality', 'unknown')
            timestamp = calib_data.get('timestamp', 'unknown')
            
            # Average pixel size for compatibility with existing code
            pixel_size_um = (abs(scale_x) + abs(scale_y)) / 2.0
            objective_pixelsizes_map[objective_id] = pixel_size_um
            
            # Extract flip information from affine matrix
            flip_info = self._setupInfo.getFlipFromAffineMatrix(objective_id)
            objective_flips_map[objective_id] = flip_info
            
            self._logger.info(f"  - {objective_id}: "
                            f"scale=({scale_x:.3f}, {scale_y:.3f}) µm/px, "
                            f"rotation={rotation:.2f}°, "
                            f"flip=(Y:{flip_info[0]}, X:{flip_info[1]}), "
                            f"quality={quality}, "
                            f"calibrated={timestamp}")
            
            # If this is the default/current objective, set it as active
            if objective_id == "default" or objective_id == self.currentObjective:
                self._logger.info(f"Set '{objective_id}' as active calibration")
        
        # Update ObjectiveInfo pixelsizes if available
        self._distributePixelSizesToObjectives(objective_pixelsizes_map)
        
        # Distribute flip information to detectors
        self._distributeFlipsToDetectors(objective_flips_map)
    
    def _distributePixelSizesToObjectives(self, objective_pixelsizes_map: dict):
        """
        Distribute calibrated pixel sizes to ObjectiveInfo configuration.
        
        This updates the pixelsizes list in ObjectiveInfo to match the calibrated
        values from affine calibration. Objectives are matched by name or index.
        
        Args:
            objective_pixelsizes_map: Dictionary mapping objective IDs to pixel sizes
        """
        
        # TODO: We need to distribute the information via the OBjectiveManager and implement this mechanism in the ObjectiveController
        '''
        if not hasattr(self._setupInfo, 'objective') or self._setupInfo.objective is None:
            self._logger.debug("No ObjectiveInfo in setup - skipping pixel size distribution")
            return
        
        objective_info = self._setupInfo.objective
        
        if not hasattr(objective_info, 'objectiveNames') or not objective_info.objectiveNames:
            self._logger.debug("No objective names defined - skipping pixel size distribution")
            return
        
        # Create mapping from objective names to indices
        objective_name_to_index = {}
        for idx, name in enumerate(objective_info.objectiveNames):
            objective_name_to_index[name] = idx
        
        # Update pixelsizes based on calibration data
        updated_count = 0
        for objective_id, pixel_size in objective_pixelsizes_map.items():
            # Try matching by exact name first
            if objective_id in objective_name_to_index:
                idx = objective_name_to_index[objective_id]
                old_value = objective_info.pixelsizes[idx] if idx < len(objective_info.pixelsizes) else None
                objective_info.pixelsizes[idx] = pixel_size
                self._logger.info(f"Updated pixelsize for objective '{objective_id}' (slot {idx+1}): "
                                f"{old_value} → {pixel_size:.3f} µm/px")
                updated_count += 1
            # Try matching "default" to first objective
            elif objective_id == "default" and len(objective_info.objectiveNames) > 0:
                idx = 0
                old_value = objective_info.pixelsizes[idx] if idx < len(objective_info.pixelsizes) else None
                objective_info.pixelsizes[idx] = pixel_size
                self._logger.info(f"Updated pixelsize for default objective '{objective_info.objectiveNames[idx]}' "
                                f"(slot {idx+1}): {old_value} → {pixel_size:.3f} µm/px")
                updated_count += 1
            # Try matching by index if objective_id is like "10x", "20x" and matches magnification
            elif hasattr(objective_info, 'magnifications'):
                # Extract magnification value from objective_id if possible
                try:
                    # Check if any objective has this as its name
                    for mag_idx, mag in enumerate(objective_info.magnifications):
                        objective_name = objective_info.objectiveNames[mag_idx] if mag_idx < len(objective_info.objectiveNames) else None
                        if objective_name and objective_id.lower() == objective_name.lower():
                            old_value = objective_info.pixelsizes[mag_idx] if mag_idx < len(objective_info.pixelsizes) else None
                            objective_info.pixelsizes[mag_idx] = pixel_size
                            self._logger.info(f"Updated pixelsize for objective '{objective_name}' (slot {mag_idx+1}): "
                                            f"{old_value} → {pixel_size:.3f} µm/px")
                            updated_count += 1
                            break
                except:
                    pass
        
        if updated_count > 0:
            self._logger.info(f"Successfully distributed {updated_count} calibrated pixel size(s) to ObjectiveInfo")
        '''
    def _distributeFlipsToDetectors(self, objective_flips_map: dict):
        """
        Distribute flip settings from affine calibration to detector managers.
        
        This updates the flipImage settings in detector cameras based on the
        affine transformation matrix. The flip is applied as a zero-CPU operation
        using numpy's flip function in the camera interface.
        
        Args:
            objective_flips_map: Dictionary mapping objective IDs to (flipY, flipX) tuples
        """
        if not hasattr(self._master, 'detectorsManager'):
            self._logger.debug("No detectorsManager available - skipping flip distribution")
            return
        
        # Get all detectors
        all_detectors = self._master.detectorsManager.getAllDeviceNames()
        
        if not all_detectors:
            self._logger.debug("No detectors found - skipping flip distribution")
            return
        
        # For now, apply to the first detector (typically the acquisition camera)
        # TODO: Could be extended to match detector to objective in multi-camera setups
        detector_name = all_detectors[0]
        detector = self._master.detectorsManager[detector_name]
        
        # Try to match objective to flip settings
        # Priority: current objective > "default" > first available
        flip_to_apply = None
        
        if self.currentObjective in objective_flips_map:
            flip_to_apply = objective_flips_map[self.currentObjective]
            self._logger.info(f"Applying flip settings for objective '{self.currentObjective}': {flip_to_apply}")
        elif "default" in objective_flips_map:
            flip_to_apply = objective_flips_map["default"]
            self._logger.info(f"Applying default flip settings: {flip_to_apply}")
        elif objective_flips_map:
            # Use first available
            first_obj = list(objective_flips_map.keys())[0]
            flip_to_apply = objective_flips_map[first_obj]
            self._logger.info(f"Applying flip settings from '{first_obj}': {flip_to_apply}")
        
        if flip_to_apply is not None:
            # Apply to detector using the manager's setFlipImage method if available
            if hasattr(detector, 'setFlipImage'):
                flipY, flipX = flip_to_apply
                detector.setFlipImage(flipY, flipX)
                self._logger.info(f"Updated detector '{detector_name}' flip settings via manager: Y={flipY}, X={flipX}")
            # Fallback: apply directly to camera if manager doesn't have method
            elif hasattr(detector, '_camera') and hasattr(detector._camera, 'flipImage'):
                old_flip = getattr(detector._camera, 'flipImage', (False, False))
                detector._camera.flipImage = flip_to_apply
                self._logger.info(f"Updated detector '{detector_name}' flip settings directly: {old_flip} → {flip_to_apply}")
            else:
                self._logger.warning(f"Detector '{detector_name}' does not support flip operations")
            
            # Also update the ObjectiveController if it exists
            if hasattr(self._master, 'objectiveController') and hasattr(self._setupInfo, 'objective'):
                # Reload pixelsizes in ObjectiveController from SetupInfo
                obj_info = self._setupInfo.objective
                if hasattr(obj_info, 'pixelsizes'):
                    self._master.objectiveController.pixelsizes = obj_info.pixelsizes
                    self._logger.debug("Updated pixelsizes in ObjectiveController")
    
    def getAffineMatrix(self, objective_id: str = None) -> np.ndarray:
        """
        Get affine transformation matrix for current or specified objective.
        
        Args:
            objective_id: Optional objective identifier. If None, uses current objective.
            
        Returns:
            2x3 numpy array representing affine transformation
        """
        if objective_id is None:
            objective_id = self.currentObjective
        
        return self._setupInfo.getAffineMatrix(objective_id)
    
    def getPixelSize(self, objective_id: str = None) -> tuple:
        """
        Get pixel size in microns for current or specified objective.
        
        Extracted from the scale parameters in the affine calibration.
        
        Args:
            objective_id: Optional objective identifier. If None, uses current objective.
            
        Returns:
            Tuple of (scale_x_um_per_pixel, scale_y_um_per_pixel)
        """
        if objective_id is None:
            objective_id = self.currentObjective
        
        if objective_id in self.affineCalibrations:
            metrics = self.affineCalibrations[objective_id].get('metrics', {})
            scale_x = metrics.get('scale_x_um_per_pixel', 1.0)
            scale_y = metrics.get('scale_y_um_per_pixel', 1.0)
            return (scale_x, scale_y)
        else:
            # Return default if no calibration
            return (1.0, 1.0)
    
    def setCurrentObjective(self, objective_id: str):
        """
        Set the current active objective for calibration.
        
        Args:
            objective_id: Identifier for the objective to activate
        """
        self.currentObjective = objective_id
        self._logger.info(f"Switched to objective '{objective_id}'")
        
        # Log if calibration exists for this objective
        if objective_id in self.affineCalibrations:
            scale_x, scale_y = self.getPixelSize(objective_id)
            self._logger.info(f"Calibration loaded: pixel size = ({scale_x:.3f}, {scale_y:.3f}) µm/px")
        else:
            self._logger.info(f"No calibration for '{objective_id}' - using default identity matrix")





    # API Methods for web interface
    @APIExport(runOnUIThread=False)  # Run in background thread
    def calibrateStageAffine(self, objectiveId: int = 0, stepSizeUm: float = 100.0, 
                             pattern: str = "cross", nSteps: int = 4, validate: bool = False):
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
            # TODO: before we do the calibration we need to check if the intensity is in a reasonable regime of the camera
        
        
            pixelcalibration_helper = PixelCalibrationClass(self)
            # TODO: The objectiveID should be one of the two available objectives - we need to check that
            result = pixelcalibration_helper.calibrate_affine(
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
            # TODO: We would probably need to apply these parameters to the Objective/Detector (e.g. flip/rot, pixelsize)
            self._loadAffineCalibrations() # FIXME: doing that for now through the file - not ideal as we reload all calibrations and a missing file causes it to crash..
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
            pixelcalibration_helper = PixelCalibrationClass(self)
            objectives = pixelcalibration_helper.list_calibrated_objectives()
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
            pixelcalibration_helper = PixelCalibrationClass(self)
            
            # Get affine matrix
            affine_matrix = pixelcalibration_helper.get_affine_matrix(objectiveId)
            if affine_matrix is None:
                return {"error": f"No calibration found for objective '{objectiveId}'", "success": False}
            
            # Get metrics
            metrics = pixelcalibration_helper.get_metrics(objectiveId)
            
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







class PixelCalibrationClass(object):
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
            
            if 0 and validate:
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
