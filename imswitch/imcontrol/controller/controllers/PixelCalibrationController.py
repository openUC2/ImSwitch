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
        
        # Get current objective from ObjectiveController if available
        self.currentObjective = self._getCurrentObjectiveId()
        self._loadAffineCalibrations()
    
    def _getCurrentObjectiveId(self) -> str:
        """
        Get the current objective ID from ObjectiveManager.
        
        The ObjectiveManager maintains the authoritative state for the current
        objective slot, which is synchronized by the ObjectiveController.
        
        Returns:
            Objective ID string (e.g., "10x", "20x") or "default" if not available
        """
        try:
            obj_mgr = self._master.objectiveManager
            
            # Use the manager's built-in method if available
            if hasattr(obj_mgr, 'getCurrentObjectiveID'):
                return obj_mgr.getCurrentObjectiveID()
            
            # Fallback: manual lookup
            if hasattr(obj_mgr, 'objectiveNames') and obj_mgr.objectiveNames:
                current_slot = getattr(obj_mgr, '_currentObjective', None)
                
                if current_slot is not None and current_slot > 0:
                    idx = current_slot - 1  # 1-based to 0-based
                    if idx < len(obj_mgr.objectiveNames):
                        return obj_mgr.objectiveNames[idx]
                
                # Return first objective as default
                return obj_mgr.objectiveNames[0]
        
        except Exception as e:
            self._logger.debug(f"Could not read current objective: {e}")
            return "default"

    def _loadAffineCalibrations(self):
        """
        Load affine calibrations from setup configuration on startup and distribute to components.
        
        This method reads calibration data from ImSwitchConfig/imcontrol_setups/XXX_config.json and makes it available
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
        
        # Note: Currently storing separate affine matrices per objective to support
        # potential per-objective misalignment (e.g., objective turret rotation).
        # Rotation/flip is typically the same, only scaling changes between objectives (> pixelsize!)
        # Future optimization: Store single rotation/flip + per-objective scale factors.
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
            # Store affine matrix for coordinate transformations (used by getAffineMatrix API)
            affine_matrix = np.array(calib_data.get('affine_matrix', [[1, 0, 0], [0, 1, 0]]))
            metrics = calib_data.get('metrics', {})
            
            # Extract metrics from affine matrix for distribution to components
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
        Distribute calibrated pixel sizes to ObjectiveManager configuration.
        
        This updates the pixelsizes in ObjectiveManager which is the source of truth
        for objective configuration. The ObjectiveController reads from ObjectiveManager
        and will pick up these changes.
        
        Note: For runtime updates after initialization, use the API endpoint:
        objectiveController.setObjectiveParameters(objectiveSlot, pixelsize=value)
        
        Args:
            objective_pixelsizes_map: Dictionary mapping objective IDs to pixel sizes
        """
        if not hasattr(self._master, 'objectiveManager'):
            self._logger.debug("No objectiveManager available - skipping pixel size distribution")
            return
        
        obj_mgr = self._master.objectiveManager
        
        # Get objective names (property returns copy, safe to check)
        objective_names = obj_mgr.objectiveNames
        if not objective_names:
            self._logger.debug("No objective names in ObjectiveManager - skipping pixel size distribution")
            return
        
        # Get current pixel sizes (property returns copy)
        current_pixelsizes = obj_mgr.pixelsizes
        if not current_pixelsizes:
            self._logger.debug("No pixelsizes in ObjectiveManager - skipping pixel size distribution")
            return
        
        # Update pixel sizes in ObjectiveManager based on objective names
        updated_count = 0
        for objective_id, pixel_size in objective_pixelsizes_map.items():
            # Try to find matching objective by name
            try:
                if objective_id in objective_names:
                    idx = objective_names.index(objective_id)
                    slot = idx + 1  # Convert to 1-based slot
                    old_value = current_pixelsizes[idx]
                    
                    # Update through manager method
                    obj_mgr.setObjectiveParameters(slot, pixelsize=pixel_size, emitSignal=True)
                    
                    self._logger.info(f"Updated ObjectiveManager pixelsize for '{objective_id}' (slot {slot}): "
                                    f"{old_value:.3f} → {pixel_size:.3f} µm/px")
                    updated_count += 1
                elif objective_id == "default" and len(objective_names) > 0:
                    # Apply to first objective
                    slot = 1
                    old_value = current_pixelsizes[0]
                    
                    # Update through manager method
                    obj_mgr.setObjectiveParameters(slot, pixelsize=pixel_size, emitSignal=False)
                    
                    self._logger.info(f"Updated ObjectiveManager pixelsize for default objective '{objective_names[0]}' "
                                    f"(slot {slot}): {old_value:.3f} → {pixel_size:.3f} µm/px")
                    updated_count += 1
            except Exception as e:
                self._logger.warning(f"Could not update pixelsize for objective '{objective_id}': {e}")
        
        if updated_count > 0:
            self._logger.info(f"Successfully updated {updated_count} pixel size(s) in ObjectiveManager")
            
            # If ObjectiveController exists, it will read from ObjectiveManager on next update
            if hasattr(self._master, 'objectiveController'):
                obj_ctrl = self._master.objectiveController
                # Sync the controller's copy with the manager's data
                obj_ctrl.pixelsizes = obj_mgr.pixelsizes
                self._logger.debug("Synchronized ObjectiveController pixelsizes with ObjectiveManager")
    
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
        
        # Filter the acquisition camera
        for detector_name in all_detectors:
            if self._master.detectorsManager[detector_name]._DetectorManager__forAcquisition:
                break 
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
        
        This method:
        1. Updates the internal tracking of current objective
        2. Attempts to physically move the objective turret if ObjectiveController is available
        3. Updates the ObjectiveManager state if available
        4. Applies the corresponding calibration data (pixel size, flip settings)
        
        Args:
            objective_id: Identifier for the objective to activate (e.g., "10x", "20x")
        """
        # Update local tracking
        old_objective = self.currentObjective
        self.currentObjective = objective_id
        self._logger.info(f"Switched to objective '{objective_id}' (was '{old_objective}')")
        
        # Request objective change via communication channel
        # This allows decoupled communication between controllers
        try:
            #self._commChannel.sigSetObjectiveByName.emit(objective_id)
            self._commChannel.sigSetObjectiveByID.emit(objective_id)
            self._logger.debug(f"Emitted signal to change objective to '{objective_id}'")
        except Exception as e:
            self._logger.warning(f"Could not emit objective change signal: {e}")
        
        # Apply calibration data for this objective if available
        if objective_id in self.affineCalibrations:
            scale_x, scale_y = self.getPixelSize(objective_id)
            self._logger.info(f"Calibration loaded: pixel size = ({scale_x:.3f}, {scale_y:.3f}) µm/px")
            
            # Apply the calibration results to update detector flip and objective parameters
            calib_data = self.affineCalibrations[objective_id]
            result = {
                "affine_matrix": calib_data.get('affine_matrix', [[1, 0, 0], [0, 1, 0]]),
                "metrics": calib_data.get('metrics', {})
            }
            self._applyCalibrationResults(objective_id, result)
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
            
            # Validate camera intensity before calibration
            if not self._validateCameraIntensity():
                return {
                    "error": "Camera intensity out of range (saturated or too dark). Adjust exposure or lighting before calibration.",
                    "success": False
                }
            # TODO: I think we should put this into a seperate thread, otherwise it may block the API
            pixelcalibration_helper = PixelCalibrationClass(self)

            # Perform the calibration
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
            
            # Apply calibration results immediately without file reload
            self._applyCalibrationResults(objectiveId, result_serializable)
            
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
        
        Reads directly from configuration file on disk.
        
        Returns:
            Dictionary with list of objective identifiers that have been calibrated
        """
        try:
            # Reload setup info from disk to get fresh data
            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            
            # Load setup from file
            setupInfo = configfiletools.loadSetupInfo(options, self._setupInfo.__class__)
            
            if not hasattr(setupInfo, 'PixelCalibration') or setupInfo.PixelCalibration is None:
                return {"success": True, "objectives": []}
            
            if not hasattr(setupInfo.PixelCalibration, 'affineCalibrations'):
                return {"success": True, "objectives": []}
            
            objectives = list(setupInfo.PixelCalibration.affineCalibrations.keys())
            return {"success": True, "objectives": objectives}
            
        except Exception as e:
            self._logger.error(f"Failed to get calibrated objectives: {e}")
            return {"error": str(e), "success": False}
    
    @APIExport()
    def getCalibrationData(self, objectiveId: str = "default"):
        """
        Get calibration data for a specific objective.
        
        Reads directly from configuration file on disk.
        
        Args:
            objectiveId: Identifier for the objective
            
        Returns:
            Dictionary with calibration data including affine matrix and metrics
        """
        try:
            # Reload setup info from disk to get fresh data
            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            
            # Load setup from file
            setupInfo = configfiletools.loadSetupInfo(options, self._setupInfo.__class__)
            
            if not hasattr(setupInfo, 'PixelCalibration') or setupInfo.PixelCalibration is None:
                return {"error": f"No PixelCalibration configuration found", "success": False}
            
            if not hasattr(setupInfo.PixelCalibration, 'affineCalibrations'):
                return {"error": f"No affine calibrations found", "success": False}
            
            if objectiveId not in setupInfo.PixelCalibration.affineCalibrations:
                return {"error": f"No calibration found for objective '{objectiveId}'", "success": False}
            
            # Get calibration data
            calib_data = setupInfo.PixelCalibration.affineCalibrations[objectiveId]
            
            # Convert to serializable dictionary
            result = {
                "success": True,
                "objectiveId": objectiveId,
                "affineMatrix": calib_data.get("affine_matrix", [[1, 0, 0], [0, 1, 0]]),
                "metrics": calib_data.get("metrics", {}),
                "timestamp": calib_data.get("timestamp", "unknown"),
                "objectiveInfo": calib_data.get("objective_info", {})
            }
            
            return result
            
        except Exception as e:
            self._logger.error(f"Failed to get calibration data: {e}")
            return {"error": str(e), "success": False}
    
    @APIExport(requestType="POST")
    def setCalibrationData(self, objectiveId: str, affineMatrix: list, 
                          metrics: dict = None, objectiveInfo: dict = None):
        """
        Set calibration data for a specific objective.
        
        Args:
            objectiveId: Identifier for the objective
            affineMatrix: 2x3 affine transformation matrix as nested list: [[a, b, c], [d, e, f]]
            metrics: Optional dictionary with calibration metrics: 
            objectiveInfo: Optional dictionary with objective information
            
        Returns:
            Success status
        """
        try:
            # Validate affine matrix format
            if not isinstance(affineMatrix, list):
                return {"error": "affineMatrix must be a list", "success": False}
            
            if len(affineMatrix) != 2:
                return {"error": "affineMatrix must have 2 rows", "success": False}
            
            for row in affineMatrix:
                if not isinstance(row, list) or len(row) != 3:
                    return {"error": "affineMatrix rows must have 3 elements", "success": False}
            
            # Convert to float to ensure correct data types
            try:
                affineMatrix = [[float(val) for val in row] for row in affineMatrix]
            except (ValueError, TypeError) as e:
                return {"error": f"affineMatrix contains non-numeric values: {e}", "success": False}
            
            # Create calibration data dictionary
            calibration_data = {
                "affine_matrix": affineMatrix,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": metrics if metrics is not None else {},
                "objective_info": objectiveInfo if objectiveInfo is not None else {}
            }
            
            # Update setup info
            self._setupInfo.setAffineCalibration(objectiveId, calibration_data)
            
            # Save to disk
            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
            
            self._logger.info(f"Calibration data saved for objective '{objectiveId}'")
            
            # Apply calibration immediately
            result = {
                "affine_matrix": affineMatrix,
                "metrics": metrics if metrics is not None else {}
            }
            self._applyCalibrationResults(objectiveId, result)
            
            return {
                "success": True,
                "objectiveId": objectiveId,
                "message": f"Calibration data saved for '{objectiveId}'"
            }
            
        except Exception as e:
            self._logger.error(f"Failed to set calibration data: {e}")
            return {"error": str(e), "success": False}
    
    @APIExport()
    def deleteCalibration(self, objectiveId: str):
        """
        Delete calibration data for a specific objective and reset to default values.
        
        Default values:
        - Pixel size: 1.0 µm/pixel
        - Affine matrix: Identity matrix [[1, 0, 0], [0, 1, 0]]
        
        Args:
            objectiveId: Identifier for the objective
            
        Returns:
            Success status
        """
        try:
            # Check if PixelCalibration exists
            if self._setupInfo.PixelCalibration is None:
                return {
                    "success": False,
                    "objectiveId": objectiveId,
                    "message": f"No PixelCalibration configuration found"
                }
            
            # Check if calibration exists
            if not hasattr(self._setupInfo.PixelCalibration, 'affineCalibrations') or \
               objectiveId not in self._setupInfo.PixelCalibration.affineCalibrations:
                return {
                    "success": False,
                    "objectiveId": objectiveId,
                    "message": f"No calibration found for '{objectiveId}'"
                }
            
            # Delete from setup info
            del self._setupInfo.PixelCalibration.affineCalibrations[objectiveId]
            
            # Save to disk
            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
            
            self._logger.info(f"Calibration deleted for objective '{objectiveId}'")
            
            # Apply default calibration (identity matrix, pixel size = 1.0)
            default_result = {
                "affine_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "metrics": {
                    "scale_x_um_per_pixel": 1.0,
                    "scale_y_um_per_pixel": 1.0,
                    "rotation_deg": 0.0
                }
            }
            self._applyCalibrationResults(objectiveId, default_result)
            
            return {
                "success": True,
                "objectiveId": objectiveId,
                "message": f"Calibration deleted for '{objectiveId}' and reset to default values (pixelsize=1.0, identity matrix)"
            }
            
        except Exception as e:
            self._logger.error(f"Failed to delete calibration: {e}")
            return {"error": str(e), "success": False}


    def _validateCameraIntensity(self) -> bool:
        """
        Validate that camera intensity is in a reasonable range for calibration.
        
        Returns:
            True if intensity is acceptable, False if saturated or too dark
        """
        try:
            if not hasattr(self._master, 'detectorsManager'):
                self._logger.warning("No detectorsManager available - skipping intensity check")
                return True
            
            all_detectors = self._master.detectorsManager.getAllDeviceNames()
            if not all_detectors:
                self._logger.warning("No detectors found - skipping intensity check")
                return True
            
            detector = self._master.detectorsManager[all_detectors[0]]
            
            # Get current frame
            if hasattr(detector, 'getLatestFrame'):
                frame = detector.getLatestFrame()
            elif hasattr(detector, '_camera') and hasattr(detector._camera, 'getLatestFrame'):
                frame = detector._camera.getLatestFrame()
            else:
                self._logger.warning("Cannot access camera frame - skipping intensity check")
                return True
            
            if frame is None:
                self._logger.warning("No frame available - skipping intensity check")
                return True
            
            # Check intensity range
            max_val = np.max(frame)
            mean_val = np.mean(frame)
            
            # Determine bit depth (assume 8-bit, 12-bit, or 16-bit)
            if frame.dtype == np.uint8:
                saturation_threshold = 250
                min_threshold = 10
            elif frame.dtype == np.uint16:
                # Check if it's actually 12-bit data in 16-bit container
                if max_val < 4096:
                    saturation_threshold = 4000
                    min_threshold = 50
                else:
                    saturation_threshold = 64000
                    min_threshold = 500
            else:
                # Unknown type, assume normalized
                saturation_threshold = 0.98
                min_threshold = 0.02
            
            # Check for saturation
            if max_val >= saturation_threshold:
                self._logger.warning(f"Camera saturated (max={max_val}, threshold={saturation_threshold})")
                return False
            
            # Check for too dark
            if mean_val <= min_threshold:
                self._logger.warning(f"Camera too dark (mean={mean_val}, threshold={min_threshold})")
                return False
            
            self._logger.info(f"Camera intensity OK: mean={mean_val:.1f}, max={max_val}")
            return True
            
        except Exception as e:
            self._logger.warning(f"Failed to validate camera intensity: {e}")
            # Don't block calibration on validation errors
            return True


    def _applyCalibrationResults(self, objective_id: str, result: dict):
        """
        Apply calibration results immediately without file reload.
        
        This updates the in-memory calibration data and applies flip/pixel size
        to the active detector and objective.
        
        Args:
            objective_id: Objective identifier that was calibrated
            result: Calibration result dictionary with metrics and affine matrix
        """
        try:
            # Update in-memory calibration storage
            if not hasattr(self, 'affineCalibrations') or self.affineCalibrations is None:
                self.affineCalibrations = {}
            
            # Store the calibration data
            metrics = result.get('metrics', {})
            affine_matrix = result.get('affine_matrix', [[1, 0, 0], [0, 1, 0]])
            
            # Update affineCalibrations dict
            if objective_id not in self.affineCalibrations:
                self.affineCalibrations[objective_id] = {}
            
            self.affineCalibrations[objective_id]['affine_matrix'] = affine_matrix
            self.affineCalibrations[objective_id]['metrics'] = metrics
            self.affineCalibrations[objective_id]['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract flip and pixel size from metrics
            scale_x = metrics.get('scale_x_um_per_pixel', 1.0)
            scale_y = metrics.get('scale_y_um_per_pixel', 1.0)
            pixel_size = (abs(scale_x) + abs(scale_y)) / 2.0
            
            flipY = scale_y < 0
            flipX = scale_x < 0
            
            self._logger.info(f"Applying calibration results for '{objective_id}': "
                            f"pixelsize={pixel_size:.3f} µm/px, flip=(Y:{flipY}, X:{flipX})")
            
            # Apply flip to detector immediately
            if hasattr(self._master, 'detectorsManager'):
                all_detectors = self._master.detectorsManager.getAllDeviceNames()
                if all_detectors:
                    detector = self._master.detectorsManager[all_detectors[0]]
                    if hasattr(detector, 'setFlipImage'):
                        detector.setFlipImage(flipY, flipX)
                        self._logger.info(f"Applied flip to detector: Y={flipY}, X={flipX}")
            
            # Update objective pixel size if this is the current objective
            if hasattr(self._master, 'objectiveManager'):
                obj_manager = self._master.objectiveManager
                current_objective_name = obj_manager.getCurrentObjectiveID()
                
                # If calibrated objective matches current objective, update immediately
                if objective_id == current_objective_name or objective_id == "default":
                    current_slot = obj_manager.getCurrentObjective()
                    if current_slot is not None:
                        obj_manager.setObjectiveParameters(current_slot, pixelsize=pixel_size, emitSignal=True)
                        self._logger.info(f"Updated objective slot {current_slot} pixelsize to {pixel_size:.3f} µm/px")
            
            self._logger.info(f"Successfully applied calibration results for '{objective_id}'")
            
        except Exception as e:
            self._logger.error(f"Failed to apply calibration results: {e}", exc_info=True)







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
        for i in range(3): 
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
        objective_id: int = None,
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
        
        if objective_id is None:
            objective_id = self._parent.currentObjective
        else:
            # move to specified objective if possible
            self._parent.setCurrentObjective(objective_id)
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
