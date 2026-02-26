import os
import cv2
import numpy as np
import time
import threading
from fastapi.responses import StreamingResponse
from fastapi import Response, HTTPException
from imswitch.imcommon.model import initLogger, dirtools, APIExport
from imswitch.imcommon.framework import Signal
import NanoImagingPack as nip
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.affine_stage_calibration import (
    measure_pixel_shift, compute_affine_matrix, validate_calibration
)
from imswitch.imcontrol.controller.controllers.pixelcalibration.overview_calibrator import OverviewCalibrator
from imswitch.imcontrol.controller.controllers.pixelcalibration.apriltag_grid_calibrator import (
    AprilTagGridCalibrator, GridConfig
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

        # Load the observation Camera
        if hasattr(self._setupInfo.PixelCalibration, 'ObservationCamera') and self._setupInfo.PixelCalibration.ObservationCamera is not None:
            self.observationCameraName = self._setupInfo.PixelCalibration.ObservationCamera
            # load detector from name
            if self.observationCameraName in allDetectorNames:
                self.observationCamera = self._master.detectorsManager[self.observationCameraName]
            else:
                self.observationCamera = None
                self._logger.warning(f"Observation camera '{self.observationCameraName}' not found among detectors")
        else:
            self.observationCamera = None
            self.observationCameraName = None

        # Get flip settings for observation camera
        self.observationFlipX = True
        self.observationFlipY = True
        if hasattr(self._setupInfo.PixelCalibration, 'ObservationCameraFlip'):
            flip_settings = self._setupInfo.PixelCalibration.ObservationCameraFlip
            if isinstance(flip_settings, dict):
                self.observationFlipX = flip_settings.get('flipX', False)
                self.observationFlipY = flip_settings.get('flipY', False)
            elif isinstance(flip_settings, (list, tuple)) and len(flip_settings) >= 2:
                self.observationFlipY = flip_settings[0]
                self.observationFlipX = flip_settings[1]

        # Initialize overview calibrator with flip settings
        self.overviewCalibrator = OverviewCalibrator(
            logger=self._logger,
            flip_x=self.observationFlipX,
            flip_y=self.observationFlipY
        )

        # Initialize AprilTag grid calibrator
        self.gridCalibrator = None
        self._gridRotated180 = False  # Flag for 180° rotated calibration sample
        self._loadGridCalibration()

        # AprilTag overlay flag for MJPEG stream
        self._aprilTagOverlayEnabled = True
        self._overlay_lock = threading.Lock()

        # Stream state for overview camera
        self.overviewStreamRunning = False
        self.overviewStreamStarted = False
        self.overviewStreamQueue = None

    @APIExport() # return image via fastapi API
    def returnObservationCameraImage(self) -> Response:
        try:
            mFrame = self.observationCamera.getLatestFrame()

            # Apply flip settings
            if self.observationFlipY:
                mFrame = np.flip(mFrame, 0)
            if self.observationFlipX:
                mFrame = np.flip(mFrame, 1)

            from PIL import Image
            import io
            # using an in-memory image
            im = Image.fromarray(mFrame)

            # save image to an in-memory bytes buffer
            # save image to an in-memory bytes buffer
            with io.BytesIO() as buf:
                im = im.convert("L")  # convert image to 'L' mode
                im.save(buf, format="PNG")
                im_bytes = buf.getvalue()

            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            return Response(im_bytes, headers=headers, media_type="image/png")
        except Exception as e:
            self._logger.error(f"Failed to return observation camera image: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving image: {e}")

    def _getCurrentObjectiveId(self) -> str:
        """
        Get the current objective ID (name) from ObjectiveManager.
        
        The ObjectiveManager maintains the authoritative state for the current
        objective slot, which is synchronized by the ObjectiveController.
        
        INDEXING CONVENTION:
        - ObjectiveManager._currentObjective: 0-based (0 or 1)
        - objectiveNames array: 0-indexed
        - Return value: objective name string (e.g., "10x", "20x")
        
        Returns:
            Objective ID string (e.g., "10x", "20x") or "default" if not available
        """
        try:
            obj_mgr = self._master.objectiveManager

            # Use the manager's built-in method for getting objective name
            if hasattr(obj_mgr, 'getCurrentObjectiveName'):
                name = obj_mgr.getCurrentObjectiveName()
                if name and name != "default":
                    return name

            # Fallback: manual lookup using 0-based index
            if hasattr(obj_mgr, 'objectiveNames') and obj_mgr.objectiveNames:
                current_slot_0based = getattr(obj_mgr, '_currentObjective', None)

                if current_slot_0based is not None and 0 <= current_slot_0based < len(obj_mgr.objectiveNames):
                    return obj_mgr.objectiveNames[current_slot_0based]

                # Return first objective as default
                return obj_mgr.objectiveNames[0]

            return "default"

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
                    slot = idx  # Use 0-based index (0 or 1)
                    old_value = current_pixelsizes[idx]

                    # Update through manager method (expects 0-based index)
                    obj_mgr.setObjectiveParameters(slot, pixelsize=pixel_size, emitSignal=True)

                    self._logger.info(f"Updated ObjectiveManager pixelsize for '{objective_id}' (slot {slot}): "
                                    f"{old_value:.3f} → {pixel_size:.3f} µm/px")
                    updated_count += 1
                elif objective_id == "default" and len(objective_names) > 0:
                    # Apply to first objective
                    slot = 0
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

    def setCurrentObjective(self, objective_id: int):
        """
        Set the current active objective for calibration.
        
        This method:
        1. Updates the internal tracking of current objective
        2. Uses ObjectiveController directly to physically move the objective turret
        3. Applies the corresponding calibration data (pixel size, flip settings)
        
        INDEXING CONVENTION:
        - objective_id: 0-based index (0 or 1)
        - ObjectiveController.moveToObjective: expects 0-based slot (0 or 1)
        - ObjectiveManager._currentObjective: 0-based (0 or 1)
        
        Args:
            objective_id: Identifier for the objective to activate (0 or 1)
        """
        # Update local tracking
        old_objective = self.currentObjective
        self.currentObjective = objective_id
        self._logger.info(f"Switched to objective '{objective_id}' (was '{old_objective}')")

        # Use ObjectiveController directly instead of signals for more reliable communication
        try:
            # Get ObjectiveController from main controllers
            obj_ctrl = self._master._controllersRegistry["Objective"]
            self._logger.debug(f"Moving to objective '{objective_id}' (slot {objective_id})")
            obj_ctrl.moveToObjective(objective_id)

        except Exception as e:
            self._logger.warning(f"Could not switch objective via controller: {e}")

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
    @APIExport(runOnUIThread=True)  # Run in background thread
    def calibrateStageAffine(self, objectiveId: int = None, stepSizeUm: float = 100.0,
                             pattern: str = "cross", nSteps: int = 1, validate: bool = False,
                             crop_size: int = 1024, isDEBUG: bool = False):
        """
        Perform affine stage-to-camera calibration via API.
        
        This runs in a background thread and can be monitored via signals.
        
        INDEXING CONVENTION:
        - objectiveId: 0-based slot number (0 or 1) for API consistency
        
        Args:
            objectiveId: Objective slot number (0 or 1, 0-based)
            stepSizeUm: Step size in microns (50-200 recommended)
            pattern: Movement pattern - "cross" or "grid"
            nSteps: Number of steps in each direction
            validate: Whether to validate the calibration
            
        Returns:
            Dictionary with calibration results including metrics
        """

        # Validate 0-based slot input
        if objectiveId not in [0, 1, None]:
            self._logger.error("Invalid objective slot: %s", objectiveId)
            return {"error": "Invalid objective slot", "success": False}

        # Validate camera intensity before calibration
        if not self._validateCameraIntensity():
            return {
                "error": "Camera intensity out of range (saturated or too dark). Adjust exposure or lighting before calibration.",
                "success": False
            }

        mThread = threading.Thread(target=self.calibrateStageAffineInThread, args=(
            objectiveId,
            stepSizeUm,
            pattern,
            nSteps,
            validate,
            crop_size,
            isDEBUG)
        )
        mThread.start()
        return {"success": True, "message": "Calibration started in background thread"}

    def calibrateStageAffineInThread(self, objectiveId: int = 0, stepSizeUm: float = 100.0,
                             pattern: str = "cross", nSteps: int = 4, validate: bool = False,
                             crop_size: int = 1024,
                             isDEBUG: bool = False):
        """
        Perform affine stage-to-camera calibration in a separate thread.
        
        INDEXING CONVENTION:
        - objectiveId: 0-based slot number (0 or 1)
        """
        try:


            pixelcalibration_helper = PixelCalibrationClass(self)

            # Perform the calibration
            result = pixelcalibration_helper.calibrate_affine(
                objective_id=objectiveId,
                step_size_um=stepSizeUm,
                pattern=pattern,
                n_steps=nSteps,
                validate=validate,
                crop_size=crop_size,
                isDEBUG=isDEBUG
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
                return {"error": "No PixelCalibration configuration found", "success": False}

            if not hasattr(setupInfo.PixelCalibration, 'affineCalibrations'):
                return {"error": "No affine calibrations found", "success": False}

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
                    "message": "No PixelCalibration configuration found"
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
            if False and max_val >= saturation_threshold:
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
            # TODO: Is this the corrected way to determine flip from scale signs?
            flipY = scale_y > 0
            flipX = scale_x > 0

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
            # INDEXING CONVENTION:
            # - ObjectiveManager stores config; ObjectiveController owns the
            #   current-slot state.  We resolve the current slot from the
            #   controller registry when available, falling back to matching
            #   objective names directly on the manager.
            if hasattr(self._master, 'objectiveManager'):
                obj_manager = self._master.objectiveManager

                # Resolve current objective name via ObjectiveController (single source of truth)
                current_objective_name = "default"
                current_slot_0based = None
                try:
                    obj_ctrl = self._master._controllersRegistry.get("Objective", None)
                    if obj_ctrl is not None and hasattr(obj_ctrl, '_getCurrentObjectiveName'):
                        current_objective_name = obj_ctrl._getCurrentObjectiveName()
                        current_slot_0based = getattr(obj_ctrl, '_currentObjective', None)
                except Exception:
                    pass

                # Fallback: look up by name in ObjectiveManager
                if current_slot_0based is None:
                    names = obj_manager.objectiveNames
                    if objective_id in names:
                        current_slot_0based = names.index(objective_id)
                        current_objective_name = objective_id
                    elif names:
                        current_slot_0based = 0
                        current_objective_name = names[0]

                # If calibrated objective matches current objective, update immediately
                if objective_id == current_objective_name or objective_id == "default":
                    if current_slot_0based is not None:
                        obj_manager.setObjectiveParameters(current_slot_0based, pixelsize=pixel_size, emitSignal=True)
                        self._logger.info(f"Updated objective slot {current_slot_0based} pixelsize to {pixel_size:.3f} µm/px")

            self._logger.info(f"Successfully applied calibration results for '{objective_id}'")

        except Exception as e:
            self._logger.error(f"Failed to apply calibration results: {e}", exc_info=True)

    # Overview Calibration API Endpoints

    @APIExport()
    def overviewIsObservationCameraAvailable(self):
        """
        Check if observation camera is available.
        
        Returns:
            Dictionary with:
                - available: bool - whether camera is available
                - name: str|null - camera name if available
        """
        return {
            "available": self.observationCamera is not None,
            "name": self.observationCameraName if self.observationCamera is not None else None
        }

    @APIExport(runOnUIThread=False)
    def overviewIdentifyAxes(self, stepUm: float = 2000.0, debug_dir: str = None):
        """
        Identify stage axis directions and signs using AprilTag tracking.
        
        Args:
            stepUm: Step size in micrometers (default 2000)
            
        Returns:
            Dictionary with mapping, sign, and samples or error
        """
        if self.observationCamera is None:
            raise HTTPException(status_code=409, detail="Observation camera not available")

        # Get positioner
        positioner_names = self._master.positionersManager.getAllDeviceNames()
        if not positioner_names:
            raise HTTPException(status_code=409, detail="No positioner available")

        positioner = self._master.positionersManager[positioner_names[0]]

        # Run calibration
        result = self.overviewCalibrator.identify_axes(
            self.observationCamera, positioner, step_um=stepUm, save_debug_images=debug_dir is not None, debug_dir=debug_dir
        )

        # Save to config if successful
        if "error" not in result:
            if not hasattr(self._setupInfo.PixelCalibration, 'overviewCalibration'):
                self._setupInfo.PixelCalibration.overviewCalibration = {}

            self._setupInfo.PixelCalibration.overviewCalibration['axes'] = {
                'mapping': result['mapping'],
                'sign': result['sign']
            }

            # Save to disk
            try:
                import imswitch.imcontrol.model.configfiletools as configfiletools
                options, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(options, self._setupInfo)
                self._logger.info("Saved axis identification to config")
            except Exception as e:
                self._logger.warning(f"Could not save config: {e}")

        return result

    @APIExport(runOnUIThread=False)
    def overviewMapIlluminationChannels(self):
        """
        Map illumination channels to colors/wavelengths using image differencing.
        
        Returns:
            Dictionary with illuminationMap, darkStats, or error
        """
        if self.observationCamera is None:
            raise HTTPException(status_code=409, detail="Observation camera not available")

        # Get laser and LED managers
        lasers_manager = getattr(self._master, 'lasersManager', None)
        leds_manager = getattr(self._master, 'LEDsManager', None)

        if lasers_manager is None and leds_manager is None:
            raise HTTPException(status_code=409, detail="No illumination sources available")

        # Run calibration
        result = self.overviewCalibrator.map_illumination_channels(
            self.observationCamera, lasers_manager, leds_manager
        )

        # Save to config if successful
        if "error" not in result:
            if not hasattr(self._setupInfo.PixelCalibration, 'overviewCalibration'):
                self._setupInfo.PixelCalibration.overviewCalibration = {}

            self._setupInfo.PixelCalibration.overviewCalibration['illuminationMap'] = result['illuminationMap']

            # Save to disk
            try:
                import imswitch.imcontrol.model.configfiletools as configfiletools
                options, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(options, self._setupInfo)
                self._logger.info("Saved illumination mapping to config")
            except Exception as e:
                self._logger.warning(f"Could not save config: {e}")

        return result

    @APIExport(runOnUIThread=False)
    def overviewVerifyHoming(self, maxTimeS: float = 20.0):
        """
        Verify homing behavior and detect inverted motor directions.
        
        Args:
            maxTimeS: Maximum time to wait for homing (seconds, default 20)
            
        Returns:
            Dictionary with X and Y homing verification results
        """
        if self.observationCamera is None:
            raise HTTPException(status_code=409, detail="Observation camera not available")

        # Get positioner
        positioner_names = self._master.positionersManager.getAllDeviceNames()
        if not positioner_names:
            raise HTTPException(status_code=409, detail="No positioner available")

        positioner = self._master.positionersManager[positioner_names[0]]

        # Run verification
        result = self.overviewCalibrator.verify_homing(
            self.observationCamera, positioner, max_time_s=maxTimeS
        )

        # Save to config
        if not hasattr(self._setupInfo.PixelCalibration, 'overviewCalibration'):
            self._setupInfo.PixelCalibration.overviewCalibration = {}

        self._setupInfo.PixelCalibration.overviewCalibration['homing'] = {
            axis: {
                'inverted': data.get('inverted', False),
                'lastCheck': data.get('lastCheck', time.strftime("%Y-%m-%dT%H:%M:%S"))
            }
            for axis, data in result.items()
            if 'error' not in data
        }

        # Save to disk
        try:
            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
            self._logger.info("Saved homing verification to config")
        except Exception as e:
            self._logger.warning(f"Could not save config: {e}")

        return result

    @APIExport(runOnUIThread=False)
    def overviewFixStepSign(self, rectSizeUm: float = 20000.0):
        """
        Determine and fix step size sign by visiting rectangle corners.
        
        Args:
            rectSizeUm: Rectangle size in micrometers (default 20000)
            
        Returns:
            Dictionary with sign corrections and samples or error
        """
        if self.observationCamera is None:
            raise HTTPException(status_code=409, detail="Observation camera not available")

        # Get positioner
        positioner_names = self._master.positionersManager.getAllDeviceNames()
        if not positioner_names:
            raise HTTPException(status_code=409, detail="No positioner available")

        positioner = self._master.positionersManager[positioner_names[0]]

        # Run calibration
        result = self.overviewCalibrator.fix_step_sign(
            self.observationCamera, positioner, rect_size_um=rectSizeUm
        )

        # Save to config if successful
        if "error" not in result:
            if not hasattr(self._setupInfo.PixelCalibration, 'overviewCalibration'):
                self._setupInfo.PixelCalibration.overviewCalibration = {}

            if 'axes' not in self._setupInfo.PixelCalibration.overviewCalibration:
                self._setupInfo.PixelCalibration.overviewCalibration['axes'] = {}

            self._setupInfo.PixelCalibration.overviewCalibration['axes']['sign'] = result['sign']

            # Apply sign correction to positioner stepsize
            # Get positioner name
            positioner_names = self._master.positionersManager.getAllDeviceNames()
            if positioner_names:
                positioner_name = positioner_names[0]

                # Get sign corrections from result
                sign_x = result['sign'].get('X', 1)
                sign_y = result['sign'].get('Y', 1)

                self._logger.info(f"Applying sign correction to positioner '{positioner_name}': X={sign_x}, Y={sign_y}")

                # Update stepsize in setupInfo by multiplying with sign
                if hasattr(self._setupInfo, 'positioners') and positioner_name in self._setupInfo.positioners:
                    positioner_info = self._setupInfo.positioners[positioner_name]

                    # Get current stepsizes
                    current_stepsize_x = positioner_info.managerProperties.get('stepsizeX', 1)
                    current_stepsize_y = positioner_info.managerProperties.get('stepsizeY', 1)

                    # Apply sign correction by multiplying
                    corrected_stepsize_x = current_stepsize_x * sign_x
                    corrected_stepsize_y = current_stepsize_y * sign_y

                    # Update in setupInfo
                    positioner_info.managerProperties['stepsizeX'] = corrected_stepsize_x
                    positioner_info.managerProperties['stepsizeY'] = corrected_stepsize_y

                    self._logger.info(
                        f"Updated stepsize: X: {current_stepsize_x} -> {corrected_stepsize_x}, "
                        f"Y: {current_stepsize_y} -> {corrected_stepsize_y}"
                    )

                    # Also update the live positioner manager if it has stepSizes attribute
                    try:
                        positioner = self._master.positionersManager[positioner_name]
                        if hasattr(positioner, 'stepsizeX'):
                            positioner.stepsizeX = corrected_stepsize_x
                        if hasattr(positioner, 'stepsizeY'):
                            positioner.stepsizeY = corrected_stepsize_y
                            self._logger.info("Applied stepsize correction to live positioner manager")
                    except Exception as e:
                        self._logger.warning(f"Could not update live positioner: {e}")

            # Save to disk
            try:
                import imswitch.imcontrol.model.configfiletools as configfiletools
                options, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(options, self._setupInfo)
                self._logger.info("Saved step sign correction and updated stepsizes to config")
            except Exception as e:
                self._logger.warning(f"Could not save config: {e}")

        return result

    @APIExport(runOnUIThread=False)
    def overviewCaptureObjectiveImage(self, slot: int):
        """
        Capture reference image for a specific objective slot.
        
        Args:
            slot: Objective slot number
            
        Returns:
            Dictionary with slot number, saved path, or error
        """
        if self.observationCamera is None:
            raise HTTPException(status_code=409, detail="Observation camera not available")

        # Get config directory
        import imswitch.imcontrol.model.configfiletools as configfiletools
        options, _ = configfiletools.loadOptions()
        config_dir = os.path.dirname(options.setupFileName)
        # TODO: we wuold need to move to the respective objective slot
        # Run capture
        result = self.overviewCalibrator.capture_objective_image(
            self.observationCamera, slot, save_dir=config_dir
        )

        # Save to config if successful
        if "error" not in result:
            if not hasattr(self._setupInfo.PixelCalibration, 'overviewCalibration'):
                self._setupInfo.PixelCalibration.overviewCalibration = {}

            if 'objectiveImages' not in self._setupInfo.PixelCalibration.overviewCalibration:
                self._setupInfo.PixelCalibration.overviewCalibration['objectiveImages'] = {}

            self._setupInfo.PixelCalibration.overviewCalibration['objectiveImages'][f'slot{slot}'] = result['path']

            # Save to disk
            try:
                configfiletools.saveSetupInfo(options, self._setupInfo)
                self._logger.info(f"Saved objective {slot} image path to config")
            except Exception as e:
                self._logger.warning(f"Could not save config: {e}")

        return result

    @APIExport()
    def overviewGetConfig(self):
        """
        Get current overview calibration configuration.
        
        Returns:
            Dictionary with overview calibration data
        """
        if not hasattr(self._setupInfo.PixelCalibration, 'overviewCalibration'):
            return {}

        return self._setupInfo.PixelCalibration.overviewCalibration

    '''
    ONLY DEBUGGING / DEVELOPMENT USE for APRIL TAG OVERLAY
    '''
    @APIExport()
    def stopOverviewStream(self):
        """Stop the overview camera MJPEG stream."""
        self.overviewStreamRunning = False
        self.overviewStreamStarted = False
        self.overviewStreamQueue = None

    def startOverviewStream(self):
        """
        Background thread that converts observation camera frames to JPEG and queues them.
        Supports optional AprilTag overlay.
        """

        if self.observationCamera is None:
            self._logger.error("Observation camera not available")
            return

        # Wait for first valid frame (up to 2s); fall back to black frame
        deadline = time.time() + 2.0
        output_frame = None
        while self.overviewStreamRunning and output_frame is None and time.time() < deadline:
            try:
                output_frame = self.observationCamera.getLatestFrame()
            except Exception:
                output_frame = None
            if output_frame is None:
                time.sleep(0.05)

        if output_frame is None:
            # Default black frame if nothing available (grayscale)
            output_frame = np.zeros((480, 640), dtype=np.uint8)

        # Adaptive resize: Keep frames below 640x480
        try:
            if output_frame.shape[0] > 640 or output_frame.shape[1] > 480:
                everyNthsPixel = int(
                    np.min(
                        [
                            max(1, output_frame.shape[0] // 480),
                            max(1, output_frame.shape[1] // 640),
                        ]
                    )
                )
            else:
                everyNthsPixel = 1
        except Exception:
            everyNthsPixel = 1

        try:
            while self.overviewStreamRunning:
                output_frame = self.observationCamera.getLatestFrame()
                if output_frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    # Downsample if needed
                    output_frame = output_frame[::everyNthsPixel, ::everyNthsPixel]
                except Exception:
                    output_frame = np.zeros((480, 640), dtype=np.uint8)

                # Apply flip settings
                if self.observationFlipY:
                    output_frame = np.flip(output_frame, 0)
                if self.observationFlipX:
                    output_frame = np.flip(output_frame, 1)

                # Check if AprilTag overlay is enabled
                with self._overlay_lock:
                    overlay_enabled = self._aprilTagOverlayEnabled

                # Apply AprilTag overlay if enabled
                if overlay_enabled and self.gridCalibrator is not None:
                    output_frame = self._draw_apriltag_overlay(output_frame)
                else:
                    # Convert grayscale to BGR if needed (for consistent processing)
                    if len(output_frame.shape) == 2:
                        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

                # Ensure uint8 image for JPEG; normalize if needed
                if output_frame.dtype != np.uint8:
                    try:
                        vmin = float(np.min(output_frame))
                        vmax = float(np.max(output_frame))
                        if vmax > vmin:
                            output_frame = (
                                (output_frame - vmin) / (vmax - vmin) * 255.0
                            ).astype(np.uint8)
                        else:
                            output_frame = np.zeros_like(output_frame, dtype=np.uint8)
                    except Exception:
                        output_frame = np.zeros_like(output_frame, dtype=np.uint8)

                # JPEG compression
                quality = 90  # Quality level (0-100)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                flag, encodedImage = cv2.imencode(".jpg", output_frame, encode_params)
                if not flag:
                    continue

                # Put raw JPEG bytes into queue; avoid blocking forever if queue is full
                try:
                    self.overviewStreamQueue.put(encodedImage.tobytes(), timeout=0.5)
                except Exception:
                    # Drop frame if queue is full or unavailable
                    pass

                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            self._logger.error(f"Overview stream error: {e}", exc_info=True)
            self.overviewStreamRunning = False

    def overviewStreamer(self):
        """
        Generator that yields JPEG frames from the queue.
        Starts the background streaming thread if not already running.
        """
        import queue

        # Start the streaming worker thread once and create a thread-safe queue
        if not self.overviewStreamStarted:
            import threading

            self.overviewStreamQueue = queue.Queue(maxsize=10)
            self.overviewStreamRunning = True
            self.overviewStreamStarted = True
            t = threading.Thread(target=self.startOverviewStream, daemon=True)
            t.start()

        try:
            while self.overviewStreamRunning:
                try:
                    # Use timeout to allow graceful shutdown
                    jpeg_bytes = self.overviewStreamQueue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Build proper MJPEG part with Content-Length for better client compatibility
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii")
                )
                yield header + jpeg_bytes + b"\r\n"
        except GeneratorExit:
            self._logger.debug("Overview stream connection closed by client.")
            self.stopOverviewStream()

    @APIExport(runOnUIThread=False)
    def overviewStream(self, startStream: bool = True):
        """
        Get MJPEG stream from observation camera with optional AprilTag overlay.
        
        Uses efficient JPEG encoding and queue-based architecture for low latency.
        Enable/disable AprilTag detection overlay using gridSetStreamOverlay endpoint.
        
        Args:
            startStream: Whether to start the stream (default True)
            
        Returns:
            StreamingResponse with multipart/x-mixed-replace for MJPEG stream
        """
        if not startStream:
            self.stopOverviewStream()
            return {"status": "success", "message": "stream stopped"}

        if self.observationCamera is None:
            raise HTTPException(status_code=409, detail="Observation camera not available")

        headers = {
            # Disable buffering and caching to reduce latency
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }

        return StreamingResponse(
            self.overviewStreamer(),
            media_type="multipart/x-mixed-replace;boundary=frame",
            headers=headers,
        )

    # ========================================================================
    # AprilTag Grid Calibration API Endpoints
    # ========================================================================

    def _draw_apriltag_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw AprilTag detection overlay on frame.
        
        Draws detected markers with:
        - Green bounding boxes
        - Tag ID labels
        - Centroid markers
        - Grid position info (if grid is configured)
        
        Args:
            frame: Input frame (grayscale or BGR)
            
        Returns:
            Frame with overlay drawn
        """
        try:
            # Ensure frame is BGR for color drawing
            if len(frame.shape) == 2:
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_color = frame.copy()
            # convert to uint8
            if frame_color.dtype != np.uint8:
                frame_color = cv2.normalize(frame_color, None, 10, 200, cv2.NORM_MINMAX).astype(np.uint8)

            # Detect tags
            tags = self.gridCalibrator.detect_tags(np.mean(frame_color, axis=2).astype(np.uint8))

            if not tags:
                return frame_color

            # Draw each detected tag
            for tag_id, (cx, cy) in tags.items():
                # Get grid position if available
                rowcol = self.gridCalibrator._grid.id_to_rowcol(tag_id)

                # Draw centroid marker
                cv2.drawMarker(
                    frame_color,
                    (int(cx), int(cy)),
                    (0, 255, 0),  # Green
                    cv2.MARKER_CROSS,
                    20, 2
                )

                # Draw tag ID label with grid position
                if rowcol is not None:
                    row, col = rowcol
                    label = f"ID:{tag_id} (R{row},C{col})"
                else:
                    label = f"ID:{tag_id}"

                # Add background rectangle for text
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

                text_x = int(cx) + 10
                text_y = int(cy) - 10

                cv2.rectangle(
                    frame_color,
                    (text_x - 2, text_y - text_height - 2),
                    (text_x + text_width + 2, text_y + baseline + 2),
                    (0, 0, 0),  # Black background
                    -1
                )

                # Draw text
                cv2.putText(
                    frame_color,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),  # Green text
                    2,
                    cv2.LINE_AA
                )

            # Draw tag count in top-left corner
            count_label = f"Tags: {len(tags)}"
            cv2.putText(
                frame_color,
                count_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),  # Yellow
                2,
                cv2.LINE_AA
            )

            return frame_color

        except Exception as e:
            self._logger.error(f"Error drawing AprilTag overlay: {e}")
            # Return original frame on error
            if len(frame.shape) == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame

    @APIExport()
    def gridSetStreamOverlay(self, enabled: bool = True):
        """
        Enable or disable AprilTag detection overlay on MJPEG stream.
        
        When enabled, the overviewStream will show detected AprilTags with:
        - Tag ID labels
        - Grid positions (row, col)
        - Centroid markers
        - Tag count
        
        Args:
            enabled: True to enable overlay, False to disable (default: True)
            
        Returns:
            Dictionary with status
        """
        try:
            with self._overlay_lock:
                self._aprilTagOverlayEnabled = enabled

            status = "enabled" if enabled else "disabled"
            self._logger.info(f"AprilTag overlay {status}")

            return {
                "success": True,
                "overlay_enabled": enabled,
                "message": f"AprilTag overlay {status}"
            }

        except Exception as e:
            self._logger.error(f"Failed to set overlay: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    @APIExport()
    def gridGetOverlay(self):
        """
        Get current AprilTag overlay status.
        
        Returns:
            Dictionary with overlay enabled status
        """
        with self._overlay_lock:
            enabled = self._aprilTagOverlayEnabled

        return {
            "success": True,
            "overlay_enabled": enabled
        }

    @APIExport()
    def gridSetRotation180(self, rotated: bool = False):
        """
        Set whether the calibration sample is rotated 180 degrees.
        
        When rotated, the tag numbering is reversed:
        - Normal: 0, 1, 2, ... 424
        - Rotated 180°: 424, 423, 422, ... 0
        
        This is useful when the calibration grid is accidentally inserted upside down.
        The grid layout adjusts automatically:
        - Row/column positions are flipped
        - Tag ID mapping is reversed
        
        Args:
            rotated: True if calibration sample is rotated 180°, False for normal orientation
            
        Returns:
            Dictionary with status and current rotation state
        """
        try:
            self._gridRotated180 = rotated

            # Update the grid calibrator if it exists
            if self.gridCalibrator is not None:
                self.gridCalibrator.set_rotation_180(rotated)

            # Save to config
            self._saveGridCalibration()

            orientation = "rotated 180°" if rotated else "normal"
            self._logger.info(f"Grid orientation set to: {orientation}")

            return {
                "success": True,
                "rotated_180": rotated,
                "message": f"Grid orientation: {orientation}"
            }

        except Exception as e:
            self._logger.error(f"Failed to set grid rotation: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    @APIExport()
    def gridGetRotation180(self):
        """
        Get current 180° rotation state of the calibration grid.
        
        Returns:
            Dictionary with rotation status
        """
        try:
            return {
                "success": True,
                "rotated_180": self._gridRotated180,
                "message": "Rotated 180°" if self._gridRotated180 else "Normal orientation"
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _loadGridCalibration(self):
        """
        Load AprilTag grid configuration from setup info.
        
        Expected structure in config JSON:
        {
            "PixelCalibration": {
                "aprilTagGrid": {
                    "rows": 17,
                    "cols": 25,
                    "start_id": 0,
                    "pitch_mm": 40.0,
                    "transform": [[a, b, tx], [c, d, ty]]  // Optional: saved calibration
                }
            }
        }
        """
        if not hasattr(self._setupInfo.PixelCalibration, 'aprilTagGrid'):
            self._logger.info("No AprilTag grid configuration found, using defaults")
            # Create default grid: 17 rows x 25 cols, 4mm pitch
            grid_config = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=4.0)
            self.gridCalibrator = AprilTagGridCalibrator(grid_config, logger=self._logger)
            return

        try:
            grid_data = self._setupInfo.PixelCalibration.aprilTagGrid
            grid_config = GridConfig.from_dict(grid_data)
            self.gridCalibrator = AprilTagGridCalibrator(grid_config, logger=self._logger)

            # Load saved transformation if available
            if 'transform' in grid_data and grid_data['transform'] is not None:
                T = np.array(grid_data['transform'], dtype=np.float64)
                if T.shape == (2, 3):
                    self.gridCalibrator.set_transform(T)
                    self._logger.info("Loaded saved grid calibration transform")

            # Load rotation state if available
            if 'rotated_180' in grid_data:
                self._gridRotated180 = grid_data['rotated_180']
                self.gridCalibrator.set_rotation_180(self._gridRotated180)
                orientation = "rotated 180°" if self._gridRotated180 else "normal"
                self._logger.info(f"Grid orientation: {orientation}")

            self._logger.info(f"Loaded AprilTag grid: {grid_config.rows}x{grid_config.cols}, pitch={grid_config.pitch_mm}mm")

        except Exception as e:
            self._logger.error(f"Failed to load grid configuration: {e}", exc_info=True)
            # Fallback to default
            grid_config = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=4.0)
            self.gridCalibrator = AprilTagGridCalibrator(grid_config, logger=self._logger)

    def _saveGridCalibration(self):
        """Save grid configuration and calibration to setup info."""
        try:
            import imswitch.imcontrol.model.configfiletools as configfiletools

            # Get grid config
            grid_dict = self.gridCalibrator.get_grid_config()

            # Add transformation if calibrated
            T = self.gridCalibrator.get_transform()
            if T is not None:
                grid_dict['transform'] = T.tolist()

            # Add rotation state
            grid_dict['rotated_180'] = self._gridRotated180

            # Update setup info in memory
            if not hasattr(self._setupInfo, 'PixelCalibration'):
                return

            self._setupInfo.PixelCalibration.aprilTagGrid = grid_dict

            # Save to file
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
            self._logger.info("Saved AprilTag grid calibration to config")

        except Exception as e:
            self._logger.error(f"Failed to save grid calibration: {e}", exc_info=True)

    @APIExport()
    def gridSetConfig(self, rows: int, cols: int, start_id: int = 0, pitch_mm: float = 4.0):
        """
        Configure the AprilTag grid layout.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            start_id: Starting tag ID (default 0)
            pitch_mm: Physical spacing between tag centers in millimeters (default 40.0)
            
        Returns:
            Dictionary with updated configuration
        """
        try:
            if self.gridCalibrator is None:
                self._loadGridCalibration()

            # Create new config
            grid_config = GridConfig(rows=rows, cols=cols, start_id=start_id, pitch_mm=pitch_mm)

            # Preserve existing transform if dimensions match
            old_transform = self.gridCalibrator.get_transform()

            # Update calibrator
            self.gridCalibrator.set_grid_config(grid_config)

            # Restore transform (it's independent of grid layout)
            if old_transform is not None:
                self.gridCalibrator.set_transform(old_transform)

            # Save to config
            self._saveGridCalibration()

            return {
                "success": True,
                "config": grid_config.to_dict(),
                "transform_preserved": old_transform is not None
            }

        except Exception as e:
            self._logger.error(f"Failed to set grid config: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    @APIExport()
    def gridGetConfig(self):
        """
        Get current AprilTag grid configuration.
        
        Returns:
            Dictionary with grid configuration and calibration status
        """
        try:
            if self.gridCalibrator is None:
                self._loadGridCalibration()

            config = self.gridCalibrator.get_grid_config()
            T = self.gridCalibrator.get_transform()

            return {
                "success": True,
                "config": config,
                "calibrated": T is not None,
                "transform": T.tolist() if T is not None else None
            }

        except Exception as e:
            self._logger.error(f"Failed to get grid config: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    @APIExport()
    def gridDetectTags(self, save_annotated: bool = False):
        """
        Detect AprilTags in the current observation camera frame.
        
        Args:
            save_annotated: If True, saves an annotated image to disk
            
        Returns:
            Dictionary with detected tags and their positions
        """
        try:
            if self.observationCamera is None:
                return {"error": "Observation camera not available", "success": False}

            if self.gridCalibrator is None:
                self._loadGridCalibration()

            # Get frame
            frame = self.observationCamera.getLatestFrame()

            # Apply flips
            if self.observationFlipY:
                frame = np.flip(frame, 0)
            if self.observationFlipX:
                frame = np.flip(frame, 1)

            # Detect tags
            tags = self.gridCalibrator.detect_tags(frame)

            # Convert to serializable format
            tags_list = [
                {
                    "id": int(tag_id),
                    "cx": float(cx),
                    "cy": float(cy),
                    "grid_position": self.gridCalibrator._grid.id_to_rowcol(tag_id)
                }
                for tag_id, (cx, cy) in tags.items()
            ]

            # Optional: save annotated image
            if save_annotated and tags:
                save_path = os.path.join(dirtools.UserFileDirs.Root,
                                        "imcontrol_slm",
                                        "grid_detection.png")
                # Re-detect with save enabled
                import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                self.gridCalibrator.detect_tags(gray)  # This will trigger save in the method

            return {
                "success": True,
                "num_tags": len(tags),
                "tags": tags_list
            }

        except Exception as e:
            self._logger.error(f"Tag detection failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    @APIExport()
    def gridCalibrateTransform(self):
        """
        Calibrate camera-to-stage transformation using currently visible AprilTags.
        
        Requires at least 3 visible tags with known grid positions.
        
        Returns:
            Dictionary with calibration results including transform matrix and residual error
        """
        try:
            if self.observationCamera is None:
                return {"error": "Observation camera not available", "success": False}

            if self.gridCalibrator is None:
                self._loadGridCalibration()

            # Get frame
            frame = self.observationCamera.getLatestFrame()

            # Apply flips
            if self.observationFlipY:
                frame = np.flip(frame, 0)
            if self.observationFlipX:
                frame = np.flip(frame, 1)

            # Detect tags
            tags = self.gridCalibrator.detect_tags(frame)

            if len(tags) < 3:
                return {
                    "error": f"Need at least 3 visible grid tags for calibration, found {len(tags)}",
                    "success": False,
                    "num_tags": len(tags)
                }

            # Perform calibration
            result = self.gridCalibrator.calibrate_from_frame(tags)

            if "error" in result:
                return {"success": False, **result}

            # Save calibration
            self._saveGridCalibration()

            return {
                "success": True,
                **result
            }

        except Exception as e:
            self._logger.error(f"Grid calibration failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    @APIExport(runOnUIThread=False)
    def gridMoveToTag(self, target_id: int,
                     roi_tolerance_px: float = 8.0,
                     max_iterations: int = 30,
                     max_step_um: float = 1000.0,
                     settle_time: float = 0.3):
        """
        Navigate stage to center a specific AprilTag ID using grid-aware iterative navigation.
        
        This method uses axis calibration data from overviewIdentifyAxes to correctly map
        movements. It compares detected tag grid positions with the stored grid topology
        to decide movement direction. Runs in a background thread to avoid blocking the server.
        
        Algorithm:
        1. Check if axis calibration is available, run overviewIdentifyAxes if needed
        2. Detect all visible tags in current frame
        3. Compare detected tag grid positions with target position
        4. Move by max_step_um in the computed direction (X or Y)
        5. Repeat until target is visible and centered
        
        Args:
            target_id: Tag ID to navigate to (must be within grid range)
            roi_tolerance_px: Acceptable pixel offset for convergence (default 8.0)
            max_iterations: Maximum iteration count (default 30)
            max_step_um: Maximum step size per iteration in micrometers (default 1000.0)
            settle_time: Wait time after movement in seconds (default 0.3)
            
        Returns:
            Dictionary with status message - actual navigation runs in background thread
        """

        return
        try:
            if self.observationCamera is None:
                return {"error": "Observation camera not available", "success": False}

            if self.gridCalibrator is None:
                self._loadGridCalibration()

            # Get positioner
            positioner_names = self._master.positionersManager.getAllDeviceNames()
            if not positioner_names:
                return {"error": "No positioner available", "success": False}

            positioner = self._master.positionersManager[positioner_names[0]]

            # Check for axis calibration data
            axis_calibration = None
            if hasattr(self._setupInfo.PixelCalibration, 'overviewCalibration'):
                overview_cal = self._setupInfo.PixelCalibration.overviewCalibration
                if 'axes' in overview_cal:
                    axis_calibration = overview_cal['axes']

            # Run axis calibration if not available
            if axis_calibration is None:
                self._logger.info("No axis calibration found, running overviewIdentifyAxes...")
                cal_result = self.overviewCalibrator.identify_axes(
                    self.observationCamera, positioner, step_um=max_step_um
                )

                if "error" in cal_result:
                    return {
                        "error": f"Axis calibration failed: {cal_result['error']}",
                        "success": False
                    }

                # Save calibration
                if not hasattr(self._setupInfo.PixelCalibration, 'overviewCalibration'):
                    self._setupInfo.PixelCalibration.overviewCalibration = {}

                self._setupInfo.PixelCalibration.overviewCalibration['axes'] = {
                    'mapping': cal_result['mapping'],
                    'sign': cal_result['sign']
                }

                # Save to disk
                try:
                    import imswitch.imcontrol.model.configfiletools as configfiletools
                    options, _ = configfiletools.loadOptions()
                    configfiletools.saveSetupInfo(options, self._setupInfo)
                    self._logger.info("Saved axis calibration to config")
                except Exception as e:
                    self._logger.warning(f"Could not save config: {e}")

                axis_calibration = self._setupInfo.PixelCalibration.overviewCalibration['axes']

            # Start navigation in background thread
            def navigation_worker():
                result = self.gridCalibrator.move_to_tag(
                    target_id=target_id,
                    observation_camera=self.observationCamera,
                    positioner=positioner,
                    axis_calibration=axis_calibration,
                    roi_center=None,  # Use image center
                    roi_tolerance_px=roi_tolerance_px,
                    max_iterations=max_iterations,
                    max_step_um=max_step_um,
                    settle_time=settle_time
                )

                if result.get('success'):
                    self._logger.info(
                        f"Navigation to tag {target_id} completed successfully in "
                        f"{result['iterations']} iterations"
                    )
                else:
                    self._logger.error(
                        f"Navigation to tag {target_id} failed: {result.get('error', 'Unknown error')}"
                    )

            import threading
            nav_thread = threading.Thread(target=navigation_worker, daemon=True)
            nav_thread.start()

            return {
                "success": True,
                "message": f"Navigation to tag {target_id} started in background",
                "target_id": target_id,
                "axis_calibration": axis_calibration
            }

        except Exception as e:
            self._logger.error(f"Grid navigation failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    @APIExport()
    def gridGetTagInfo(self, tag_id: int):
        """
        Get information about a specific tag ID.
        
        Args:
            tag_id: Tag ID to query
            
        Returns:
            Dictionary with tag information including grid position
        """
        try:
            if self.gridCalibrator is None:
                self._loadGridCalibration()

            rowcol = self.gridCalibrator._grid.id_to_rowcol(tag_id)

            if rowcol is None:
                return {
                    "error": f"Tag ID {tag_id} is outside grid range",
                    "success": False
                }

            row, col = rowcol

            # Compute physical position relative to grid origin
            x_mm = col * self.gridCalibrator._grid.pitch_mm
            y_mm = row * self.gridCalibrator._grid.pitch_mm

            return {
                "success": True,
                "tag_id": tag_id,
                "row": row,
                "col": col,
                "position_mm": {"x": x_mm, "y": y_mm}
            }

        except Exception as e:
            self._logger.error(f"Failed to get tag info: {e}", exc_info=True)
            return {"error": str(e), "success": False}

    # ------------------------------------------------------------------ #
    #  Manual Pixel-Size Calibration (two-point method)                   #
    # ------------------------------------------------------------------ #

    @APIExport(requestType="POST")
    def manualPixelSizeCalibration(
        self,
        point1X: float = 0.0,
        point1Y: float = 0.0,
        point2X: float = 0.0,
        point2Y: float = 0.0,
        movementDistanceUm: float = 100.0,
        movementAxis: str = "X",
        objectiveId: str = None,
    ):
        """
        Compute and store the pixel size from a manual two-point calibration.

        Workflow (done on the frontend):
        1. User marks a recognisable feature in the live image  → (point1X, point1Y)
        2. Stage is moved by *movementDistanceUm* along *movementAxis*
        3. User marks the **same** feature in the new image       → (point2X, point2Y)

        The pixel displacement is calculated as the Euclidean distance between
        the two points.  The pixel size is then:

            pixel_size = movementDistanceUm / displacement_px   [µm / pixel]

        The result is saved to the setup configuration as a simplified affine
        calibration (scale-only, no rotation) so it is immediately available
        to the rest of ImSwitch.

        Args:
            point1X:  X-pixel coordinate of the feature BEFORE movement
            point1Y:  Y-pixel coordinate of the feature BEFORE movement
            point2X:  X-pixel coordinate of the feature AFTER movement
            point2Y:  Y-pixel coordinate of the feature AFTER movement
            movementDistanceUm:  Known stage movement in micrometres (> 0)
            movementAxis:  Axis of movement ("X" or "Y")
            objectiveId:  Objective name/ID; defaults to current objective

        Returns:
            Dictionary with computed pixel size, displacement, and status
        """
        try:
            # --- resolve objective ------------------------------------------------
            if objectiveId is None or objectiveId == "":
                objectiveId = self._getCurrentObjectiveId()

            # --- validate inputs --------------------------------------------------
            if movementDistanceUm <= 0:
                return {"error": "movementDistanceUm must be > 0", "success": False}

            if movementAxis not in ("X", "Y"):
                return {"error": "movementAxis must be 'X' or 'Y'", "success": False}

            # --- compute displacement in pixels -----------------------------------
            dx = point2X - point1X
            dy = point2Y - point1Y
            displacement_px = np.sqrt(dx ** 2 + dy ** 2)

            if displacement_px < 1.0:
                return {
                    "error": "Pixel displacement too small (< 1 px). "
                             "Mark two distinct positions or increase the movement distance.",
                    "success": False,
                }

            pixel_size_um = abs(movementDistanceUm) / displacement_px

            self._logger.info(
                f"Manual pixel-size calibration for '{objectiveId}': "
                f"displacement = {displacement_px:.1f} px, "
                f"movement = {movementDistanceUm:.1f} µm ({movementAxis}), "
                f"→ pixel size = {pixel_size_um:.4f} µm/px"
            )

            # --- build a simplified affine matrix (scale-only) --------------------
            # Convention: [[scaleX, 0, 0], [0, scaleY, 0]]
            # We set both axes to the same scale since we only measured one axis.
            affine_matrix = [
                [pixel_size_um, 0.0, 0.0],
                [0.0, pixel_size_um, 0.0],
            ]

            metrics = {
                "scale_x_um_per_pixel": pixel_size_um,
                "scale_y_um_per_pixel": pixel_size_um,
                "rotation_deg": 0.0,
                "quality": "manual",
                "method": "manual_two_point",
                "displacement_px": float(displacement_px),
                "movement_distance_um": float(movementDistanceUm),
                "movement_axis": movementAxis,
                "point1": [float(point1X), float(point1Y)],
                "point2": [float(point2X), float(point2Y)],
            }

            calibration_data = {
                "affine_matrix": affine_matrix,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metrics": metrics,
                "objective_info": {"name": objectiveId, "method": "manual"},
            }

            # --- persist to setup config ------------------------------------------
            self._setupInfo.setAffineCalibration(objectiveId, calibration_data)

            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)

            # --- apply immediately -------------------------------------------------
            result = {"affine_matrix": affine_matrix, "metrics": metrics}
            self._applyCalibrationResults(objectiveId, result)

            return {
                "success": True,
                "objectiveId": objectiveId,
                "pixelSizeUm": float(pixel_size_um),
                "displacementPx": float(displacement_px),
                "affineMatrix": affine_matrix,
                "metrics": metrics,
                "message": (
                    f"Pixel size calibrated to {pixel_size_um:.4f} µm/px "
                    f"for objective '{objectiveId}'"
                ),
            }

        except Exception as e:
            self._logger.error(f"Manual pixel-size calibration failed: {e}", exc_info=True)
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

    def _grab_image(self, crop_size:int=1024, frameSync: int = 3, returnFrameNumber: bool = False):
        # ensure we get a fresh frame
        timeoutFrameRequest = 1 # seconds # TODO: Make dependent on exposure time
        cTime = time.time()

        lastFrameNumber=-1
        currentFrameNumber = None
        while True:
            # get frame and frame number to get one that is newer than the one with illumination off eventually
            mFrame, currentFrameNumber = self._parent.detector.getLatestFrame(returnFrameNumber=True)
            if lastFrameNumber==-1:
                # first round
                lastFrameNumber = currentFrameNumber
            if time.time()-cTime> timeoutFrameRequest:
                # in case exposure time is too long we need break at one point
                if mFrame is None:
                    mFrame = self._parent.detector.getLatestFrame(returnFrameNumber=False)
                break
            if currentFrameNumber <= lastFrameNumber+frameSync:
                time.sleep(0.01) # off-load CPU
            else:
                break

        if len(mFrame.shape) > 2: mFrame = np.mean(mFrame, axis=2)  # convert to grayscale
        if returnFrameNumber:
            return np.array(nip.extract(mFrame, (crop_size, crop_size))), currentFrameNumber
        return np.array(nip.extract(mFrame, (crop_size, crop_size)))


    def _get_stage_position(self):
        """Get current stage position in microns [X, Y, Z]."""
        stage = self._parent._master.positionersManager[self._parent._master.positionersManager.getAllDeviceNames()[0]]
        posDict = stage.getPosition()
        return np.array([posDict["X"], posDict["Y"], posDict["Z"]])

    def _move_stage(self, position_um):
        """Move stage to absolute position in microns [X, Y, Z]."""
        stage = self._parent._master.positionersManager[self._parent._master.positionersManager.getAllDeviceNames()[0]]
        #stage.move(value=position_um[0], axis="X", is_absolute=True, is_blocking=True)
        #stage.move(value=position_um[1], axis="Y", is_absolute=True, is_blocking=True)
        stage.move(value=position_um, axis="XY", is_absolute=True, is_blocking=True)
        if len(position_um) > 2:
            stage.move(value=position_um[2], axis="Z", is_absolute=True, is_blocking=True)

    def calibrate_affine(
        self,
        objective_id: int = None,
        step_size_um: float = 100.0,
        pattern: str = "cross",
        n_steps: int = 4,
        validate: bool = False,
        settle_time: float = 0.2,
        crop_size: int = 1024,
        isDEBUG: bool = False
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
            objective_id = self._parent.currentObjective # TODO: This is a string of the objective name but perhaps we need the ID?
        else:
            # move to specified objective if possible
            self._parent.setCurrentObjective(objective_id)
        try:
            # 1. Get starting position and capture reference image
            start_position = self._get_stage_position()
            self._parent._logger.info(f"Starting position: {start_position[:2]} µm")

            time.sleep(settle_time)
            ref_image = self._grab_image(crop_size=crop_size)
            self._parent._logger.info(f"Reference image captured: {ref_image.shape}")

            # 2. Generate movement pattern
            if pattern == "cross":
                # Cross pattern: center + 4 cardinal + 4 diagonal = 9 positions
                offsets = [
                    (0, 0),
                    (step_size_um, 0),
                    (-step_size_um, 0),
                    (0, 0),
                    (0, step_size_um),
                    (0, -step_size_um),
                    (0, 0)]
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
                image = self._grab_image(crop_size=crop_size)

                # Measure pixel shift using phase correlation
                shift, correlation = measure_pixel_shift(np.array(ref_image), np.array(image))

                pixel_shifts.append(shift)
                stage_shifts.append([dx, dy])
                correlation_values.append(correlation)

                self._parent._logger.debug(f"Position {i+1}/{len(offsets)}: stage=({dx:.1f}, {dy:.1f}), "
                                          f"pixels=({shift[0]:.2f}, {shift[1]:.2f}), corr={correlation:.3f}")
                # save images if debugging
                if isDEBUG:
                    import tifffile
                    tifffile.imwrite("affine_calib_image_.tif", image.astype(np.float32), append=True)
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
            # TODO: We need to check if the affine_matrix makes sense if not discard and inform the user!
            # Add correlation info to metrics
            metrics["mean_correlation"] = float(np.mean(correlation_values))
            metrics["min_correlation"] = float(np.min(correlation_values))

            self._parent._logger.info(f"Calibration quality: {metrics.get('quality', 'unknown')}")
            self._parent._logger.info(f"RMSE: {metrics.get('rmse_um', 0):.3f} µm")
            self._parent._logger.info(f"Rotation: {metrics.get('rotation_deg', 0):.2f}°")
            self._parent._logger.info(f"Scale: X={metrics.get('scale_x_um_per_pixel', 0):.3f} µm/px, "
                                      f"Y={metrics.get('scale_y_um_per_pixel', 0):.3f} µm/px")
            self._parent._logger.info(f"Shear: {metrics.get('shear_xy', 0):.4f}, {metrics.get('shear_yx', 0):.4f}")
            # print Stage Shift and pixelshifts
            self._parent._logger.info(f"Stage shifts (µm): {stage_shifts}")
            self._parent._logger.info(f"Pixel shifts (px): {pixel_shifts}")

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
