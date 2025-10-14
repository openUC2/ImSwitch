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
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.calibration_storage import CalibrationStorage
from ..basecontrollers import LiveUpdatedController

#import NanoImagingPack as nip


class PixelCalibrationController(LiveUpdatedController):
    """Linked to PixelCalibrationWidget."""

    sigImageReceived = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self.pixelSize=500 # defaul FIXME: Load from json?

        # select detectors # TODO: Bad practice, but how can we access the pixelsize then?
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]


    def snapPreview(self):
        self._logger.info("Snap preview...")
        previewImage = self._master.detectorsManager.execOnCurrent(lambda c: c.getLatestFrame())



    # API Methods for web interface
    
    @APIExport(runOnUIThread=True)
    def calibrateStageAffine(self, objectiveId: str = "default", stepSizeUm: float = 100.0, 
                             pattern: str = "cross", nSteps: int = 4, validate: bool = True):
        """
        Perform affine stage-to-camera calibration via API.
        
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
            # TODO: This has to run in a thread and has to be stopable from the API
            # TODO: the result has to be send to the socket via the Signal
            # TODO: we want to access the results via the API as well
            result = csm_extension.calibrate_affine(
                objective_id=objectiveId,
                step_size_um=stepSizeUm,
                pattern=pattern,
                n_steps=nSteps,
                validate=validate
            )
            
            # Convert numpy arrays to lists for JSON serialization
            if "affine_matrix" in result:
                result["affine_matrix"] = result["affine_matrix"].tolist()
            if "pixel_displacements" in result:
                result["pixel_displacements"] = result["pixel_displacements"].tolist()
            if "stage_displacements" in result:
                result["stage_displacements"] = result["stage_displacements"].tolist()
            if "correlation_values" in result:
                result["correlation_values"] = result["correlation_values"].tolist()
            if "inlier_mask" in result:
                result["inlier_mask"] = result["inlier_mask"].tolist()
            if "starting_position" in result:
                result["starting_position"] = result["starting_position"].tolist()
            
            '''
            TOTO: We get a serialization error here:
            >>>>> Sending message {"id":"f99a6822-5f67-4eb6-b456-28ec89b277f9","type":"HEARTBEAT_ANSWER"}
                INFO:     100.71.92.70:59867 - "GET /PixelCalibrationController/calibrateStageAffine?objectiveId=default&stepSizeUm=100&pattern=cross&nSteps=4&validate=true HTTP/1.1" 500 Internal Server Error
                ERROR:    Exception in ASGI application
                Traceback (most recent call last):
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/fastapi/encoders.py", line 324, in jsonable_encoder
                    data = dict(obj)
                        ^^^^^^^^^
                TypeError: 'numpy.int64' object is not iterable

                During handling of the above exception, another exception occurred:

                Traceback (most recent call last):
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/fastapi/encoders.py", line 329, in jsonable_encoder
                    data = vars(obj)
                        ^^^^^^^^^
                TypeError: vars() argument must have __dict__ attribute

                The above exception was the direct cause of the following exception:

                Traceback (most recent call last):
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/uvicorn/protocols/http/httptools_impl.py", line 409, in run_asgi
                    result = await app(  # type: ignore[func-returns-value]
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
                    return await self.app(scope, receive, send)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/fastapi/applications.py", line 1054, in __call__
                    await super().__call__(scope, receive, send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/applications.py", line 113, in __call__
                    await self.middleware_stack(scope, receive, send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/middleware/errors.py", line 186, in __call__
                    raise exc
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/middleware/errors.py", line 164, in __call__
                    await self.app(scope, receive, _send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/middleware/cors.py", line 85, in __call__
                    await self.app(scope, receive, send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 63, in __call__
                    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
                    raise exc
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
                    await app(scope, receive, sender)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/routing.py", line 716, in __call__
                    await self.middleware_stack(scope, receive, send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/routing.py", line 736, in app
                    await route.handle(scope, receive, send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/routing.py", line 290, in handle
                    await self.app(scope, receive, send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/routing.py", line 78, in app
                    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
                    raise exc
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
                    await app(scope, receive, sender)
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/starlette/routing.py", line 75, in app
                    response = await f(request)
                            ^^^^^^^^^^^^^^^^
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/fastapi/routing.py", line 328, in app
                    content = await serialize_response(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/fastapi/routing.py", line 202, in serialize_response
                    return jsonable_encoder(response_content)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/fastapi/encoders.py", line 289, in jsonable_encoder
                    encoded_value = jsonable_encoder(
                                    ^^^^^^^^^^^^^^^^^
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/fastapi/encoders.py", line 289, in jsonable_encoder
                    encoded_value = jsonable_encoder(
                                    ^^^^^^^^^^^^^^^^^
                File "/Users/bene/mambaforge/envs/intel_env/lib/python3.11/site-packages/fastapi/encoders.py", line 332, in jsonable_encoder
                    raise ValueError(errors) from e
                ValueError: [TypeError("'numpy.int64' object is not iterable"), TypeError('vars() argument must have __dict__ attribute')]
                Received message {"id":"38ba3f5f-7a88-44d3-8862-1363aa2decf7","type":"HEARTBEAT"}
                >>>>> Sending message {"id":"6acfacf3-a3d2-465e-93a3-30a0b8a367cc","type":"HEARTBEAT_ANSWER"}

                ''' 
            return {
                "success": True,
                "objectiveId": objectiveId,
                "metrics": result.get("metrics", {}),
                "validation": result.get("validation", {})
            }
        except Exception as e:
            self._logger.error(f"API calibration failed: {e}")
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
            csm_extension = CSMExtension(self)
            success = csm_extension._calibration_storage.delete_calibration(objectiveId)
            return {
                "success": success,
                "objectiveId": objectiveId,
                "message": f"Calibration deleted for '{objectiveId}'" if success else f"No calibration found for '{objectiveId}'"
            }
        except Exception as e:
            self._logger.error(f"Failed to delete calibration: {e}")
            return {"error": str(e), "success": False}







class CSMExtension(object):
    """
    Use the camera as an encoder, so we can relate camera and stage coordinates
    """

    def __init__(self, parent):
        self._parent = parent
        # Initialize calibration storage
        calib_file_path = os.path.join(dirtools.UserFileDirs.Root, "camera_stage_calibration.json")
        self._calibration_storage = CalibrationStorage(calib_file_path, logger=self._parent._logger)


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
            
            # 7. Store calibration data
            objective_info = {
                "name": objective_id,
                "detector": self._parent.detector._camera.model if hasattr(self._parent.detector._camera, 'model') else "unknown"
            }
            
            self._calibration_storage.save_calibration(
                objective_id=objective_id,
                affine_matrix=affine_matrix,
                metrics=metrics,
                objective_info=objective_info
            )
            
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
    
    def get_metrics(self, objective_id: str = "default"):
        """
        Get calibration metrics for a specific objective.
        
        Args:
            objective_id: Identifier for the objective
            
        Returns:
            Dictionary of metrics or None if not found
        """
        return self._calibration_storage.get_metrics(objective_id)

    @property
    def image_to_stage_displacement_matrix(self):
        """A 2x2 matrix that converts displacement in image coordinates to stage coordinates."""
        if  self._calibration_storage:
            try:
                # Try to get affine calibration first
                objectives = self.list_calibrated_objectives()
                if objectives:
                    # Use first available objective
                    affine_matrix = self.get_affine_matrix(objectives[0])
                    # Return just the 2x2 part (ignore translation)
                    return affine_matrix[:, :2]
            except:
                pass
        
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
