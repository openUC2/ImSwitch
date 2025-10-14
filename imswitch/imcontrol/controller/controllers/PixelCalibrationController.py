import json
import os

import numpy as np
import time
import tifffile as tif
import threading
from datetime import datetime
import threading
import cv2
from skimage.registration import phase_cross_correlation
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex, Timer
from imswitch.imcontrol.model import configfiletools, initLogger
import time
from imswitch import IS_HEADLESS


from fractions import Fraction
from uuid import UUID
import logging
import time
import numpy as np
import PIL
import io
import os
import json
from collections import namedtuple

try:
    from camera_stage_mapping.camera_stage_tracker import Tracker
    from camera_stage_mapping.closed_loop_move import closed_loop_move, closed_loop_scan
    from camera_stage_mapping.scan_coords_times import ordered_spiral
    from camera_stage_mapping.affine_stage_calibration import (
        calibrate_affine_transform, validate_calibration, apply_affine_transform
    )
    from camera_stage_mapping.calibration_storage import CalibrationStorage
    IS_CAMERA_STAGE_MAPPING_INSTALLED = True
except ImportError:
    IS_CAMERA_STAGE_MAPPING_INSTALLED = False


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

    def startPixelCalibration(self):
        """Start affine calibration (replaces old calibrate_xy)."""
        if IS_CAMERA_STAGE_MAPPING_INSTALLED:
            self.stageCalibrationAffine(objective_id="default", step_size_um=100.0)

    def stageCalibration(self):
        stageCalibrationT = threading.Thread(target=self.stageCalibrationThread, args=())
        stageCalibrationT.start()

    def stageCalibrationAffine(self, objective_id: str = "default", step_size_um: float = 100.0):
        """
        Start affine calibration in a background thread.
        
        Args:
            objective_id: Identifier for the objective being calibrated
            step_size_um: Step size in microns
        """
        def affine_calibration_thread():
            try:
                csm_extension = CSMExtension(self)
                csm_extension.calibrate_affine(
                    objective_id=objective_id,
                    step_size_um=step_size_um,
                    pattern="cross",
                    n_steps=4,
                    validate=True
                )
            except Exception as e:
                self._logger.error(f"Affine calibration thread failed: {e}")

        calibrationThread = threading.Thread(target=affine_calibration_thread)
        calibrationThread.start()

    def stageCalibrationThread(self, stageName=None, scanMax=100, scanMin=-100, scanStep = 50, rescalingFac=10.0, gridScan=True):
        """Legacy method - now calls affine calibration."""
        csm_extension = CSMExtension(self)
        csm_extension.calibrate_affine(objective_id="default", step_size_um=100.0)







class CSMExtension(object):
    """
    Use the camera as an encoder, so we can relate camera and stage coordinates
    """

    def __init__(self, parent):
        self._parent = parent
        # Initialize calibration storage
        calib_file_path = os.path.join(dirtools.UserFileDirs.Root, "camera_stage_calibration.json")
        if IS_CAMERA_STAGE_MAPPING_INSTALLED:
            self._calibration_storage = CalibrationStorage(calib_file_path, logger=self._parent._logger)
        else:
            self._calibration_storage = None


    def update_settings(self, settings):
        """Update the stored extension settings dictionary"""
        if 0:
            pass
            '''
            keys = ["extensions", self.name]
            dictionary = create_from_path(keys)
            set_by_path(dictionary, keys, settings)
            logging.info(f"Updating settings with {dictionary}")
            self.microscope.update_settings(dictionary)
            self.microscope.save_settings()
            '''

    def get_settings(self):
        """Retrieve the settings for this extension"""
        if 0:
            keys = ["extensions", self.name]
            #return get_by_path(self.microscope.read_settings(), keys)
        return {}

    def camera_stage_functions(self):
        """Return functions that allow us to interface with the microscope"""

        def grabCroppedFrame(crop_size=512):
            marray = self._parent.detector.getLatestFrame()
            center_x, center_y = marray.shape[1] // 2, marray.shape[0] // 2

            # Calculate the starting and ending indices for cropping
            x_start = center_x - crop_size // 2
            x_end = x_start + crop_size
            y_start = center_y - crop_size // 2
            y_end = y_start + crop_size

            # Crop the center region
            return marray[y_start:y_end, x_start:x_end]

        def getPositionList():
            posDict = self._parent._master.positionersManager[self._parent._master.positionersManager.getAllDeviceNames()[0]].getPosition()
            return (posDict["X"], posDict["Y"], posDict["Z"])

        def movePosition(posList):
            stage = self._parent._master.positionersManager[self._parent._master.positionersManager.getAllDeviceNames()[0]]
            stepSizeX = 1
            stepSizeY = 1
            stage.move(value=posList[0]/stepSizeX, axis="X", is_absolute=True, is_blocking=True)
            stage.move(value=posList[1]/stepSizeY, axis="Y", is_absolute=True, is_blocking=True)
            self._parent._logger.info("Moving to: "+str(posList))
            if len(posList)>2:
                stage.move(value=posList[2], axis="Z", is_absolute=True, is_blocking=True)

        grab_image = grabCroppedFrame
        get_position = getPositionList
        move = movePosition
        wait = time.sleep(0.1)

        return grab_image, get_position, move, wait

    def calibrate_affine(
        self,
        objective_id: str = "default",
        step_size_um: float = 100.0,
        pattern: str = "cross",
        n_steps: int = 4,
        validate: bool = True
    ):
        """
        Perform robust affine calibration using the new protocol.
        
        This method provides a more robust approach than calibrate_xy:
        - Uses phase correlation for sub-pixel accuracy
        - Computes full 2x3 affine transformation
        - Includes outlier detection and validation
        - Stores per-objective calibration data
        
        Args:
            objective_id: Identifier for the objective being calibrated
            step_size_um: Step size in microns (50-200 recommended)
            pattern: Movement pattern - "cross" or "grid"
            n_steps: Number of steps in each direction
            validate: Whether to validate the calibration
        
        Returns:
            Dictionary with calibration results
        """
        if not IS_CAMERA_STAGE_MAPPING_INSTALLED:
            raise ImportError("Camera stage mapping module is not available")
        
        self._parent._logger.info(f"Starting affine calibration for objective '{objective_id}'")
        
        # Get camera and stage interface functions
        grab_image, get_position, move, wait = self.camera_stage_functions()
        
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
                logger=self._parent._logger
            )
            
            # Validate calibration if requested
            if validate:
                is_valid, message = validate_calibration(
                    result["affine_matrix"],
                    result["metrics"],
                    logger=self._parent._logger
                )
                result["validation"] = {
                    "is_valid": is_valid,
                    "message": message
                }
                
                if not is_valid:
                    self._parent._logger.warning("Calibration validation failed but data will still be saved")
            
            # Store calibration data
            objective_info = {
                "name": objective_id,
                "detector": self._parent.detector._camera.model if hasattr(self._parent.detector._camera, 'model') else "unknown"
            }
            
            self._calibration_storage.save_calibration(
                objective_id=objective_id,
                affine_matrix=result["affine_matrix"],
                metrics=result["metrics"],
                objective_info=objective_info
            )
            
            self._parent._logger.info(f"Affine calibration completed for objective '{objective_id}'")
            
            # Update widget with results
            if 1:
                metrics = result["metrics"]
                info_text = (
                    f"Calibration completed for {objective_id}\n"
                    f"Quality: {metrics.get('quality', 'unknown')}\n"
                    f"RMSE: {metrics.get('rmse_um', 0):.3f} µm\n"
                    f"Rotation: {metrics.get('rotation_deg', 0):.2f}°\n"
                    f"Scale X: {metrics.get('scale_x_um_per_pixel', 0):.3f} µm/px\n"
                    f"Scale Y: {metrics.get('scale_y_um_per_pixel', 0):.3f} µm/px"
                )
                # TODO: report as socket/signal?

            return result
            
        except Exception as e:
            self._parent._logger.error(f"Affine calibration failed: {e}")

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
        if not IS_CAMERA_STAGE_MAPPING_INSTALLED:
            raise ImportError("Camera stage mapping module is not available")
        
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
        if not IS_CAMERA_STAGE_MAPPING_INSTALLED:
            return []
        return self._calibration_storage.list_objectives()

    @property
    def image_to_stage_displacement_matrix(self):
        """A 2x2 matrix that converts displacement in image coordinates to stage coordinates."""
        if IS_CAMERA_STAGE_MAPPING_INSTALLED and self._calibration_storage:
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
        
        # Fallback to legacy calibration
        try:
            settings = self.get_settings()
            return settings["image_to_stage_displacement"]
        except KeyError:
            raise ValueError("The microscope has not yet been calibrated.")

    def move_in_image_coordinates(self, displacement_in_pixels):
        """Move by a given number of pixels on the camera"""
        p = np.array(displacement_in_pixels)
        relative_move = np.dot(p, self.image_to_stage_displacement_matrix)
        self.microscope.stage.move_rel([relative_move[0], relative_move[1], 0])

    def closed_loop_move_in_image_coordinates(self, displacement_in_pixels, **kwargs):
        """Move by a given number of pixels on the camera, using the camera as an encoder."""
        grab_image, get_position, move, wait = self.camera_stage_functions()

        tracker = Tracker(grab_image, get_position, settle=wait)
        tracker.acquire_template()
        closed_loop_move(tracker, self.move_in_image_coordinates, displacement_in_pixels, **kwargs)

    def closed_loop_scan(self, scan_path, **kwargs):
        """Perform closed-loop moves to each point defined in scan_path.

        This returns a generator, which will move the stage to each point in
        ``scan_path``, then yield ``i, pos`` where ``i``
        is the index of the scan point, and ``pos`` is the estimated position
        in pixels relative to the starting point.  To use it properly, you
        should iterate over it, for example::

            for i, pos in csm_extension.closed_loop_scan(scan_path):
                capture_image(f"image_{i}.jpg")

        ``scan_path`` should be an Nx2 numpy array defining
        the points to visit in pixels relative to the current position.

        If an exception occurs during the scan, we automatically return to the
        starting point.  Keyword arguments are passed to
        ``closed_loop_move.closed_loop_scan``.
        """
        grab_image, get_position, move, wait = self.camera_stage_functions()

        tracker = Tracker(grab_image, get_position, settle=wait)
        tracker.acquire_template()

        return closed_loop_scan(tracker, self.move_in_image_coordinates, move, np.array(scan_path), **kwargs)


    def test_closed_loop_spiral_scan(self, step_size, N, **kwargs):
        """Move the microscope in a spiral scan, and return the positions."""
        scan_path = ordered_spiral(0,0, N, *step_size)

        for i, pos in self.closed_loop_scan(np.array(scan_path), **kwargs):
            pass




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
