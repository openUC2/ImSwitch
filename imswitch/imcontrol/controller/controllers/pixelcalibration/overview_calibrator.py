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

"""
Overview camera-based calibration for stage axes, illumination channels, and homing.

This module provides methods to:
- Identify stage axis directions and signs using AprilTag tracking
- Map illumination channels to colors/wavelengths via image differencing
- Verify homing polarity via AprilTag motion detection
- Fix step size sign via rectangular position moves
- Capture per-objective reference images
"""

import time
import numpy as np
import cv2
import threading
from typing import Dict, List, Tuple, Optional, Any
from imswitch.imcommon.model import initLogger


class OverviewCalibrator:
    """
    Stateless calibration methods using observation camera.
    
    This class provides calibration algorithms that use an overhead/observation camera
    to automatically detect stage configuration and illumination settings.
    All methods are stateless and return result dictionaries without performing disk I/O.
    """
    
    def __init__(self, logger=None, flip_x: bool = False, flip_y: bool = False):
        """
        Initialize the calibrator.
        
        Args:
            logger: Optional logger instance. If None, creates a new logger.
            flip_x: Whether to flip images horizontally (default False)
            flip_y: Whether to flip images vertically (default False)
        """
        self._logger = logger if logger is not None else initLogger(self)
        self._flip_x = flip_x
        self._flip_y = flip_y
        
        # AprilTag detector setup
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        try:
            # OpenCV >= 4.7
            self._aruco_params = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        except AttributeError:
            # Legacy OpenCV
            self._aruco_params = cv2.aruco.DetectorParameters_create()
            self._aruco_detector = None
        
        # Live tracking state
        self._tracking_thread = None
        self._tracking_active = False
        self._tracking_lock = threading.Lock()
    
    def _get_cleared_frame(self, observation_camera, num_clears: int = 5):
        """
        Get latest frame from observation camera with buffer clearing.
        
        Repeatedly calls getLatestFrame() to clear the camera buffer and ensure
        we get the most recent frame, not a buffered one. Applies flip settings
        configured at initialization.
        
        Args:
            observation_camera: Camera detector instance with getLatestFrame() method
            num_clears: Number of frames to retrieve (default 5)
            
        Returns:
            Latest camera frame as numpy array with flips applied
        """
        frame = None
        for _ in range(num_clears):
            frame = observation_camera.getLatestFrame()
        
        # Apply flip settings
        if self._flip_y:
            frame = np.flip(frame, 0)
        if self._flip_x:
            frame = np.flip(frame, 1)
        
        return frame
    
    def detect_tag_centroid(self, img: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect AprilTag centroid in image.
        
        Uses OpenCV's ArUco/AprilTag detection to find tag corners and compute
        the mean position. If multiple tags are detected, returns the average
        of all centroids.
        
        Args:
            img: Image array (grayscale or BGR)
            
        Returns:
            (u, v) centroid coordinates in pixels, or None if no tags detected
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Detect markers
        if self._aruco_detector is not None:
            # OpenCV >= 4.7
            corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        else:
            # Legacy OpenCV
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict, 
                                                       parameters=self._aruco_params)
        
        if ids is None or len(ids) == 0:
            return None
        
        # Compute average centroid of all detected tags
        centroids = []
        for corner in corners:
            pts = corner[0]
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            centroids.append((cx, cy))
        
        if len(centroids) == 0:
            return None
        
        # Average all centroids
        avg_u = np.mean([c[0] for c in centroids])
        avg_v = np.mean([c[1] for c in centroids])
        
        return (float(avg_u), float(avg_v))
    
    def detect_tags_with_ids(self, img: np.ndarray, save_path: Optional[str] = None) -> Dict[int, Tuple[float, float]]:
        """
        Detect AprilTags and return dictionary mapping tag IDs to centroids.
        
        Args:
            img: Image array (grayscale or BGR)
            save_path: Optional path to save annotated image for debugging
            
        Returns:
            Dictionary mapping tag_id -> (u, v) centroid coordinates
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Detect markers
        if self._aruco_detector is not None:
            # OpenCV >= 4.7
            corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        else:
            # Legacy OpenCV
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict, 
                                                       parameters=self._aruco_params)
        
        if ids is None or len(ids) == 0:
            return {}
        
        # Build tag_id -> centroid mapping
        tag_centroids = {}
        for i, tag_id in enumerate(ids.flatten()):
            pts = corners[i][0]
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            tag_centroids[int(tag_id)] = (float(cx), float(cy))
        
        # Optionally save annotated image for debugging
        if save_path is not None:
            self._save_annotated_image(img, corners, ids, save_path)
        
        return tag_centroids
    
    def _save_annotated_image(self, img: np.ndarray, corners, ids, save_path: str):
        """Save image with detected tags annotated."""
        import os
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create color image for annotation
            if len(img.shape) == 2:
                annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                annotated = img.copy()
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            
            # Save image
            cv2.imwrite(save_path, annotated)
            self._logger.debug(f"Saved annotated tag image to {save_path}")
            
        except Exception as e:
            self._logger.warning(f"Failed to save annotated image: {e}")
    
    def _live_tracking_worker(self, observation_camera, positioner, update_interval: float):
        """
        Worker thread for continuous tag tracking.
        
        Continuously captures frames, detects tags, and prints movement information
        relative to the initial position and tags.
        
        Args:
            observation_camera: Camera detector instance
            positioner: Stage positioner instance
            update_interval: Time between updates in seconds
        """
        try:
            # Get initial state
            initial_frame = self._get_cleared_frame(observation_camera)
            initial_tags = self.detect_tags_with_ids(initial_frame)
            initial_pos = positioner.getPosition()
            
            if not initial_tags:
                self._logger.warning("Live tracking: No tags detected in initial frame")
                return
            
            self._logger.info(f"Live tracking started with {len(initial_tags)} tags")
            self._logger.info(f"Initial stage position: X={initial_pos.get('X', 0):.1f}, Y={initial_pos.get('Y', 0):.1f}")
            
            # Print header
            print("\n" + "="*80)
            print("LIVE TAG TRACKING - Press stop_live_tracking() to end")
            print("="*80)
            print(f"{'Time(s)':<10} {'Stage X(um)':<15} {'Stage Y(um)':<15} {'Avg ΔU(px)':<15} {'Avg ΔV(px)':<15} {'Tags':<10}")
            print("-"*80)
            
            start_time = time.time()
            
            while True:
                with self._tracking_lock:
                    if not self._tracking_active:
                        break
                
                # Capture current frame
                current_frame = self._get_cleared_frame(observation_camera, num_clears=1)
                current_tags = self.detect_tags_with_ids(current_frame)
                current_pos = positioner.getPosition()
                
                # Calculate stage displacement
                stage_dx = current_pos.get("X", 0) - initial_pos.get("X", 0)
                stage_dy = current_pos.get("Y", 0) - initial_pos.get("Y", 0)
                
                # Calculate average tag displacement
                common_tags = set(initial_tags.keys()) & set(current_tags.keys())
                
                if common_tags:
                    shifts = []
                    for tag_id in common_tags:
                        u0, v0 = initial_tags[tag_id]
                        u1, v1 = current_tags[tag_id]
                        shifts.append((u1 - u0, v1 - v0))
                    
                    avg_du = np.mean([s[0] for s in shifts])
                    avg_dv = np.mean([s[1] for s in shifts])
                    
                    elapsed = time.time() - start_time
                    
                    # Print update
                    print(f"{elapsed:<10.1f} {stage_dx:<15.1f} {stage_dy:<15.1f} {avg_du:<15.1f} {avg_dv:<15.1f} {len(common_tags):<10}")
                else:
                    elapsed = time.time() - start_time
                    print(f"{elapsed:<10.1f} {stage_dx:<15.1f} {stage_dy:<15.1f} {'N/A':<15} {'N/A':<15} {0:<10}")
                
                time.sleep(update_interval)
            
            print("-"*80)
            print("Live tracking stopped")
            print("="*80 + "\n")
            
        except Exception as e:
            self._logger.error(f"Live tracking error: {e}", exc_info=True)
            with self._tracking_lock:
                self._tracking_active = False
    
    def start_live_tracking(self, observation_camera, positioner, 
                           update_interval: float = 0.5) -> Dict[str, Any]:
        """
        Start continuous live tracking of tags and stage position.
        
        Launches a background thread that continuously monitors tag positions
        and prints movement information relative to the starting position.
        
        Args:
            observation_camera: Camera detector instance with getLatestFrame() method
            positioner: Stage positioner instance with getPosition() method
            update_interval: Time between updates in seconds (default 0.5)
            
        Returns:
            Dictionary with:
                - status: "started" or "already_running"
                - message: Status message
                - error: Error message if startup failed
        """
        try:
            with self._tracking_lock:
                if self._tracking_active:
                    return {
                        "status": "already_running",
                        "message": "Live tracking is already active"
                    }
                
                self._tracking_active = True
            
            # Start tracking thread
            self._tracking_thread = threading.Thread(
                target=self._live_tracking_worker,
                args=(observation_camera, positioner, update_interval),
                daemon=True
            )
            self._tracking_thread.start()
            
            self._logger.info("Live tracking started")
            
            return {
                "status": "started",
                "message": f"Live tracking started with {update_interval}s update interval",
                "updateInterval": update_interval
            }
            
        except Exception as e:
            self._logger.error(f"Failed to start live tracking: {e}", exc_info=True)
            with self._tracking_lock:
                self._tracking_active = False
            return {"error": str(e), "status": "failed"}
    
    def stop_live_tracking(self) -> Dict[str, Any]:
        """
        Stop continuous live tracking.
        
        Signals the tracking thread to stop and waits for it to finish.
        
        Returns:
            Dictionary with:
                - status: "stopped" or "not_running"
                - message: Status message
        """
        try:
            with self._tracking_lock:
                if not self._tracking_active:
                    return {
                        "status": "not_running",
                        "message": "Live tracking is not currently active"
                    }
                
                self._tracking_active = False
            
            # Wait for thread to finish
            if self._tracking_thread is not None:
                self._tracking_thread.join(timeout=2.0)
                if self._tracking_thread.is_alive():
                    self._logger.warning("Tracking thread did not stop cleanly")
            
            self._logger.info("Live tracking stopped")
            
            return {
                "status": "stopped",
                "message": "Live tracking stopped successfully"
            }
            
        except Exception as e:
            self._logger.error(f"Error stopping live tracking: {e}", exc_info=True)
            return {"error": str(e), "status": "error"}
    
    def get_live_tracking_status(self) -> Dict[str, Any]:
        """
        Get the current status of live tracking.
        
        Returns:
            Dictionary with:
                - active: Boolean indicating if tracking is active
                - thread_alive: Boolean indicating if thread is running
        """
        with self._tracking_lock:
            active = self._tracking_active
        
        thread_alive = self._tracking_thread is not None and self._tracking_thread.is_alive()
        
        return {
            "active": active,
            "thread_alive": thread_alive
        }
    
    def identify_axes(self, observation_camera, positioner, step_um: float = 2000.0,
                     settle_time: float = 0.3, save_debug_images: bool = False,
                     debug_dir: Optional[str] = None, speed:float = 15000.0) -> Dict[str, Any]:
        """
        Identify stage axis mapping and signs using AprilTag tracking.
        
        Moves the stage in +X and +Y directions and tracks the AprilTag displacement
        in camera coordinates to determine:
        - Which camera axis (width/height) corresponds to stage X/Y
        - The sign of each axis (±1)
        
        Uses multi-tag tracking: detects all tags in the initial frame, then tracks
        individual tags across movements and averages their shifts for robustness.
        
        Args:
            observation_camera: Camera detector instance with getLatestFrame() method
            positioner: Stage positioner instance with move() and getPosition() methods
            step_um: Step size in micrometers (default 2000)
            settle_time: Time to wait after movement for settling (seconds)
            save_debug_images: If True, save annotated images for debugging
            debug_dir: Directory for debug images (required if save_debug_images=True)
            
        Returns:
            Dictionary with:
                - mapping: {stageX_to_cam: "width"|"height", stageY_to_cam: "width"|"height"}
                - sign: {X: +1|-1, Y: +1|-1}
                - samples: List of movement samples with stage_move and cam_shift
                - tracked_tags: List of tag IDs that were successfully tracked
                - error: Error message if detection failed
                
        response from http request (without the samples):
        {
        "mapping": {
            "stageX_to_cam": "width",
            "stageY_to_cam": "height"
        },
        "sign": {
            "X": 1,
            "Y": 1
        }
        }
        """
        import os
        
        try:
            # Get initial position
            initial_pos = positioner.getPosition()
            p0_x = initial_pos.get("X", 0)
            p0_y = initial_pos.get("Y", 0)
            
            time.sleep(settle_time)
            frame0 = self._get_cleared_frame(observation_camera)
            
            # Detect all tags in initial frame
            save_path_0 = None
            if save_debug_images and debug_dir:
                save_path_0 = os.path.join(debug_dir, "identify_axes_initial.png")
            
            tags_initial = self.detect_tags_with_ids(frame0, save_path=save_path_0)
            
            if not tags_initial:
                return {"error": "No AprilTags detected in initial frame"}
            
            self._logger.info(f"Detected {len(tags_initial)} tags in initial frame: {list(tags_initial.keys())}")
            
            samples = []
            
            # Test +X movement
            self._logger.info(f"Moving +{step_um} µm in X direction")
            positioner.move(value=p0_x + step_um, axis="X", is_absolute=True, is_blocking=True, speed=speed)
            time.sleep(settle_time)
            
            frame_x = self._get_cleared_frame(observation_camera)
                
            save_path_x = None
            if save_debug_images and debug_dir:
                save_path_x = os.path.join(debug_dir, "identify_axes_x_move.png")
            
            tags_after_x = self.detect_tags_with_ids(frame_x, save_path=save_path_x)
            
            if not tags_after_x:
                positioner.move(value=p0_x, axis="X", is_absolute=True, is_blocking=True, speed=speed)
                return {"error": "All AprilTags lost after X movement"}
            self._logger.info(f"Detected {len(tags_after_x)} tags after X movement: {list(tags_after_x.keys())}")
            # Track common tags between initial and post-X frames
            common_tags_x = set(tags_initial.keys()) & set(tags_after_x.keys())
            
            if not common_tags_x:
                positioner.move(value=p0_x, axis="X", is_absolute=True, is_blocking=True, speed=speed)
                return {"error": "No common tags found after X movement"}
            
            # Compute average shift for tracked tags
            shifts_x = []
            for tag_id in common_tags_x:
                u0, v0 = tags_initial[tag_id]
                u_x, v_x = tags_after_x[tag_id]
                du_x = u_x - u0
                dv_x = v_x - v0
                shifts_x.append((du_x, dv_x))
            
            # Average shifts across all tracked tags
            du_x = np.mean([s[0] for s in shifts_x])
            dv_x = np.mean([s[1] for s in shifts_x])
            
            self._logger.info(f"X movement: tracked {len(common_tags_x)} tags, avg shift: ({du_x:.1f}, {dv_x:.1f}) px")
            # e.g. 2025-11-09 13:07:16 INFO [PixelCalibrationController] X movement: tracked 17 tags, avg shift: (34.9, -0.8) px
            samples.append({
                "stageMove": [step_um, 0],
                "camShift": [float(du_x), float(dv_x)],
                "trackedTags": list(common_tags_x),
                "individualShifts": [{"tag_id": int(tid), "du": float(s[0]), "dv": float(s[1])} 
                                     for tid, s in zip(common_tags_x, shifts_x)]
            })
            
            # Return to origin
            positioner.move(value=p0_x, axis="X", is_absolute=True, is_blocking=True, speed=speed)
            time.sleep(settle_time)
            
            # Test +Y movement
            self._logger.info(f"Moving +{step_um} µm in Y direction")
            positioner.move(value=p0_y + step_um, axis="Y", is_absolute=True, is_blocking=True, speed=speed)
            time.sleep(settle_time)
            
            frame_y = self._get_cleared_frame(observation_camera)
            
            save_path_y = None
            if save_debug_images and debug_dir:
                save_path_y = os.path.join(debug_dir, "identify_axes_y_move.png")
            
            tags_after_y = self.detect_tags_with_ids(frame_y, save_path=save_path_y)
            self._logger.info(f"Detected {len(tags_after_y)} tags after Y movement: {list(tags_after_y.keys())}")
            if not tags_after_y:
                positioner.move(value=p0_y, axis="Y", is_absolute=True, is_blocking=True, speed=speed)
                return {"error": "All AprilTags lost after Y movement"}
            
            # Track common tags between initial and post-Y frames
            common_tags_y = set(tags_initial.keys()) & set(tags_after_y.keys())
            
            if not common_tags_y:
                positioner.move(value=p0_y, axis="Y", is_absolute=True, is_blocking=True, speed=speed)
                return {"error": "No common tags found after Y movement"}
            
            # Compute average shift for tracked tags
            shifts_y = []
            for tag_id in common_tags_y:
                u0, v0 = tags_initial[tag_id]
                u_y, v_y = tags_after_y[tag_id]
                du_y = u_y - u0
                dv_y = v_y - v0
                shifts_y.append((du_y, dv_y))
            
            # Average shifts across all tracked tags
            du_y = np.mean([s[0] for s in shifts_y])
            dv_y = np.mean([s[1] for s in shifts_y])
            
            self._logger.info(f"Y movement: tracked {len(common_tags_y)} tags, avg shift: ({du_y:.1f}, {dv_y:.1f}) px")
            # e.g. 2025-11-09 13:07:35 INFO [PixelCalibrationController] Y movement: tracked 16 tags, avg shift: (-0.0, 27.1) px
            samples.append({
                "stageMove": [0, step_um],
                "camShift": [float(du_y), float(dv_y)],
                "trackedTags": list(common_tags_y),
                "individualShifts": [{"tag_id": int(tid), "du": float(s[0]), "dv": float(s[1])} 
                                     for tid, s in zip(common_tags_y, shifts_y)]
            })
            
            # Return to origin
            positioner.move(value=p0_y, axis="Y", is_absolute=True, is_blocking=True, speed=speed)
            
            # Determine axis mapping
            # X movement: if |du_x| > |dv_x| => X aligns with camera width (u)
            if abs(du_x) >= abs(dv_x):
                stage_x_maps_to = "width"
                sign_x = 1 if du_x > 0 else -1
            else:
                stage_x_maps_to = "height"
                sign_x = 1 if dv_x > 0 else -1
            
            # Y movement: if |dv_y| > |du_y| => Y aligns with camera height (v)
            if abs(dv_y) >= abs(du_y):
                stage_y_maps_to = "height"
                sign_y = 1 if dv_y > 0 else -1
            else:
                stage_y_maps_to = "width"
                sign_y = 1 if du_y > 0 else -1
            
            # Collect all successfully tracked tags
            all_tracked = list(set(common_tags_x) | set(common_tags_y))
            
            result = {
                "mapping": {
                    "stageX_to_cam": stage_x_maps_to,
                    "stageY_to_cam": stage_y_maps_to
                },
                "sign": {
                    "X": int(sign_x),
                    "Y": int(sign_y)
                },
                "samples": samples,
                "trackedTags": all_tracked
            }
            
            self._logger.info(f"Axis identification complete: {result['mapping']}, signs: {result['sign']}")
            return result
            
        except Exception as e:
            self._logger.error(f"Axis identification failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def map_illumination_channels(self, observation_camera, lasers_manager=None, 
                                  leds_manager=None, settle_time: float = 0.3) -> Dict[str, Any]:
        """
        Map illumination channels to colors/wavelengths using image differencing.
        
        Acquires a dark reference frame, then for each illumination channel:
        1. Turns on the channel
        2. Captures an image
        3. Computes difference from dark frame
        4. Analyzes color/intensity to classify the channel
        
        Args:
            observation_camera: Camera detector with getLatestFrame() method
            lasers_manager: Optional laser manager with device enumeration
            leds_manager: Optional LED manager with device enumeration
            settle_time: Time to wait after turning on illumination (seconds)
            
        Returns:
            Dictionary with:
                - illuminationMap: List of {channel, wavelength_nm, color, mean_intensity}
                - darkStats: Statistics from dark frame
                - error: Error message if something failed
        """
        try:
            illumination_map = []
            
            # Turn off all illumination and capture dark frame
            self._logger.info("Capturing dark reference frame")
            if lasers_manager is not None:
                for laser_name in lasers_manager.getAllDeviceNames():
                    lasers_manager[laser_name].setValue(0)
            
            if leds_manager is not None:
                for led_name in leds_manager.getAllDeviceNames():
                    leds_manager[led_name].setValue(0)
            
            time.sleep(settle_time)
            dark_frame = self._get_cleared_frame(observation_camera)
            dark_mean = float(np.mean(dark_frame))
            dark_max = float(np.max(dark_frame))
            
            dark_stats = {
                "mean": dark_mean,
                "max": dark_max,
                "shape": list(dark_frame.shape)
            }
            
            # Process laser channels
            if lasers_manager is not None:
                for laser_name in lasers_manager.getAllDeviceNames():
                    laser = lasers_manager[laser_name]
                    
                    # Get nominal wavelength from laser info
                    laser_info = lasers_manager[laser_name]
                    nominal_wavelength = laser_info.wavelength if laser_info else None
                    
                    # Turn on laser at moderate power
                    max_value = laser_info.valueRangeMax if laser_info else 100
                    test_value = max_value * 0.5  # Use 50% power
                    
                    self._logger.info(f"Testing laser: {laser_name} at {test_value}")
                    laser.setValue(test_value)
                    time.sleep(settle_time)
                    
                    # Capture frame
                    frame = self._get_cleared_frame(observation_camera)
                    diff = frame.astype(np.float32) - dark_frame.astype(np.float32)
                    diff = np.clip(diff, 0, None)
                    
                    # Compute statistics
                    mean_intensity = float(np.mean(diff))
                    max_intensity = float(np.max(diff))
                    
                    # Classify color based on wavelength
                    if nominal_wavelength:
                        color = self._classify_wavelength(nominal_wavelength)
                    else:
                        color = self._classify_from_intensity(diff)
                    
                    illumination_map.append({
                        "channel": laser_name,
                        "type": "laser",
                        "wavelength_nm": float(nominal_wavelength) if nominal_wavelength else None,
                        "color": color,
                        "mean_intensity": mean_intensity,
                        "max_intensity": max_intensity
                    })
                    
                    # Turn off laser
                    laser.setValue(0)
                    time.sleep(settle_time * 0.5)
            
            # Process LED channels
            if leds_manager is not None:
                for led_name in leds_manager.getAllDeviceNames():
                    led = leds_manager[led_name]
                    
                    # Get nominal wavelength from LED info if available
                    led_info = leds_manager[led_name]
                    # LEDs might not have wavelength info
                    nominal_wavelength = None
                    
                    # Turn on LED at moderate power
                    max_value = led_info.valueRangeMax if led_info else 100
                    test_value = max_value * 0.5
                    
                    self._logger.info(f"Testing LED: {led_name} at {test_value}")
                    led.setValue(test_value)
                    time.sleep(settle_time)
                    
                    # Capture frame
                    frame = self._get_cleared_frame(observation_camera)
                    diff = frame.astype(np.float32) - dark_frame.astype(np.float32)
                    diff = np.clip(diff, 0, None)
                    
                    # Compute statistics
                    mean_intensity = float(np.mean(diff))
                    max_intensity = float(np.max(diff))
                    
                    # Classify color from image content
                    color = self._classify_from_intensity(diff)
                    
                    illumination_map.append({
                        "channel": led_name,
                        "type": "led",
                        "wavelength_nm": float(nominal_wavelength) if nominal_wavelength else None,
                        "color": color,
                        "mean_intensity": mean_intensity,
                        "max_intensity": max_intensity
                    })
                    
                    # Turn off LED
                    led.setValue(0)
                    time.sleep(settle_time * 0.5)
            
            return {
                "illuminationMap": illumination_map,
                "darkStats": dark_stats
            }
            
        except Exception as e:
            self._logger.error(f"Illumination mapping failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _classify_wavelength(self, wavelength_nm: float) -> str:
        """Classify color based on wavelength."""
        if wavelength_nm < 410:
            return "uv"
        elif wavelength_nm < 500:
            return "blue"
        elif wavelength_nm < 580:
            return "green"
        elif wavelength_nm < 600:
            return "yellow"
        elif wavelength_nm < 700:
            return "red"
        elif wavelength_nm < 780:
            return "far-red"
        else:
            return "ir"
    
    def _classify_from_intensity(self, diff_image: np.ndarray) -> str:
        """
        Classify color from difference image.
        
        For color images, analyzes RGB channels.
        For grayscale, returns "white" or "unknown".
        """
        if len(diff_image.shape) == 3 and diff_image.shape[2] >= 3:
            # Color image - analyze BGR channels
            b = np.mean(diff_image[:, :, 0])
            g = np.mean(diff_image[:, :, 1])
            r = np.mean(diff_image[:, :, 2])
            
            # Normalize
            total = b + g + r
            if total < 1e-6:
                return "unknown"
            
            b_norm = b / total
            g_norm = g / total
            r_norm = r / total
            
            # Simple classification
            if r_norm > 0.5:
                return "red"
            elif g_norm > 0.5:
                return "green"
            elif b_norm > 0.5:
                return "blue"
            elif r_norm > 0.35 and g_norm > 0.35:
                return "yellow"
            elif r_norm > 0.3 and g_norm > 0.3 and b_norm > 0.3:
                return "white"
            else:
                return "unknown"
        else:
            # Grayscale - just check if there's signal
            mean_val = np.mean(diff_image)
            if mean_val > 10:
                return "white"
            else:
                return "unknown"
    
    def verify_homing(self, observation_camera, positioner, max_time_s: float = 20.0,
                     check_interval: float = 0.5) -> Dict[str, Any]:
        """
        Verify homing behavior and detect inverted motor directions.
        
        Attempts to home each axis and monitors AprilTag motion. If homing times out
        but the tag motion stops near field edge, recommends inverting the motor direction.
        
        Args:
            observation_camera: Camera detector with getLatestFrame() method
            positioner: Stage positioner with home() and getPosition() methods
            max_time_s: Maximum time to wait for homing (seconds)
            check_interval: Time between motion checks (seconds)
            
        Returns:
            Dictionary with X and Y results:
                {axis: {inverted: bool, evidence: samples, error: str}}
        """
        result = {}
        
        for axis in ["X", "Y"]:
            try:
                # TODO: Instead of 
                self._logger.info(f"Verifying homing for axis {axis}")
                
                # Get initial centroid
                frame0 = self._get_cleared_frame(observation_camera)
                c0 = self.detect_tag_centroid(frame0)
                
                if c0 is None:
                    result[axis] = {"error": "No AprilTag detected", "inverted": False}
                    continue
                
                # Start homing
                start_time = time.time()
                homing_complete = False
                motion_samples = []
                
                # Trigger homing (non-blocking if possible)
                try:
                    # Try non-blocking home
                    positioner.home(axis=axis, is_blocking=False) # TODO: Need to do home_x, home_y 
                except:
                    # Fallback to blocking home with timeout handling
                    pass
                
                # Monitor motion
                last_centroid = c0
                motion_stopped_time = None
                
                while (time.time() - start_time) < max_time_s:
                    time.sleep(check_interval)
                    
                    # Check current position
                    frame = self._get_cleared_frame(observation_camera, num_clears=1)
                    centroid = self.detect_tag_centroid(frame)
                    
                    if centroid is None:
                        # Tag lost - might have moved out of frame
                        motion_samples.append({
                            "time": time.time() - start_time,
                            "centroid": None,
                            "note": "tag_lost"
                        })
                        break
                    
                    # Compute motion
                    du = centroid[0] - last_centroid[0]
                    dv = centroid[1] - last_centroid[1]
                    motion = np.sqrt(du**2 + dv**2)
                    
                    motion_samples.append({
                        "time": time.time() - start_time,
                        "centroid": [float(centroid[0]), float(centroid[1])],
                        "motion": float(motion)
                    })
                    
                    # Check if motion stopped
                    if motion < 2.0:  # pixels
                        if motion_stopped_time is None:
                            motion_stopped_time = time.time()
                        elif (time.time() - motion_stopped_time) > 1.0:
                            # Motion stopped for >1 second
                            homing_complete = True
                            break
                    else:
                        motion_stopped_time = None
                    
                    last_centroid = centroid
                
                # Analyze results
                if homing_complete:
                    result[axis] = {
                        "inverted": False,
                        "evidence": motion_samples,
                        "lastCheck": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                    self._logger.info(f"Axis {axis} homing completed normally")
                else:
                    # Check if tag was lost (possible sign of reaching edge)
                    tag_lost = any(s.get("note") == "tag_lost" for s in motion_samples)
                    
                    result[axis] = {
                        "inverted": tag_lost,
                        "evidence": motion_samples,
                        "lastCheck": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "recommendation": "Invert motor direction" if tag_lost else "Check homing setup"
                    }
                    
                    if tag_lost:
                        self._logger.warning(f"Axis {axis} homing may need inversion (tag lost)")
                    else:
                        self._logger.warning(f"Axis {axis} homing timed out")
                
            except Exception as e:
                self._logger.error(f"Homing verification failed for {axis}: {e}", exc_info=True)
                result[axis] = {"error": str(e), "inverted": False}
        
        return result
    
    def fix_step_sign(self, observation_camera, positioner, rect_size_um: float = 20000.0,
                     settle_time: float = 0.3) -> Dict[str, Any]:
        """
        Determine and fix step size sign by visiting rectangle corners.
        
        Moves to 4 corners of a rectangle and compares stage displacement with
        camera displacement using AprilTag tracking. Detects if the sign needs
        to be inverted.
        
        Args:
            observation_camera: Camera detector with getLatestFrame() method
            positioner: Stage positioner with move() and getPosition() methods
            rect_size_um: Rectangle size in micrometers (default 20000)
            settle_time: Time to wait after movement (seconds)
            
        Returns:
            Dictionary with:
                - sign: {X: ±1, Y: ±1}
                - samples: List of position samples with stage_pos and cam_centroid
                - error: Error message if detection failed
        """
        try:
            # Get initial position
            initial_pos = positioner.getPosition()
            x0 = initial_pos.get("X", 0)
            y0 = initial_pos.get("Y", 0)
            
            # Define rectangle corners relative to start
            corners = [
                (0, 0),                          # Origin
                (rect_size_um, 0),              # Right
                (rect_size_um, rect_size_um),   # Top-right
                (0, rect_size_um)               # Top
            ]
            
            samples = []
            
            for i, (dx, dy) in enumerate(corners):
                # Move to corner
                target_x = x0 + dx
                target_y = y0 + dy
                
                self._logger.info(f"Moving to corner {i+1}/4: ({dx}, {dy}) µm offset")
                positioner.move(value=target_x, axis="X", is_absolute=True, is_blocking=True)
                positioner.move(value=target_y, axis="Y", is_absolute=True, is_blocking=True)
                time.sleep(settle_time)
                
                # Get actual position and centroid
                pos = positioner.getPosition()
                frame = self._get_cleared_frame(observation_camera)
                centroid = self.detect_tag_centroid(frame)
                
                if centroid is None:
                    # Try to return to start
                    positioner.move(value=x0, axis="X", is_absolute=True, is_blocking=True)
                    positioner.move(value=y0, axis="Y", is_absolute=True, is_blocking=True)
                    return {"error": f"AprilTag lost at corner {i+1}"}
                
                samples.append({
                    "stage_pos": [float(pos.get("X", 0)), float(pos.get("Y", 0))],
                    "cam_centroid": [float(centroid[0]), float(centroid[1])],
                    "target": [dx, dy]
                })
            
            # Return to origin
            positioner.move(value=x0, axis="X", is_absolute=True, is_blocking=True)
            positioner.move(value=y0, axis="Y", is_absolute=True, is_blocking=True)
            
            # Analyze displacement patterns
            # Compare stage movement with camera movement
            stage_dx = samples[1]["stage_pos"][0] - samples[0]["stage_pos"][0]
            cam_du = samples[1]["cam_centroid"][0] - samples[0]["cam_centroid"][0]
            cam_dv = samples[1]["cam_centroid"][1] - samples[0]["cam_centroid"][1]
            
            stage_dy = samples[3]["stage_pos"][1] - samples[0]["stage_pos"][1]
            cam_du_y = samples[3]["cam_centroid"][0] - samples[0]["cam_centroid"][0]
            cam_dv_y = samples[3]["cam_centroid"][1] - samples[0]["cam_centroid"][1]
            
            # Determine which camera axis aligns with which stage axis
            # and check sign consistency
            
            # For X: check if camera displacement is consistent with stage X
            if abs(cam_du) > abs(cam_dv):
                # X aligns with camera u (width)
                sign_x = 1 if (stage_dx * cam_du) > 0 else -1
            else:
                # X aligns with camera v (height)
                sign_x = 1 if (stage_dx * cam_dv) > 0 else -1
            
            # For Y: check if camera displacement is consistent with stage Y
            if abs(cam_dv_y) > abs(cam_du_y):
                # Y aligns with camera v (height)
                sign_y = 1 if (stage_dy * cam_dv_y) > 0 else -1
            else:
                # Y aligns with camera u (width)
                sign_y = 1 if (stage_dy * cam_du_y) > 0 else -1
            
            result = {
                "sign": {
                    "X": int(sign_x),
                    "Y": int(sign_y)
                },
                "samples": samples
            }
            
            self._logger.info(f"Step sign determination complete: {result['sign']}")
            return result
            
        except Exception as e:
            self._logger.error(f"Step sign determination failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def capture_objective_image(self, observation_camera, slot: int, 
                               save_dir: str) -> Dict[str, Any]:
        """
        Capture reference image for a specific objective slot.
        
        Args:
            observation_camera: Camera detector with getLatestFrame() method
            slot: Objective slot number
            save_dir: Directory to save the image
            
        Returns:
            Dictionary with:
                - slot: Objective slot number
                - path: Saved image path
                - error: Error message if capture failed
        """
        import os
        
        try:
            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Capture frame
            frame = self._get_cleared_frame(observation_camera)
            
            # Generate filename
            filename = f"objective{slot}_calibration.png"
            filepath = os.path.join(save_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, frame)
            
            self._logger.info(f"Captured objective {slot} image: {filepath}")
            
            return {
                "slot": slot,
                "path": filepath
            }
            
        except Exception as e:
            self._logger.error(f"Image capture failed for slot {slot}: {e}", exc_info=True)
            return {"error": str(e), "slot": slot}
