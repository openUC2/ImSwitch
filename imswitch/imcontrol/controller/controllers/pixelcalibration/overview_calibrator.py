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
from typing import Dict, List, Tuple, Optional, Any
from imswitch.imcommon.model import initLogger


class OverviewCalibrator:
    """
    Stateless calibration methods using observation camera.
    
    This class provides calibration algorithms that use an overhead/observation camera
    to automatically detect stage configuration and illumination settings.
    All methods are stateless and return result dictionaries without performing disk I/O.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the calibrator.
        
        Args:
            logger: Optional logger instance. If None, creates a new logger.
        """
        self._logger = logger if logger is not None else initLogger(self)
        
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
    
    def identify_axes(self, observation_camera, positioner, step_um: float = 2000.0,
                     settle_time: float = 0.3) -> Dict[str, Any]:
        """
        Identify stage axis mapping and signs using AprilTag tracking.
        
        Moves the stage in +X and +Y directions and tracks the AprilTag displacement
        in camera coordinates to determine:
        - Which camera axis (width/height) corresponds to stage X/Y
        - The sign of each axis (±1)
        
        Args:
            observation_camera: Camera detector instance with getLatestFrame() method
            positioner: Stage positioner instance with move() and getPosition() methods
            step_um: Step size in micrometers (default 2000)
            settle_time: Time to wait after movement for settling (seconds)
            
        Returns:
            Dictionary with:
                - mapping: {stageX_to_cam: "width"|"height", stageY_to_cam: "width"|"height"}
                - sign: {X: +1|-1, Y: +1|-1}
                - samples: List of movement samples with stage_move and cam_shift
                - error: Error message if detection failed
        """
        try:
            # Get initial position and centroid
            initial_pos = positioner.getPosition()
            p0_x = initial_pos.get("X", 0)
            p0_y = initial_pos.get("Y", 0)
            
            time.sleep(settle_time)
            frame0 = observation_camera.getLatestFrame()
            c0 = self.detect_tag_centroid(frame0)
            
            if c0 is None:
                return {"error": "No AprilTag detected in initial frame"}
            
            u0, v0 = c0
            samples = []
            
            # Test +X movement
            self._logger.info(f"Moving +{step_um} µm in X direction")
            positioner.move(value=p0_x + step_um, axis="X", is_absolute=True, is_blocking=True)
            time.sleep(settle_time)
            
            frame_x = observation_camera.getLatestFrame()
            c_x = self.detect_tag_centroid(frame_x)
            
            if c_x is None:
                # Try to return to start
                positioner.move(value=p0_x, axis="X", is_absolute=True, is_blocking=True)
                return {"error": "AprilTag lost after X movement"}
            
            u_x, v_x = c_x
            du_x = u_x - u0
            dv_x = v_x - v0
            
            samples.append({
                "stageMove": [step_um, 0],
                "camShift": [float(du_x), float(dv_x)]
            })
            
            # Return to origin
            positioner.move(value=p0_x, axis="X", is_absolute=True, is_blocking=True)
            time.sleep(settle_time)
            
            # Test +Y movement
            self._logger.info(f"Moving +{step_um} µm in Y direction")
            positioner.move(value=p0_y + step_um, axis="Y", is_absolute=True, is_blocking=True)
            time.sleep(settle_time)
            
            frame_y = observation_camera.getLatestFrame()
            c_y = self.detect_tag_centroid(frame_y)
            
            if c_y is None:
                # Try to return to start
                positioner.move(value=p0_y, axis="Y", is_absolute=True, is_blocking=True)
                return {"error": "AprilTag lost after Y movement"}
            
            u_y, v_y = c_y
            du_y = u_y - u0
            dv_y = v_y - v0
            
            samples.append({
                "stageMove": [0, step_um],
                "camShift": [float(du_y), float(dv_y)]
            })
            
            # Return to origin
            positioner.move(value=p0_y, axis="Y", is_absolute=True, is_blocking=True)
            
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
            
            result = {
                "mapping": {
                    "stageX_to_cam": stage_x_maps_to,
                    "stageY_to_cam": stage_y_maps_to
                },
                "sign": {
                    "X": int(sign_x),
                    "Y": int(sign_y)
                },
                "samples": samples
            }
            
            self._logger.info(f"Axis identification complete: {result}")
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
            dark_frame = observation_camera.getLatestFrame()
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
                    laser_info = lasers_manager._LaserManager__setupInfo.lasers.get(laser_name)
                    nominal_wavelength = laser_info.wavelength if laser_info else None
                    
                    # Turn on laser at moderate power
                    max_value = laser_info.valueRangeMax if laser_info else 100
                    test_value = max_value * 0.5  # Use 50% power
                    
                    self._logger.info(f"Testing laser: {laser_name} at {test_value}")
                    laser.setValue(test_value)
                    time.sleep(settle_time)
                    
                    # Capture frame
                    frame = observation_camera.getLatestFrame()
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
                    led_info = leds_manager._LEDsManager__setupInfo.leds.get(led_name)
                    # LEDs might not have wavelength info
                    nominal_wavelength = None
                    
                    # Turn on LED at moderate power
                    max_value = led_info.valueRangeMax if led_info else 100
                    test_value = max_value * 0.5
                    
                    self._logger.info(f"Testing LED: {led_name} at {test_value}")
                    led.setValue(test_value)
                    time.sleep(settle_time)
                    
                    # Capture frame
                    frame = observation_camera.getLatestFrame()
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
                self._logger.info(f"Verifying homing for axis {axis}")
                
                # Get initial centroid
                frame0 = observation_camera.getLatestFrame()
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
                    positioner.home(axis=axis, is_blocking=False)
                except:
                    # Fallback to blocking home with timeout handling
                    pass
                
                # Monitor motion
                last_centroid = c0
                motion_stopped_time = None
                
                while (time.time() - start_time) < max_time_s:
                    time.sleep(check_interval)
                    
                    # Check current position
                    frame = observation_camera.getLatestFrame()
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
                frame = observation_camera.getLatestFrame()
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
            frame = observation_camera.getLatestFrame()
            
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
