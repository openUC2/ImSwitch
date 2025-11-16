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
AprilTag grid-based stage calibration and navigation.

This module provides:
- Grid LUT for mapping tag IDs to (row, col) positions
- Camera-to-stage affine transformation calibration
- Automatic navigation to specific tag IDs with closed-loop feedback
"""

import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for AprilTag grid layout."""
    rows: int
    cols: int
    start_id: int
    pitch_mm: float  # Physical spacing between tag centers
    
    def id_to_rowcol(self, tag_id: int) -> Optional[Tuple[int, int]]:
        """
        Convert tag ID to (row, col) position in grid.
        
        Args:
            tag_id: AprilTag ID
            
        Returns:
            (row, col) tuple or None if ID is out of range
        """
        offset = tag_id - self.start_id
        if offset < 0 or offset >= (self.rows * self.cols):
            return None
        
        row = offset // self.cols
        col = offset % self.cols
        return (row, col)
    
    def rowcol_to_id(self, row: int, col: int) -> Optional[int]:
        """
        Convert (row, col) position to tag ID.
        
        Args:
            row: Grid row index
            col: Grid column index
            
        Returns:
            Tag ID or None if position is out of range
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        
        offset = row * self.cols + col
        return self.start_id + offset
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "start_id": self.start_id,
            "pitch_mm": self.pitch_mm
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridConfig':
        """Deserialize from dictionary."""
        return cls(
            rows=data["rows"],
            cols=data["cols"],
            start_id=data["start_id"],
            pitch_mm=data["pitch_mm"]
        )


class AprilTagGridCalibrator:
    """
    AprilTag grid-based stage calibration and navigation.
    
    This class provides methods to:
    1. Calibrate camera-to-stage transformation using detected tags
    2. Navigate to specific tag IDs using closed-loop feedback
    3. Handle oblique/trapezoidal views via affine transforms
    """
    
    def __init__(self, grid_config: GridConfig, logger=None):
        """
        Initialize the calibrator.
        
        Args:
            grid_config: Grid configuration (rows, cols, pitch, etc.)
            logger: Optional logger instance
        """
        from imswitch.imcommon.model import initLogger
        
        self._grid = grid_config
        self._logger = logger if logger is not None else initLogger(self)
        
        # Camera-to-stage affine transform (2x3 matrix: [R|t])
        # Maps camera pixel coordinates to stage micrometers
        # stage_xy = T @ [pixel_u, pixel_v, 1]
        self._T_cam2stage: Optional[np.ndarray] = None
        
        # AprilTag detector
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        try:
            # OpenCV >= 4.7
            self._aruco_params = cv2.aruco.DetectorParameters()
            self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        except AttributeError:
            # Legacy OpenCV
            self._aruco_params = cv2.aruco.DetectorParameters_create()
            self._aruco_detector = None
    
    def detect_tags(self, img: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """
        Detect AprilTags and return dictionary mapping tag IDs to centroids.
        
        Args:
            img: Image array (grayscale or BGR)
            
        Returns:
            Dictionary mapping tag_id -> (cx, cy) centroid in pixels
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
        
        return tag_centroids
    
    def get_current_tag(self, img: np.ndarray, 
                       roi_center: Optional[Tuple[float, float]] = None) -> Optional[Tuple[int, float, float]]:
        """
        Find the tag closest to the ROI center.
        
        Args:
            img: Image array
            roi_center: (cx, cy) ROI center in pixels. If None, uses image center.
            
        Returns:
            (tag_id, cx, cy) for the closest tag, or None if no tags detected
        """
        tags = self.detect_tags(img)
        if not tags:
            return None
        
        # Determine ROI center
        if roi_center is None:
            h, w = img.shape[:2]
            roi_center = (w / 2.0, h / 2.0)
        
        # Find closest tag to ROI center
        min_dist = float('inf')
        closest_tag = None
        
        for tag_id, (cx, cy) in tags.items():
            dist = np.sqrt((cx - roi_center[0])**2 + (cy - roi_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_tag = (tag_id, cx, cy)
        
        return closest_tag
    
    def calibrate_from_frame(self, tags: Dict[int, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Calibrate camera-to-stage transformation from detected tags.
        
        Uses least-squares fitting to compute affine transform that maps
        camera pixel coordinates to stage micrometers based on known grid positions.
        
        Requires at least 3 detected tags with known grid positions.
        
        Args:
            tags: Dictionary mapping tag_id -> (cx_px, cy_px)
            
        Returns:
            Dictionary with:
                - T_cam2stage: 2x3 affine transformation matrix (as list)
                - num_tags: Number of tags used
                - residual_um: RMS residual error in micrometers
                - tag_ids: List of tag IDs used
                - error: Error message if calibration failed
        """
        try:
            # Filter tags that are in the grid
            valid_tags = []
            cam_points = []
            grid_points = []
            
            for tag_id, (cx, cy) in tags.items():
                rowcol = self._grid.id_to_rowcol(tag_id)
                if rowcol is None:
                    continue
                
                row, col = rowcol
                
                # Convert grid position to physical coordinates (mm)
                grid_x_mm = col * self._grid.pitch_mm
                grid_y_mm = row * self._grid.pitch_mm
                
                valid_tags.append(tag_id)
                cam_points.append([cx, cy])
                grid_points.append([grid_x_mm, grid_y_mm])
            
            if len(valid_tags) < 3:
                return {
                    "error": f"Need at least 3 valid grid tags for calibration, found {len(valid_tags)}",
                    "num_tags": len(valid_tags)
                }
            
            # Convert to numpy arrays
            cam_pts = np.array(cam_points, dtype=np.float64)
            grid_pts = np.array(grid_points, dtype=np.float64)
            
            # Solve for affine transform using least squares
            # We want: grid_pts = T @ [cam_pts; 1]
            # Set up system: [cx cy 1] @ T.T = [gx gy]
            
            # Add homogeneous coordinate
            cam_pts_h = np.hstack([cam_pts, np.ones((len(cam_pts), 1))])
            
            # Solve separately for X and Y (each is a linear system)
            # T is 2x3 matrix: [[a, b, tx], [c, d, ty]]
            # We solve: cam_pts_h @ T[0, :].T = grid_pts[:, 0]
            #          cam_pts_h @ T[1, :].T = grid_pts[:, 1]
            
            T = np.zeros((2, 3))
            
            # Solve for X row
            T[0, :], residuals_x, _, _ = np.linalg.lstsq(cam_pts_h, grid_pts[:, 0], rcond=None)
            
            # Solve for Y row
            T[1, :], residuals_y, _, _ = np.linalg.lstsq(cam_pts_h, grid_pts[:, 1], rcond=None)
            
            # Compute residual error
            predicted = cam_pts_h @ T.T
            errors = predicted - grid_pts
            residual_um = float(np.sqrt(np.mean(errors**2)) * 1000.0)  # mm to um
            
            # Store transformation
            self._T_cam2stage = T
            
            self._logger.info(f"Calibration complete: {len(valid_tags)} tags, residual={residual_um:.2f} µm")
            self._logger.info(f"Transform matrix:\n{T}")
            
            return {
                "T_cam2stage": T.tolist(),
                "num_tags": len(valid_tags),
                "residual_um": residual_um,
                "tag_ids": valid_tags
            }
            
        except Exception as e:
            self._logger.error(f"Calibration failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def set_transform(self, T: np.ndarray):
        """
        Set the camera-to-stage transformation matrix.
        
        Args:
            T: 2x3 affine transformation matrix
        """
        if T.shape != (2, 3):
            raise ValueError(f"Transform must be 2x3 matrix, got {T.shape}")
        self._T_cam2stage = T.copy()
        self._logger.info("Set camera-to-stage transformation")
    
    def get_transform(self) -> Optional[np.ndarray]:
        """Get the current camera-to-stage transformation matrix."""
        return self._T_cam2stage.copy() if self._T_cam2stage is not None else None
    
    def pixel_to_stage_delta(self, du_px: float, dv_px: float) -> Tuple[float, float]:
        """
        Convert pixel displacement to stage displacement in micrometers.
        
        Args:
            du_px: Horizontal pixel displacement
            dv_px: Vertical pixel displacement
            
        Returns:
            (dx_um, dy_um) stage displacement in micrometers
            
        Raises:
            RuntimeError: If transformation is not calibrated
        """
        if self._T_cam2stage is None:
            raise RuntimeError("Camera-to-stage transformation not calibrated")
        
        # Apply only the linear part (rotation + scale)
        # Displacement doesn't use translation
        delta_px = np.array([du_px, dv_px])
        delta_mm = self._T_cam2stage[:, :2] @ delta_px
        
        # Convert mm to um
        dx_um = float(delta_mm[0] * 1000.0)
        dy_um = float(delta_mm[1] * 1000.0)
        
        return (dx_um, dy_um)
    
    def grid_to_stage_delta(self, from_id: int, to_id: int) -> Optional[Tuple[float, float]]:
        """
        Compute stage displacement between two tag IDs based on grid positions.
        
        Args:
            from_id: Starting tag ID
            to_id: Target tag ID
            
        Returns:
            (dx_um, dy_um) stage displacement in micrometers, or None if IDs invalid
        """
        from_pos = self._grid.id_to_rowcol(from_id)
        to_pos = self._grid.id_to_rowcol(to_id)
        
        if from_pos is None or to_pos is None:
            return None
        
        # Compute grid displacement
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        
        # Convert to mm
        dx_mm = dc * self._grid.pitch_mm
        dy_mm = dr * self._grid.pitch_mm
        
        # Convert to um
        dx_um = dx_mm * 1000.0
        dy_um = dy_mm * 1000.0
        
        return (dx_um, dy_um)
    
    def move_to_tag(self, target_id: int, observation_camera, positioner,
                   roi_center: Optional[Tuple[float, float]] = None,
                   roi_tolerance_px: float = 8.0,
                   max_iterations: int = 20,
                   step_fraction: float = 0.8,
                   settle_time: float = 0.3,
                   search_enabled: bool = True,
                   search_step_um: float = 5000.0,
                   search_pattern_size: int = 3) -> Dict[str, Any]:
        """
        Navigate to a specific tag ID using closed-loop feedback.
        
        Algorithm:
        1. Detect current tag closest to ROI center
        2. If target not visible, perform coarse search pattern
        3. Compute displacement from current tag to target using grid geometry
        4. Move stage by fraction of computed displacement
        5. Re-detect and iterate until target is centered within tolerance
        6. Final micro-centering based on pixel offset
        
        Args:
            target_id: Desired tag ID to center
            observation_camera: Camera with getLatestFrame() method
            positioner: Stage with move() and getPosition() methods
            roi_center: (cx, cy) ROI center in pixels. If None, uses image center.
            roi_tolerance_px: Acceptable pixel offset for convergence (default 8.0)
            max_iterations: Maximum iteration count (default 20)
            step_fraction: Fraction of computed displacement to apply per step (0-1, default 0.8)
            settle_time: Wait time after movement (seconds, default 0.3)
            search_enabled: Enable coarse search if target not initially visible
            search_step_um: Step size for search pattern (micrometers, default 5000)
            search_pattern_size: Search pattern grid size (default 3x3)
            
        Returns:
            Dictionary with:
                - success: Boolean indicating if target was centered
                - final_offset_px: Final pixel offset from ROI center
                - iterations: Number of iterations used
                - final_tag_id: Tag ID at final position
                - trajectory: List of iteration data (tag_id, offset_px, stage_move_um)
                - error: Error message if navigation failed
        """
        try:
            if self._T_cam2stage is None:
                return {"error": "Camera-to-stage transformation not calibrated", "success": False}
            
            # Validate target ID
            if self._grid.id_to_rowcol(target_id) is None:
                return {"error": f"Target ID {target_id} is outside grid range", "success": False}
            
            trajectory = []
            
            # Determine ROI center
            frame = observation_camera.getLatestFrame()
            h, w = frame.shape[:2]
            if roi_center is None:
                roi_center = (w / 2.0, h / 2.0)
            
            # Initial detection
            current_tag_info = self.get_current_tag(frame, roi_center)
            
            # Search if target not visible
            if current_tag_info is None and search_enabled:
                self._logger.info(f"No tags detected, starting search pattern")
                search_result = self._search_for_tags(observation_camera, positioner, 
                                                      search_step_um, search_pattern_size,
                                                      settle_time)
                if not search_result.get("success", False):
                    return {
                        "error": "Search failed to find any tags",
                        "success": False,
                        "search_trajectory": search_result.get("trajectory", [])
                    }
                
                # Re-detect after search
                frame = observation_camera.getLatestFrame()
                current_tag_info = self.get_current_tag(frame, roi_center)
            
            if current_tag_info is None:
                return {"error": "No tags detected and search disabled", "success": False}
            
            current_id, cx, cy = current_tag_info
            self._logger.info(f"Starting navigation from tag {current_id} to tag {target_id}")
            
            # Iterative navigation
            for iteration in range(max_iterations):
                # Check if we've reached the target
                if current_id == target_id:
                    # Micro-centering: use pixel offset
                    offset_x = cx - roi_center[0]
                    offset_y = cy - roi_center[1]
                    offset_mag = np.sqrt(offset_x**2 + offset_y**2)
                    
                    if offset_mag <= roi_tolerance_px:
                        # Success!
                        self._logger.info(f"Target {target_id} centered in {iteration} iterations, offset={offset_mag:.1f}px")
                        return {
                            "success": True,
                            "final_offset_px": float(offset_mag),
                            "iterations": iteration,
                            "final_tag_id": current_id,
                            "trajectory": trajectory
                        }
                    
                    # Micro-centering move based on pixel offset
                    dx_um, dy_um = self.pixel_to_stage_delta(offset_x, offset_y)
                    
                    # Move to center
                    current_pos = positioner.getPosition()
                    new_x = current_pos.get("X", 0) + dx_um
                    new_y = current_pos.get("Y", 0) + dy_um
                    
                    positioner.move(value=new_x, axis="X", is_absolute=True, is_blocking=True)
                    positioner.move(value=new_y, axis="Y", is_absolute=True, is_blocking=True)
                    time.sleep(settle_time)
                    
                    trajectory.append({
                        "iteration": iteration,
                        "mode": "micro_centering",
                        "current_tag": current_id,
                        "offset_px": float(offset_mag),
                        "move_um": [float(dx_um), float(dy_um)]
                    })
                    
                    # Re-detect
                    frame = observation_camera.getLatestFrame()
                    current_tag_info = self.get_current_tag(frame, roi_center)
                    
                    if current_tag_info is None:
                        return {
                            "error": f"Lost tag during micro-centering at iteration {iteration}",
                            "success": False,
                            "trajectory": trajectory
                        }
                    
                    current_id, cx, cy = current_tag_info
                    
                else:
                    # Coarse navigation: move toward target using grid geometry
                    delta = self.grid_to_stage_delta(current_id, target_id)
                    
                    if delta is None:
                        return {
                            "error": f"Failed to compute grid displacement at iteration {iteration}",
                            "success": False,
                            "trajectory": trajectory
                        }
                    
                    dx_um, dy_um = delta
                    
                    # Apply step fraction to avoid overshooting
                    dx_um *= step_fraction
                    dy_um *= step_fraction
                    
                    # Move stage
                    current_pos = positioner.getPosition()
                    new_x = current_pos.get("X", 0) + dx_um
                    new_y = current_pos.get("Y", 0) + dy_um
                    
                    positioner.move(value=new_x, axis="X", is_absolute=True, is_blocking=True)
                    positioner.move(value=new_y, axis="Y", is_absolute=True, is_blocking=True)
                    time.sleep(settle_time)
                    
                    offset_x = cx - roi_center[0]
                    offset_y = cy - roi_center[1]
                    offset_mag = np.sqrt(offset_x**2 + offset_y**2)
                    
                    trajectory.append({
                        "iteration": iteration,
                        "mode": "coarse_navigation",
                        "current_tag": current_id,
                        "target_tag": target_id,
                        "offset_px": float(offset_mag),
                        "move_um": [float(dx_um), float(dy_um)]
                    })
                    
                    # Re-detect
                    frame = observation_camera.getLatestFrame()
                    current_tag_info = self.get_current_tag(frame, roi_center)
                    
                    if current_tag_info is None:
                        return {
                            "error": f"Lost all tags during navigation at iteration {iteration}",
                            "success": False,
                            "trajectory": trajectory
                        }
                    
                    current_id, cx, cy = current_tag_info
                    self._logger.debug(f"Iteration {iteration}: moved to tag {current_id}")
            
            # Max iterations reached
            offset_x = cx - roi_center[0]
            offset_y = cy - roi_center[1]
            offset_mag = np.sqrt(offset_x**2 + offset_y**2)
            
            return {
                "error": f"Max iterations ({max_iterations}) reached",
                "success": False,
                "final_offset_px": float(offset_mag),
                "iterations": max_iterations,
                "final_tag_id": current_id,
                "trajectory": trajectory
            }
            
        except Exception as e:
            self._logger.error(f"Navigation failed: {e}", exc_info=True)
            return {"error": str(e), "success": False}
    
    def _search_for_tags(self, observation_camera, positioner,
                        step_um: float, pattern_size: int,
                        settle_time: float) -> Dict[str, Any]:
        """
        Perform coarse raster search to find any tags.
        
        Args:
            observation_camera: Camera detector
            positioner: Stage positioner
            step_um: Step size for search pattern (micrometers)
            pattern_size: Grid size (e.g., 3 for 3x3 pattern)
            settle_time: Wait time after movement (seconds)
            
        Returns:
            Dictionary with success status and trajectory
        """
        try:
            start_pos = positioner.getPosition()
            start_x = start_pos.get("X", 0)
            start_y = start_pos.get("Y", 0)
            
            trajectory = []
            
            # Search in a spiral/raster pattern
            for i in range(pattern_size):
                for j in range(pattern_size):
                    # Compute offset from center
                    offset_i = i - pattern_size // 2
                    offset_j = j - pattern_size // 2
                    
                    target_x = start_x + offset_j * step_um
                    target_y = start_y + offset_i * step_um
                    
                    # Move to position
                    positioner.move(value=target_x, axis="X", is_absolute=True, is_blocking=True)
                    positioner.move(value=target_y, axis="Y", is_absolute=True, is_blocking=True)
                    time.sleep(settle_time)
                    
                    # Check for tags
                    frame = observation_camera.getLatestFrame()
                    tags = self.detect_tags(frame)
                    
                    trajectory.append({
                        "position": [float(target_x), float(target_y)],
                        "tags_found": len(tags)
                    })
                    
                    if tags:
                        self._logger.info(f"Search found {len(tags)} tags at position ({i},{j})")
                        return {"success": True, "trajectory": trajectory, "tags": list(tags.keys())}
            
            # Return to start
            positioner.move(value=start_x, axis="X", is_absolute=True, is_blocking=True)
            positioner.move(value=start_y, axis="Y", is_absolute=True, is_blocking=True)
            
            return {"success": False, "trajectory": trajectory}
            
        except Exception as e:
            self._logger.error(f"Search failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def get_grid_config(self) -> Dict[str, Any]:
        """Get current grid configuration as dictionary."""
        return self._grid.to_dict()
    
    def set_grid_config(self, config: GridConfig):
        """Update grid configuration."""
        self._grid = config
        self._logger.info(f"Updated grid config: {config.rows}x{config.cols}, pitch={config.pitch_mm}mm")
