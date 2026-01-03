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
        self._rotated_180 = False  # Flag for 180° rotated calibration sample

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

    def set_rotation_180(self, rotated: bool):
        """
        Set whether the calibration sample is rotated 180 degrees.
        
        When rotated, tag IDs are mapped in reverse:
        - Normal: tag 0 = (row 0, col 0), tag 424 = (row 16, col 24)
        - Rotated: tag 0 = (row 16, col 24), tag 424 = (row 0, col 0)
        
        Args:
            rotated: True if sample is rotated 180°
        """
        self._rotated_180 = rotated
        self._logger.info(f"Grid rotation set to: {'180°' if rotated else 'normal'}")

    def get_rotation_180(self) -> bool:
        """
        Get current rotation state.
        
        Returns:
            True if grid is rotated 180°, False otherwise
        """
        return self._rotated_180

    def _map_tag_id(self, tag_id: int) -> int:
        """
        Map detected tag ID to logical ID based on rotation state.
        
        If rotated 180°, reverses the ID numbering:
        - detected_id -> (max_id - detected_id + min_id)
        
        Args:
            tag_id: Detected physical tag ID
            
        Returns:
            Logical tag ID for grid lookup
        """
        if not self._rotated_180:
            return tag_id

        # Calculate max possible ID
        max_id = self._grid.start_id + (self._grid.rows * self._grid.cols) - 1

        # Reverse the ID: 0->424, 1->423, etc.
        return max_id - tag_id + self._grid.start_id

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

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 20, 255, cv2.NORM_MINMAX).astype(np.uint8)
        try:
            # Detect markers
            if self._aruco_detector is not None:
                # OpenCV >= 4.7
                corners, ids, _ = self._aruco_detector.detectMarkers(gray)
                if ids is None: # inverted image
                    corners, ids, _ = self._aruco_detector.detectMarkers(255 - gray)
            else:
                # Legacy OpenCV
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict,
                                                        parameters=self._aruco_params)
        except Exception as e:
            self._logger.error(f"AprilTag detection failed: {e}")
            return {}
        if ids is None or len(ids) == 0:
            return {}

        # Build tag_id -> centroid mapping
        tag_centroids = {}
        for i, tag_id in enumerate(ids.flatten()):
            pts = corners[i][0]
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])

            # Apply ID mapping if grid is rotated 180°
            logical_id = self._map_tag_id(int(tag_id))
            tag_centroids[logical_id] = (float(cx), float(cy))

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
                   axis_calibration: Dict[str, Any],
                   roi_center: Optional[Tuple[float, float]] = None,
                   roi_tolerance_px: float = 8.0,
                   max_iterations: int = 30,
                   max_step_um: float = 1000.0,
                   settle_time: float = 0.3) -> Dict[str, Any]:
        """
        Navigate to a specific tag ID using iterative grid-aware navigation.
        
        This method uses continuous feedback from detected tags, comparing current tag
        positions with stored grid topology to decide movement direction. It uses axis
        calibration data from overviewIdentifyAxes to correctly map pixel movements to
        stage movements.
        
        Algorithm:
        1. Detect all visible tags in current frame
        2. Determine which direction (X/Y) to move based on target grid position
        3. Move by max_step_um in the computed direction
        4. Detect tags again and validate we're moving toward target
        5. Repeat until target is visible and centered
        
        Args:
            target_id: Desired tag ID to center
            observation_camera: Camera with getLatestFrame() method
            positioner: Stage with move() and getPosition() methods
            axis_calibration: Axis calibration data from overviewIdentifyAxes with:
                - mapping: {stageX_to_cam: "width"|"height", stageY_to_cam: "width"|"height"}
                - sign: {X: +1|-1, Y: +1|-1}
            roi_center: (cx, cy) ROI center in pixels. If None, uses image center.
            roi_tolerance_px: Acceptable pixel offset for convergence (default 8.0)
            max_iterations: Maximum iteration count (default 30)
            max_step_um: Maximum step size per iteration in micrometers (default 1000.0)
            settle_time: Wait time after movement (seconds, default 0.3)
            
        Returns:
            Dictionary with:
                - success: Boolean indicating if target was centered
                - final_offset_px: Final pixel offset from ROI center
                - iterations: Number of iterations used
                - final_tag_id: Tag ID at final position
                - trajectory: List of iteration data
                - error: Error message if navigation failed
        """
        try:
            # Validate target ID
            target_pos = self._grid.id_to_rowcol(target_id)
            if target_pos is None:
                return {"error": f"Target ID {target_id} is outside grid range", "success": False}

            target_row, target_col = target_pos
            trajectory = []

            # Extract axis calibration data
            mapping = axis_calibration.get('mapping', {})
            sign = axis_calibration.get('sign', {})

            if not mapping or not sign:
                return {
                    "error": "Invalid axis calibration data. Run overviewIdentifyAxes first.",
                    "success": False
                }

            # Determine ROI center
            frame = observation_camera.getLatestFrame()
            h, w = frame.shape[:2]
            if roi_center is None:
                roi_center = (w / 2.0, h / 2.0)

            self._logger.info(
                f"Starting navigation to tag {target_id} (grid row={target_row}, col={target_col})"
            )

            # Iterative navigation comparing grid positions
            for iteration in range(max_iterations):
                # Detect all tags in current frame
                frame = observation_camera.getLatestFrame()
                current_tags = self.detect_tags(frame)

                if not current_tags:
                    return {
                        "error": f"No tags detected at iteration {iteration}",
                        "success": False,
                        "trajectory": trajectory
                    }

                self._logger.info(
                    f"Iteration {iteration}: Detected {len(current_tags)} tags: {list(current_tags.keys())}"
                )

                # Check if target is visible
                if target_id in current_tags:
                    cx, cy = current_tags[target_id]
                    offset_x = cx - roi_center[0]
                    offset_y = cy - roi_center[1]
                    offset_mag = np.sqrt(offset_x**2 + offset_y**2)

                    if offset_mag <= roi_tolerance_px:
                        # Success!
                        self._logger.info(
                            f"Target {target_id} centered in {iteration} iterations, "
                            f"offset={offset_mag:.1f}px"
                        )
                        return {
                            "success": True,
                            "final_offset_px": float(offset_mag),
                            "iterations": iteration,
                            "final_tag_id": target_id,
                            "trajectory": trajectory
                        }

                    # Micro-centering: move to center the target tag
                    dx_um, dy_um = self._pixel_offset_to_stage_move(
                        offset_x, offset_y, mapping, sign
                    )

                    # Limit movement to max_step_um
                    move_mag = np.sqrt(dx_um**2 + dy_um**2)
                    if move_mag > max_step_um:
                        scale = max_step_um / move_mag
                        dx_um *= scale
                        dy_um *= scale

                    # Move to center (invert signs because we want to move stage opposite to pixel offset)
                    positioner.move(value=-dx_um, axis="X", is_absolute=False, is_blocking=True)
                    positioner.move(value=-dy_um, axis="Y", is_absolute=False, is_blocking=True)
                    time.sleep(settle_time)

                    trajectory.append({
                        "iteration": iteration,
                        "mode": "centering",
                        "current_tag": target_id,
                        "offset_px": float(offset_mag),
                        "move_um": [float(-dx_um), float(-dy_um)]
                    })

                else:
                    # Target not visible - compute direction based on detected tag grid positions
                    move_decision = self._compute_grid_based_movement(
                        current_tags, target_row, target_col, max_step_um, sign
                    )

                    if move_decision is None:
                        return {
                            "error": f"Cannot determine movement direction at iteration {iteration}",
                            "success": False,
                            "trajectory": trajectory,
                            "detected_tags": list(current_tags.keys())
                        }

                    dx_um, dy_um, decision_info = move_decision

                    # Move stage
                    positioner.move(value=dx_um, axis="X", is_absolute=False, is_blocking=True)
                    positioner.move(value=dy_um, axis="Y", is_absolute=False, is_blocking=True)
                    time.sleep(settle_time)

                    self._logger.debug(
                        f"Iteration {iteration}: Moving toward target "
                        f"(grid_offset: row={decision_info['row_offset']}, col={decision_info['col_offset']})"
                    )

                    trajectory.append({
                        "iteration": iteration,
                        "mode": "grid_navigation",
                        "target_tag": target_id,
                        "move_um": [float(dx_um), float(dy_um)],
                        "decision_info": decision_info
                    })

            # Max iterations reached
            return {
                "error": f"Max iterations ({max_iterations}) reached without centering target",
                "success": False,
                "iterations": max_iterations,
                "trajectory": trajectory
            }

        except Exception as e:
            self._logger.error(f"Navigation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "success": False,
                "trajectory": trajectory if 'trajectory' in locals() else []
            }

    def _pixel_offset_to_stage_move(self, pixel_dx: float, pixel_dy: float,
                                    mapping: Dict[str, str], sign: Dict[str, int]) -> Tuple[float, float]:
        """
        Convert pixel offset to stage movement using axis calibration.
        
        Uses affine transform if available, otherwise uses axis calibration mapping.
        
        Args:
            pixel_dx: Pixel offset in width direction
            pixel_dy: Pixel offset in height direction
            mapping: Axis mapping from calibration {stageX_to_cam: "width"|"height", stageY_to_cam: "width"|"height"}
            sign: Axis sign from calibration {X: +1|-1, Y: +1|-1}
            
        Returns:
            (dx_um, dy_um) stage movement in micrometers
        """
        # Use affine transform if available
        if self._T_cam2stage is not None:
            return self.pixel_to_stage_delta(pixel_dx, pixel_dy)

        # Otherwise use axis calibration with pitch-based scaling
        # Assume pixel-to-mm ratio from grid pitch (rough estimate)
        # This will be refined by the iterative feedback loop
        pixel_to_mm = self._grid.pitch_mm / 100.0  # Rough estimate: ~40mm pitch / ~100px

        # Map camera axes to stage axes
        stage_x_um = 0.0
        stage_y_um = 0.0

        if mapping.get('stageX_to_cam') == 'width':
            stage_x_um = pixel_dx * pixel_to_mm * 1000.0 * sign.get('X', 1)
        elif mapping.get('stageX_to_cam') == 'height':
            stage_x_um = pixel_dy * pixel_to_mm * 1000.0 * sign.get('X', 1)

        if mapping.get('stageY_to_cam') == 'width':
            stage_y_um = pixel_dx * pixel_to_mm * 1000.0 * sign.get('Y', 1)
        elif mapping.get('stageY_to_cam') == 'height':
            stage_y_um = pixel_dy * pixel_to_mm * 1000.0 * sign.get('Y', 1)

        return (stage_x_um, stage_y_um)

    def _compute_grid_based_movement(self, detected_tags: Dict[int, Tuple[float, float]],
                                    target_row: int, target_col: int,
                                    max_step_um: float,
                                    sign: Dict[str, int]) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """
        Compute stage movement direction based on detected tag grid positions vs target.
        
        Compares current tag positions in the grid to target position and decides
        which direction to move on the stage.
        
        Args:
            detected_tags: Dictionary mapping tag_id -> (cx, cy) for detected tags
            target_row: Target tag grid row
            target_col: Target tag grid column
            max_step_um: Maximum step size in micrometers
            sign: Axis sign from calibration {X: +1|-1, Y: +1|-1}
            
        Returns:
            (dx_um, dy_um, decision_info) or None if cannot determine direction
        """
        # Compute average grid position of detected tags
        total_row = 0
        total_col = 0
        valid_count = 0

        for tag_id in detected_tags.keys():
            pos = self._grid.id_to_rowcol(tag_id)
            if pos is not None:
                row, col = pos
                total_row += row
                total_col += col
                valid_count += 1

        if valid_count == 0:
            return None

        avg_row = total_row / valid_count
        avg_col = total_col / valid_count

        # Compute offset to target in grid coordinates
        row_offset = target_row - avg_row
        col_offset = target_col - avg_col

        # Decide primary direction: move in dimension with larger offset
        move_in_rows = abs(row_offset) > abs(col_offset)

        dx_um = 0.0
        dy_um = 0.0

        if move_in_rows:
            # Move in Y direction (rows)
            # Positive row offset means move down (+Y on stage)
            dy_um = max_step_um * np.sign(row_offset) * sign.get('Y', 1)
        else:
            # Move in X direction (cols)
            # Positive col offset means move right (+X on stage)
            dx_um = max_step_um * np.sign(col_offset) * sign.get('X', 1)

        decision_info = {
            "avg_detected_row": float(avg_row),
            "avg_detected_col": float(avg_col),
            "target_row": target_row,
            "target_col": target_col,
            "row_offset": float(row_offset),
            "col_offset": float(col_offset),
            "move_direction": "rows" if move_in_rows else "cols",
            "num_detected_tags": valid_count
        }

        return (dx_um, dy_um, decision_info)

    def _find_best_neighbor_toward_target(self, detected_tags: Dict[int, Tuple[float, float]],
                                         target_id: int) -> Optional[Tuple[int, float, float, Dict[str, Any]]]:
        """
        Find the best tag among detected tags that moves us toward the target.
        
        Uses grid topology to validate neighbors and compute grid distance.
        Prefers tags that are:
        1. Closer to target in grid coordinates
        2. Have more expected neighbors visible (indicates good position)
        3. Target itself if visible
        
        Args:
            detected_tags: Dictionary mapping tag_id -> (cx, cy) for detected tags
            target_id: Target tag ID we want to reach
            
        Returns:
            (tag_id, cx, cy, nav_info) for best next tag, or None if no valid path
            nav_info contains: grid_distance, expected_neighbors, actual_neighbors
        """
        if target_id in detected_tags:
            # Target is visible, return it directly
            cx, cy = detected_tags[target_id]
            return (target_id, cx, cy, {
                "grid_distance": 0,
                "expected_neighbors": [],
                "actual_neighbors": []
            })

        target_pos = self._grid.id_to_rowcol(target_id)
        if target_pos is None:
            return None

        target_row, target_col = target_pos

        # Score each detected tag
        best_tag = None
        best_score = float('inf')
        best_info = None

        for tag_id, (cx, cy) in detected_tags.items():
            tag_pos = self._grid.id_to_rowcol(tag_id)
            if tag_pos is None:
                continue

            tag_row, tag_col = tag_pos

            # Compute grid distance to target (Manhattan distance)
            grid_dist = abs(target_row - tag_row) + abs(target_col - tag_col)

            # Get expected neighbors based on grid topology
            expected_neighbors = self._get_expected_neighbors(tag_id)

            # Count how many expected neighbors are actually detected
            actual_neighbors = [n for n in expected_neighbors if n in detected_tags]
            neighbor_coverage = len(actual_neighbors) / max(len(expected_neighbors), 1)

            # Score: prioritize smaller grid distance and better neighbor coverage
            # Weight grid distance more heavily
            score = grid_dist * 10.0 - neighbor_coverage * 2.0

            if score < best_score:
                best_score = score
                best_tag = tag_id
                best_info = {
                    "grid_distance": grid_dist,
                    "expected_neighbors": expected_neighbors,
                    "actual_neighbors": actual_neighbors,
                    "neighbor_coverage": neighbor_coverage,
                    "score": score
                }

        if best_tag is None:
            return None

        cx, cy = detected_tags[best_tag]
        return (best_tag, cx, cy, best_info)

    def _get_expected_neighbors(self, tag_id: int) -> List[int]:
        """
        Get list of expected neighbor tag IDs based on grid topology.
        
        A neighbor is defined as an adjacent tag in the grid (up, down, left, right).
        
        Args:
            tag_id: Tag ID to get neighbors for
            
        Returns:
            List of neighbor tag IDs (may be empty if tag is at grid edge)
        """
        pos = self._grid.id_to_rowcol(tag_id)
        if pos is None:
            return []

        row, col = pos
        neighbors = []

        # Check all 4 cardinal directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_row = row + dr
            neighbor_col = col + dc

            neighbor_id = self._grid.rowcol_to_id(neighbor_row, neighbor_col)
            if neighbor_id is not None:
                neighbors.append(neighbor_id)

        return neighbors

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
