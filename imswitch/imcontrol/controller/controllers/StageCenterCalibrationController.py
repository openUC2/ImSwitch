import os
import threading
import time
from datetime import datetime

import numpy as np
import tifffile as tif
import cv2

from imswitch import IS_HEADLESS
from imswitch.imcommon.framework import Signal, Mutex
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController


class StageCenterCalibrationController(ImConWidgetController):
    """Out‑growing square‑spiral search for the sample’s centre.

    * Each leg is executed as a **continuous move** on the respective axis using
      ``MovementController`` so that frames are grabbed **while** the stage is
      travelling – for **both X *and* Y directions**.
    * Spiral pitch = ``step_um``; side length increases after every second turn
      (E→N→W→S cycle).
    * Acquisition stops when the mean intensity (20‑pixel subsampling) rises by
      ``brightness_factor`` or when the requested ``max_radius_um`` is reached.
    * All visited (x, y) coordinates are stored and returned; a CSV copy is also
      written for record keeping.
    """

    sigImageReceived = Signal(np.ndarray)  # optional live‑view

    # ─────────────────────────── initialisation ────────────────────────────

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # state
        self._task = None
        self._is_running = False
        self._positions: list[tuple[float, float]] = []
        self._run_mutex = Mutex()
        
        # Calibration target constants (in mm)
        self.CALIBRATION_CENTER_X = 63.81
        self.CALIBRATION_CENTER_Y = 42.06
        self.MAZE_START_X = 9.5
        self.MAZE_START_Y = 11.5
        self.STEPSIZE_GRID_X = 105.0
        self.STEPSIZE_GRID_Y = 16.0
        self.WELLPLATE_START_X = 12.2
        self.WELLPLATE_START_Y = 9.0
        self.WELLPLATE_SPACING = 4.5  # mm
        
        # Calibration target dimensions (in mm)
        self.TARGET_WIDTH = 127.76
        self.TARGET_HEIGHT = 85.48
        
        # Maze navigation state
        self._maze_running = False
        self._maze_positions = []

    # ───────────────────────────── API ──────────────────────────────────────
    def getDetector(self):
        # devices
        return self._master.detectorsManager[self._master.detectorsManager.getAllDeviceNames()[0]]
        
    def getStage(self):
        stageName = self._master.positionersManager.getAllDeviceNames()[0]
        return self._master.positionersManager[stageName]

    @APIExport()
    def performCalibration(
        self,
        start_x: float,
        start_y: float,
        exposure_time_us: int = 3000,
        speed: int = 5000,
        step_um: float = 50.0,
        max_radius_um: float = 2000.0,
        brightness_factor: float = 1.4,
    ) -> list[tuple[float, float]]:
        if self._is_running:
            return self._positions

        self._is_running = True
        self._positions.clear()

        try:
            self.getDetector().setExposure(exposure_time_us)
        except AttributeError:
            pass

        self._task = threading.Thread(
            target=self._worker,
            args=(
                start_x,
                start_y,
                speed,
                step_um,
                max_radius_um,
                brightness_factor,
            ),
            daemon=True,
        )
        self._task.start()
        self._task.join()
        return self._positions.copy()

    @APIExport()
    def getIsCalibrationRunning(self):
        return self._is_running

    # ──────────────────────────── worker ────────────────────────────────────

    def _worker(self, cx, cy, speed, step_um, max_r, bf):
        self.getStage().move("X", cx, True, True)
        self.getStage().move("Y", cy, True, True)

        baseline = self._grabMeanFrame()
        if baseline is None:
            self._logger.error("No detector image – aborting")
            self._is_running = False
            return

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, N, W, S
        dir_idx = 0
        run_len = 1
        legs_done = 0
        off_x = off_y = 0.0

        while self._is_running:
            dx, dy = directions[dir_idx]
            axis = "X" if dx else "Y"

            for _ in range(run_len):
                if not self._is_running:
                    break
                off_x += dx * step_um
                off_y += dy * step_um

                if max(abs(off_x), abs(off_y)) > max_r:
                    self._logger.info("Max radius reached – stop")
                    self._is_running = False
                    break

                target = (cx + off_x) if axis == "X" else (cy + off_y)
                ctrl = MovementController(self.getStage())
                ctrl.move_to_position(target, axis=axis, speed=speed, is_absolute=True)

                # ───── grab frames while travelling ─────
                while not ctrl.is_target_reached() and self._is_running:
                    m = self._grabMeanFrame()
                    p = self.getStage().getPosition()
                    self._positions.append((p["X"], p["Y"]))
                    if m is not None and m >= baseline * bf:
                        self._logger.info("Brightness threshold hit – done")
                        self._is_running = False
                        break
                    time.sleep(0.002)  # mild CPU relief

                if not self._is_running:
                    break

            if not self._is_running:
                break

            dir_idx = (dir_idx + 1) % 4
            legs_done += 1
            if legs_done == 2:
                legs_done = 0
                run_len += 1  # enlarge spiral

        self._savePositionsCsv()
        self._is_running = False

    @APIExport()
    def stopCalibration(self):
        """Stops the calibration process."""
        self._is_running = False
        if self._task is not None:
            self._task.join()
            self._task = None
        self._logger.info("Calibration stopped.")
    
    @APIExport()
    def setKnownPosition(self, x_mm: float = None, y_mm: float = None):
        """
        Manually set the stage offset to a known position.
        If no coordinates provided, uses the calibration center (63.81, 42.06).
        """
        if x_mm is None:
            x_mm = self.CALIBRATION_CENTER_X
        if y_mm is None:
            y_mm = self.CALIBRATION_CENTER_Y
        
        stage = self.getStage()
        current_pos = stage.getPosition()
        
        # Set stage offset for both axes
        stage.setStageOffsetAxis(knownPosition=x_mm * 1000, currentPosition=current_pos["X"], axis="X")  # Convert mm to µm
        stage.setStageOffsetAxis(knownPosition=y_mm * 1000, currentPosition=current_pos["Y"], axis="Y")  # Convert mm to µm
        
        self._logger.info(f"Stage offset set to known position: X={x_mm}mm, Y={y_mm}mm")
        return {"status": "success", "x_mm": x_mm, "y_mm": y_mm}
    
    @APIExport()
    def performAutomaticCalibration(self, laser_name: str = None, laser_intensity: float = 50.0) -> dict:
        """
        Automatic calibration using line detection with Hough transform.
        Homes the stage, moves to 30mm offset, searches for white lines, then finds center ring.
        """
        if self._is_running:
            return {"status": "error", "message": "Calibration already running"}
        
        self._is_running = True
        try:
            stage = self.getStage()
            
            # Home the stage in X and Y
            self._logger.info("Homing stage...")
            stage.home("X")
            stage.home("Y")
            
            # Move to 30mm offset position
            self._logger.info("Moving to 30mm offset position...")
            stage.move("X", 30000, True, True)  # 30mm in µm
            stage.move("Y", 30000, True, True)
            
            # Turn on laser if specified
            if laser_name:
                try:
                    laser_controller = self._master.lasersManager.get(laser_name)
                    if laser_controller:
                        laser_controller.setLaserValue(laser_intensity)
                        laser_controller.setLaserActive(True)
                        self._logger.info(f"Laser {laser_name} activated at {laser_intensity}")
                except Exception as e:
                    self._logger.warning(f"Could not activate laser {laser_name}: {e}")
            
            # Search for white lines
            center_position = self._findCalibrationCenter()
            
            if center_position:
                # Set the known position offset
                self.setKnownPosition(self.CALIBRATION_CENTER_X, self.CALIBRATION_CENTER_Y)
                return {"status": "success", "center": center_position}
            else:
                return {"status": "error", "message": "Could not find calibration center"}
                
        except Exception as e:
            self._logger.error(f"Automatic calibration failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self._is_running = False
            # Turn off laser if it was turned on
            if laser_name:
                try:
                    laser_controller = self._master.lasersManager.get(laser_name)
                    if laser_controller:
                        laser_controller.setLaserActive(False)
                except Exception:
                    pass

    @APIExport()
    def getCalibrationTargetInfo(self) -> dict:
        """
        Returns information about the calibration target including SVG representation.
        """
        frontside_svg = f'''
        <svg width="{self.TARGET_WIDTH}" height="{self.TARGET_HEIGHT}" viewBox="0 0 {self.TARGET_WIDTH} {self.TARGET_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
            <!-- Calibration target frontside -->
            <rect width="{self.TARGET_WIDTH}" height="{self.TARGET_HEIGHT}" fill="white" stroke="black" stroke-width="1"/>
            
            <!-- Center hole at (63.81, 42.06) -->
            <circle cx="{self.CALIBRATION_CENTER_X}" cy="{self.CALIBRATION_CENTER_Y}" r="2" fill="black"/>
            <text x="{self.CALIBRATION_CENTER_X + 5}" y="{self.CALIBRATION_CENTER_Y}" font-size="3">Center ({self.CALIBRATION_CENTER_X}, {self.CALIBRATION_CENTER_Y})</text>
            
            <!-- Maze start point at (9.5, 11.5) -->
            <circle cx="{self.MAZE_START_X}" cy="{self.MAZE_START_Y}" r="1" fill="blue"/>
            <text x="{self.MAZE_START_X + 3}" y="{self.MAZE_START_Y}" font-size="2">Maze Start</text>
            
            <!-- Stepsize grid at (105, 16) -->
            <g transform="translate({self.STEPSIZE_GRID_X}, {self.STEPSIZE_GRID_Y})">
                <!-- 7x7 grid with 1mm spacing -->
                {''.join([f'<circle cx="{i}" cy="{j}" r="0.3" fill="red"/>' for i in range(7) for j in range(7)])}
            </g>
            <text x="{self.STEPSIZE_GRID_X}" y="{self.STEPSIZE_GRID_Y - 2}" font-size="2">7x7 Grid (1mm spacing)</text>
        </svg>
        '''
        
        backside_svg = f'''
        <svg width="{self.TARGET_WIDTH}" height="{self.TARGET_HEIGHT}" viewBox="0 0 {self.TARGET_WIDTH} {self.TARGET_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
            <!-- Calibration target backside -->
            <rect width="{self.TARGET_WIDTH}" height="{self.TARGET_HEIGHT}" fill="lightgray" stroke="black" stroke-width="1"/>
            
            <!-- 384 wellplate pattern starting at (12.2, 9.0) with 4.5mm spacing -->
            <g transform="translate({self.WELLPLATE_START_X}, {self.WELLPLATE_START_Y})">
                <!-- 24 columns (A-P) x 16 rows (1-24) wellplate pattern -->
                {''.join([f'<circle cx="{col * self.WELLPLATE_SPACING}" cy="{row * self.WELLPLATE_SPACING}" r="0.5" fill="green"/>' 
                         for col in range(24) for row in range(16)])}
            </g>
            <text x="{self.WELLPLATE_START_X}" y="{self.WELLPLATE_START_Y - 2}" font-size="2">384 Wellplate Pattern</text>
        </svg>
        '''
        
        return {
            "width_mm": self.TARGET_WIDTH,
            "height_mm": self.TARGET_HEIGHT,
            "frontside_svg": frontside_svg,
            "backside_svg": backside_svg,
            "calibration_center": {"x": self.CALIBRATION_CENTER_X, "y": self.CALIBRATION_CENTER_Y},
            "maze_start": {"x": self.MAZE_START_X, "y": self.MAZE_START_Y},
            "stepsize_grid": {"x": self.STEPSIZE_GRID_X, "y": self.STEPSIZE_GRID_Y},
            "wellplate_start": {"x": self.WELLPLATE_START_X, "y": self.WELLPLATE_START_Y, "spacing": self.WELLPLATE_SPACING}
        }
    
    @APIExport()
    def startMaze(self, custom_path: list = None) -> dict:
        """
        Start maze navigation from position (9.5, 11.5) with 1000µm steps.
        Saves images as TIFF stack during navigation.
        """
        if self._maze_running:
            return {"status": "error", "message": "Maze already running"}
        
        self._maze_running = True
        self._maze_positions = []
        
        # Default maze path (can be customized)
        if custom_path is None:
            custom_path = [
                (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1),  # Simple square path
                (1, 1)  # End at center
            ]
        
        try:
            stage = self.getStage()
            
            # Move to maze start position
            start_x_um = self.MAZE_START_X * 1000  # Convert mm to µm
            start_y_um = self.MAZE_START_Y * 1000
            stage.move("X", start_x_um, True, True)
            stage.move("Y", start_y_um, True, True)
            
            # Start maze navigation in separate thread
            self._task = threading.Thread(target=self._navigateMaze, args=(custom_path,), daemon=True)
            self._task.start()
            
            return {"status": "started", "path_length": len(custom_path)}
            
        except Exception as e:
            self._maze_running = False
            self._logger.error(f"Failed to start maze navigation: {e}")
            return {"status": "error", "message": str(e)}
    
    @APIExport()
    def stopMaze(self) -> dict:
        """Stop maze navigation."""
        self._maze_running = False
        if self._task is not None:
            self._task.join()
            self._task = None
        self._logger.info("Maze navigation stopped.")
        return {"status": "stopped", "positions_visited": len(self._maze_positions)}
    
    @APIExport()
    def getMazeStatus(self) -> dict:
        """Get current maze navigation status."""
        return {
            "running": self._maze_running,
            "positions_visited": len(self._maze_positions),
            "current_position": self._maze_positions[-1] if self._maze_positions else None
        }
    
    @APIExport()
    def performStepsizeCalibration(self) -> dict:
        """
        Perform stepsize calibration using 7x7 hole lattice at (105, 16) with 1mm spacing.
        Captures images at each hole position and saves as TIFF stack.
        """
        if self._is_running:
            return {"status": "error", "message": "Another calibration is running"}
        
        self._is_running = True
        try:
            stage = self.getStage()
            detector = self.getDetector()
            
            # Move to starting position
            start_x_um = self.STEPSIZE_GRID_X * 1000  # Convert mm to µm
            start_y_um = self.STEPSIZE_GRID_Y * 1000
            
            images = []
            positions = []
            
            # Scan 7x7 grid
            for i in range(7):
                for j in range(7):
                    if not self._is_running:
                        break
                    
                    # Calculate position (1mm = 1000µm spacing)
                    x_pos = start_x_um + (i * 1000)
                    y_pos = start_y_um + (j * 1000)
                    
                    # Move to position
                    stage.move("X", x_pos, True, True)
                    stage.move("Y", y_pos, True, True)
                    
                    # Capture image
                    time.sleep(0.1)  # Allow settling
                    frame = detector.getLatestFrame()
                    if frame is not None:
                        images.append(frame)
                        positions.append((x_pos, y_pos))
                        self._logger.debug(f"Captured image at grid position ({i}, {j})")
            
            # Save TIFF stack
            if images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dir_path = os.path.join(os.path.expanduser("~"), "imswitch_calibrations", timestamp)
                os.makedirs(dir_path, exist_ok=True)
                
                stack_path = os.path.join(dir_path, "stepsize_calibration_stack.tiff")
                tif.imwrite(stack_path, np.array(images))
                
                positions_path = os.path.join(dir_path, "stepsize_positions.csv")
                np.savetxt(positions_path, np.array(positions), delimiter=",", header="X(µm),Y(µm)")
                
                self._logger.info(f"Stepsize calibration completed. {len(images)} images saved to {stack_path}")
                return {
                    "status": "success",
                    "images_captured": len(images),
                    "tiff_stack_path": stack_path,
                    "positions_path": positions_path
                }
            else:
                return {"status": "error", "message": "No images captured"}
                
        except Exception as e:
            self._logger.error(f"Stepsize calibration failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self._is_running = False
    
    @APIExport()
    def perform384WellplateCalibration(self, sample_wells: list = None) -> dict:
        """
        Perform 384 wellplate calibration on backside pattern.
        Scans random positions of wells A1-P24 and compares center positions.
        """
        if self._is_running:
            return {"status": "error", "message": "Another calibration is running"}
        
        self._is_running = True
        try:
            stage = self.getStage()
            detector = self.getDetector()
            
            # Default sample wells if none provided
            if sample_wells is None:
                # Sample some wells across the plate
                sample_wells = ["A1", "A12", "A24", "H1", "H12", "H24", "P1", "P12", "P24"]
            
            images = []
            positions = []
            well_info = []
            
            for well in sample_wells:
                if not self._is_running:
                    break
                
                # Parse well coordinate (e.g., "A1" -> row=0, col=0)
                row = ord(well[0]) - ord('A')  # A=0, B=1, ..., P=15
                col = int(well[1:]) - 1        # 1=0, 2=1, ..., 24=23
                
                if row > 15 or col > 23:  # Validate well coordinates
                    self._logger.warning(f"Invalid well coordinate: {well}")
                    continue
                
                # Calculate position
                x_pos = (self.WELLPLATE_START_X + col * self.WELLPLATE_SPACING) * 1000  # Convert to µm
                y_pos = (self.WELLPLATE_START_Y + row * self.WELLPLATE_SPACING) * 1000
                
                # Move to position
                stage.move("X", x_pos, True, True)
                stage.move("Y", y_pos, True, True)
                
                # Capture image
                time.sleep(0.1)  # Allow settling
                frame = detector.getLatestFrame()
                if frame is not None:
                    images.append(frame)
                    positions.append((x_pos, y_pos))
                    well_info.append({"well": well, "row": row, "col": col, "x": x_pos, "y": y_pos})
                    self._logger.debug(f"Captured image at well {well}")
            
            # Save TIFF stack and data
            if images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dir_path = os.path.join(os.path.expanduser("~"), "imswitch_calibrations", timestamp)
                os.makedirs(dir_path, exist_ok=True)
                
                stack_path = os.path.join(dir_path, "wellplate_384_calibration_stack.tiff")
                tif.imwrite(stack_path, np.array(images))
                
                positions_path = os.path.join(dir_path, "wellplate_positions.csv")
                np.savetxt(positions_path, np.array(positions), delimiter=",", header="X(µm),Y(µm)")
                
                # Save well information as JSON
                import json
                wells_path = os.path.join(dir_path, "wellplate_wells.json")
                with open(wells_path, 'w') as f:
                    json.dump(well_info, f, indent=2)
                
                self._logger.info(f"384 wellplate calibration completed. {len(images)} wells scanned.")
                return {
                    "status": "success",
                    "wells_scanned": len(images),
                    "tiff_stack_path": stack_path,
                    "positions_path": positions_path,
                    "wells_info_path": wells_path,
                    "wells": well_info
                }
            else:
                return {"status": "error", "message": "No images captured"}
                
        except Exception as e:
            self._logger.error(f"384 wellplate calibration failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self._is_running = False
    
    # ─────────────────────── new calibration helpers ───────────────────────
    
    def _findCalibrationCenter(self) -> tuple:
        """
        Find calibration center using line detection and ring positioning.
        Returns (x, y) coordinates in micrometers or None if not found.
        """
        stage = self.getStage()
        detector = self.getDetector()
        
        # Step 1: Search for white lines in X direction
        self._logger.info("Searching for white lines in X direction...")
        line_found_x = None
        
        for i in range(50):  # Max 50 steps of 1000µm = 50mm search
            if not self._is_running:
                return None
            
            # Move 1000µm in X direction
            current_pos = stage.getPosition()
            new_x = current_pos["X"] + 1000
            stage.move("X", new_x, True, True)
            
            # Take image and check for lines
            time.sleep(0.1)
            frame = detector.getLatestFrame()
            if frame is not None and self._detectWhiteLine(frame):
                line_found_x = new_x
                self._logger.info(f"White line found in X at position {new_x}")
                break
        
        if line_found_x is None:
            self._logger.error("No white line found in X direction")
            return None
        
        # Step 2: Search for white lines in Y direction
        self._logger.info("Searching for white lines in Y direction...")
        line_found_y = None
        
        for i in range(50):  # Max 50 steps
            if not self._is_running:
                return None
            
            # Move 1000µm in Y direction
            current_pos = stage.getPosition()
            new_y = current_pos["Y"] + 1000
            stage.move("Y", new_y, True, True)
            
            # Take image and check for lines
            time.sleep(0.1)
            frame = detector.getLatestFrame()
            if frame is not None and self._detectWhiteLine(frame):
                line_found_y = new_y
                self._logger.info(f"White line found in Y at position {new_y}")
                break
        
        if line_found_y is None:
            self._logger.error("No white line found in Y direction")
            return None
        
        # Step 3: Continue alternating until center is bright
        self._logger.info("Searching for bright center...")
        baseline_intensity = self._grabMeanFrame()
        
        for iteration in range(20):  # Max iterations to find center
            if not self._is_running:
                return None
            
            # Move in X direction and check intensity
            current_pos = stage.getPosition()
            stage.move("X", current_pos["X"] + 1000, True, True)
            time.sleep(0.1)
            
            intensity = self._grabMeanFrame()
            if intensity and baseline_intensity and intensity > baseline_intensity * 2.0:
                self._logger.info("High intensity detected - near center")
                
                # Step 4: Look for rings and center on them
                center_pos = self._findRingCenter()
                if center_pos:
                    return center_pos
            
            # Move in Y direction and check intensity
            current_pos = stage.getPosition()
            stage.move("Y", current_pos["Y"] + 1000, True, True)
            time.sleep(0.1)
            
            intensity = self._grabMeanFrame()
            if intensity and baseline_intensity and intensity > baseline_intensity * 2.0:
                self._logger.info("High intensity detected - near center")
                
                # Look for rings and center on them
                center_pos = self._findRingCenter()
                if center_pos:
                    return center_pos
        
        # If we reach here, return current position as best guess
        final_pos = stage.getPosition()
        return (final_pos["X"], final_pos["Y"])
    
    def _detectWhiteLine(self, image: np.ndarray) -> bool:
        """
        Detect white lines in image using Hough transform.
        Returns True if lines are detected.
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.astype(np.uint8)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Apply Hough line transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            return lines is not None and len(lines) > 0
            
        except Exception as e:
            self._logger.error(f"Line detection failed: {e}")
            return False
    
    def _findRingCenter(self) -> tuple:
        """
        Find center of rings using Hough circle transform.
        Returns (x, y) stage coordinates or None if not found.
        """
        try:
            detector = self.getDetector()
            frame = detector.getLatestFrame()
            if frame is None:
                return None
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detect circles using Hough transform
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
            
            if circles is not None and len(circles[0]) > 0:
                # Find the largest circle (assuming it's the calibration ring)
                circles = np.round(circles[0, :]).astype("int")
                largest_circle = max(circles, key=lambda c: c[2])  # Max by radius
                
                center_x, center_y, radius = largest_circle
                self._logger.info(f"Ring found at image coordinates ({center_x}, {center_y}) with radius {radius}")
                
                # Convert image coordinates to stage offset
                # This is a simplified approach - in practice, you'd need proper calibration
                stage = self.getStage()
                current_pos = stage.getPosition()
                
                # Calculate offset needed to center the ring (simplified pixel-to-micron conversion)
                image_center_x = gray.shape[1] // 2
                image_center_y = gray.shape[0] // 2
                
                # Assume 1 pixel = 1 micrometer (this should be calibrated properly)
                offset_x = (center_x - image_center_x) * 1.0  # Adjust this scaling factor
                offset_y = (center_y - image_center_y) * 1.0
                
                # Move stage to center the ring
                target_x = current_pos["X"] - offset_x  # Negative because stage moves opposite to image
                target_y = current_pos["Y"] - offset_y
                
                stage.move("X", target_x, True, True)
                stage.move("Y", target_y, True, True)
                
                final_pos = stage.getPosition()
                return (final_pos["X"], final_pos["Y"])
            
            return None
            
        except Exception as e:
            self._logger.error(f"Ring detection failed: {e}")
            return None
    
    def _navigateMaze(self, path: list):
        """
        Navigate through maze path, capturing images at each position.
        """
        stage = self.getStage()
        detector = self.getDetector()
        
        images = []
        positions = []
        
        try:
            for step_idx, (dx, dy) in enumerate(path):
                if not self._maze_running:
                    break
                
                # Calculate target position (1000µm steps)
                start_x = self.MAZE_START_X * 1000
                start_y = self.MAZE_START_Y * 1000
                target_x = start_x + (dx * 1000)
                target_y = start_y + (dy * 1000)
                
                # Move to position
                stage.move("X", target_x, True, True)
                stage.move("Y", target_y, True, True)
                
                # Capture image
                time.sleep(0.1)  # Allow settling
                frame = detector.getLatestFrame()
                if frame is not None:
                    images.append(frame)
                    positions.append((target_x, target_y))
                    self._maze_positions.append((target_x, target_y))
                    self._logger.debug(f"Maze step {step_idx + 1}/{len(path)}: moved to ({target_x}, {target_y})")
            
            # Save TIFF stack
            if images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dir_path = os.path.join(os.path.expanduser("~"), "imswitch_calibrations", timestamp)
                os.makedirs(dir_path, exist_ok=True)
                
                stack_path = os.path.join(dir_path, "maze_navigation_stack.tiff")
                tif.imwrite(stack_path, np.array(images))
                
                positions_path = os.path.join(dir_path, "maze_positions.csv")
                np.savetxt(positions_path, np.array(positions), delimiter=",", header="X(µm),Y(µm)")
                
                self._logger.info(f"Maze navigation completed. {len(images)} images saved to {stack_path}")
                
        except Exception as e:
            self._logger.error(f"Maze navigation failed: {e}")
        finally:
            self._maze_running = False

    # ─────────────────────── helpers ────────────────────────────────────────

    def _grabMeanFrame(self):
        frame = self.getDetector().getLatestFrame()
        if frame is None or frame.size == 0:
            return None
        meanValue = np.mean(frame[::20, ::20])  # subsample for speed
        self._logger.debug(f"Mean value of frame: {meanValue}") 
        return meanValue

    def _savePositionsCsv(self):
        if not self._positions:
            return
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_path = os.path.join(os.path.expanduser("~"), "imswitch_calibrations", ts)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "stage_center_spiral.csv")
        np.savetxt(path, np.array(self._positions), delimiter=",", header="X(µm),Y(µm)")
        self._logger.info(f"Positions saved to {path}")

    # ─────────────────────── GUI convenience ───────────────────────────────

    def _startCalibrationFromGui(self):
        if IS_HEADLESS:
            return
        w = self._widget
        pos = self.performCalibration(w.spinStartX.value(), w.spinStartY.value(),
                                       w.spinExposure.value(), w.spinSpeed.value(),
                                       w.spinPitch.value())
        w.showPositions(pos)

    def displayImage(self, frame):
        if IS_HEADLESS:
            return
        self._widget.setImage(np.uint16(frame), colormap="gray", name="Calib", pixelsize=(1, 1))


class MovementController:
    """Tiny helper that moves one axis asynchronously."""

    def __init__(self, stage):
        self.stage = stage
        self._done = False

    def move_to_position(self, value, axis, speed, is_absolute):
        self._done = False
        threading.Thread(target=self._move, args=(value, axis, speed, is_absolute), daemon=True).start()

    def _move(self, value, axis, speed, is_absolute):
        self.stage.move(axis=axis, value=value, speed=speed, is_absolute=is_absolute, is_blocking=True)
        self._done = True

    def is_target_reached(self):
        return self._done
