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
        self.MAX_SPEED = 20000  # µm/s
        
        # TODO: This is FRAME Specific - make this configurable through the configs
        self.HOME_POS_X = 0*1000  # in µm in x we are roughly at the wellplatepos 0
        self.HOME_POS_Y = 87*1000  # in µm in y we are roughly at the wellplatepos 87mm (opposite site )

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
    def getIsCalibrationRunning(self):
        return self._is_running

    @APIExport()
    def getCalibrationStatus(self) -> dict:
        """
        Get the current status of any running calibration process.
        """
        return {
            "is_running": self._is_running,
            "positions_collected": len(self._positions),
            "last_position": self._positions[-1] if self._positions else None
        }

    @APIExport()
    def stopCalibration(self) -> dict:
        """
        Stop any running calibration process.
        """
        if not self._is_running:
            return {"status": "info", "message": "No calibration is currently running"}
        
        self._is_running = False
        self._logger.info("Calibration process stopped by user request")
        return {"status": "stopped", "message": "Calibration process has been stopped"}

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
        
        # Set stage offset for both axes # TODO: NOT CORRECT 
        stage.setStageOffsetAxis(knownPosition=x_mm * 1000, currentPosition=current_pos["X"], axis="X")  # Convert mm to µm
        stage.setStageOffsetAxis(knownPosition=y_mm * 1000, currentPosition=current_pos["Y"], axis="Y")  # Convert mm to µm
        
        self._logger.info(f"Stage offset set to known position: X={x_mm}mm, Y={y_mm}mm")
        return {"status": "success", "x_mm": x_mm, "y_mm": y_mm}

    @APIExport()
    def findCalibrationCenter(
        self,
        unit_um: float = 1000.0,
        increment_units: int = 1,
        start_len_units: int = 1,
        min_x: float = None,
        max_x: float = None,
        min_y: float = None,
        max_y: float = None,
        intensity_factor: float = 1.5,
        settle_s: float = 0.1,
        max_legs: int = 50, 
        laser_name: str = None,
        laser_intensity: float = None,
        homing_procedure: bool = False
    ) -> dict:
        """
        API export for spiral search calibration center finding.
        Starts the search in a separate thread and returns immediately.
        
        Args:
            unit_um: Base grid step in µm
            increment_units: Increase after every two legs
            start_len_units: Starting leg length in units
            min_x, max_x, min_y, max_y: Absolute stage limits in µm
            intensity_factor: Stop when mean >= factor * baseline
            settle_s: Dwell after each move
            max_legs: Safety cap on spiral legs
            
        Returns:
            dict: Status information
        """
        if self._is_running:
            return {"status": "error", "message": "Another calibration is already running"}
        self.findCalibrationCenterThread = threading.Thread(
            target=self._findCalibrationCenterForThread,
            args=(unit_um, increment_units, start_len_units, min_x, max_x, min_y, max_y, intensity_factor, settle_s, max_legs, laser_name, laser_intensity, homing_procedure),
            daemon=True,
        )
        self.findCalibrationCenterThread.start()
        return {"status": "started", "message": "Calibration center search started"}

    def _findCalibrationCenterForThread(
        self,
        unit_um: float = 1000.0,
        increment_units: int = 1,
        start_len_units: int = 1,
        min_x: float = None,
        max_x: float = None,
        min_y: float = None,
        max_y: float = None,
        intensity_factor: float = 1.5,
        settle_s: float = 0.1,
        max_legs: int = 400,
        laser_name: str = None,
        laser_intensity: float = None, 
        homing_procedure: bool = False
    ) -> dict: # TODO: I think this interface can be neglected, we can direclty use this to start _findCalibrationCenter
        """
        Thread implementation for calibration center finding.
        """
        if self._is_running:
            return {"status": "error", "message": "Another calibration is already running"}
        
        self._is_running = True
        try:
            center_position = self._findCalibrationCenter(
                unit_um=unit_um,
                increment_units=increment_units,
                start_len_units=start_len_units,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                intensity_factor=intensity_factor,
                settle_s=settle_s,
                max_legs=max_legs, 
                laser_name=laser_name,
                laser_intensity=laser_intensity
            )
            
            if center_position:
                self._logger.info(f"Calibration center found at: {center_position}")
                # Save positions to CSV for record keeping
                self._savePositionsCsv()
                return {"status": "success", "center_position": center_position}
            else:
                self._logger.warning("Calibration center search completed but no center found")
                return {"status": "completed", "message": "No center found within search parameters"}
                
        except Exception as e:
            self._logger.error(f"Calibration center search failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            self._is_running = False

    @APIExport()
    def getCalibrationTargetInfo(self) -> dict:
        """
        Returns information about the calibration target including SVG representation.
        SVG files are served from disk via the ImSwitch server.
        """
        # Get SVG files from disk
        svg_files = self._getCalibrationSVGFiles()
        
        # Generate fallback SVG content if files are not found
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
            "svg_file_paths": svg_files,  # Paths to SVG files served by ImSwitch server
            #"frontside_svg": frontside_svg,  # Fallback SVG content
            #"backside_svg": backside_svg,   # Fallback SVG content
            "calibration_center": {"x": self.CALIBRATION_CENTER_X, "y": self.CALIBRATION_CENTER_Y},
            "maze_start": {"x": self.MAZE_START_X, "y": self.MAZE_START_Y},
            "stepsize_grid": {"x": self.STEPSIZE_GRID_X, "y": self.STEPSIZE_GRID_Y},
            "wellplate_start": {"x": self.WELLPLATE_START_X, "y": self.WELLPLATE_START_Y, "spacing": self.WELLPLATE_SPACING}
        }

    @APIExport()
    def stopFindCalibrationCenter(self) -> dict:
        """
        Stop the ongoing calibration center finding process.
        """
        if not self._is_running:
            return {"status": "info", "message": "No calibration center search is currently running"}
        
        self._is_running = False
        self._logger.info("Calibration center search stopped by user request")
        return {"status": "stopped", "message": "Calibration center search has been stopped"}
    
    def _getCalibrationSVGFiles(self) -> dict:
        """
        Get paths to calibration SVG files from disk.
        Returns dictionary with file paths that can be served via ImSwitch server.
        """
        try:
            _baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_data')
            images_dir = os.path.join(_baseDataFilesDir, 'images')
            
            svg_files = {
                "frontside_svg_path": None,
                "backside_svg_path": None,
                "available_svg_files": []
            }
            
            # Check if images directory exists
            if not os.path.exists(images_dir):
                self._logger.warning(f"Images directory not found: {images_dir}")
                return svg_files
            
            # Find all SVG files in directory and subfolders
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    if file.lower().endswith('.svg'):
                        # Get relative path from _data directory for server serving
                        relative_path = os.path.join(root.split("_data/")[-1], file)
                        svg_files["available_svg_files"].append(relative_path)
                        
                        # Check for specific calibration files
                        if 'calibration_front' in file.lower() or 'front' in file.lower():
                            svg_files["frontside_svg_path"] = relative_path
                        elif 'calibration_back' in file.lower() or 'back' in file.lower():
                            svg_files["backside_svg_path"] = relative_path
            
            self._logger.info(f"Found {len(svg_files['available_svg_files'])} SVG files in {images_dir}")
            return svg_files
            
        except Exception as e:
            self._logger.error(f"Failed to get SVG files: {e}")
            return {
                "frontside_svg_path": None,
                "backside_svg_path": None,
                "available_svg_files": []
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
            stage.move(axis="X", value=start_x_um, is_absolute=True, is_blocking=True)
            stage.move(axis="Y", value=start_y_um, is_absolute=True, is_blocking=True)
            
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
        # TODO: This has to be moved to a thread as well and cancellable 
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
                    stage.move(axis="X", value=x_pos, is_absolute=True, is_blocking=True)
                    stage.move(axis="Y", value=y_pos, is_absolute=True, is_blocking=True)
                    
                    # Capture image
                    time.sleep(0.1)  # Allow settling
                    frame = detector.getLatestFrame()
                    if frame is not None:
                        images.append(frame)
                        positions.append((x_pos, y_pos))
                        self._logger.debug(f"Captured image at grid position ({i}, {j})")
                    if not self._is_running:
                        return {"status": "stopped", "message": "Stepsize calibration stopped by user"}
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
                stage.move(axis="X", value=x_pos, is_absolute=True, is_blocking=True)
                stage.move(axis="Y", value=y_pos, is_absolute=True, is_blocking=True)
                
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
    
    def _findCalibrationCenter(
        self,
        unit_um: float = 1000.0,           # base grid step in µm
        increment_units: int = 1,          # increase after every two legs (set to 2 for 1,1,3,3,...)
        start_len_units: int = 1,          # starting leg length in units
        min_x: float | None = None,        # absolute stage limits in µm
        max_x: float | None = None,
        min_y: float | None = None,
        max_y: float | None = None,
        intensity_factor: float = 1.5,     # stop when mean >= factor * baseline
        settle_s: float = 0.1,             # dwell after each move
        max_legs: int = 400,                # safety cap on spiral legs
        laser_name: str = None,
        laser_intensity: float = None,
        homing_procedure = False
    ) -> tuple[float, float] | None:
        """
        Spiral search around current position. Moves in a square-spiral:
        (+X), (+Y), (-X), (-Y), increasing leg length after every two legs.
        Leg lengths (in 'units') follow: start_len_units, start_len_units,
        start_len_units+increment_units, start_len_units+increment_units, ...
        Each unit corresponds to 'unit_um' micrometers.

        Stops when mean intensity rises by 'intensity_factor' over the initial baseline.
        Returns (X, Y) in µm, or None if aborted.
        """
        stage = self.getStage()
        if homing_procedure:
            self._logger.info("Homing stage...")
            stage.home_x(isBlocking=False)
            stage.home_y(isBlocking=True)

        # Home the stage in X and Y
        stage.resetStageOffsetAxis(axis="X")
        stage.resetStageOffsetAxis(axis="Y")

        # Set position to reasonable value
        stage.setStageOffsetAxis(knownPosition = self.HOME_POS_X, currentPosition = 0, axis="X")  # in µm
        stage.setStageOffsetAxis(knownPosition = self.HOME_POS_Y, currentPosition = 0, axis="Y")  # in µm

        # Move to calibration center position
        self._logger.info("Moving to calibration center position...")
        stage.move(axis="XY", value=(self.CALIBRATION_CENTER_X*1000,self.CALIBRATION_CENTER_Y*1000), is_absolute=True, is_blocking=True, speed=self.MAX_SPEED)  # in µm

        # Turn on laser if specified
        if laser_name is not None and laser_intensity is not None:
            try:
                if hasattr(self._master, 'lasersManager'):
                    laser_controller = self._master.lasersManager.get(laser_name)
                    if laser_controller:
                        laser_controller.setLaserValue(laser_intensity)
                        laser_controller.setLaserActive(True)
                        self._logger.info(f"Laser {laser_name} activated at {laser_intensity}")
                else:
                    self._logger.warning("Laser manager not available")
            except Exception as e:
                self._logger.warning(f"Could not activate laser {laser_name}: {e}")
    
        if not self._is_running:
            return None

        # ensure camera is in livemode to grab frames continuously
        detector = self.getDetector()
        self._commChannel.sigStartLiveAcquistion.emit(True)

        # Helpers
        def clamp(val: float, lo: float | None, hi: float | None) -> float:
            if lo is not None and val < lo:
                return lo
            if hi is not None and val > hi:
                return hi
            return val

        def move_abs(axis: str, target: float) -> None:
            stage.move(axis=axis, value=target, is_absolute=True, is_blocking=True, speed=self.MAX_SPEED)


        threshold = 20 # We expect at least 20 pixels to be saturated

        # Spiral state
        dirs = [(+1, 0), (0, +1), (-1, 0), (0, -1)]  # +X, +Y, -X, -Y
        dir_idx = 0
        len_units = start_len_units
        legs_done = 0

        # Start from current absolute position
        pos = stage.getPosition()
        x = float(pos["X"])
        y = float(pos["Y"])

        # Main loop
        while self._is_running and legs_done < max_legs:
            dx_units, dy_units = dirs[dir_idx]
            leg_len_um = len_units * unit_um
            # TODO: Here we should move in background using MovementController/move_to_position -> then continously grab frames and analyse
            # TOdO we should also add the speed as an argument via APIExport 
            # Determine target on ONE axis per leg
            if dx_units != 0:
                target_x = clamp(x + dx_units * leg_len_um, min_x, max_x)
                if target_x != x:
                    move_abs("X", target_x)
                    x = target_x
            else:
                target_y = clamp(y + dy_units * leg_len_um, min_y, max_y)
                if target_y != y:
                    move_abs("Y", target_y)
                    y = target_y

            # Measure
            time.sleep(settle_s)
            
            # Baseline intensity (use a short average if available)
            nSaturatedPixels = self._grabAndProcessFrame()
            if nSaturatedPixels is not None and nSaturatedPixels >= threshold:
                # Optional refinement if available
                if hasattr(self, "_findRingCenter"):
                    try:
                        center = self._findRingCenter()
                        if center:
                            return center
                    except Exception:
                        pass
                return (x, y)

            # Next leg
            dir_idx = (dir_idx + 1) % 4
            legs_done += 1

            # Increase leg length after every two legs
            if legs_done % 2 == 0:
                len_units += increment_units

        # Safety exit: return last position
        return (x, y)
    
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
            import NanoImagingPack as nip
            blurred = nip.gaussf(gray, 100)
            maxY, maxY =  np.unravel_index(np.argmax(blurred, axis=None), blurred.shape)
            # now let's move in the opposite direction of the maximum
            mPixelSize = detector.pixelSizeUm[0]
            center_y, center_x = maxY*mPixelSize, maxY*mPixelSize
            stage = self.getStage()
            
            # Convert image coordinates to stage offset
            # This is a simplified approach - in practice, you'd need proper calibration
            current_pos = stage.getPosition()
        
            self._logger.info(f"Ring detected at image coords ({center_x}, {center_y})")

            # Calculate offset needed to center the ring (simplified pixel-to-micron conversion)
            image_center_x = gray.shape[1] // 2
            image_center_y = gray.shape[0] // 2
            
            # Assume 1 pixel = 1 micrometer (this should be calibrated properly)
            offset_x = (center_x - image_center_x) * 1.0  # Adjust this scaling factor
            offset_y = (center_y - image_center_y) * 1.0
            
            # Move stage to center the ring
            target_x = current_pos["X"] - offset_x  # Negative because stage moves opposite to image
            target_y = current_pos["Y"] - offset_y
            
            stage.move(axis="X", value=target_x, is_absolute=True, is_blocking=True)
            stage.move(axis="Y", value=target_y, is_absolute=True, is_blocking=True)
            
            final_pos = stage.getPosition()
            return (final_pos["X"], final_pos["Y"])
            
            
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
                stage.move(axis="X", value=target_x, is_absolute=True, is_blocking=True)
                stage.move(axis="Y", value=target_y, is_absolute=True, is_blocking=True)
                
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

    def _grabAndProcessFrame(self, threshold=250):
        '''returns the number of saturated pixels in the latest frame'''
        frame = self.getDetector().getLatestFrame()
        if frame is None or frame.size == 0:
            return 0 #None
        processedValue = np.sum(frame[::20, ::20]>threshold)
        self._logger.debug(f"Processed value of frame: {processedValue}")
        return processedValue

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
