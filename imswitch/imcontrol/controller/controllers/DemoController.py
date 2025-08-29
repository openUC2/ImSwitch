import time
import threading
import random
import math
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel

try:
    from imswitch.imcommon.model import initLogger, APIExport
    from imswitch.imcommon.framework import Signal
    from ..basecontrollers import ImConWidgetController
    _HAS_IMSWITCH = True
except ImportError:
    # Fallback for testing without full ImSwitch environment
    _HAS_IMSWITCH = False

    class APIExport:
        def __init__(self, **kwargs):
            pass  # Accept any arguments

        def __call__(self, func):
            return func

    class Signal:
        def emit(self, *args):
            pass

    class ImConWidgetController:
        def __init__(self, *args, **kwargs):
            pass

    def initLogger(obj):
        import logging
        return logging.getLogger(__name__)

# Import scan coordinate functions
try:
    from .camera_stage_mapping.scan_coords_times import ordered_spiral, raster
    _HAS_SCAN_COORDS = True
except ImportError:
    _HAS_SCAN_COORDS = False

    def ordered_spiral(starting_x, starting_y, number_of_shells, x_move, y_move):
        """Fallback spiral function"""
        return [(starting_x, starting_y)]

    def raster(starting_x, starting_y, x_move, y_move, rows, columns):
        """Fallback raster function"""
        return [(starting_x, starting_y)]


class DemoParams(BaseModel):
    """
    Pydantic model for demo parameters.
    """
    minPosX: float = 0.0          # minimum X position in micrometers
    maxPosX: float = 10000.0      # maximum X position in micrometers
    minPosY: float = 0.0          # minimum Y position in micrometers
    maxPosY: float = 10000.0      # maximum Y position in micrometers
    scanningScheme: str = "random"  # scanning scheme: "spiral", "random", "grid"
    illuminationMode: str = "random"  # illumination mode: "random", "continuous"

    # Grid/spiral specific parameters
    gridRows: int = 3             # number of rows for grid scanning
    gridColumns: int = 3          # number of columns for grid scanning
    spiralShells: int = 3         # number of shells for spiral scanning

    # Random specific parameters
    numRandomPositions: int = 10  # number of random positions

    # Demo control parameters
    dwellTime: float = 2.0        # time to dwell at each position in seconds
    totalRunTime: float = 60.0    # total demo run time in seconds

    class Config:
        # Allows arbitrary Python types if necessary
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        """
        Override dict() to convert to dictionary for JSON serialization.
        Calls the parent dict() and returns the result.
        """
        return super().dict(*args, **kwargs)


class DemoResults(BaseModel):
    """Pydantic model for demo results."""
    totalPositions: int = 0
    currentPosition: int = 0
    currentCoordinates: List[float] = [0.0, 0.0]
    isRunning: bool = False
    elapsedTime: float = 0.0
    startTime: str = ""
    currentIllumination: Dict[str, Any] = {}

    class Config:
        # Allows arbitrary Python types if necessary
        arbitrary_types_allowed = True

    def dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'totalPositions': self.totalPositions,
            'currentPosition': self.currentPosition,
            'currentCoordinates': self.currentCoordinates,
            'isRunning': self.isRunning,
            'elapsedTime': self.elapsedTime,
            'startTime': self.startTime,
            'currentIllumination': self.currentIllumination
        }


class DemoController(ImConWidgetController):
    """Controller for trade fair demonstrations.

    This controller performs automated demonstrations with stage motion and
    illumination control, supporting different scanning patterns and
    illumination modes.
    """

    sigDemoUpdate = Signal()
    sigDemoComplete = Signal()
    sigPositionUpdate = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # Initialize parameters and results
        self.params = DemoParams()
        self.results = DemoResults()

        # Hardware managers
        self.stages = None
        self.ledMatrix = None
        self.lasers = None

        # State management
        self.isRunning = False
        self.shouldStop = False
        self.demo_thread = None

        # Position tracking
        self.demo_positions = []
        self.current_position_index = 0
        self.start_time = None

        # Initialize hardware
        self._initializeHardware()

        if _HAS_IMSWITCH:
            self._logger.info("DemoController initialized")
        else:
            print("DemoController initialized (testing mode)")

    def _initializeHardware(self):
        """Initialize hardware managers"""
        if not _HAS_IMSWITCH:
            # Mock hardware for testing
            self.stages = None
            self.ledMatrix = None
            self.lasers = None
            return

        try:
            # Get stage/positioner
            self._initializeStages()

            # Get LED matrix
            self._initializeLEDMatrix()

            # Get lasers
            self._initializeLasers()

        except Exception as e:
            self._logger.error(f"Error initializing hardware: {e}")

    def _initializeStages(self):
        """Initialize stage hardware"""
        positioner_names = self._master.positionersManager.getAllDeviceNames()
        if positioner_names:
            self.stages = self._master.positionersManager[positioner_names[0]]
            self._logger.info(f"Using positioner: {positioner_names[0]}")
        else:
            self._logger.warning("No positioners found")

    def _initializeLEDMatrix(self):
        """Initialize LED matrix hardware"""
        try:
            if hasattr(self._master, 'LEDMatrixManager'):
                ledmatrix_names = self._master.LEDMatrixManager.getAllDeviceNames()
                if ledmatrix_names:
                    led_name = ledmatrix_names[0]
                    self.ledMatrix = self._master.LEDMatrixManager[led_name]
                    self._logger.info(f"Using LED matrix: {led_name}")
                else:
                    self._logger.warning("No LED matrix found")
            else:
                self._logger.warning("LEDMatrixManager not available")
        except Exception as e:
            self._logger.warning(f"Could not initialize LED matrix: {e}")

    def _initializeLasers(self):
        """Initialize laser hardware"""
        try:
            laser_names = self._master.lasersManager.getAllDeviceNames()
            if laser_names:
                self.lasers = {}
                for laser_name in laser_names:
                    self.lasers[laser_name] = self._master.lasersManager[laser_name]
                self._logger.info(f"Using lasers: {list(self.lasers.keys())}")
            else:
                self._logger.warning("No lasers found")
        except Exception as e:
            self._logger.warning(f"Could not initialize lasers: {e}")

    def _generatePositions(self) -> List[List[float]]:
        """Generate positions based on scanning scheme"""
        positions = []

        if self.params.scanningScheme == "random":
            # Generate random positions
            for _ in range(self.params.numRandomPositions):
                x = random.uniform(self.params.minPosX, self.params.maxPosX)
                y = random.uniform(self.params.minPosY, self.params.maxPosY)
                positions.append([x, y])

        elif self.params.scanningScheme == "grid":
            positions = self._generateGridPositions()

        elif self.params.scanningScheme == "spiral":
            positions = self._generateSpiralPositions()

        return positions

    def _generateGridPositions(self) -> List[List[float]]:
        """Generate grid positions using raster function"""
        if _HAS_SCAN_COORDS:
            x_range = self.params.maxPosX - self.params.minPosX
            y_range = self.params.maxPosY - self.params.minPosY
            x_step = x_range / max(1, self.params.gridColumns - 1)
            y_step = y_range / max(1, self.params.gridRows - 1)

            grid_coords = raster(
                self.params.minPosX, self.params.minPosY,
                x_step, y_step,
                self.params.gridRows, self.params.gridColumns
            )
            return [[float(coord[0]), float(coord[1])] for coord in grid_coords]
        else:
            # Fallback grid generation
            positions = []
            for i in range(self.params.gridColumns):
                for j in range(self.params.gridRows):
                    x_range = self.params.maxPosX - self.params.minPosX
                    y_range = self.params.maxPosY - self.params.minPosY
                    x_denom = max(1, self.params.gridColumns - 1)
                    y_denom = max(1, self.params.gridRows - 1)
                    x = self.params.minPosX + i * x_range / x_denom
                    y = self.params.minPosY + j * y_range / y_denom
                    positions.append([x, y])
            return positions

    def _generateSpiralPositions(self) -> List[List[float]]:
        """Generate spiral positions"""
        if _HAS_SCAN_COORDS:
            center_x = (self.params.minPosX + self.params.maxPosX) / 2
            center_y = (self.params.minPosY + self.params.maxPosY) / 2
            x_range = self.params.maxPosX - self.params.minPosX
            y_range = self.params.maxPosY - self.params.minPosY
            x_step = x_range / (2 * self.params.spiralShells)
            y_step = y_range / (2 * self.params.spiralShells)

            spiral_coords = ordered_spiral(
                center_x, center_y,
                self.params.spiralShells,
                x_step, y_step
            )
            return [[float(coord[0]), float(coord[1])] for coord in spiral_coords]
        else:
            # Fallback spiral generation - simple circular pattern
            positions = []
            center_x = (self.params.minPosX + self.params.maxPosX) / 2
            center_y = (self.params.minPosY + self.params.maxPosY) / 2
            radius_x = (self.params.maxPosX - center_x) / 2
            radius_y = (self.params.maxPosY - center_y) / 2
            radius = min(radius_x, radius_y)

            positions.append([center_x, center_y])  # Center point
            for shell in range(1, self.params.spiralShells + 1):
                shell_radius = shell * radius / self.params.spiralShells
                points_in_shell = shell * 8  # More points in outer shells
                for i in range(points_in_shell):
                    angle = 2 * math.pi * i / points_in_shell
                    x = center_x + shell_radius * math.cos(angle)
                    y = center_y + shell_radius * math.sin(angle)
                    # Ensure positions stay within bounds
                    x = max(self.params.minPosX, min(self.params.maxPosX, x))
                    y = max(self.params.minPosY, min(self.params.maxPosY, y))
                    positions.append([x, y])
            return positions

    def _moveToPosition(self, position: List[float]):
        """Move stage to specified position"""
        if not _HAS_IMSWITCH or self.stages is None:
            # Mock movement for testing
            time.sleep(0.1)
            return

        try:
            # Move to position
            self.stages.move(position[0], "X")
            self.stages.move(position[1], "Y")
            self._logger.debug(f"Moved to position: {position}")
        except Exception as e:
            self._logger.error(f"Error moving to position {position}: {e}")

    def _setRandomIllumination(self):
        """Set random LED and laser illumination"""
        if self.params.illuminationMode != "random":
            return

        # Random LED color at intensity 255
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 255)  # White
        ]
        random_color = random.choice(colors)

        try:
            if self.ledMatrix and _HAS_IMSWITCH:
                self.ledMatrix.setAll(random_color)
                self._logger.debug(f"Set LED to color: {random_color}")

            # Turn on random laser
            if self.lasers and _HAS_IMSWITCH:
                laser_names = list(self.lasers.keys())
                if laser_names:
                    random_laser = random.choice(laser_names)
                    self.lasers[random_laser].setEnabled(True)
                    self._logger.debug(f"Enabled laser: {random_laser}")

        except Exception as e:
            self._logger.error(f"Error setting random illumination: {e}")

    def _setContinuousIllumination(self, enable: bool = True):
        """Set continuous illumination"""
        if self.params.illuminationMode != "continuous":
            return

        try:
            if self.ledMatrix and _HAS_IMSWITCH:
                if enable:
                    self.ledMatrix.setAll((255, 255, 255))  # White light
                else:
                    self.ledMatrix.setAll((0, 0, 0))  # Turn off
                self._logger.debug(f"Set continuous LED: {enable}")

        except Exception as e:
            self._logger.error(f"Error setting continuous illumination: {e}")

    def _turnOffIllumination(self):
        """Turn off all illumination"""
        try:
            # Turn off LED matrix
            if self.ledMatrix and _HAS_IMSWITCH:
                self.ledMatrix.setAll((0, 0, 0))

            # Turn off all lasers
            if self.lasers and _HAS_IMSWITCH:
                for laser in self.lasers.values():
                    laser.setEnabled(False)

            self._logger.debug("Turned off all illumination")
        except Exception as e:
            self._logger.error(f"Error turning off illumination: {e}")

    def _runDemo(self):
        """Main demo execution loop"""
        try:
            self.start_time = time.time()
            self.results.startTime = datetime.now().isoformat()
            self.results.isRunning = True

            # Generate positions
            self.demo_positions = self._generatePositions()
            self.results.totalPositions = len(self.demo_positions)

            if _HAS_IMSWITCH:
                message = f"Starting demo with {len(self.demo_positions)} positions"
                self._logger.info(message)
            else:
                print(f"Starting demo with {len(self.demo_positions)} positions")

            # Set continuous illumination if needed
            if self.params.illuminationMode == "continuous":
                self._setContinuousIllumination(True)

            # Main demo loop
            for i, position in enumerate(self.demo_positions):
                if self.shouldStop:
                    break

                # Update status
                self.results.currentPosition = i + 1
                self.results.currentCoordinates = position
                self.results.elapsedTime = time.time() - self.start_time

                # Move to position
                self._moveToPosition(position)

                # Set illumination for this position
                if self.params.illuminationMode == "random":
                    self._setRandomIllumination()

                # Dwell at position
                time.sleep(self.params.dwellTime)

                # Turn off illumination after dwell time in random mode
                if self.params.illuminationMode == "random":
                    self._turnOffIllumination()

                # Check if total run time exceeded
                if time.time() - self.start_time >= self.params.totalRunTime:
                    break

                # Emit update signal
                self.sigDemoUpdate.emit()

        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error in demo execution: {e}")
            else:
                print(f"Error in demo execution: {e}")
        finally:
            # Clean up
            self._turnOffIllumination()
            self.isRunning = False
            self.results.isRunning = False
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.results.elapsedTime = elapsed
            self.sigDemoComplete.emit()

            if _HAS_IMSWITCH:
                self._logger.info("Demo completed")
            else:
                print("Demo completed")

    @APIExport()
    def getDemoParams(self) -> DemoParams:
        """Get current demo parameters"""
        return self.params

    @APIExport(requestType="POST")
    def setDemoParams(self, params: DemoParams) -> bool:
        """Set demo parameters"""
        try:
            self.params = params

            if _HAS_IMSWITCH:
                self._logger.info("Updated demo parameters")
            else:
                print("Updated demo parameters")
            return True
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error setting parameters: {e}")
            else:
                print(f"Error setting parameters: {e}")
            return False

    @APIExport()
    def getDemoResults(self) -> DemoResults:
        """Get current demo results"""
        return self.results

    @APIExport()
    def startDemo(self) -> bool:
        """Start the demo"""
        if self.isRunning:
            if _HAS_IMSWITCH:
                self._logger.warning("Demo is already running")
            else:
                print("Demo is already running")
            return False

        try:
            self.isRunning = True
            self.shouldStop = False

            # Reset results
            self.results = DemoResults()

            # Start demo thread
            self.demo_thread = threading.Thread(target=self._runDemo, daemon=True)
            self.demo_thread.start()

            if _HAS_IMSWITCH:
                self._logger.info("Demo started")
            else:
                print("Demo started")
            return True

        except Exception as e:
            self.isRunning = False
            if _HAS_IMSWITCH:
                self._logger.error(f"Error starting demo: {e}")
            else:
                print(f"Error starting demo: {e}")
            return False

    @APIExport()
    def stopDemo(self) -> bool:
        """Stop the demo"""
        if not self.isRunning:
            if _HAS_IMSWITCH:
                self._logger.warning("Demo is not running")
            else:
                print("Demo is not running")
            return False

        try:
            self.shouldStop = True

            # Wait for thread to complete
            if self.demo_thread and self.demo_thread.is_alive():
                self.demo_thread.join(timeout=5.0)

            # Ensure illumination is turned off
            self._turnOffIllumination()

            self.isRunning = False

            if _HAS_IMSWITCH:
                self._logger.info("Demo stopped")
            else:
                print("Demo stopped")
            return True

        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error stopping demo: {e}")
            else:
                print(f"Error stopping demo: {e}")
            return False
