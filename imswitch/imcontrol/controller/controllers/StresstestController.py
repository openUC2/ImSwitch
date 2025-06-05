import json
import os
import time
import threading
import random
import math
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    from imswitch.imcommon.model import initLogger, APIExport, dirtools
    from imswitch.imcommon.framework import Signal, Timer
    from ..basecontrollers import ImConWidgetController
    from imswitch import IS_HEADLESS
    _HAS_IMSWITCH = True
except ImportError:
    # Fallback for testing without full ImSwitch environment
    _HAS_IMSWITCH = False
    
    class APIExport:
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

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class StresstestParams:
    """Parameters for stress testing"""
    def __init__(self, minPosX=0.0, maxPosX=10000.0, minPosY=0.0, maxPosY=10000.0,
                 numRandomPositions=10, numCycles=5, timeInterval=60.0,
                 illuminationIntensity=50, exposureTime=0.1, saveImages=True, outputPath=""):
        self.minPosX = minPosX  # minimum X position in micrometers
        self.maxPosX = maxPosX  # maximum X position in micrometers
        self.minPosY = minPosY  # minimum Y position in micrometers
        self.maxPosY = maxPosY  # maximum Y position in micrometers
        self.numRandomPositions = numRandomPositions  # number of random positions per cycle
        self.numCycles = numCycles  # number of repetition cycles
        self.timeInterval = timeInterval  # time interval between cycles in seconds
        self.illuminationIntensity = illuminationIntensity  # illumination intensity (0-100)
        self.exposureTime = exposureTime  # camera exposure time in seconds
        self.saveImages = saveImages  # whether to save captured images
        self.outputPath = outputPath  # output directory for results
    
    def dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'minPosX': self.minPosX,
            'maxPosX': self.maxPosX,
            'minPosY': self.minPosY,
            'maxPosY': self.maxPosY,
            'numRandomPositions': self.numRandomPositions,
            'numCycles': self.numCycles,
            'timeInterval': self.timeInterval,
            'illuminationIntensity': self.illuminationIntensity,
            'exposureTime': self.exposureTime,
            'saveImages': self.saveImages,
            'outputPath': self.outputPath
        }


class StresstestResults:
    """Results from stress testing"""
    def __init__(self):
        self.totalPositions = 0
        self.completedPositions = 0
        self.averagePositionError = 0.0
        self.maxPositionError = 0.0
        self.positionErrors = []
        self.timestamps = []
        self.targetPositions = []
        self.actualPositions = []
        self.isRunning = False
    
    def dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'totalPositions': self.totalPositions,
            'completedPositions': self.completedPositions,
            'averagePositionError': self.averagePositionError,
            'maxPositionError': self.maxPositionError,
            'positionErrors': self.positionErrors,
            'timestamps': self.timestamps,
            'targetPositions': self.targetPositions,
            'actualPositions': self.actualPositions,
            'isRunning': self.isRunning
        }


class StresstestController(ImConWidgetController):
    """Controller for stage stress testing and camera calibration.
    
    This controller periodically moves to different random locations within a 
    specified range, takes images, and quantifies variation in position over time.
    It combines stage motion, camera acquisition, and illumination control
    similar to HistoScanController.
    """
    
    sigStresttestUpdate = Signal()
    sigStresttestComplete = Signal()
    sigPositionUpdate = Signal()  # signal for position updates
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        
        # Initialize parameters and results
        self.params = StresstestParams()
        self.results = StresstestResults()
        
        # Hardware managers
        self.stages = None
        self.detector = None
        self.illumination = None
        
        # State management
        self.isRunning = False
        self.shouldStop = False
        self.stresstest_thread = None
        
        # Position tracking
        self.target_positions = []
        self.actual_positions = []
        self.position_errors = []
        
        # Initialize hardware
        self._initializeHardware()
        
        # Set default output path
        self._setDefaultOutputPath()
        
        if _HAS_IMSWITCH:
            self._logger.info("StresstestController initialized")
        else:
            print("StresstestController initialized (testing mode)")
        
    def _initializeHardware(self):
        """Initialize hardware managers"""
        if not _HAS_IMSWITCH:
            # Mock hardware for testing
            self.stages = None
            self.detector = None  
            self.illumination = None
            return
            
        try:
            # Get stage/positioner
            positioner_names = self._master.positionersManager.getAllDeviceNames()
            if positioner_names:
                self.stages = self._master.positionersManager[positioner_names[0]]
                self._logger.info(f"Using positioner: {positioner_names[0]}")
            else:
                self._logger.warning("No positioners found")
                
            # Get detector/camera
            detector_names = self._master.detectorsManager.getAllDeviceNames()
            if detector_names:
                self.detector = self._master.detectorsManager[detector_names[0]]
                self._logger.info(f"Using detector: {detector_names[0]}")
            else:
                self._logger.warning("No detectors found")
                
            # Get illumination (laser or LED)
            laser_names = self._master.lasersManager.getAllDeviceNames()
            if laser_names:
                self.illumination = self._master.lasersManager[laser_names[0]]
                self._logger.info(f"Using laser: {laser_names[0]}")
            else:
                self._logger.warning("No lasers found")
                
        except Exception as e:
            self._logger.error(f"Error initializing hardware: {e}")
            
    def _setDefaultOutputPath(self):
        """Set default output path for results"""
        try:
            if _HAS_IMSWITCH:
                default_path = os.path.join(dirtools.UserFileDirs.Root, 'stresstest_results')
            else:
                default_path = os.path.join(os.path.expanduser("~"), 'stresstest_results')
            os.makedirs(default_path, exist_ok=True)
            self.params.outputPath = default_path
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error setting default output path: {e}")
            else:
                print(f"Error setting default output path: {e}")
            self.params.outputPath = ""
    
    @APIExport()
    def getStresstestParams(self) -> StresstestParams:
        """Get current stress test parameters"""
        return self.params
    
    @APIExport()
    def setStresstestParams(self, params: StresstestParams) -> bool:
        """Set stress test parameters"""
        try:
            self.params = params
            if _HAS_IMSWITCH:
                self._logger.info(f"Updated stress test parameters: {params}")
            else:
                print(f"Updated stress test parameters")
            return True
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error setting parameters: {e}")
            else:
                print(f"Error setting parameters: {e}")
            return False
    
    @APIExport()
    def getStresstestResults(self) -> StresstestResults:
        """Get current stress test results"""
        return self.results
    
    @APIExport()
    def startStresstest(self) -> bool:
        """Start the stress test"""
        if self.isRunning:
            if _HAS_IMSWITCH:
                self._logger.warning("Stress test already running")
            else:
                print("Stress test already running")
            return False
            
        if not self._validateHardware():
            return False
            
        try:
            self.isRunning = True
            self.shouldStop = False
            self.results = StresstestResults()
            self.results.isRunning = True
            
            # Generate random positions for testing
            self._generateRandomPositions()
            
            # Start stress test in background thread
            self.stresstest_thread = threading.Thread(target=self._runStresstest)
            self.stresstest_thread.start()
            
            if _HAS_IMSWITCH:
                self._logger.info("Stress test started")
            else:
                print("Stress test started")
            return True
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error starting stress test: {e}")
            else:
                print(f"Error starting stress test: {e}")
            self.isRunning = False
            self.results.isRunning = False
            return False
    
    @APIExport()
    def stopStresstest(self) -> bool:
        """Stop the stress test"""
        try:
            self.shouldStop = True
            if self.stresstest_thread and self.stresstest_thread.is_alive():
                self.stresstest_thread.join(timeout=5.0)
                
            self.isRunning = False
            self.results.isRunning = False
            if _HAS_IMSWITCH:
                self._logger.info("Stress test stopped")
            else:
                print("Stress test stopped")
            return True
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error stopping stress test: {e}")
            else:
                print(f"Error stopping stress test: {e}")
            return False
    
    def _validateHardware(self) -> bool:
        """Validate that required hardware is available"""
        if not _HAS_IMSWITCH:
            # In testing mode, always validate as True
            return True
            
        if not self.stages:
            self._logger.error("No stage/positioner available")
            return False
            
        if not self.detector:
            self._logger.error("No detector/camera available")
            return False
            
        return True
    
    def _generateRandomPositions(self):
        """Generate random positions within the specified range"""
        self.target_positions = []
        
        for cycle in range(self.params.numCycles):
            cycle_positions = []
            for _ in range(self.params.numRandomPositions):
                x = random.uniform(self.params.minPosX, self.params.maxPosX)
                y = random.uniform(self.params.minPosY, self.params.maxPosY)
                cycle_positions.append([x, y])
            self.target_positions.append(cycle_positions)
            
        total_positions = self.params.numCycles * self.params.numRandomPositions
        self.results.totalPositions = total_positions
        if _HAS_IMSWITCH:
            self._logger.info(f"Generated {total_positions} random positions across {self.params.numCycles} cycles")
        else:
            print(f"Generated {total_positions} random positions across {self.params.numCycles} cycles")
    
    def _runStresstest(self):
        """Main stress test execution loop"""
        try:
            self._setupIllumination()
            start_time = time.time()
            
            for cycle in range(self.params.numCycles):
                if self.shouldStop:
                    break
                    
                if _HAS_IMSWITCH:
                    self._logger.info(f"Starting cycle {cycle + 1}/{self.params.numCycles}")
                else:
                    print(f"Starting cycle {cycle + 1}/{self.params.numCycles}")
                
                # Process positions in this cycle
                for pos_idx, target_pos in enumerate(self.target_positions[cycle]):
                    if self.shouldStop:
                        break
                        
                    self._processPosition(target_pos, cycle, pos_idx)
                    
                # Wait for next cycle if not the last one
                if cycle < self.params.numCycles - 1:
                    self._waitForNextCycle(start_time, cycle + 1)
            
            # Finalize results
            self._finalizeResults()
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error during stress test execution: {e}")
            else:
                print(f"Error during stress test execution: {e}")
        finally:
            self._cleanup()
    
    def _setupIllumination(self):
        """Setup illumination for imaging"""
        if self.illumination:
            try:
                self.illumination.setEnabled(True)
                # Set intensity as percentage of max value
                max_intensity = getattr(self.illumination, 'valueRangeMax', 100)
                intensity_value = (self.params.illuminationIntensity / 100.0) * max_intensity
                self.illumination.setValue(intensity_value)
                if _HAS_IMSWITCH:
                    self._logger.info(f"Illumination set to {self.params.illuminationIntensity}%")
                else:
                    print(f"Illumination set to {self.params.illuminationIntensity}%")
            except Exception as e:
                if _HAS_IMSWITCH:
                    self._logger.warning(f"Could not setup illumination: {e}")
                else:
                    print(f"Could not setup illumination: {e}")
    
    def _processPosition(self, target_pos: List[float], cycle: int, pos_idx: int):
        """Process a single position: move, capture, analyze"""
        try:
            # Move to target position
            if _HAS_IMSWITCH:
                self._logger.debug(f"Moving to position {target_pos}")
                if self.stages:
                    self.stages.move(value=target_pos, axis="XY", is_absolute=True, is_blocking=True)
            else:
                print(f"Moving to position {target_pos}")
            
            # Wait for stage to settle
            time.sleep(0.1)
            
            # Get actual position
            if _HAS_IMSWITCH and self.stages:
                actual_pos_dict = self.stages.getPosition()
                actual_pos = [actual_pos_dict.get("X", 0), actual_pos_dict.get("Y", 0)]
            else:
                # Mock slight position error for testing
                actual_pos = [target_pos[0] + random.uniform(-1, 1), target_pos[1] + random.uniform(-1, 1)]
            
            # Calculate position error using basic math instead of numpy
            error = math.sqrt((target_pos[0] - actual_pos[0])**2 + (target_pos[1] - actual_pos[1])**2)
            
            # Capture image
            image = None
            if _HAS_IMSWITCH and self.detector:
                try:
                    if not self.detector._running:
                        self.detector.startAcquisition()
                    image = self.detector.getLatestFrame()
                except Exception as e:
                    self._logger.warning(f"Could not capture image: {e}")
            elif not _HAS_IMSWITCH:
                # Create mock image data for testing
                if _HAS_NUMPY:
                    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                else:
                    image = [[random.randint(0, 255) for _ in range(100)] for _ in range(100)]
            
            # Store results
            self.actual_positions.append(actual_pos)
            self.position_errors.append(error)
            self.results.targetPositions.append(target_pos)
            self.results.actualPositions.append(actual_pos)
            self.results.positionErrors.append(error)
            self.results.timestamps.append(datetime.now().isoformat())
            self.results.completedPositions += 1
            
            # Save image if requested
            if self.params.saveImages and image is not None:
                self._saveImage(image, target_pos, actual_pos, cycle, pos_idx)
                
            # Emit position update signal
            position_data = {
                'target': target_pos,
                'actual': actual_pos,
                'error': error,
                'cycle': cycle,
                'position_idx': pos_idx
            }
            self.sigPositionUpdate.emit(position_data)
            
            if _HAS_IMSWITCH:
                self._logger.debug(f"Position error: {error:.2f} µm")
            else:
                print(f"Position error: {error:.2f} µm")
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error processing position {target_pos}: {e}")
            else:
                print(f"Error processing position {target_pos}: {e}")
    
    def _saveImage(self, image, target_pos: List[float], actual_pos: List[float], 
                   cycle: int, pos_idx: int):
        """Save captured image with metadata"""
        try:
            # Try to import tifffile, fall back to basic file saving
            try:
                import tifffile
                has_tifffile = True
            except ImportError:
                has_tifffile = False
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if has_tifffile:
                filename = f"stresstest_c{cycle:02d}_p{pos_idx:02d}_{timestamp}.tif"
            else:
                filename = f"stresstest_c{cycle:02d}_p{pos_idx:02d}_{timestamp}.txt"
            filepath = os.path.join(self.params.outputPath, filename)
            
            # Create metadata
            metadata = {
                'target_position_x': target_pos[0],
                'target_position_y': target_pos[1],
                'actual_position_x': actual_pos[0],
                'actual_position_y': actual_pos[1],
                'position_error': math.sqrt((target_pos[0] - actual_pos[0])**2 + (target_pos[1] - actual_pos[1])**2),
                'cycle': cycle,
                'position_index': pos_idx,
                'timestamp': datetime.now().isoformat(),
                'illumination_intensity': self.params.illuminationIntensity,
                'exposure_time': self.params.exposureTime
            }
            
            # Save image with metadata
            if has_tifffile and _HAS_NUMPY:
                tifffile.imwrite(filepath, image, metadata=metadata)
            else:
                # Save metadata only as JSON file if we can't save image
                with open(filepath, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            if _HAS_IMSWITCH:
                self._logger.debug(f"Saved image: {filename}")
            else:
                print(f"Saved image: {filename}")
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.warning(f"Could not save image: {e}")
            else:
                print(f"Could not save image: {e}")
    
    def _waitForNextCycle(self, start_time: float, next_cycle: int):
        """Wait for the specified time interval before next cycle"""
        elapsed_time = time.time() - start_time
        target_time = next_cycle * self.params.timeInterval
        wait_time = target_time - elapsed_time
        
        if wait_time > 0:
            if _HAS_IMSWITCH:
                self._logger.info(f"Waiting {wait_time:.1f}s before next cycle")
            else:
                print(f"Waiting {wait_time:.1f}s before next cycle")
            time.sleep(wait_time)
    
    def _finalizeResults(self):
        """Calculate final statistics"""
        if self.position_errors:
            if _HAS_NUMPY:
                self.results.averagePositionError = float(np.mean(self.position_errors))
                self.results.maxPositionError = float(np.max(self.position_errors))
            else:
                # Use basic Python for statistics
                self.results.averagePositionError = sum(self.position_errors) / len(self.position_errors)
                self.results.maxPositionError = max(self.position_errors)
            
            # Save results to JSON file
            self._saveResults()
            
            if _HAS_IMSWITCH:
                self._logger.info(f"Stress test completed. Average error: {self.results.averagePositionError:.2f} µm, "
                                f"Max error: {self.results.maxPositionError:.2f} µm")
            else:
                print(f"Stress test completed. Average error: {self.results.averagePositionError:.2f} µm, "
                      f"Max error: {self.results.maxPositionError:.2f} µm")
        
        self.sigStresttestComplete.emit()
    
    def _saveResults(self):
        """Save results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stresstest_results_{timestamp}.json"
            filepath = os.path.join(self.params.outputPath, filename)
            
            # Create comprehensive results dictionary
            results_dict = {
                'parameters': self.params.dict(),
                'results': self.results.dict(),
                'summary': {
                    'total_positions': len(self.position_errors),
                    'average_error_um': self.results.averagePositionError,
                    'max_error_um': self.results.maxPositionError,
                    'min_error_um': min(self.position_errors) if self.position_errors else 0,
                    'std_error_um': self._calculate_std(self.position_errors) if self.position_errors else 0,
                    'test_duration_minutes': (len(self.position_errors) * self.params.timeInterval) / 60.0
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)
                
            if _HAS_IMSWITCH:
                self._logger.info(f"Results saved to: {filename}")
            else:
                print(f"Results saved to: {filename}")
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Could not save results: {e}")
            else:
                print(f"Could not save results: {e}")
    
    def _calculate_std(self, values):
        """Calculate standard deviation without numpy"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _cleanup(self):
        """Cleanup after stress test completion"""
        try:
            # Turn off illumination
            if _HAS_IMSWITCH and self.illumination:
                self.illumination.setEnabled(False)
                
            self.isRunning = False
            self.results.isRunning = False
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error during cleanup: {e}")
            else:
                print(f"Error during cleanup: {e}")