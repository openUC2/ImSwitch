import json
import os
import time
import threading
import random
import math
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from typing import List
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

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Import off-the-shelf image registration libraries if available
try:
    import cv2
    from skimage import registration
    from skimage.feature import match_template
    import numpy as np
    _HAS_IMAGE_REGISTRATION = True
except ImportError:
    _HAS_IMAGE_REGISTRATION = False


def displacement_between_images_skimage(image1, image2):
    """
    Calculate displacement between two images using scikit-image phase cross-correlation.
    
    Args:
        image1: Reference image (numpy array)
        image2: Image to compare (numpy array)
    
    Returns:
        numpy array: [y_shift, x_shift] displacement in pixels
    """
    if not _HAS_IMAGE_REGISTRATION:
        raise ImportError("Required libraries (scikit-image, opencv) not available for image registration")
    
    # Convert to grayscale if needed
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # Use scikit-image phase cross-correlation
    shift, error, diffphase = registration.phase_cross_correlation(image1, image2)
    
    return shift


def locate_feature_in_image_opencv(image, template):
    """
    Locate a template in an image using OpenCV template matching.
    
    Args:
        image: Large image to search in (numpy array)
        template: Template to find (numpy array)
    
    Returns:
        numpy array: [y, x] position of template center in image
    """
    if not _HAS_IMAGE_REGISTRATION:
        raise ImportError("Required libraries (opencv) not available for template matching")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    
    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Calculate center position of the matched template
    template_center = np.array([max_loc[1] + template.shape[0] // 2, 
                               max_loc[0] + template.shape[1] // 2])
    
    return template_center


class StresstestParams(BaseModel):
    """
    Pydantic model for stress test parameters.
    """
    minPosX: float = 0.0          # minimum X position in micrometers
    maxPosX: float = 10000.0      # maximum X position in micrometers
    minPosY: float = 0.0          # minimum Y position in micrometers
    maxPosY: float = 10000.0      # maximum Y position in micrometers
    numRandomPositions: int = 5  # number of random positions per cycle
    numCycles: int = 3            # number of repetition cycles
    timeInterval: float = 10.0    # time interval between cycles in seconds
    illuminationIntensity: float = 50.0  # illumination intensity (0-100)
    exposureTime: float = 0.1     # camera exposure time in seconds
    saveImages: bool = True       # whether to save captured images
    outputPath: str = ""          # output directory for results
    
    # Image-based error estimation parameters
    enableImageBasedError: bool = False  # enable image registration-based error measurement
    numImagesPerPosition: int = 5        # number of images to capture per position for registration
    imageRegistrationMethod: str = "fft" # registration method: "fft" or "correlation"
    pixelSizeUM: float = 0.1            # pixel size in micrometers for converting pixel shifts to distance

    class Config:
        # Allows arbitrary Python types if necessary
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        """
        Override dict() to convert to dictionary for JSON serialization.
        Calls the parent dict() and returns the result.
        """
        return super().dict(*args, **kwargs)

class StresstestResults(BaseModel):
    """Pydantic model for stress test results."""
    totalPositions: int = 0
    completedPositions: int = 0
    averagePositionError: float = 0.0
    maxPositionError: float = 0.0
    positionErrors: List[float] = []
    timestamps: List[str] = []
    targetPositions: List[List[float]] = []
    actualPositions: List[List[float]] = []
    isRunning: bool = False
    
    # Image-based error measurement results
    imageBasedErrors: List[float] = []      # average image registration errors per position 
    imageShifts: List[List[float]] = []     # pixel shifts [x, y] for each position
    imageRegistrationResults: List[Dict] = []  # detailed registration results per position
    averageImageError: float = 0.0
    maxImageError: float = 0.0

    class Config:
        # Allows arbitrary Python types if necessary
        arbitrary_types_allowed = True
    
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
            'isRunning': self.isRunning,
            'imageBasedErrors': self.imageBasedErrors,
            'imageShifts': self.imageShifts,
            'imageRegistrationResults': self.imageRegistrationResults,
            'averageImageError': self.averageImageError,
            'maxImageError': self.maxImageError
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
        
        # Load default parameters from manager if available
        self._loadDefaultParams()
        
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
            if not self.params.outputPath:  # Only set if not already set from manager
                self.params.outputPath = default_path
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error setting default output path: {e}")
            else:
                print(f"Error setting default output path: {e}")
            if not self.params.outputPath:
                self.params.outputPath = ""

    def _loadDefaultParams(self):
        """Load default parameters from StresstestManager if available"""
        if not _HAS_IMSWITCH:
            return
            
        try:
            # Check if StresstestManager is available
            if hasattr(self._master, 'StresstestManager'):
                manager = self._master.StresstestManager
                default_params = manager.getDefaultParams()
                
                # Update params with default values from manager
                self.params.minPosX = default_params.get('minPosX', self.params.minPosX)
                self.params.maxPosX = default_params.get('maxPosX', self.params.maxPosX)
                self.params.minPosY = default_params.get('minPosY', self.params.minPosY)
                self.params.maxPosY = default_params.get('maxPosY', self.params.maxPosY)
                self.params.numRandomPositions = default_params.get('numRandomPositions', self.params.numRandomPositions)
                self.params.numCycles = default_params.get('numCycles', self.params.numCycles)
                self.params.timeInterval = default_params.get('timeInterval', self.params.timeInterval)
                self.params.illuminationIntensity = default_params.get('illuminationIntensity', self.params.illuminationIntensity)
                self.params.exposureTime = default_params.get('exposureTime', self.params.exposureTime)
                self.params.saveImages = default_params.get('saveImages', self.params.saveImages)
                self.params.outputPath = default_params.get('outputPath', self.params.outputPath)
                self.params.enableImageBasedError = default_params.get('enableImageBasedError', self.params.enableImageBasedError)
                self.params.numImagesPerPosition = default_params.get('numImagesPerPosition', self.params.numImagesPerPosition)
                self.params.imageRegistrationMethod = default_params.get('imageRegistrationMethod', self.params.imageRegistrationMethod)
                self.params.pixelSizeUM = default_params.get('pixelSizeUM', self.params.pixelSizeUM)
                
                self._logger.info("Loaded default parameters from StresstestManager")
            else:
                self._logger.debug("StresstestManager not available, using built-in defaults")
                
        except Exception as e:
            self._logger.warning(f"Could not load default parameters from manager: {e}")

    def _saveParamsToManager(self):
        """Save current parameters to StresstestManager if available"""
        if not _HAS_IMSWITCH:
            return
            
        try:
            # Check if StresstestManager is available
            if hasattr(self._master, 'StresstestManager'):
                manager = self._master.StresstestManager
                params_dict = self.params.dict()
                manager.updateParams(params_dict)
                self._logger.debug("Saved parameters to StresstestManager")
            else:
                self._logger.debug("StresstestManager not available, cannot save parameters")
                
        except Exception as e:
            self._logger.warning(f"Could not save parameters to manager: {e}")
    
    @APIExport()
    def getStresstestParams(self) -> StresstestParams:
        """Get current stress test parameters"""
        return self.params
    
    @APIExport(requestType="POST")
    def setStresstestParams(self, params: StresstestParams) -> bool:
        """Set stress test parameters"""
        try:
            self.params = params
            
            # Save parameters to manager if available
            self._saveParamsToManager()
            
            if _HAS_IMSWITCH:
                self._logger.info(f"Updated stress test parameters")
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
            
            # Capture image(s) and perform image-based error estimation if enabled
            image = None
            image_based_error = 0.0
            image_shift = [0.0, 0.0]
            registration_results = {}
            
            if self.params.enableImageBasedError:
                image_based_results = self._performImageBasedErrorEstimation()
                image_based_error = image_based_results['average_error']
                image_shift = image_based_results['average_shift']
                registration_results = image_based_results['detailed_results']
                image = image_based_results.get('reference_image')
            else:
                # Capture single image
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
            
            # Store image-based results if enabled
            if self.params.enableImageBasedError:
                self.results.imageBasedErrors.append(image_based_error)
                self.results.imageShifts.append(image_shift)
                self.results.imageRegistrationResults.append(registration_results)
            
            # Save image if requested
            if self.params.saveImages and image is not None:
                self._saveImage(image, target_pos, actual_pos, cycle, pos_idx, image_based_error, image_shift)
                
            # Emit position update signal
            position_data = {
                'target': target_pos,
                'actual': actual_pos,
                'error': error,
                'cycle': cycle,
                'position_idx': pos_idx,
                'image_based_error': image_based_error,
                'image_shift': image_shift
            }
            self.sigPositionUpdate.emit(position_data)
            
            if _HAS_IMSWITCH:
                if self.params.enableImageBasedError:
                    self._logger.debug(f"Position error: {error:.2f} µm, Image error: {image_based_error:.2f} µm")
                else:
                    self._logger.debug(f"Position error: {error:.2f} µm")
            else:
                if self.params.enableImageBasedError:
                    print(f"Position error: {error:.2f} µm, Image error: {image_based_error:.2f} µm")
                else:
                    print(f"Position error: {error:.2f} µm")
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error processing position {target_pos}: {e}")
            else:
                print(f"Error processing position {target_pos}: {e}")
    
    def _performImageBasedErrorEstimation(self):
        """Perform image-based error estimation by capturing multiple images and computing registration shifts"""
        try:
            if not _HAS_IMAGE_REGISTRATION or not _HAS_NUMPY:
                if _HAS_IMSWITCH:
                    self._logger.warning("Image registration libraries (scikit-image, opencv) not available, skipping image-based error estimation")
                else:
                    print("Image registration libraries (scikit-image, opencv) not available, skipping image-based error estimation")
                return {
                    'average_error': 0.0,
                    'average_shift': [0.0, 0.0],
                    'detailed_results': {},
                    'reference_image': None
                }
            
            images = []
            shifts = []
            
            # Capture multiple images
            for img_idx in range(self.params.numImagesPerPosition):
                if _HAS_IMSWITCH and self.detector:
                    try:
                        if not self.detector._running:
                            self.detector.startAcquisition()
                        # Wait a bit between captures to allow for any micro-movements
                        if img_idx > 0:
                            time.sleep(0.05)
                        image = self.detector.getLatestFrame()
                        if image is not None:
                            images.append(image)
                    except Exception as e:
                        if _HAS_IMSWITCH:
                            self._logger.warning(f"Could not capture image {img_idx}: {e}")
                        else:
                            print(f"Could not capture image {img_idx}: {e}")
                elif not _HAS_IMSWITCH:
                    # Create mock images with slight shifts for testing
                    base_shift_x = random.uniform(-2, 2)
                    base_shift_y = random.uniform(-2, 2)
                    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                    # Add some structure to make registration meaningful
                    image[30:70, 30:70] = np.random.randint(100, 200, (40, 40), dtype=np.uint8)
                    # Simulate small shifts for each image
                    shift_x = base_shift_x + random.uniform(-0.5, 0.5)
                    shift_y = base_shift_y + random.uniform(-0.5, 0.5)
                    shifted_img = np.roll(image, int(shift_x), axis=0)
                    shifted_img = np.roll(shifted_img, int(shift_y), axis=1)
                    images.append(shifted_img)
            
            if len(images) < 2:
                if _HAS_IMSWITCH:
                    self._logger.warning("Not enough images captured for registration analysis")
                else:
                    print("Not enough images captured for registration analysis")
                return {
                    'average_error': 0.0,
                    'average_shift': [0.0, 0.0],
                    'detailed_results': {},
                    'reference_image': images[0] if images else None
                }
            
            # Use first image as reference
            reference_image = images[0]
            
            # Calculate registration shifts for each subsequent image
            detailed_results = {
                'num_images': len(images),
                'registration_method': self.params.imageRegistrationMethod,
                'pixel_size_um': self.params.pixelSizeUM,
                'individual_shifts': []
            }
            
            for i, img in enumerate(images[1:], 1):
                try:
                    if self.params.imageRegistrationMethod == "fft":
                        # Use scikit-image FFT-based registration
                        displacement = displacement_between_images_skimage(reference_image, img)
                    else:
                        # Use OpenCV template matching (correlation-based)
                        # Extract a central region for template matching
                        h, w = reference_image.shape[:2]
                        template_size = min(h//4, w//4, 50)  # Use a reasonable template size
                        template = reference_image[h//2-template_size//2:h//2+template_size//2,
                                                 w//2-template_size//2:w//2+template_size//2]
                        
                        # Find template position in the image
                        position = locate_feature_in_image_opencv(img, template)
                        expected_pos = np.array([h//2, w//2])
                        displacement = position - expected_pos
                    
                    # Convert displacement to micrometers
                    displacement_um = displacement * self.params.pixelSizeUM
                    distance_um = np.sqrt(np.sum(displacement_um**2))
                    
                    shifts.append(displacement.tolist())
                    detailed_results['individual_shifts'].append({
                        'image_index': i,
                        'displacement_pixels': displacement.tolist(),
                        'displacement_um': displacement_um.tolist(), 
                        'distance_um': float(distance_um)
                    })
                    
                except Exception as e:
                    if _HAS_IMSWITCH:
                        self._logger.warning(f"Registration failed for image {i}: {e}")
                    else:
                        print(f"Registration failed for image {i}: {e}")
                    # Use zero displacement for failed registrations
                    shifts.append([0.0, 0.0])
                    detailed_results['individual_shifts'].append({
                        'image_index': i,
                        'displacement_pixels': [0.0, 0.0],
                        'displacement_um': [0.0, 0.0],
                        'distance_um': 0.0,
                        'error': str(e)
                    })
            
            # Calculate average shift and error
            if shifts:
                avg_shift = np.mean(shifts, axis=0)
                # Calculate average error in micrometers  
                errors_um = [result['distance_um'] for result in detailed_results['individual_shifts'] if 'error' not in result]
                avg_error = np.mean(errors_um) if errors_um else 0.0
                
                detailed_results['average_shift_pixels'] = avg_shift.tolist()
                detailed_results['average_shift_um'] = (avg_shift * self.params.pixelSizeUM).tolist()
                detailed_results['average_error_um'] = float(avg_error)
                detailed_results['max_error_um'] = float(max(errors_um)) if errors_um else 0.0
                detailed_results['std_error_um'] = float(np.std(errors_um)) if len(errors_um) > 1 else 0.0
            else:
                avg_shift = np.array([0.0, 0.0])
                avg_error = 0.0
                detailed_results['average_shift_pixels'] = [0.0, 0.0]
                detailed_results['average_shift_um'] = [0.0, 0.0]
                detailed_results['average_error_um'] = 0.0
                detailed_results['max_error_um'] = 0.0
                detailed_results['std_error_um'] = 0.0
            
            return {
                'average_error': avg_error,
                'average_shift': avg_shift.tolist(),
                'detailed_results': detailed_results,
                'reference_image': reference_image
            }
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Error in image-based error estimation: {e}")
            else:
                print(f"Error in image-based error estimation: {e}")
            return {
                'average_error': 0.0,
                'average_shift': [0.0, 0.0],
                'detailed_results': {'error': str(e)},
                'reference_image': None
            }
    
    def _saveImage(self, image, target_pos: List[float], actual_pos: List[float], 
                   cycle: int, pos_idx: int, image_based_error: float = 0.0, image_shift: List[float] = None):
        """Save captured image with metadata"""
        if image_shift is None:
            image_shift = [0.0, 0.0]
            
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
                'exposure_time': self.params.exposureTime,
                'image_based_error_um': image_based_error,
                'image_shift_pixels_x': image_shift[0],
                'image_shift_pixels_y': image_shift[1],
                'image_shift_um_x': image_shift[0] * self.params.pixelSizeUM,
                'image_shift_um_y': image_shift[1] * self.params.pixelSizeUM,
                'image_registration_enabled': self.params.enableImageBasedError,
                'num_images_per_position': self.params.numImagesPerPosition,
                'pixel_size_um': self.params.pixelSizeUM
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
            while True:
                time.sleep(wait_time)
                if not self.shouldStop:
                    break
    
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
            
            # Calculate image-based statistics if enabled
            if self.params.enableImageBasedError and self.results.imageBasedErrors:
                if _HAS_NUMPY:
                    self.results.averageImageError = float(np.mean(self.results.imageBasedErrors))
                    self.results.maxImageError = float(np.max(self.results.imageBasedErrors))
                else:
                    self.results.averageImageError = sum(self.results.imageBasedErrors) / len(self.results.imageBasedErrors)
                    self.results.maxImageError = max(self.results.imageBasedErrors)
            
            # Save results to JSON file
            self._saveResults()
            
            log_msg = f"Stress test completed. Average position error: {self.results.averagePositionError:.2f} µm, Max position error: {self.results.maxPositionError:.2f} µm"
            if self.params.enableImageBasedError and self.results.imageBasedErrors:
                log_msg += f", Average image error: {self.results.averageImageError:.2f} µm, Max image error: {self.results.maxImageError:.2f} µm"
            
            if _HAS_IMSWITCH:
                self._logger.info(log_msg)
            else:
                print(log_msg)
        
        self.sigStresttestComplete.emit()
    
    def _saveResults(self):
        """Save results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stresstest_results_{timestamp}.json"
            filepath = os.path.join(self.params.outputPath, filename)
            
            # Create comprehensive results dictionary
            summary = {
                'total_positions': len(self.position_errors),
                'average_error_um': self.results.averagePositionError,
                'max_error_um': self.results.maxPositionError,
                'min_error_um': min(self.position_errors) if self.position_errors else 0,
                'std_error_um': self._calculate_std(self.position_errors) if self.position_errors else 0,
                'test_duration_minutes': (len(self.position_errors) * self.params.timeInterval) / 60.0
            }
            
            # Add image-based statistics if available
            if self.params.enableImageBasedError and self.results.imageBasedErrors:
                summary.update({
                    'image_based_enabled': True,
                    'average_image_error_um': self.results.averageImageError,
                    'max_image_error_um': self.results.maxImageError,
                    'min_image_error_um': min(self.results.imageBasedErrors),
                    'std_image_error_um': self._calculate_std(self.results.imageBasedErrors),
                    'num_images_per_position': self.params.numImagesPerPosition,
                    'registration_method': self.params.imageRegistrationMethod,
                    'pixel_size_um': self.params.pixelSizeUM
                })
            else:
                summary['image_based_enabled'] = False
            
            results_dict = {
                'parameters': self.params.dict(),
                'results': self.results.dict(),
                'summary': summary
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            # Save detailed image registration results if available
            if self.params.enableImageBasedError and self.results.imageRegistrationResults:
                self._saveImageRegistrationResults()
                
            if _HAS_IMSWITCH:
                self._logger.info(f"Results saved to: {filename}")
            else:
                print(f"Results saved to: {filename}")
            
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Could not save results: {e}")
            else:
                print(f"Could not save results: {e}")
    
    def _saveImageRegistrationResults(self):
        """Save detailed image registration results to a separate JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stresstest_image_registration_{timestamp}.json"
            filepath = os.path.join(self.params.outputPath, filename)
            
            # Create detailed image registration results
            detailed_results = {
                'test_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_positions': len(self.results.imageRegistrationResults),
                    'num_images_per_position': self.params.numImagesPerPosition,
                    'registration_method': self.params.imageRegistrationMethod,
                    'pixel_size_um': self.params.pixelSizeUM
                },
                'position_results': []
            }
            
            # Add results for each position
            for i, (pos_results, target_pos, actual_pos, timestamp_str) in enumerate(zip(
                self.results.imageRegistrationResults,
                self.results.targetPositions,
                self.results.actualPositions,
                self.results.timestamps
            )):
                position_data = {
                    'position_index': i,
                    'target_position': target_pos,
                    'actual_position': actual_pos,
                    'timestamp': timestamp_str,
                    'image_registration_results': pos_results
                }
                detailed_results['position_results'].append(position_data)
            
            # Calculate aggregate statistics
            if self.results.imageBasedErrors:
                detailed_results['aggregate_statistics'] = {
                    'average_error_um': self.results.averageImageError,
                    'max_error_um': self.results.maxImageError,
                    'min_error_um': min(self.results.imageBasedErrors),
                    'std_error_um': self._calculate_std(self.results.imageBasedErrors),
                    'total_registrations': sum(len(result.get('individual_shifts', [])) for result in self.results.imageRegistrationResults)
                }
            
            with open(filepath, 'w') as f:
                json.dump(detailed_results, f, indent=2)
                
            if _HAS_IMSWITCH:
                self._logger.info(f"Detailed image registration results saved to: {filename}")
            else:
                print(f"Detailed image registration results saved to: {filename}")
                
        except Exception as e:
            if _HAS_IMSWITCH:
                self._logger.error(f"Could not save image registration results: {e}")
            else:
                print(f"Could not save image registration results: {e}")
    
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