import numpy as np
import time
import tifffile as tif
import os
from datetime import datetime
import queue
from typing import Generator, Optional, Dict, Any
from dataclasses import dataclass
import threading
from pathlib import Path

from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex
from imswitch.imcommon.model import initLogger, dirtools
from ..basecontrollers import LiveUpdatedController
from imswitch.imcommon.model import APIExport

# Import pydantic models for parameter management
try:
    from imswitch.imcontrol.model.storm_models import (
        STORMProcessingParameters, 
        STORMAcquisitionParameters,
        STORMReconstructionResult,
        STORMStatusResponse
    )
    HAS_STORM_MODELS = True
except ImportError:
    HAS_STORM_MODELS = False

try:
    from .microEye.Filters import BandpassFilter
    from .microEye.fitting.fit import CV_BlobDetector
    from .microEye.fitting.results import FittingMethod
    from .microEye.fitting.fit import localize_frame
    isMicroEye = True
except ImportError:
    isMicroEye = False

# Arkitekt integration
try:
    from arkitekt_next import easy, startup, state, progress
    from mikro_next.api.schema import (Image, from_array_like, Stage,
                                       create_stage)
    from mikro_next.api.schema import (PartialRGBViewInput, ColorMap,
                                       PartialAffineTransformationViewInput)
    import xarray as xr
    IS_ARKITEKT = True
except ImportError:
    IS_ARKITEKT = False
    # Create dummy decorators and functions when Arkitekt is not available
    Stage = object
    Image = object
    def state(cls):
        return cls

    def startup(func):
        return func

    def progress(value, message=""):
        pass


@state
@dataclass
class STORMState:
    """State management for STORM acquisition via Arkitekt"""
    stage: Optional[Stage] = None
    acquisition_active: bool = False
    session_id: Optional[str] = None
    total_frames_acquired: int = 0
    current_parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.current_parameters is None:
            self.current_parameters = {}


@startup
def init_storm_state(instance_id: str) -> STORMState:
    """Initialize the STORM state for Arkitekt"""
    stage = None
    if IS_ARKITEKT:
        try:stage = create_stage(name=f"STORM_Stage_{instance_id}")
        except Exception as e:
            print(f"Failed to create STORM stage: {e}")
    return STORMState(
        stage=stage,
        acquisition_active=False,
        session_id=None,
        total_frames_acquired=0,
        current_parameters={}
    )


class STORMReconController(LiveUpdatedController):
    """ Linked to STORMReconWidget."""

    sigImageReceived = Signal()
    sigNSTORMImageAcquired = Signal(int)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._logger = initLogger(self, tryInheritParent=True)

        # get the detector
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]

        # API-related properties for async acquisition
        self._acquisition_active = False
        self._frame_queue = queue.Queue()
        self._acquisition_thread = None
        self._cropping_params = None
        self._ome_zarr_store = None
        self._current_session_id = None
        self._direct_saving_mode = False  # True when saving directly without generators
        self._save_format = "omezarr"
        self._saveDirectory = "STORM"
        self._frame_count = 0

        # Initialize Arkitekt integration if available
        self._arkitekt_app = None
        self._arkitekt_handle = None
        if IS_ARKITEKT:
            self._initializeArkitekt()

        # Enhanced microEye integration without widget dependencies
        if isMicroEye:
            self._initializeMicroEyeComponents()
            
        # Initialize processing parameters via pydantic models
        self._initializeProcessingParameters()

    def _initializeMicroEyeComponents(self):
        """Initialize microEye components without widget dependencies."""
        try:
            # Create modern worker without thread-based approach
            self.imageComputationWorker = STORMReconImageComputationWorker()
            
            # Setup reconstructor components
            self.peakDetector = CV_BlobDetector()
            self.preFilter = BandpassFilter()
            
            # Set default parameters
            self.imageComputationWorker.setDetector(self.peakDetector)
            self.imageComputationWorker.setFilter(self.preFilter)
            self.imageComputationWorker.setThreshold(0.2)
            self.imageComputationWorker.setFitRoiSize(13)
            
            # Initialize reconstruction storage
            self._last_reconstruction_path = None
            self._session_directory = None
            
            self._logger.info("MicroEye components initialized without widget dependencies")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize microEye components: {e}")

    def _initializeProcessingParameters(self):
        """Initialize processing parameters using pydantic models."""
        if HAS_STORM_MODELS:
            self._processing_params = STORMProcessingParameters()
            self._acquisition_params = STORMAcquisitionParameters()
        else:
            # Fallback to basic parameters
            self._processing_params = {
                'threshold': 0.2,
                'fit_roi_size': 13,
                'update_rate': 10
            }
            self._acquisition_params = {
                'session_id': None,
                'save_enabled': False
            }
        
        self._logger.debug("Processing parameters initialized")

    def _initializeArkitekt(self):
        """Initialize Arkitekt connection and register functions"""
        try:
            self._logger.debug("Initializing Arkitekt integration for STORM")

            # Create Arkitekt app
            self._arkitekt_app = easy("STORM_Service", url="http://go.arkitekt.io",)

            # Register STORM-specific functions for remote access
            self._arkitekt_app.register(self.arkitekt_start_storm_acquisition)
            self._arkitekt_app.register(self.arkitekt_stop_storm_acquisition)
            self._arkitekt_app.register(self.arkitekt_get_storm_frames)
            self._arkitekt_app.register(self.arkitekt_get_storm_status)
            self._arkitekt_app.register(self.arkitekt_set_storm_parameters)
            self._arkitekt_app.register(self.arkitekt_capture_storm_image)
            self._arkitekt_app.register(self.arkitekt_trigger_reconstruction)

            # Enter the app context
            self._arkitekt_app.enter()

            # Start the app in detached mode
            self._arkitekt_handle = self._arkitekt_app.run_detached()

            self._logger.debug("Arkitekt STORM service started successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize Arkitekt: {e}")
            self._arkitekt_app = None
            self._arkitekt_handle = None

    def __del__(self):
        """Cleanup resources."""
        try:
            if self._acquisition_active:
                self.stopFastSTORMAcquisition()
                
            if hasattr(self, '_arkitekt_handle') and self._arkitekt_handle is not None:
                self._arkitekt_app.cancel()
                self._arkitekt_app.exit()
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")

    # Arkitekt remote callable functions
    def arkitekt_start_storm_acquisition(self,
                                          storm_state: STORMState,
                                          session_id: str = None,
                                          crop_x: int = None,
                                          crop_y: int = None,
                                          crop_width: int = None,
                                          crop_height: int = None,
                                          saveDirectory: str = None,
                                          save_format: str = "tiff",
                                          exposure_time: float = None
                                          ) -> Dict[str, Any]:
        """
        Start STORM acquisition via Arkitekt.

        This function starts fast STORM frame acquisition with optional
        cropping and saving. It can be called remotely via the Arkitekt
        framework.

        Parameters:
        - storm_state: STORMState object for tracking acquisition state
        - session_id: Unique identifier for this acquisition session
        - crop_x, crop_y: Top-left corner of crop region (None to disable)
        - crop_width, crop_height: Dimensions of crop region
        - saveDirectory: Path to save acquired frames (None to disable saving)
        - save_format: Format to save frames ('omezarr', 'tiff')
        - exposure_time: Exposure time for frames (None to use current)

        Returns:
        - Dictionary with session info and status
        """
        progress(10, "Starting STORM acquisition...")

        result = self.startFastSTORMAcquisition(
            session_id=session_id,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            saveDirectory=saveDirectory,
            save_format=save_format,
            exposure_time=exposure_time,
            process_arkitekt=True
        )

        if result.get("success"):
            storm_state.acquisition_active = True
            storm_state.session_id = result.get("session_id")
            storm_state.total_frames_acquired = 0
            progress(100, f"STORM acquisition started: {storm_state.session_id}")
        else:
            progress(0, f"Failed to start: {result.get('message')}")

        return result

    def arkitekt_stop_storm_acquisition(self,
                                         storm_state: STORMState
                                         ) -> Dict[str, Any]:
        """
        Stop STORM acquisition via Arkitekt.

        Parameters:
        - storm_state: STORMState object for tracking acquisition state

        Returns:
        - Dictionary with session info and status
        """
        progress(50, "Stopping STORM acquisition...")

        result = self.stopFastSTORMAcquisition()

        if result.get("success"):
            storm_state.acquisition_active = False
            storm_state.session_id = None
            total_frames = storm_state.total_frames_acquired
            progress(100, f"STORM acquisition stopped. Total frames: {total_frames}")
        else:
            progress(0, f"Failed to stop: {result.get('message')}")

        return result

    def arkitekt_get_storm_frames(self,
                                  storm_state: STORMState,
                                  num_frames: int = 100,
                                  timeout: float = 10.0,
                                  image_name_prefix: str = "storm_frame"
                                  ) -> Generator[Image, None, None]:
        """
        Get STORM frames via Arkitekt as Mikro Images.

        This generator yields acquired STORM frames converted to Mikro Image
        format for integration with the Arkitekt/Mikro ecosystem.

        Parameters:
        - storm_state: STORMState object for tracking acquisition state
        - num_frames: Maximum number of frames to yield
        - timeout: Timeout for waiting for each frame
        - image_name_prefix: Prefix for generated image names

        Yields:
        - Mikro Image objects containing frame data
        """
        frame_count = 0

        for frame_data in self.getSTORMFrameGenerator(num_frames=num_frames,
                                                      timeout=timeout):
            if 'error' in frame_data:
                progress(0, f"Error acquiring frame: {frame_data['error']}")
                break

            frame = frame_data['raw_frame']
            metadata = frame_data['metadata']

            # Update state
            storm_state.total_frames_acquired += 1
            frame_count += 1

            # Convert to RGB if needed for Mikro
            if len(frame.shape) == 2:
                frame_rgb = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
            else:
                frame_rgb = frame

            # Create image name
            image_name = f"{image_name_prefix}_{frame_count:04d}"

            # Create affine transformation for spatial context
            affine_view = PartialAffineTransformationViewInput(
                affineMatrix=[
                    [1.0, 0, 0, 0],  # Could use pixel size if available
                    [0, 1.0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ],
                stage=storm_state.stage,
            )

            # Create RGB views for visualization
            rgb_views = [
                PartialRGBViewInput(
                    cMin=0, cMax=1,
                    contrastLimitMax=float(frame_rgb.max()),
                    contrastLimitMin=float(frame_rgb.min()),
                    colorMap=ColorMap.RED,
                    baseColor=[0, 0, 0]
                ),
                PartialRGBViewInput(
                    cMin=1, cMax=2,
                    contrastLimitMax=float(frame_rgb.max()),
                    contrastLimitMin=float(frame_rgb.min()),
                    colorMap=ColorMap.GREEN,
                    baseColor=[0, 0, 0]
                ),
                PartialRGBViewInput(
                    cMin=2, cMax=3,
                    contrastLimitMax=float(frame_rgb.max()),
                    contrastLimitMin=float(frame_rgb.min()),
                    colorMap=ColorMap.BLUE,
                    baseColor=[0, 0, 0]
                )
            ]

            progress_val = int((frame_count / num_frames) * 100)
            progress(progress_val,
                     f"Processing STORM frame {frame_count}/{num_frames}")

            # Convert to Mikro Image and yield
            yield from_array_like(
                xr.DataArray(frame_rgb, dims=list("yxc")),
                name=image_name,
                rgb_views=rgb_views,
                transformation_views=[affine_view]
            )

    def arkitekt_get_storm_status(self,
                                  storm_state: STORMState
                                  ) -> Dict[str, Any]:
        """
        Get STORM acquisition status via Arkitekt.

        Parameters:
        - storm_state: STORMState object for tracking acquisition state

        Returns:
        - Dictionary with current status information including state info
        """
        base_status = self.getSTORMStatus()

        # Add Arkitekt-specific state information
        base_status.update({
            "arkitekt_session_active": storm_state.acquisition_active,
            "arkitekt_session_id": storm_state.session_id,
            "total_frames_acquired": storm_state.total_frames_acquired,
            "arkitekt_available": IS_ARKITEKT
        })

        return base_status

    def arkitekt_set_storm_parameters(self,
                                      storm_state: STORMState,
                                      threshold: float = None,
                                      roi_size: int = None,
                                      update_rate: int = None
                                      ) -> Dict[str, Any]:
        """
        Set STORM processing parameters via Arkitekt.

        Parameters:
        - storm_state: STORMState object for tracking acquisition state
        - threshold: Detection threshold for localization
        - roi_size: ROI size for fitting
        - update_rate: Update rate for live processing

        Returns:
        - Dictionary with current parameter values
        """
        result = self.setSTORMParameters(
            threshold=threshold,
            roi_size=roi_size,
            update_rate=update_rate
        )

        # Update state parameters
        storm_state.current_parameters.update(result)

        progress(100, f"STORM parameters updated: {result}")

        return result

    def arkitekt_capture_storm_image(self,
                                     storm_state: STORMState,
                                     image_name: str = "storm_capture"
                                     ) -> Image:
        """
        Capture a single STORM image via Arkitekt.

        This function captures a single frame and processes it through the
        STORM reconstruction pipeline, returning a Mikro Image.

        Parameters:
        - storm_state: STORMState object for tracking acquisition state
        - image_name: Name for the captured image

        Returns:
        - Mikro Image object containing the captured and processed frame
        """
        progress(25, "Capturing STORM image...")

        # Trigger reconstruction and get frame
        self.triggerSTORMReconstruction()
        frames_chunk = self.detector.getChunk()

        if frames_chunk is None:
            raise ValueError("No frame available from detector")

        # Use normalized frame processing and take the most recent frame
        frames_list = self._normalizeFrameChunk(frames_chunk)
        if not frames_list:
            raise ValueError("No frames in chunk")
        frame = frames_list[-1]

        progress(50, "Processing frame through STORM pipeline...")

        # Apply cropping if active
        if self._cropping_params is not None:
            crop = self._cropping_params
            frame = frame[crop['y']:crop['y']+crop['height'],
                         crop['x']:crop['x']+crop['width']]

        # Convert to RGB format for Mikro
        if len(frame.shape) == 2:
            frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)

        progress(75, "Converting to Mikro Image...")

        # Create affine transformation
        affine_view = PartialAffineTransformationViewInput(
            affineMatrix=[
                [1.0, 0, 0, 0],
                [0, 1.0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            stage=storm_state.stage,
        )

        # Create RGB views
        rgb_views = [
            PartialRGBViewInput(
                cMin=0, cMax=1,
                contrastLimitMax=float(frame.max()),
                contrastLimitMin=float(frame.min()),
                colorMap=ColorMap.RED,
                baseColor=[0, 0, 0]
            ),
            PartialRGBViewInput(
                cMin=1, cMax=2,
                contrastLimitMax=float(frame.max()),
                contrastLimitMin=float(frame.min()),
                colorMap=ColorMap.GREEN,
                baseColor=[0, 0, 0]
            ),
            PartialRGBViewInput(
                cMin=2, cMax=3,
                contrastLimitMax=float(frame.max()),
                contrastLimitMin=float(frame.min()),
                colorMap=ColorMap.BLUE,
                baseColor=[0, 0, 0]
            )
        ]

        progress(100, "STORM image captured and processed")

        return from_array_like(
            xr.DataArray(frame, dims=list("yxc")),
            name=image_name,
            rgb_views=rgb_views,
            transformation_views=[affine_view]
        )

    def arkitekt_trigger_reconstruction(self,
                                        storm_state: STORMState
                                        ) -> str:
        """
        Trigger STORM reconstruction via Arkitekt.

        Parameters:
        - storm_state: STORMState object for tracking acquisition state

        Returns:
        - Status message
        """
        try:
            self.triggerSTORMReconstruction()
            progress(100, "STORM reconstruction triggered successfully")
            return "STORM reconstruction triggered successfully"
        except Exception as e:
            progress(0, f"Failed to trigger reconstruction: {str(e)}")
            return f"Failed to trigger reconstruction: {str(e)}"

    def __del__(self):
        # Clean up Arkitekt resources
        if self._arkitekt_handle is not None:
            try:
                self._arkitekt_app.cancel()
                self._arkitekt_app.exit()
            except Exception as e:
                self._logger.error(f"Error cleaning up Arkitekt: {e}")

        # Clean up existing resources
        if hasattr(self, 'imageComputationThread'):
            self.imageComputationThread.quit()
            self.imageComputationThread.wait()
        if hasattr(super(), '__del__'):
            super().__del__()

    @APIExport()
    def triggerSTORMReconstruction(self, frame=None):
        """ Trigger reconstruction. """
        if frame is None:
            frames_chunk = self.detector.getChunk()
            if frames_chunk is not None:
                # Use normalized frame processing and take the most recent frame
                frames_list = self._normalizeFrameChunk(frames_chunk)
                frame = frames_list[-1] if frames_list else None
            else:
                frame = None
        self.imageComputationWorker.reconSTORMFrame(frame=frame)

    @APIExport(runOnUIThread=False)
    def startFastSTORMAcquisition(self,
                                  session_id: str = None,
                                  crop_x: int = None,
                                  crop_y: int = None,
                                  crop_width: int = None,
                                  crop_height: int = None,
                                  saveDirectory: str = "STORM",
                                  save_format: str = "tiff",
                                  exposure_time: float = None,
                                  max_frames: int = -1,
                                  process_arkitekt: bool = False) -> Dict[str, Any]:
        """
        Start fast STORM frame acquisition with optional cropping and saving.

        When not using Arkitekt, this will initialize frame saving and start
        continuous acquisition to disk. When using Arkitekt, frames are made
        available via generators.

        Parameters:
        - session_id: Unique identifier for this acquisition session
        - crop_x, crop_y: Top-left corner of crop region (None to disable cropping)
        - crop_width, crop_height: Dimensions of crop region
        - saveDirectory: Path to save acquired frames (None to disable saving)
        - save_format: Format to save frames ('omezarr', 'tiff')
        - exposure_time: Exposure time for frames (None to use current)

        Returns:
        - Dictionary with session info and status
        """
        if self._acquisition_active:
            return {"success": False, "message": "Acquisition already active"}

        # Generate session ID if not provided
        if session_id is None:
            session_id = f"storm_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._current_session_id = session_id
        self._frame_count = 0

        # Set cropping parameters
        crop_params_given = (crop_x is not None and crop_y is not None and
                             crop_width is not None and crop_height is not None)
        if crop_params_given:
            self._cropping_params = {
                'x': crop_x, 'y': crop_y,
                'width': crop_width, 'height': crop_height
            }
            # Apply cropping to detector if supported
            if hasattr(self.detector, 'crop'):
                self.detector.crop(crop_x, crop_y, crop_width, crop_height)
        else:
            self._cropping_params = None

        # Set exposure time if provided
        if exposure_time is not None and hasattr(self.detector, 'setParameter'):
            self.detector.setParameter('ExposureTime', exposure_time)

        # Determine acquisition mode and initialize saving
        if saveDirectory is not None:
            self._saveDirectory = saveDirectory
            self._save_format = save_format
            self._initializeSaving(saveDirectory, save_format)

            # If not using Arkitekt, start direct saving mode
            if (not IS_ARKITEKT or self._arkitekt_app is None) or not process_arkitekt:
                self._direct_saving_mode = True
                self._startDirectSavingAcquisition(max_frames=max_frames)
            else:
                self._direct_saving_mode = False
        else:
            self._direct_saving_mode = False

        # Start acquisition
        self._acquisition_active = True
        if not self._direct_saving_mode:
            self.detector.startAcquisition()

        return {
            "success": True,
            "session_id": session_id,
            "message": "Fast STORM acquisition started" + (" (direct saving mode)" if self._direct_saving_mode else ""),
            "cropping": self._cropping_params,
            "saveDirectory": saveDirectory,
            "save_format": save_format,
            "direct_saving": self._direct_saving_mode
        }

    @APIExport(runOnUIThread=False)
    def stopFastSTORMAcquisition(self) -> Dict[str, Any]:
        """
        Stop fast STORM frame acquisition and finalize saving.

        Returns:
        - Dictionary with session info and status
        """
        if not self._acquisition_active:
            return {"success": False, "message": "No acquisition active"}

        self._acquisition_active = False

        # Stop direct saving acquisition thread if running
        if self._direct_saving_mode and self._acquisition_thread is not None:
            self._acquisition_thread.join(timeout=5.0)  # Wait up to 5 seconds
            self._acquisition_thread = None

        # Stop detector acquisition
        if hasattr(self.detector, 'stopAcquisition'):
            self.detector.stopAcquisition()

        # Finalize saving
        frames_saved = self._frame_count
        if self._ome_zarr_store is not None or hasattr(self, '_tiff_writer'):
            self._finalizeSaving()

        session_id = self._current_session_id
        self._current_session_id = None
        self._direct_saving_mode = False
        self._saveDirectory = None
        self._frame_count = 0

        return {
            "success": True,
            "session_id": session_id,
            "message": f"Fast STORM acquisition stopped. {frames_saved} frames saved.",
            "frames_saved": frames_saved
        }

    @APIExport(runOnUIThread=False)
    def getSTORMFrameGenerator(self,
                               num_frames: int = 100,
                               timeout: float = 10.0) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields acquired STORM frames with metadata.

        Parameters:
        - num_frames: Maximum number of frames to yield
        - timeout: Timeout for waiting for each frame

        Yields:
        - Dictionary containing frame data, timestamp, and metadata
        """
        frames_yielded = 0

        while frames_yielded < num_frames and self._acquisition_active:
            try:
                # Get all frames from detector since last call to avoid losing frames
                frames_chunk = self.detector.getChunk()

                if frames_chunk is not None:
                    # Use normalized frame processing
                    frames_to_process = self._normalizeFrameChunk(frames_chunk)

                    for frame in frames_to_process:
                        if frames_yielded >= num_frames:
                            break

                        # Apply cropping if specified
                        if self._cropping_params is not None:
                            crop = self._cropping_params
                            frame = frame[crop['y']:crop['y']+crop['height'],
                                         crop['x']:crop['x']+crop['width']]

                        # Process with microEye if available and enabled
                        processed_frame = None
                        localization_params = None
                        if isMicroEye and self.active:
                            processed_frame, localization_params = self.imageComputationWorker.reconSTORMFrame(frame)

                        # Create metadata
                        metadata = {
                            'timestamp': datetime.now().isoformat(),
                            'frame_number': frames_yielded,
                            'session_id': self._current_session_id,
                            'original_shape': frame.shape,
                            'cropping_params': self._cropping_params
                        }

                        if localization_params is not None:
                            metadata['num_localizations'] = len(localization_params)

                        # Save frame if saving is enabled
                        if self._ome_zarr_store is not None:
                            self._saveFrameToZarr(frame, frames_yielded, metadata)

                        yield {
                            'raw_frame': frame,
                            'processed_frame': processed_frame,
                            'localization_params': localization_params,
                            'metadata': metadata
                        }

                        frames_yielded += 1

                # Small delay to prevent excessive CPU usage when no frames available
                time.sleep(0.001)

            except Exception as e:
                yield {
                    'error': str(e),
                    'frame_number': frames_yielded,
                    'timestamp': datetime.now().isoformat()
                }
                break

        # Cleanup
        if self._acquisition_active:
            self.stopFastSTORMAcquisition()

    @APIExport()
    def getSTORMStatus(self) -> Dict[str, Any]:
        """
        Get current STORM acquisition status.

        Returns:
        - Dictionary with current status information
        """
        return {
            "acquisition_active": self._acquisition_active,
            "session_id": self._current_session_id,
            "cropping_params": self._cropping_params,
            "microeye_available": isMicroEye,
            "processing_active": self.active if hasattr(self, 'active') else False,
            "detector_running": getattr(self.detector, '_running', False) if hasattr(self.detector, '_running') else None,
            "direct_saving_mode": self._direct_saving_mode,
            "saveDirectory": self._saveDirectory,
            "save_format": self._save_format,
            "frames_saved": self._frame_count,
            "arkitekt_available": IS_ARKITEKT and self._arkitekt_app is not None
        }

    @APIExport()
    def setSTORMProcessingParameters(self, **kwargs) -> Dict[str, Any]:
        """
        Set STORM processing parameters via pydantic model validation.
        
        Args:
            **kwargs: Processing parameters to update
            
        Returns:
            Updated processing parameters
        """
        try:
            if HAS_STORM_MODELS:
                # Update using pydantic validation
                updated_params = self._processing_params.copy(update=kwargs)
                self._processing_params = updated_params
                
                # Update microEye worker if available
                if isMicroEye:
                    self.imageComputationWorker.setThreshold(updated_params.threshold)
                    self.imageComputationWorker.setFitRoiSize(updated_params.fit_roi_size)
                
                result = updated_params.dict()
            else:
                # Fallback to basic parameter handling
                self._processing_params.update(kwargs)
                
                if isMicroEye:
                    if 'threshold' in kwargs:
                        self.imageComputationWorker.setThreshold(kwargs['threshold'])
                    if 'fit_roi_size' in kwargs:
                        self.imageComputationWorker.setFitRoiSize(kwargs['fit_roi_size'])
                
                result = self._processing_params.copy()
            
            self._logger.info(f"Processing parameters updated: {kwargs}")
            return {"success": True, "parameters": result}
            
        except Exception as e:
            self._logger.error(f"Failed to set processing parameters: {e}")
            return {"success": False, "error": str(e)}

    @APIExport()
    def startSTORMReconstructionLocal(self, **kwargs) -> Dict[str, Any]:
        """
        Start local STORM reconstruction with microeye processing.
        
        Args:
            **kwargs: Acquisition and processing parameters
            
        Returns:
            Status dictionary
        """
        if not isMicroEye:
            return {"success": False, "error": "MicroEye not available"}
            
        if self._acquisition_active:
            return {"success": False, "error": "Acquisition already active"}

        try:
            # Update parameters if provided
            if 'processing' in kwargs:
                self.setSTORMProcessingParameters(**kwargs['processing'])
            if 'acquisition' in kwargs:
                self._updateAcquisitionParameters(kwargs['acquisition'])
            
            # Setup session and data storage
            session_id = kwargs.get('session_id', f"storm_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self._current_session_id = session_id
            self._setupLocalDataDirectory()
            
            # Start acquisition with local processing
            result = self.startFastSTORMAcquisition(
                session_id=session_id,
                saveDirectory=str(self._session_directory / "frames"),
                save_format="tiff",
                process_locally=True,
                **kwargs
            )
            
            if result.get("success"):
                # Activate enhanced local processing
                self.imageComputationWorker.setActive(True)
                
                self._logger.info(f"Local STORM reconstruction started: {session_id}")
                return {
                    "success": True,
                    "session_id": session_id,
                    "data_directory": str(self._session_directory),
                    "message": "Local STORM reconstruction started"
                }
            else:
                return result
                
        except Exception as e:
            self._logger.error(f"Failed to start local STORM reconstruction: {e}")
            return {"success": False, "error": str(e)}

    @APIExport()
    def stopSTORMReconstructionLocal(self) -> Dict[str, Any]:
        """
        Stop local STORM reconstruction and save final results.
        
        Returns:
            Status dictionary with final statistics
        """
        if not self._acquisition_active:
            return {"success": False, "error": "No acquisition active"}

        try:
            # Stop acquisition
            result = self.stopFastSTORMAcquisition()
            
            # Finalize local processing
            if isMicroEye:
                self.imageComputationWorker.setActive(False)
                final_path = self.imageComputationWorker.saveImage()
                self._last_reconstruction_path = final_path
            
            session_id = self._current_session_id
            
            self._logger.info(f"Local STORM reconstruction stopped: {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "final_reconstruction_path": self._last_reconstruction_path,
                "data_directory": str(self._session_directory) if self._session_directory else None,
                "message": "Local STORM reconstruction completed"
            }
            
        except Exception as e:
            self._logger.error(f"Failed to stop local STORM reconstruction: {e}")
            return {"success": False, "error": str(e)}

    @APIExport()
    def getLastReconstructedImagePath(self) -> Optional[str]:
        """
        Get the filepath of the last reconstructed STORM image.
        
        Returns:
            Filepath to last reconstructed image (TIF format) or None
        """
        return self._last_reconstruction_path

    @APIExport()
    def getSTORMReconstructionStatus(self) -> Dict[str, Any]:
        """
        Get enhanced STORM reconstruction status including local processing info.
        
        Returns:
            Comprehensive status information
        """
        base_status = self.getSTORMStatus()
        
        # Add local processing specific information
        enhanced_status = {
            **base_status,
            "local_processing_active": isMicroEye and getattr(self.imageComputationWorker, 'active', False) if hasattr(self, 'imageComputationWorker') else False,
            "last_reconstruction_path": self._last_reconstruction_path,
            "session_directory": str(self._session_directory) if self._session_directory else None,
            "processing_parameters": self._processing_params.dict() if HAS_STORM_MODELS else self._processing_params,
            "microeye_worker_available": hasattr(self, 'imageComputationWorker')
        }
        
        return enhanced_status

    def _updateAcquisitionParameters(self, params: Dict[str, Any]):
        """Update acquisition parameters."""
        if HAS_STORM_MODELS:
            self._acquisition_params = self._acquisition_params.copy(update=params)
        else:
            self._acquisition_params.update(params)

    def _setupLocalDataDirectory(self):
        """Setup local data directory following experimentcontroller pattern."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = dirtools.UserFileDirs.Data
        
        # Create STORM-specific directory structure similar to experimentcontroller
        storm_base = Path(base_path) / "STORMController"
        self._data_directory = storm_base / timestamp
        self._session_directory = self._data_directory / self._current_session_id
        
        # Create necessary subdirectories
        self._session_directory.mkdir(parents=True, exist_ok=True)
        (self._session_directory / "raw_frames").mkdir(exist_ok=True)
        (self._session_directory / "reconstructed_frames").mkdir(exist_ok=True)
        (self._session_directory / "localizations").mkdir(exist_ok=True)
        
        self._logger.info(f"Local data directory setup: {self._session_directory}")

    def _enhancedProcessFrame(self, frame: np.ndarray) -> Optional[str]:
        """
        Enhanced frame processing that saves results locally.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Path to saved reconstructed frame or None
        """
        if not isMicroEye or not hasattr(self, 'imageComputationWorker'):
            return None

        try:
            # Process frame with microEye
            reconstructed_frame, localizations = self.imageComputationWorker.reconSTORMFrame(frame)
            
            if reconstructed_frame is not None and self._session_directory:
                # Save reconstructed frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"recon_{timestamp}.tif"
                filepath = self._session_directory / "reconstructed_frames" / filename
                
                # Convert to appropriate format for saving
                if reconstructed_frame.dtype == np.float32 or reconstructed_frame.dtype == np.float64:
                    frame_to_save = (reconstructed_frame * 65535).astype(np.uint16)
                else:
                    frame_to_save = reconstructed_frame.astype(np.uint16)
                
                tif.imwrite(str(filepath), frame_to_save)
                self._last_reconstruction_path = str(filepath)
                
                # Save localizations if available
                if localizations is not None and len(localizations) > 0:
                    loc_filename = f"localizations_{timestamp}.csv"
                    loc_filepath = self._session_directory / "localizations" / loc_filename
                    np.savetxt(str(loc_filepath), localizations, delimiter=',', 
                              header='x,y,background,intensity,magnitude_x,magnitude_y')
                
                return str(filepath)
                
        except Exception as e:
            self._logger.error(f"Error in enhanced frame processing: {e}")
            
        return None

    # Legacy API compatibility - kept for backward compatibility but deprecated
    @APIExport()
    def setSTORMParameters(self,
                          threshold: float = None,
                          roi_size: int = None,
                          update_rate: int = None) -> Dict[str, Any]:
        """
        Legacy method for setting STORM parameters (deprecated).
        Use setSTORMProcessingParameters instead.
        """
        self._logger.warning("setSTORMParameters is deprecated, use setSTORMProcessingParameters instead")
        
        params = {}
        if threshold is not None:
            params['threshold'] = threshold
        if roi_size is not None:
            params['fit_roi_size'] = roi_size
        if update_rate is not None:
            params['update_rate'] = update_rate
            
        return self.setSTORMProcessingParameters(**params)

    def _initializeSaving(self, saveDirectory: str, save_format: str):
        """Initialize saving mechanism based on format."""
        try:
            if save_format.lower() == "omezarr":
                self._initializeOMEZarrSaving(saveDirectory)
            elif save_format.lower() == "tiff":
                self._initializeTiffSaving(saveDirectory)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
        except Exception as e:
            self._logger.error(f"Failed to initialize saving: {e}")
            self._ome_zarr_store = None

    def _startDirectSavingAcquisition(self, max_frames: int = -1):
        """Start direct frame acquisition and saving in background thread.
        max_frames: Maximum number of frames to save (-1 for unlimited)."""
        def acquisition_worker():
            """Background worker for continuous frame acquisition and saving."""
            self._logger.info("Starting direct saving acquisition worker")

            # Start detector acquisition
            self.detector.startAcquisition()

            while self._acquisition_active:
                try:
                    # Get all frames from detector since last call
                    frames_chunk, frame_indices = self.detector.getChunk()

                    if frames_chunk is not None:
                        # Handle different chunk formats from different cameras
                        frames_to_process = self._normalizeFrameChunk(frames_chunk)

                        for frame in frames_to_process:
                            if not self._acquisition_active:
                                break

                            # Apply cropping if specified
                            if self._cropping_params is not None:
                                crop = self._cropping_params
                                frame = frame[crop['y']:crop['y']+crop['height'],
                                             crop['x']:crop['x']+crop['width']]

                            # Create metadata
                            metadata = {
                                'timestamp': datetime.now().isoformat(),
                                'frame_number': self._frame_count,
                                'session_id': self._current_session_id,
                                'original_shape': frame.shape,
                                'cropping_params': self._cropping_params
                            }

                            # Save frame based on format
                            if self._save_format.lower() == "omezarr" and self._ome_zarr_store is not None:
                                self._saveFrameToZarr(frame, self._frame_count, metadata)
                            elif self._save_format.lower() == "tiff":
                                self._saveFrameToTiff(frame, self._frame_count, metadata)
                            else:
                                self._logger.error(f"Unsupported save format: {self._save_format}")
                                self._acquisition_active = False
                                break

                            self._frame_count += 1

                            # Optional: Log progress periodically
                            if self._frame_count % 20 == 0:
                                self.sigNSTORMImageAcquired.emit(self._frame_count)
                                #self._logger.debug(f"Saved {self._frame_count} frames")
                            if max_frames > 0 and self._frame_count >= max_frames:
                                self._logger.info(f"Reached max frames limit: {max_frames}")
                                self._acquisition_active = False
                                break
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.001)

                except Exception as e:
                    self._logger.error(f"Error in acquisition worker: {e}")
                    self._acquisition_active = False
                    break

            self._logger.info(f"Direct saving acquisition worker stopped. Total frames: {self._frame_count}")

        # Start the acquisition worker in a separate thread
        self._acquisition_thread = threading.Thread(target=acquisition_worker)
        self._acquisition_thread.daemon = True
        self._acquisition_thread.start()

    def _normalizeFrameChunk(self, frames_chunk):
        """
        Normalize frame chunk format from different camera implementations.

        Different cameras return different formats:
        - Some return (nBuffer, height, width)
        - Some return (height, width, nBuffer)
        - Some return a list of frames
        - Some return a single frame

        Returns a list of individual frames.
        """
        if frames_chunk is None:
            return []

        # Handle single frame case
        if len(frames_chunk.shape) == 2:
            return [frames_chunk]

        # Handle multiple frames case
        if len(frames_chunk.shape) == 3:
            # Check which dimension is likely the buffer dimension
            h, w = frames_chunk.shape[0], frames_chunk.shape[1]
            z = frames_chunk.shape[2]

            # If third dimension is much smaller, it's likely the buffer dimension
            # Common case: (height, width, nBuffer) where nBuffer is small
            if z < min(h, w) and z < 50:  # Reasonable buffer size limit
                # Shape is (height, width, nBuffer)
                return [frames_chunk[:, :, i] for i in range(z)]
            else:
                # Shape is likely (nBuffer, height, width)
                return [frames_chunk[i, :, :] for i in range(frames_chunk.shape[0])]

        # Handle cases where chunk is already a list/sequence
        if hasattr(frames_chunk, '__iter__') and not isinstance(frames_chunk, np.ndarray):
            return list(frames_chunk)

        # Default: assume it's (nBuffer, height, width)
        return [frames_chunk[i] for i in range(frames_chunk.shape[0])]

    def _initializeOMEZarrSaving(self, saveDirectory: str):
        """Initialize simple OME-Zarr saving for time series without pyramids."""
        try:
            # Use standard data directory structure
            timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drivePath = dirtools.UserFileDirs.Data
            dirPath = os.path.join(drivePath, 'STORMController', timeStamp)
            saveDirectory = os.path.join(dirPath, saveDirectory)

            # Ensure directory exists
            os.makedirs(os.path.dirname(saveDirectory), exist_ok=True)

            # Initialize simplified OME-Zarr store using MinimalZarrDataSource
            from imswitch.imcontrol.controller.controllers.experiment_controller.zarr_data_source import MinimalZarrDataSource
            self._ome_zarr_store = MinimalZarrDataSource(saveDirectory, mode="w")

            # Get frame shape for configuration
            sample_chunk = self.detector.getChunk()
            if sample_chunk is not None:
                sample_frames = self._normalizeFrameChunk(sample_chunk)
                if sample_frames:
                    sample_frame = sample_frames[0]
                    if self._cropping_params is not None:
                        crop = self._cropping_params
                        shape_y, shape_x = crop['height'], crop['width']
                    else:
                        shape_y, shape_x = sample_frame.shape

                    # Configure metadata for simple time-series OME-Zarr
                    config = {
                        'shape_t': 10000,  # Large number, will be trimmed later
                        'shape_c': 1,
                        'shape_z': 1,
                        'shape_y': shape_y,
                        'shape_x': shape_x,
                        'dtype': sample_frame.dtype,
                        "experiment": {
                            "MicroscopeState": {
                                "number_z_steps": 1,
                                "timepoints": 10000,
                                "channels": {
                                    "channel_1": {
                                        "is_selected": True,
                                        "name": "STORM Channel",
                                        "color": "#FF0000"
                                    }
                                },
                                "microscope_name": "storm_detector",
                                "stack_cycling_mode": "per_stack"
                            },
                            "CameraParameters": {
                                "storm_detector": {
                                    "x_pixels": shape_x,
                                    "y_pixels": shape_y
                                }
                            }
                        }
                    }
                    self._ome_zarr_store.set_metadata_from_configuration_experiment(config)
                    self._ome_zarr_store.new_position(pos_index=0, pos_name="STORM Position", pos_x=0, pos_y=0, pos_z=0)

        except ImportError:
            self._logger.warning("OME-Zarr dependencies not available, falling back to TIFF")
            self._initializeTiffSaving(saveDirectory.replace('.zarr', '.tiff'))
        except Exception as e:
            self._logger.error(f"Failed to initialize OME-Zarr saving: {e}")
            self._ome_zarr_store = None

    def _initializeTiffSaving(self, saveDirectory: str):
        """Initialize single TIFF saving following ExperimentController pattern."""
        # Use standard data directory structure
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        drivePath = dirtools.UserFileDirs.Data
        dirPath = os.path.join(drivePath, 'STORMController', timeStamp)

        # Ensure directory exists
        os.makedirs(dirPath, exist_ok=True)

        # Create single TIFF file path
        self._tiff_save_path = os.path.join(dirPath, f"{saveDirectory}_{timeStamp}.tiff")
        self._tiff_writer = None
        self._logger.info(f"TIFF saving initialized: {self._tiff_save_path}")

    def _saveFrameToZarr(self, frame: np.ndarray, frame_number: int, metadata: dict):
        """Save frame to OME-Zarr store."""
        try:
            if self._ome_zarr_store is not None:
                # Write frame to zarr store using the simple time-series approach
                self._ome_zarr_store.write(frame, ti=frame_number, ci=0)
        except Exception as e:
            self._logger.error(f"Failed to save frame to Zarr: {e}")

    def _saveFrameToTiff(self, frame: np.ndarray, frame_number: int, metadata: dict):
        """Save frame to single TIFF file using append mode."""
        try:
            # Initialize writer on first frame
            if self._tiff_writer is None:
                # Use tifffile to create a single multi-page TIFF
                self._tiff_writer = tif.TiffWriter(self._tiff_save_path, bigtiff=True)

            # Write frame to TIFF with metadata
            self._tiff_writer.write(
                frame.astype(np.uint16),
                metadata={'FrameNumber': frame_number,
                         'Timestamp': metadata.get('timestamp', ''),
                         'SessionID': metadata.get('session_id', '')},
                contiguous=True
            )

        except Exception as e:
            self._logger.error(f"Failed to save frame to TIFF: {e}")

    def _finalizeSaving(self):
        """Finalize and close saving."""
        try:
            # Close OME-Zarr store properly
            if self._ome_zarr_store is not None:
                self._ome_zarr_store.close()
                self._ome_zarr_store = None

            # Close single TIFF writer properly
            if hasattr(self, '_tiff_writer') and self._tiff_writer is not None:
                self._tiff_writer.close()
                self._tiff_writer = None
                self._logger.info(f"TIFF file finalized: {self._tiff_save_path}")

        except Exception as e:
            self._logger.error(f"Failed to finalize saving: {e}")

    # Enhanced Worker Class for microEye processing without Qt threading
    class STORMReconImageComputationWorker:
        """
        Modern STORM reconstruction worker without Qt dependencies.
        Handles frame processing, localization, and result storage.
        """

        def __init__(self):
            self._logger = initLogger(self, tryInheritParent=False)
            
            # Processing parameters
            self.threshold = 0.2
            self.fit_roi_size = 13
            self.active = False
            
            # Storage for reconstruction results
            self.sumReconstruction = None
            self.allParameters = []
            
            # Processing components (will be set externally)
            self.preFilter = None
            self.peakDetector = None
            self.fittingMethod = None
            self.tempEnabled = False

        def reconSTORMFrame(self, frame, preFilter=None, peakDetector=None,
                            rel_threshold=0.4, PSFparam=np.array([1.5]),
                            roiSize=13, method=None):
            """Reconstruct STORM frame with localization."""
            if not isMicroEye:
                return frame, None
                
            # Use provided or default components
            if method is None:
                method = FittingMethod._2D_Phasor_CPU if 'FittingMethod' in globals() else None
            if preFilter is None:
                preFilter = self.preFilter
            if peakDetector is None:
                peakDetector = self.peakDetector

            try:
                index = 1
                filtered = frame.copy()
                varim = None

                # Perform localization
                frames, params, crlbs, loglike = localize_frame(
                    index, frame, filtered, varim,
                    preFilter, peakDetector, rel_threshold,
                    PSFparam, roiSize, method
                )

                # Create simple reconstruction visualization
                frameLocalized = np.zeros_like(frame, dtype=np.float32)
                if params is not None and len(params) > 0:
                    try:
                        allX = np.clip(params[:, 0].astype(int), 0, frame.shape[1] - 1)
                        allY = np.clip(params[:, 1].astype(int), 0, frame.shape[0] - 1)
                        frameLocalized[allY, allX] = 1.0
                    except Exception as e:
                        self._logger.warning(f"Error creating localization visualization: {e}")

                return frameLocalized, params
                
            except Exception as e:
                self._logger.error(f"Error in STORM frame reconstruction: {e}")
                return frame, None

        def processFrame(self, frame):
            """Process a single frame (modern interface)."""
            if not self.active:
                return None, None
                
            return self.reconSTORMFrame(
                frame=frame,
                preFilter=self.preFilter,
                peakDetector=self.peakDetector,
                rel_threshold=self.threshold,
                roiSize=self.fit_roi_size
            )

        def setThreshold(self, threshold):
            """Set detection threshold."""
            self.threshold = threshold

        def setFitRoiSize(self, roiSize):
            """Set fitting ROI size."""
            self.fit_roi_size = roiSize

        def setFittingMethod(self, method):
            """Set fitting method."""
            self.fittingMethod = method
            
        def setFilter(self, filter):
            """Set preprocessing filter."""
            self.preFilter = filter

        def setTempEnabled(self, tempEnabled):
            """Set temporal filtering enabled."""
            self.tempEnabled = tempEnabled

        def setDetector(self, detector):
            """Set peak detector."""
            self.peakDetector = detector

        def setActive(self, enabled):
            """Set processing active state."""
            self.active = enabled

        def saveImage(self, filename="STORMRecon", fileExtension="tif"):
            """Save accumulated reconstruction to file."""
            if self.sumReconstruction is None:
                return None

            try:
                timestamp = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
                filepath = self.getSaveFilePath(
                    date=timestamp,
                    filename=filename,
                    extension=fileExtension
                )

                # Convert to appropriate format for saving
                if self.sumReconstruction.dtype == np.float32 or self.sumReconstruction.dtype == np.float64:
                    image_to_save = (self.sumReconstruction / self.sumReconstruction.max() * 65535).astype(np.uint16)
                else:
                    image_to_save = self.sumReconstruction.astype(np.uint16)

                tif.imwrite(filepath, image_to_save, append=False)
                self._logger.info(f"STORM reconstruction saved: {filepath}")

                # Reset reconstruction
                self.sumReconstruction = None
                self.allParameters = []

                return filepath
                
            except Exception as e:
                self._logger.error(f"Error saving STORM reconstruction: {e}")
                return None

        def getSaveFilePath(self, date, filename, extension):
            """Get save file path following data directory structure."""
            mFilename = f"{date}_{filename}.{extension}"
            dirPath = os.path.join(dirtools.UserFileDirs.Data, 'STORMController', 'reconstructions', date)

            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            return os.path.join(dirPath, mFilename)

# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
