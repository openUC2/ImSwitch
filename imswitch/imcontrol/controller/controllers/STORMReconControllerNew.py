"""
Renewed STORM Reconstruction Controller with enhanced microeye integration
and removed widget dependencies.
"""

import numpy as np
import time
import tifffile as tif
import os
import threading
import queue
from datetime import datetime
from typing import Generator, Optional, Dict, Any, Tuple
from pathlib import Path

from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex
from imswitch.imcommon.model import initLogger, dirtools
from ..basecontrollers import LiveUpdatedController
from imswitch.imcommon.model import APIExport
from imswitch.imcontrol.model.storm_models import (
    STORMProcessingParameters, 
    STORMAcquisitionParameters,
    STORMReconstructionResult,
    STORMStatusResponse,
    STORMErrorResponse,
    FittingMethodType,
    FilterType
)

# microEye integration
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


class STORMReconControllerNew(LiveUpdatedController):
    """
    Renewed STORM Reconstruction Controller without widget dependencies.
    
    Provides both synchronous (microeye) and asynchronous (Arkitekt) processing
    capabilities via REST API endpoints.
    """

    # Signals for frontend updates
    sigFrameAcquired = Signal(int)  # frame_number
    sigFrameProcessed = Signal(object)  # STORMReconstructionResult
    sigAcquisitionStateChanged = Signal(bool)  # active
    sigProcessingStateChanged = Signal(bool)  # active
    sigErrorOccurred = Signal(str)  # error_message

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._logger = initLogger(self, tryInheritParent=True)

        # Get detector
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]

        # State management
        self._acquisition_active = False
        self._processing_active = False
        self._current_session_id = None
        self._frame_count = 0
        self._processed_count = 0
        self._total_localizations = 0

        # Parameters
        self._processing_params = STORMProcessingParameters()
        self._acquisition_params = STORMAcquisitionParameters()

        # Processing infrastructure
        self._frame_queue = queue.Queue(maxsize=100)
        self._acquisition_thread = None
        self._processing_thread = None
        self._stop_processing = threading.Event()

        # Data storage
        self._data_directory = None
        self._session_directory = None
        self._last_reconstruction_path = None

        # microEye components
        if isMicroEye:
            self._initializeMicroEye()

        # Arkitekt integration
        if IS_ARKITEKT:
            self._initializeArkitekt()

        self._logger.info("STORMReconControllerNew initialized")

    def _initializeMicroEye(self):
        """Initialize microEye processing components."""
        try:
            # Initialize detection and filtering components
            self.peakDetector = CV_BlobDetector()
            self.preFilter = BandpassFilter()
            
            # Storage for accumulated reconstruction
            self.accumulated_reconstruction = None
            self.all_localizations = []
            
            self._logger.info("MicroEye components initialized")
        except Exception as e:
            self._logger.error(f"Failed to initialize microEye: {e}")
            raise

    def _initializeArkitekt(self):
        """Initialize Arkitekt connection."""
        try:
            self._arkitekt_app = easy("STORM_Service_New", url="http://go.arkitekt.io")
            
            # Register functions
            self._arkitekt_app.register(self.arkitekt_start_acquisition)
            self._arkitekt_app.register(self.arkitekt_stop_acquisition)
            self._arkitekt_app.register(self.arkitekt_get_status)
            self._arkitekt_app.register(self.arkitekt_set_parameters)
            
            self._arkitekt_app.enter()
            self._arkitekt_handle = self._arkitekt_app.run_detached()
            
            self._logger.info("Arkitekt service initialized")
        except Exception as e:
            self._logger.error(f"Failed to initialize Arkitekt: {e}")
            self._arkitekt_app = None

    @APIExport()
    def setSTORMProcessingParameters(self, params: Dict[str, Any]) -> STORMProcessingParameters:
        """
        Set STORM processing parameters using pydantic model.
        
        Args:
            params: Dictionary of parameters to update
            
        Returns:
            Updated processing parameters
        """
        try:
            # Update parameters using pydantic validation
            updated_params = self._processing_params.copy(update=params)
            self._processing_params = updated_params
            
            # Update microEye components if available
            if isMicroEye:
                self._updateMicroEyeParameters()
            
            self._logger.info(f"Processing parameters updated: {params}")
            return self._processing_params
            
        except Exception as e:
            self._logger.error(f"Failed to set processing parameters: {e}")
            raise

    def _updateMicroEyeParameters(self):
        """Update microEye components with current parameters."""
        if not isMicroEye:
            return
            
        # Update detection threshold
        # Note: This would typically update the detector's threshold
        # Implementation depends on microEye API
        
        # Update fitting ROI size
        # Note: This would typically update the fitter's ROI size
        
        self._logger.debug("MicroEye parameters updated")

    @APIExport()
    def startSTORMReconstruction(self, 
                               acquisition_params: Optional[Dict[str, Any]] = None,
                               processing_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start STORM frame reconstruction with given parameters.
        
        Args:
            acquisition_params: Acquisition parameters
            processing_params: Processing parameters
            
        Returns:
            Status dictionary
        """
        if self._acquisition_active:
            return {"success": False, "message": "Acquisition already active"}

        try:
            # Update parameters if provided
            if acquisition_params:
                self._acquisition_params = STORMAcquisitionParameters(**acquisition_params)
            if processing_params:
                self._processing_params = STORMProcessingParameters(**processing_params)

            # Setup session
            session_id = self._acquisition_params.session_id or f"storm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._current_session_id = session_id
            
            # Setup data directory
            self._setupDataDirectory()
            
            # Reset counters
            self._frame_count = 0
            self._processed_count = 0
            self._total_localizations = 0
            
            # Start acquisition and processing
            self._startAcquisition()
            if self._acquisition_params.process_locally and isMicroEye:
                self._startProcessing()
            
            self._acquisition_active = True
            self.sigAcquisitionStateChanged.emit(True)
            
            self._logger.info(f"STORM reconstruction started: {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "message": "STORM reconstruction started",
                "data_directory": str(self._session_directory)
            }
            
        except Exception as e:
            self._logger.error(f"Failed to start STORM reconstruction: {e}")
            return {"success": False, "message": str(e)}

    @APIExport()
    def stopSTORMReconstruction(self) -> Dict[str, Any]:
        """
        Stop STORM frame reconstruction.
        
        Returns:
            Status dictionary with final statistics
        """
        if not self._acquisition_active:
            return {"success": False, "message": "No acquisition active"}

        try:
            # Stop acquisition
            self._stopAcquisition()
            
            # Stop processing
            if self._processing_active:
                self._stopProcessing()
            
            # Save final reconstruction
            final_path = None
            if isMicroEye and self.accumulated_reconstruction is not None:
                final_path = self._saveFinalReconstruction()
            
            session_id = self._current_session_id
            frames_acquired = self._frame_count
            frames_processed = self._processed_count
            total_localizations = self._total_localizations
            
            # Reset state
            self._acquisition_active = False
            self._current_session_id = None
            self.sigAcquisitionStateChanged.emit(False)
            
            self._logger.info(f"STORM reconstruction stopped: {frames_acquired} frames, {total_localizations} localizations")
            
            return {
                "success": True,
                "session_id": session_id,
                "frames_acquired": frames_acquired,
                "frames_processed": frames_processed,
                "total_localizations": total_localizations,
                "final_reconstruction_path": final_path,
                "message": f"STORM reconstruction completed"
            }
            
        except Exception as e:
            self._logger.error(f"Failed to stop STORM reconstruction: {e}")
            return {"success": False, "message": str(e)}

    @APIExport()
    def getSTORMStatus(self) -> STORMStatusResponse:
        """
        Get current STORM reconstruction status.
        
        Returns:
            Current status information
        """
        return STORMStatusResponse(
            acquisition_active=self._acquisition_active,
            session_id=self._current_session_id,
            frames_acquired=self._frame_count,
            processing_active=self._processing_active,
            frames_processed=self._processed_count,
            total_localizations=self._total_localizations,
            microeye_available=isMicroEye,
            arkitekt_available=IS_ARKITEKT and self._arkitekt_app is not None,
            current_processing_params=self._processing_params if self._acquisition_active else None,
            current_acquisition_params=self._acquisition_params if self._acquisition_active else None,
            last_reconstruction_path=self._last_reconstruction_path
        )

    @APIExport()
    def getLastReconstructedImage(self) -> Optional[str]:
        """
        Get the filepath of the last reconstructed image.
        
        Returns:
            Filepath to last reconstructed image or None
        """
        return self._last_reconstruction_path

    @APIExport()
    def triggerSingleFrameReconstruction(self) -> STORMReconstructionResult:
        """
        Trigger reconstruction of a single frame.
        
        Returns:
            Reconstruction result
        """
        if not isMicroEye:
            raise ValueError("MicroEye not available for single frame reconstruction")

        try:
            # Get latest frame from detector
            frame = self.detector.getLatestFrame()
            if frame is None:
                raise ValueError("No frame available from detector")

            # Process frame
            processed_frame, localizations = self._processFrame(frame)
            
            # Save frame
            frame_path = self._saveFrame(frame, self._frame_count, is_processed=False)
            recon_path = self._saveFrame(processed_frame, self._frame_count, is_processed=True)
            
            # Create result
            result = STORMReconstructionResult(
                frame_number=self._frame_count,
                timestamp=datetime.now().isoformat(),
                session_id=self._current_session_id or "single_frame",
                num_localizations=len(localizations) if localizations is not None else 0,
                localization_parameters=localizations.tolist() if localizations is not None else None,
                raw_frame_path=frame_path,
                reconstructed_frame_path=recon_path,
                processing_parameters=self._processing_params,
                acquisition_parameters=self._acquisition_params
            )
            
            self._last_reconstruction_path = recon_path
            self.sigFrameProcessed.emit(result)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Failed to reconstruct single frame: {e}")
            raise

    def _setupDataDirectory(self):
        """Setup data directory structure following experimentcontroller pattern."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = dirtools.UserFileDirs.Data
        
        # Create STORM-specific directory structure
        storm_base = Path(base_path) / "STORMController"
        self._data_directory = storm_base / timestamp
        self._session_directory = self._data_directory / self._current_session_id
        
        # Create directories
        self._session_directory.mkdir(parents=True, exist_ok=True)
        (self._session_directory / "raw_frames").mkdir(exist_ok=True)
        (self._session_directory / "reconstructed_frames").mkdir(exist_ok=True)
        (self._session_directory / "localizations").mkdir(exist_ok=True)
        
        self._logger.info(f"Data directory setup: {self._session_directory}")

    def _startAcquisition(self):
        """Start frame acquisition in background thread."""
        def acquisition_worker():
            self.detector.startAcquisition()
            
            while self._acquisition_active:
                try:
                    frames_chunk = self.detector.getChunk()
                    if frames_chunk is not None:
                        frames = self._normalizeFrameChunk(frames_chunk)
                        
                        for frame in frames:
                            if not self._acquisition_active:
                                break
                                
                            self._frame_count += 1
                            
                            # Add to processing queue if local processing enabled
                            if self._acquisition_params.process_locally and isMicroEye:
                                try:
                                    self._frame_queue.put_nowait((self._frame_count, frame))
                                except queue.Full:
                                    self._logger.warning("Frame queue full, dropping frame")
                            
                            # Save raw frame
                            if self._acquisition_params.save_enabled:
                                self._saveFrame(frame, self._frame_count, is_processed=False)
                            
                            self.sigFrameAcquired.emit(self._frame_count)
                            
                            if self._frame_count % 10 == 0:
                                self._logger.debug(f"Acquired {self._frame_count} frames")
                    
                    time.sleep(0.001)  # Small delay to prevent excessive CPU usage
                    
                except Exception as e:
                    self._logger.error(f"Error in acquisition worker: {e}")
                    self.sigErrorOccurred.emit(str(e))
                    break
                    
            self.detector.stopAcquisition()
            self._logger.info(f"Acquisition worker stopped. Total frames: {self._frame_count}")

        self._acquisition_thread = threading.Thread(target=acquisition_worker, daemon=True)
        self._acquisition_thread.start()

    def _startProcessing(self):
        """Start frame processing in background thread."""
        def processing_worker():
            self._processing_active = True
            self.sigProcessingStateChanged.emit(True)
            
            while not self._stop_processing.is_set():
                try:
                    # Get frame from queue with timeout
                    frame_number, frame = self._frame_queue.get(timeout=1.0)
                    
                    # Process frame
                    processed_frame, localizations = self._processFrame(frame)
                    
                    # Update accumulated reconstruction
                    if processed_frame is not None:
                        if self.accumulated_reconstruction is None:
                            self.accumulated_reconstruction = processed_frame.astype(np.float64)
                        else:
                            self.accumulated_reconstruction += processed_frame.astype(np.float64)
                    
                    # Store localizations
                    if localizations is not None:
                        self.all_localizations.append(localizations)
                        self._total_localizations += len(localizations)
                    
                    # Save processed frame
                    if self._acquisition_params.save_enabled:
                        recon_path = self._saveFrame(processed_frame, frame_number, is_processed=True)
                        self._last_reconstruction_path = recon_path
                    
                    # Create and emit result
                    result = STORMReconstructionResult(
                        frame_number=frame_number,
                        timestamp=datetime.now().isoformat(),
                        session_id=self._current_session_id,
                        num_localizations=len(localizations) if localizations is not None else 0,
                        localization_parameters=localizations.tolist() if localizations is not None else None,
                        raw_frame_path=None,  # Not saved in processing worker
                        reconstructed_frame_path=self._last_reconstruction_path,
                        processing_parameters=self._processing_params,
                        acquisition_parameters=self._acquisition_params
                    )
                    
                    self._processed_count += 1
                    self.sigFrameProcessed.emit(result)
                    
                    if self._processed_count % 10 == 0:
                        self._logger.debug(f"Processed {self._processed_count} frames, {self._total_localizations} localizations")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self._logger.error(f"Error in processing worker: {e}")
                    self.sigErrorOccurred.emit(str(e))
                    
            self._processing_active = False
            self.sigProcessingStateChanged.emit(False)
            self._logger.info(f"Processing worker stopped. Total processed: {self._processed_count}")

        self._stop_processing.clear()
        self._processing_thread = threading.Thread(target=processing_worker, daemon=True)
        self._processing_thread.start()

    def _stopAcquisition(self):
        """Stop frame acquisition."""
        self._acquisition_active = False
        if self._acquisition_thread:
            self._acquisition_thread.join(timeout=5.0)

    def _stopProcessing(self):
        """Stop frame processing."""
        self._stop_processing.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

    def _processFrame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process a single frame with microEye.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, localizations)
        """
        if not isMicroEye:
            return None, None

        try:
            # Apply preprocessing filter
            filtered_frame = frame.copy()
            
            # Localize frame using microEye
            index = 1
            varim = None
            
            # Map fitting method enum to microEye constant
            fitting_method_map = {
                FittingMethodType.PHASOR_2D_CPU: FittingMethod._2D_Phasor_CPU,
                FittingMethodType.GAUSS_MLE_FIXED_SIGMA: FittingMethod._2D_Gauss_MLE_fixed_sigma,
                FittingMethodType.GAUSS_MLE_FREE_SIGMA: FittingMethod._2D_Gauss_MLE_free_sigma,
                FittingMethodType.GAUSS_MLE_ELLIPTICAL_SIGMA: FittingMethod._2D_Gauss_MLE_elliptical_sigma,
                FittingMethodType.GAUSS_MLE_CSPLINE: FittingMethod._3D_Gauss_MLE_cspline_sigma,
            }
            
            method = fitting_method_map.get(
                self._processing_params.fitting_method, 
                FittingMethod._2D_Phasor_CPU
            )
            
            # Perform localization
            frames, params, crlbs, loglike = localize_frame(
                index,
                frame,
                filtered_frame,
                varim,
                self.preFilter,
                self.peakDetector,
                self._processing_params.threshold,
                np.array([1.5]),  # PSF parameter
                self._processing_params.fit_roi_size,
                method
            )
            
            # Create simple reconstruction visualization
            reconstructed_frame = np.zeros_like(frame, dtype=np.float32)
            if params is not None and len(params) > 0:
                try:
                    x_coords = np.clip(params[:, 0].astype(int), 0, frame.shape[1] - 1)
                    y_coords = np.clip(params[:, 1].astype(int), 0, frame.shape[0] - 1)
                    reconstructed_frame[y_coords, x_coords] = 1.0
                except Exception as e:
                    self._logger.warning(f"Error creating reconstruction visualization: {e}")
            
            return reconstructed_frame, params
            
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return None, None

    def _saveFrame(self, frame: np.ndarray, frame_number: int, is_processed: bool = False) -> Optional[str]:
        """
        Save frame to disk.
        
        Args:
            frame: Frame data
            frame_number: Frame number
            is_processed: Whether this is a processed frame
            
        Returns:
            Path to saved frame
        """
        if frame is None or self._session_directory is None:
            return None

        try:
            subdir = "reconstructed_frames" if is_processed else "raw_frames"
            prefix = "recon" if is_processed else "raw"
            
            filename = f"{prefix}_frame_{frame_number:06d}.tif"
            filepath = self._session_directory / subdir / filename
            
            # Convert to appropriate dtype for saving
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame_to_save = (frame * 65535).astype(np.uint16)
            else:
                frame_to_save = frame.astype(np.uint16)
            
            tif.imwrite(str(filepath), frame_to_save)
            
            return str(filepath)
            
        except Exception as e:
            self._logger.error(f"Error saving frame: {e}")
            return None

    def _saveFinalReconstruction(self) -> Optional[str]:
        """Save final accumulated reconstruction."""
        if self.accumulated_reconstruction is None or self._session_directory is None:
            return None

        try:
            filepath = self._session_directory / "final_reconstruction.tif"
            
            # Normalize and convert to uint16
            final_recon = self.accumulated_reconstruction.astype(np.float64)
            if final_recon.max() > 0:
                final_recon = (final_recon / final_recon.max() * 65535).astype(np.uint16)
            else:
                final_recon = final_recon.astype(np.uint16)
            
            tif.imwrite(str(filepath), final_recon)
            
            self._last_reconstruction_path = str(filepath)
            self._logger.info(f"Final reconstruction saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            self._logger.error(f"Error saving final reconstruction: {e}")
            return None

    def _normalizeFrameChunk(self, frames_chunk):
        """Normalize frame chunk format from different camera implementations."""
        if frames_chunk is None:
            return []

        # Handle single frame case
        if len(frames_chunk.shape) == 2:
            return [frames_chunk]

        # Handle multiple frames case
        if len(frames_chunk.shape) == 3:
            h, w = frames_chunk.shape[0], frames_chunk.shape[1]
            z = frames_chunk.shape[2]

            # If third dimension is much smaller, it's likely the buffer dimension
            if z < min(h, w) and z < 50:
                return [frames_chunk[:, :, i] for i in range(z)]
            else:
                return [frames_chunk[i, :, :] for i in range(frames_chunk.shape[0])]

        # Handle cases where chunk is already a list/sequence
        if hasattr(frames_chunk, '__iter__') and not isinstance(frames_chunk, np.ndarray):
            return list(frames_chunk)

        # Default: assume it's (nBuffer, height, width)
        return [frames_chunk[i] for i in range(frames_chunk.shape[0])]

    # Arkitekt remote callable functions
    def arkitekt_start_acquisition(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start STORM acquisition via Arkitekt."""
        return self.startSTORMReconstruction(acquisition_params=params)

    def arkitekt_stop_acquisition(self) -> Dict[str, Any]:
        """Stop STORM acquisition via Arkitekt."""
        return self.stopSTORMReconstruction()

    def arkitekt_get_status(self) -> STORMStatusResponse:
        """Get STORM status via Arkitekt."""
        return self.getSTORMStatus()

    def arkitekt_set_parameters(self, params: Dict[str, Any]) -> STORMProcessingParameters:
        """Set STORM parameters via Arkitekt."""
        return self.setSTORMProcessingParameters(params)

    def __del__(self):
        """Cleanup resources."""
        try:
            if self._acquisition_active:
                self.stopSTORMReconstruction()
                
            if hasattr(self, '_arkitekt_handle') and self._arkitekt_handle is not None:
                self._arkitekt_app.cancel()
                self._arkitekt_app.exit()
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.