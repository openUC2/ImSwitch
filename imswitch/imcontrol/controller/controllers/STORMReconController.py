import numpy as np
import time
import tifffile as tif
import os
from datetime import datetime
import threading
import queue
import zarr
from typing import Generator, Optional, Dict, Any

from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger, dirtools
from ..basecontrollers import LiveUpdatedController

from imswitch.imcommon.model import APIExport

import numpy as np
try:
    import microEye
    isMicroEye = True
    from microEye.Filters import BandpassFilter
    from microEye.fitting.fit import CV_BlobDetector
    from microEye.fitting.results import FittingMethod
    from microEye.fitting.fit import localize_frame

except:
    isMicroEye = False


class STORMReconController(LiveUpdatedController):
    """ Linked to STORMReconWidget."""

    sigImageReceived = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._logger = initLogger(self, tryInheritParent=True)
        
        self.updateRate = 0
        self.it = 0
        self.showPos = False
        self.threshold = 0.2

        # reconstruction related settings
        #TODO: Make parameters adaptable from Plugin
        # Prepare image computation worker
        self.imageComputationWorker = self.STORMReconImageComputationWorker()
        self.imageComputationWorker.sigSTORMReconImageComputed.connect(self.displayImage)

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

        if isMicroEye:
            self.imageComputationThread = Thread()
            self.imageComputationWorker.moveToThread(self.imageComputationThread)
            self.sigImageReceived.connect(self.imageComputationWorker.computeSTORMReconImage)
            self.imageComputationThread.start()

            # Connect CommunicationChannel signals
            self._commChannel.sigUpdateImage.connect(self.update)

            # Connect STORMReconWidget signals
            self._widget.sigShowToggled.connect(self.setShowSTORMRecon)
            self._widget.sigUpdateRateChanged.connect(self.changeRate)
            self._widget.sigSliderValueChanged.connect(self.valueChanged)

            self.changeRate(self.updateRate)
            self.setShowSTORMRecon(False)

            # setup reconstructor
            self.peakDetector = CV_BlobDetector()
            self.preFilter = BandpassFilter()

            self.imageComputationWorker.setDetector(self.peakDetector)
            self.imageComputationWorker.setFilter(self.preFilter)

    def valueChanged(self, magnitude):
        """ Change magnitude. """
        self.dz = magnitude*1e-3
        self.imageComputationWorker.set_dz(self.dz)

    def __del__(self):
        self.imageComputationThread.quit()
        self.imageComputationThread.wait()
        if hasattr(super(), '__del__'):
            super().__del__()

    def setShowSTORMRecon(self, enabled):
        """ Show or hide STORMRecon. """

        # read parameters from GUI for reconstruction the data on the fly
        # Filters + Blob detector params
        filter = self._widget.image_filter.currentData().filter
        tempEnabled = self._widget.tempMedianFilter.enabled.isChecked()
        detector = self._widget.detection_method.currentData().detector
        threshold = self._widget.th_min_slider.value()
        fit_roi_size = self._widget.fit_roi_size.value()
        fitting_method = self._widget.fitting_cbox.currentData()

        # write parameters to worker
        self.imageComputationWorker.setFilter(filter)
        self.imageComputationWorker.setTempEnabled(tempEnabled)
        self.imageComputationWorker.setDetector(detector)
        self.imageComputationWorker.setThreshold(threshold)
        self.imageComputationWorker.setFitRoiSize(fit_roi_size)
        self.imageComputationWorker.setFittingMethod(fitting_method)

        self.active = enabled

        # if it will be deactivated, trigger an image-save operation
        if not self.active:
            self.imageComputationWorker.saveImage()
        else:
            # this will activate/deactivate the live reconstruction
            self.imageComputationWorker.setActive(enabled)


    def update(self, detectorName, im, init, isCurrentDetector):
        """ Update with new detector frame. """
        if not isCurrentDetector or not self.active:
            return

        if self.it == self.updateRate:
            self.it = 0
            self.imageComputationWorker.prepareForNewImage(im)
            self.sigImageReceived.emit()
        else:
            self.it += 1

    def displayImage(self, im):
        """ Displays the image in the view. """
        self._widget.setImage(im)

    def changeRate(self, updateRate):
        """ Change update rate. """
        self.updateRate = updateRate
        self.it = 0

    @APIExport()
    def triggerSTORMReconstruction(self, frame=None):
        """ Trigger reconstruction. """
        if frame is None:
            frame = self.detector.getLatestFrame()
        self.imageComputationWorker.reconSTORMFrame(frame=frame)

    @APIExport(runOnUIThread=False)
    def startFastSTORMAcquisition(self, 
                                  session_id: str = None,
                                  crop_x: int = None, 
                                  crop_y: int = None,
                                  crop_width: int = None, 
                                  crop_height: int = None,
                                  save_path: str = None,
                                  save_format: str = "omezarr",
                                  exposure_time: float = None) -> Dict[str, Any]:
        """
        Start fast STORM frame acquisition with optional cropping and saving.
        
        Parameters:
        - session_id: Unique identifier for this acquisition session
        - crop_x, crop_y: Top-left corner of crop region (None to disable cropping)
        - crop_width, crop_height: Dimensions of crop region
        - save_path: Path to save acquired frames (None to disable saving)
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
        
        # Set cropping parameters
        if crop_x is not None and crop_y is not None and crop_width is not None and crop_height is not None:
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
        
        # Initialize saving if requested
        if save_path is not None:
            self._initializeSaving(save_path, save_format)
        
        # Start acquisition
        self._acquisition_active = True
        self.detector.startAcquisition()
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Fast STORM acquisition started",
            "cropping": self._cropping_params,
            "save_path": save_path,
            "save_format": save_format
        }

    @APIExport(runOnUIThread=False)
    def stopFastSTORMAcquisition(self) -> Dict[str, Any]:
        """
        Stop fast STORM frame acquisition.
        
        Returns:
        - Dictionary with session info and status
        """
        if not self._acquisition_active:
            return {"success": False, "message": "No acquisition active"}
        
        self._acquisition_active = False
        
        # Stop detector acquisition
        if hasattr(self.detector, 'stopAcquisition'):
            self.detector.stopAcquisition()
        
        # Finalize saving
        if self._ome_zarr_store is not None:
            self._finalizeSaving()
        
        session_id = self._current_session_id
        self._current_session_id = None
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Fast STORM acquisition stopped"
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
                # Get latest frame from detector
                frame = self.detector.getLatestFrame()
                
                if frame is not None:
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
                
                # Small delay to prevent excessive CPU usage
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
            "detector_running": getattr(self.detector, '_running', False) if hasattr(self.detector, '_running') else None
        }

    @APIExport()
    def setSTORMParameters(self, 
                          threshold: float = None,
                          roi_size: int = None,
                          update_rate: int = None) -> Dict[str, Any]:
        """
        Set STORM processing parameters.
        
        Parameters:
        - threshold: Detection threshold for localization
        - roi_size: ROI size for fitting
        - update_rate: Update rate for live processing
        
        Returns:
        - Dictionary with current parameter values
        """
        if threshold is not None:
            self.threshold = threshold
            if hasattr(self.imageComputationWorker, 'setThreshold'):
                self.imageComputationWorker.setThreshold(threshold)
        
        if roi_size is not None:
            if hasattr(self.imageComputationWorker, 'setFitRoiSize'):
                self.imageComputationWorker.setFitRoiSize(roi_size)
        
        if update_rate is not None:
            self.updateRate = update_rate
            self.changeRate(update_rate)
        
        return {
            "threshold": self.threshold,
            "roi_size": getattr(self.imageComputationWorker, 'fit_roi_size', None),
            "update_rate": self.updateRate
        }

    def _initializeSaving(self, save_path: str, save_format: str):
        """Initialize saving mechanism based on format."""
        try:
            if save_format.lower() == "omezarr":
                self._initializeOMEZarrSaving(save_path)
            elif save_format.lower() == "tiff":
                self._initializeTiffSaving(save_path)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
        except Exception as e:
            self._logger.error(f"Failed to initialize saving: {e}")
            self._ome_zarr_store = None

    def _initializeOMEZarrSaving(self, save_path: str):
        """Initialize OME-Zarr saving similar to ExperimentController."""
        try:
            # Try to import OME-Zarr dependencies
            from imswitch.imcontrol.controller.controllers.experiment_controller.zarr_data_source import MinimalZarrDataSource
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Initialize OME-Zarr store
            self._ome_zarr_store = MinimalZarrDataSource(save_path, mode="w")
            
            # Get frame shape for configuration
            sample_frame = self.detector.getLatestFrame()
            if sample_frame is not None:
                if self._cropping_params is not None:
                    crop = self._cropping_params
                    shape_y, shape_x = crop['height'], crop['width']
                else:
                    shape_y, shape_x = sample_frame.shape
                
                # Configure metadata for OME-Zarr
                config = {
                    'shape_t': 1000,  # Will be extended as needed
                    'shape_c': 1,
                    'shape_z': 1,
                    'shape_y': shape_y,
                    'shape_x': shape_x,
                    'dtype': sample_frame.dtype
                }
                
                self._ome_zarr_store.set_metadata_from_configuration_experiment(config)
                self._ome_zarr_store.new_position()
                
        except ImportError:
            self._logger.warning("OME-Zarr dependencies not available, falling back to TIFF")
            self._initializeTiffSaving(save_path.replace('.zarr', '.tiff'))
        except Exception as e:
            self._logger.error(f"Failed to initialize OME-Zarr saving: {e}")
            self._ome_zarr_store = None

    def _initializeTiffSaving(self, save_path: str):
        """Initialize TIFF saving as fallback."""
        self._tiff_save_path = save_path
        self._saved_frames = []

    def _saveFrameToZarr(self, frame: np.ndarray, frame_number: int, metadata: dict):
        """Save frame to OME-Zarr store."""
        try:
            if self._ome_zarr_store is not None:
                # Write frame to zarr store
                # This is a simplified implementation - in practice you'd need to handle
                # the proper indexing for time series
                self._ome_zarr_store.write(frame, t=frame_number, c=0, z=0)
        except Exception as e:
            self._logger.error(f"Failed to save frame to Zarr: {e}")

    def _finalizeSaving(self):
        """Finalize and close saving."""
        try:
            if self._ome_zarr_store is not None:
                # Close the store properly
                self._ome_zarr_store = None
            
            if hasattr(self, '_saved_frames') and self._saved_frames:
                # Save accumulated TIFF frames
                if hasattr(self, '_tiff_save_path'):
                    tif.imwrite(self._tiff_save_path, np.stack(self._saved_frames), append=False)
                self._saved_frames = []
                
        except Exception as e:
            self._logger.error(f"Failed to finalize saving: {e}")


    class STORMReconImageComputationWorker(Worker):
        sigSTORMReconImageComputed = Signal(np.ndarray)

        def __init__(self):
            super().__init__()

            self.threshold = 0.2 # default threshold
            self.fit_roi_size = 13 # default roi size

            self._logger = initLogger(self, tryInheritParent=False)
            self._numQueuedImages = 0
            self._numQueuedImagesMutex = Mutex()

            # store the sum of all reconstructed frames
            self.sumReconstruction = None
            self.allParameters = []

            self.active = False


        def reconSTORMFrame(self, frame, preFilter=None, peakDetector=None,
                            rel_threshold=0.4, PSFparam=np.array([1.5]),
                            roiSize=13, method=None):
            # tune parameters
            if method is None: # avoid error when microeye is not installed..
                method = FittingMethod._2D_Phasor_CPU
            if preFilter is None:
                preFilter = self.preFilter
            if peakDetector is None:
                peakDetector = self.peakDetector

            # parameters are read only once the SMLM reconstruction is initiated
            # cannot be altered during recroding
            index = 1
            filtered = frame.copy() # nip.gaussf(frame, 1.5)
            varim = None

            # localize  frame
            # params = > x,y,background, max(0, intensity), magnitudeX / magnitudeY
            frames, params, crlbs, loglike = localize_frame(
                        index,
                        frame,
                        filtered,
                        varim,
                        preFilter,
                        peakDetector,
                        rel_threshold,
                        PSFparam,
                        roiSize,
                        method)

            # create a simple render
            frameLocalized = np.zeros(frame.shape)
            try:
                allX = np.int32(params[:,0])
                allY = np.int32(params[:,1])
                frameLocalized[(allY, allX)] = 1
            except Exception as e:
                pass

            return frameLocalized, params

        def setThreshold(self, threshold):
            self.threshold = threshold

        def setFitRoiSize(self, roiSize):
            self.fit_roi_size = roiSize

        def computeSTORMReconImage(self):
            """ Compute STORMRecon of an image. """
            try:
                if self._numQueuedImages > 1 or not self.active:
                    return  # Skip this frame in order to catch up
                STORMReconrecon, params = self.reconSTORMFrame(frame=self._image,
                                                               preFilter=self.preFilter,
                                                               peakDetector=self.peakDetector,
                                                               rel_threshold=self.threshold,
                                                               roiSize=self.fit_roi_size)
                self.allParameters.append(params)

                if self.sumReconstruction is None:
                    self.sumReconstruction = STORMReconrecon
                else:
                    self.sumReconstruction += STORMReconrecon
                self.sigSTORMReconImageComputed.emit(np.array(self.sumReconstruction))
            finally:
                self._numQueuedImagesMutex.lock()
                self._numQueuedImages -= 1
                self._numQueuedImagesMutex.unlock()

        def prepareForNewImage(self, image):
            """ Must always be called before the worker receives a new image. """
            self._image = image
            self._numQueuedImagesMutex.lock()
            self._numQueuedImages += 1
            self._numQueuedImagesMutex.unlock()

        def setFittingMethod(self, method):
            self.fittingMethod = method
        def setFilter(self, filter):
            self.preFilter = filter

        def setTempEnabled(self, tempEnabled):
            self.tempEnabled = tempEnabled

        def setDetector(self, detector):
            self.peakDetector = detector

        def saveImage(self, filename="STORMRecon", fileExtension="tif"):
            if self.sumReconstruction is None:
                return

            # wait to finish all queued images
            while self._numQueuedImages > 0:
                time.sleep(0.1)

            Ntime = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
            filePath = self.getSaveFilePath(date=Ntime,
                                filename=filename,
                                extension=fileExtension)

            # self.switchOffIllumination()
            self._logger.debug(filePath)
            tif.imwrite(filePath, self.sumReconstruction, append=False)

            # Reset sumReconstruction
            self.sumReconstruction *= 0
            self.allParameters = []

        def getSaveFilePath(self, date, filename, extension):
            mFilename =  f"{date}_{filename}.{extension}"
            dirPath  = os.path.join(dirtools.UserFileDirs.Root, 'recordings', date)

            newPath = os.path.join(dirPath,mFilename)

            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            return newPath

        def setActive(self, enabled):
            self.active = enabled

# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.