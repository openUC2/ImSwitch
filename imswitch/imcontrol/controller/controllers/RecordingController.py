"""
Recording Controller - API Interface for recording operations.

This controller provides the API interface (via APIExport) for all recording
operations. All operations are delegated to RecordingService from the io module.

The legacy RecordingManager has been removed - all recording functionality
is now provided by the centralized io/writers ecosystem.
"""

import os
import time
from typing import Optional, Union, List, Dict, Any
import numpy as np
import datetime
from fastapi.responses import StreamingResponse
from fastapi import Response, HTTPException
from PIL import Image
import io as python_io
import queue  # thread-safe queue for streamer
from imswitch.imcommon.framework import Timer
from imswitch.imcommon.model import ostools, APIExport, initLogger, dirtools
from imswitch.imcontrol.model import RecMode
from imswitch.imcontrol.model.io import (
    RecordingService, 
    SaveFormat,
    SaveMode,
    get_recording_service,
    StreamingDataStoreAdapter,
    SnapResult,
)
from ..basecontrollers import ImConWidgetController


class RecordingController(ImConWidgetController):
    """
    Recording Controller - API interface for snap and recording operations.
    
    All operations are delegated to RecordingService from the io module.
    This controller serves as a thin API layer for the centralized io/writers ecosystem.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        # Define a dictionary to store variables accessible to the function
        self.shared_variables: dict[str, any] = {}

        self.settingAttr = False
        self.recording = False
        self.doneScan = False
        self.endedRecording = False


        self.streamstarted = False
        
        # Recording service (lazy initialization)
        self._recording_service: Optional[RecordingService] = None
        
        # Streaming adapter for continuous recordings (lazy initialization)
        self._streaming_adapter: Optional[StreamingDataStoreAdapter] = None

        # Connect CommunicationChannel signals
        '''
        self._commChannel.sigRecordingStarted.connect(self.recordingStarted)
        self._commChannel.sigRecordingEnded.connect(self.recordingEnded)
        self._commChannel.sigScanDone.connect(self.scanDone)
        self._commChannel.sigUpdateRecFrameNum.connect(self.updateRecFrameNum)
        self._commChannel.sigUpdateRecTime.connect(self.updateRecTime)
        self._commChannel.sigSnapImg.connect(self.snap)
        self._commChannel.sigSnapImgPrev.connect(self.snapImagePrev)
        self._commChannel.sigStartRecordingExternal.connect(self.startRecording)
        self._commChannel.sharedAttrs.sigAttributeSet.connect(self.attrChanged)
        '''
    
    @property
    def recording_service(self) -> RecordingService:
        """Get or create the recording service (lazy initialization)."""
        if self._recording_service is None:
            self._recording_service = get_recording_service()
            self._recording_service.set_detectors_manager(self._master.detectorsManager)
            if hasattr(self._master, 'metadataHub') and self._master.metadataHub is not None:
                self._recording_service.set_metadata_hub(self._master.metadataHub)
            # Set comm_channel for frame callbacks
            self._recording_service.set_comm_channel(self._commChannel)
        return self._recording_service

    def start_streaming_recording(self, folder: str, detector_names: List[str] = None,
                                   n_time_points: int = 1000, 
                                   write_zarr: bool = True, 
                                   write_tiff: bool = False) -> bool:
        """
        Start a streaming recording session using StreamingDataStoreAdapter.
        
        This is the modern approach for continuous recordings with OME-Zarr support.
        Uses automatic frame capture via signal callback for continuous recording.
        
        Args:
            folder: Base path for recording data
            detector_names: List of detectors to record (None = all)
            n_time_points: Expected number of frames
            write_zarr: Write OME-Zarr format
            write_tiff: Write OME-TIFF format
            
        Returns:
            True if session started successfully
        """
        if self._streaming_adapter is not None and self._streaming_adapter._is_open:
            self.__logger.warning("Streaming session already active")
            return False
        
        try:
            metadata_hub = getattr(self._master, 'metadataHub', None)
            
            self._streaming_adapter = StreamingDataStoreAdapter(
                base_path=folder,
                detectors_manager=self._master.detectorsManager,
                metadata_hub=metadata_hub,
                write_zarr=write_zarr,
                write_tiff=write_tiff,
                n_time_points=n_time_points,
            )
            
            self._streaming_adapter.open(detector_names)
            
            # Configure recording service for automatic frame capture
            self.recording_service.set_streaming_adapter(self._streaming_adapter)
            self.recording_service._streaming_start_time = time.time()
            self.recording_service._streaming_detector_names = detector_names or []
            self.recording_service._connect_frame_callback()
            
            self.__logger.info(f"Streaming recording started with auto-capture: {folder}")
            return True
        except Exception as e:
            self.__logger.error(f"Failed to start streaming recording: {e}")
            self._streaming_adapter = None
            return False
    
    
    def stop_streaming_recording(self) -> Dict[str, Any]:
        """
        Stop the active streaming recording session.
        
        Returns:
            Statistics about the recording session
        """
        if self._streaming_adapter is None:
            return {}
        
        try:
            # Disconnect frame callback first
            self.recording_service._disconnect_frame_callback()
            self.recording_service._streaming_adapter = None
            self.recording_service._streaming_start_time = None
            self.recording_service._streaming_detector_names = []
            
            stats = self._streaming_adapter.get_statistics()
            self._streaming_adapter.close()
            self.__logger.info(f"Streaming recording stopped. Stats: {stats}")
        except Exception as e:
            self.__logger.error(f"Error stopping streaming recording: {e}")
            stats = {}
        finally:
            self._streaming_adapter = None
        
        return stats

    def snap(self, name=None, mSaveFormat=None) -> dict:
        """
        Take a snap and save it to a file using RecordingService.
        
        This method delegates to RecordingService from the io module for
        all snap operations.
        """
        self.updateRecAttrs(isSnapping=True)

        # by default
        if mSaveFormat is None:
                mSaveFormat = SaveFormat.TIFF
        else:
                # Convert integer to SaveFormat enum (from API call)
                mSaveFormat = SaveFormat(mSaveFormat)

        timeStampDay = datetime.datetime.now().strftime("%Y_%m_%d")
        relativeFolder = os.path.join("recordings", timeStampDay)
        folder = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), relativeFolder)
        if not os.path.exists(folder):
            os.makedirs(folder)
            time.sleep(0.01)

        detectorNames = self.getDetectorNamesToCapture()
        if name is None:
            name = "_snap"
        savename = os.path.join(folder, self.getFileName() + "_" + name)

        # Collect metadata attributes
        attrs = {
            detectorName: self._get_detector_attrs(detectorName)
            for detectorName in detectorNames
        }


        # Use RecordingService for snap operations
        result = self.recording_service.snap(
            detector_names=detectorNames,
            savepath=savename,
            save_mode=SaveMode.Disk,
            format=mSaveFormat,
            attrs=attrs,
            async_write=False,  # Synchronous write to ensure file is saved immediately
        )
        self.__logger.debug(f"Snap completed: {result}")
        
        return {"fullPath": savename, "relativePath": relativeFolder}

    def snapNumpy(self) -> Dict[str, np.ndarray]:
        """
        Take a snap and return numpy arrays.
        
        This method delegates to RecordingService.snap_numpy() for
        memory-only snap operations.
        """
        self.updateRecAttrs(isSnapping=True)
        detectorNames = self.getDetectorNamesToCapture()
        
        # Collect metadata attributes (same as snap())
        attrs = {
            detectorName: self._get_detector_attrs(detectorName)
            for detectorName in detectorNames
        }
        
        return self.recording_service.snap_numpy(
            detector_names=detectorNames,
            attrs=attrs
        )

    def _start_frame_recording(self, save_format: SaveFormat):
        """Start recording for a specific number of frames."""
        if save_format == SaveFormat.MP4:
            self.recording_service.start_video_recording(
                filepath=f"{self.savename}.mp4",
                fps=30.0
            )
        else:
            # Use streaming for TIFF/other formats
            self.start_streaming_recording(
                folder=self.savename,
                detector_names=self.recordingArgs["detectorNames"],
                n_time_points=self.recordingArgs.get("recFrames", 1000),
                write_zarr=True,
                write_tiff=(save_format == SaveFormat.TIFF),
            )
        self._commChannel.sigRecordingStarted.emit()
    
    
    def _start_continuous_recording(self, save_format: SaveFormat):
        """Start continuous recording until stopped."""
        if save_format == SaveFormat.MP4:
            self.recording_service.start_video_recording(
                filepath=f"{self.savename}.mp4",
                fps=30.0
            )
        elif save_format == SaveFormat.TIFF:
            self.start_streaming_recording(
                folder=self.savename,
                detector_names=self.recordingArgs["detectorNames"],
                n_time_points=10000,  # Large buffer for continuous recording
                write_zarr=False,
                write_tiff=True,
            )
        elif save_format == SaveFormat.ZARR:
            self.start_streaming_recording(
                folder=self.savename,
                detector_names=self.recordingArgs["detectorNames"],
                n_time_points=10000,
                write_zarr=True,
                write_tiff=False,
            )
        else:
            self.__logger.warning(f"Record format {save_format} not yet implemented, using TIFF")
            self.start_streaming_recording(
                folder=self.savename,
                detector_names=self.recordingArgs["detectorNames"],
                n_time_points=10000,
                write_zarr=False,
                write_tiff=True,
            )
        self._commChannel.sigRecordingStarted.emit()
    
    def _stop_recording(self):
        """Stop any active recording."""
        if self.recording_service.is_video_recording:
            self.recording_service.stop_video_recording()
        if self._streaming_adapter is not None:
            self.stop_streaming_recording()
        self._commChannel.sigRecordingEnded.emit()


    def recordingStarted(self):
        pass

    def recordingEnded(self):
        self.endedRecording = True

    def getDetectorNamesToCapture(self, detectorMode = -1 ):
        """Returns a list of which detectors the user has selected to be captured."""
        # TODO: Later wfor multicamera modify this! 
        if detectorMode == -1:  # Current detector at start
            return [self._master.detectorsManager.getCurrentDetectorName()]
        elif detectorMode == -2:  # All acquisition detectors
            return list(
                self._master.detectorsManager.execOnAll(
                    lambda c: c.name, condition=lambda c: c.forAcquisition
                ).values()
            )
        elif detectorMode == -3:  # A specific detector
            pass 
            #return self._widget.getSelectedSpecificDetectors()

    def getFileName(self):
        """Gets the filename of the data to save."""
        filename = time.strftime("%Hh%Mm%Ss")
        return filename

    def _get_detector_attrs(self, detector_name):
        """
        Get attributes for a detector, combining SharedAttrs and MetadataHub.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            Dictionary of attributes for the detector
        """
        # Start with SharedAttrs (backwards compatible)
        attrs = self._commChannel.sharedAttrs.getSharedAttributes() # TODO: HDF5 doesn't exist anymore
        
        # Add MetadataHub snapshot if available
        if hasattr(self._master, 'metadataHub') and self._master.metadataHub is not None:
            try:
                import json
                
                # Get global metadata snapshot
                global_snapshot = self._master.metadataHub.snapshot_global()
                
                # Get detector-specific snapshot
                detector_snapshot = self._master.metadataHub.snapshot_detector(detector_name)
                
                # Serialize and add to attrs
                if global_snapshot:
                    attrs['_metadata_hub_global'] = json.dumps(global_snapshot, default=str)
                
                if detector_snapshot:
                    # Add detector context
                    if 'detector_context' in detector_snapshot:
                        ctx = detector_snapshot['detector_context']
                        attrs[f'{detector_name}:pixel_size_um'] = ctx.get('pixel_size_um')
                        attrs[f'{detector_name}:shape_px'] = json.dumps(ctx.get('shape_px'))
                        attrs[f'{detector_name}:fov_um'] = json.dumps(ctx.get('fov_um'))
                        if ctx.get('exposure_ms') is not None:
                            attrs[f'{detector_name}:exposure_ms'] = ctx.get('exposure_ms')
                        if ctx.get('gain') is not None:
                            attrs[f'{detector_name}:gain'] = ctx.get('gain')
                    
                    # Add detector-specific metadata
                    if 'metadata' in detector_snapshot:
                        for key, value_dict in detector_snapshot['metadata'].items():
                            attrs[f'{detector_name}:hub:{key}'] = value_dict.get('value')
            except Exception as e:
                self.__logger.warning(f"Error getting metadata hub snapshot: {e}")
        
        return attrs
    
    def attrChanged(self, key, value):
        pass # TODO: Not needed anymore 
    
    def setSharedAttr(self, attr, value):
        self.settingAttr = True
        try:
            self._commChannel.sharedAttrs[(_attrCategory, attr)] = value
        finally:
            self.settingAttr = False

    def updateRecAttrs(self, *, isSnapping):
        self.setSharedAttr(_framesAttr, "null")
        self.setSharedAttr(_timeAttr, "null")

        if isSnapping:
            self.setSharedAttr(_recModeAttr, "Snap")
        else:
            self.setSharedAttr(_recModeAttr, self.recMode.name)


    @APIExport(runOnUIThread=True)
    def snapImageToPath(self, fileName: Optional[str] = None, saveFormat: int = SaveFormat.TIFF) -> dict:
        """Take a snap and save it to a file at the given fileName.

        Parameters:
        - fileName: Optional suffix or filename to append to the generated name. If None,
            the default naming used by `snap()` will be applied.
        - saveFormat: Desired `SaveFormat` enum value (default: `SaveFormat.TIFF`).
        """
        '''
        numpy_array = list(self.snapNumpy().values())[0]
        deconvoled_image = self._master.arkitekt_controller.upload_and_deconvolve_image(
            numpy_array
        )
        print(deconvoled_image)
        '''
        # do nothing here
        return self.snap(name=fileName, mSaveFormat=saveFormat)

    @APIExport(runOnUIThread=False)
    def snapImage(self, output: bool = False, toList: bool = True) -> Union[None, list]:
        """
        Take a snap and save it to a .tiff file at the set file path.
        output: if True, return the numpy array of the image as a list if toList is True, or as a numpy array if toList is False
        toList: if True, return the numpy array of the image as a list, otherwise return it as a numpy array
        """
        if output:
            numpy_array_list = self.snapNumpy()
            mDetector = list(numpy_array_list.keys())[0]
            numpy_array = numpy_array_list[mDetector]
            if toList:
                return numpy_array.tolist()  # Convert the numpy array to a list
            else:
                return np.array(numpy_array)
        else:
            self.snap()

    @APIExport(runOnUIThread=False)
    def snapNumpyToFastAPI(
        self, detectorName: str = None, resizeFactor: float = 1
    ) -> Response:
        """
        Taking a snap and return it as a FastAPI Response object.
        detectorName: the name of the detector to take the snap from. If None, take the snap from the first detector.
        resizeFactor: the factor by which to resize the image. If <1, the image will be downscaled, if >1, nothing will happen.
        """
        # Create a 2D NumPy array representing the image
        images = self.snapNumpy()

        if detectorName == "ALL":
            # Capture images for all detectors and put them into a large array
            detectorNames = self.getDetectorNamesToCapture()
            images = {
                detectorName: images[detectorName] for detectorName in detectorNames
            }

            # Determine the maximum height and total width for the stitched image
            max_height = max(img.shape[0] for img in images.values())
            total_width = sum(img.shape[1] for img in images.values())

            # Check if images are RGB or grayscale
            is_rgb = len(next(iter(images.values())).shape) == 3

            # Create an empty array for the stitched image
            if is_rgb:
                image = np.zeros(
                    (max_height, total_width, 3),
                    dtype=next(iter(images.values())).dtype,
                )
            else:
                image = np.zeros(
                    (max_height, total_width), dtype=next(iter(images.values())).dtype
                )

            # Stitch images together
            current_x = 0
            for detectorName in detectorNames:
                img = images[detectorName]
                height, width = img.shape[:2]

                if is_rgb and len(img.shape) == 2:
                    # Convert grayscale to RGB if needed using numpy
                    img = np.stack([img, img, img], axis=-1)

                image[:height, current_x : current_x + width] = img
                current_x += width

            # Resize the image if needed to save bandwidth
            if resizeFactor < 1:
                image = self.resizeImage(image, resizeFactor)
        # get the image from the first detector if detectorName is not specified
        else:
            if detectorName is None:
                detectorName = self.getDetectorNamesToCapture()[0]
            # get the image from the specified detector
            image = images[detectorName]

            # eventually resize image to save bandwidth
            if resizeFactor < 1:
                image = self.resizeImage(image, resizeFactor)

        # using an in-memory image
        im = Image.fromarray(image)

        # save image to an in-memory bytes buffer
        with python_io.BytesIO() as buf:
            im = im.convert("L")  # convert image to 'L' mode
            im.save(buf, format="PNG")
            im_bytes = buf.getvalue()

        headers = {"Content-Disposition": 'inline; filename="test.png"'}
        return Response(im_bytes, headers=headers, media_type="image/png")

    @APIExport(runOnUIThread=True)
    def startRecording(self, mSaveFormat: int = SaveFormat.TIFF) -> None:
        """Starts recording with the set settings to the set file path using RecordingService.""" 
        mSaveFormat = SaveFormat(mSaveFormat)

        # we probably call from the FASTAPI server
        if self.recording:  # Already recording
            return

        timeStamp = datetime.datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
        folder = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), "recordings", timeStamp)
        if not os.path.exists(folder):
            os.makedirs(folder)
        time.sleep(0.01)
        self.savename = os.path.join(folder, self.getFileName()) + "_rec"

        detectorsBeingCaptured = self.getDetectorNamesToCapture()
        self.recMode = RecMode.UntilStop
        self.recordingArgs = {
            "detectorNames": detectorsBeingCaptured,
            "recMode": self.recMode,
            "savename": self.savename,
            "saveMode": SaveMode.Disk,
            "saveFormat": mSaveFormat,
            "attrs": {
                detectorName: self._commChannel.sharedAttrs.getSharedAttributes()
                for detectorName in detectorsBeingCaptured
            },
        }
        # Use RecordingService for recording
        self._start_continuous_recording(mSaveFormat)
        self.recording = True
        self.endedRecording = False

    @APIExport(runOnUIThread=True)
    def stopRecording(self) -> None:
        """Stops recording using RecordingService."""
        self.recording = False
        self.endedRecording = True
        self._stop_recording()

    @APIExport(runOnUIThread=False)
    def isRecording(self) -> bool:
        """
        Check if any recording is currently in progress.
        
        Returns:
            True if video or streaming recording is active
        """
        return self.recording_service.is_recording or self.recording

    @APIExport(runOnUIThread=False)
    def getRecordingDuration(self) -> float:
        """
        Get the current recording duration in seconds.
        
        Returns:
            Duration in seconds, or 0.0 if not recording
        """
        return self.recording_service.recording_duration

    @APIExport(runOnUIThread=False)
    def getRecordingFrameCount(self) -> int:
        """
        Get the number of frames recorded so far.
        
        Returns:
            Number of frames recorded
        """
        return self.recording_service.recording_frame_count

    @APIExport(runOnUIThread=False)
    def getRecordingStatus(self) -> Dict[str, Any]:
        """
        Get comprehensive recording status information.
        
        Returns:
            Dictionary containing:
            - is_recording: bool - whether recording is active
            - format: str - recording format (e.g., 'MP4', 'OME_ZARR')
            - duration_seconds: float - recording duration
            - frame_count: int - number of frames recorded
            - filepath: str - output file path (if available)
        """
        return self.recording_service.get_status_dict()

    def resizeImage(self, image, scale_factor):
        """
        Resize the input image by a given scale factor using nearest neighbor interpolation.

        Parameters:
            image (numpy.ndarray): The input image. For RGB, shape should be (height, width, 3),
                                for monochrome/grayscale, shape should be (height, width).
            scale_factor (float): The scaling factor by which to resize the image.

        Returns:
            numpy.ndarray: The resized image.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
            height, width, _ = image.shape
        elif len(image.shape) == 2:  # Monochrome/grayscale image
            height, width = image.shape
        else:
            raise ValueError(
                "Invalid image shape. Supported shapes are (height, width, 3) for RGB and (height, width) for monochrome."
            )

        new_height, new_width = int(height * scale_factor), int(width * scale_factor)

        # Use PIL's resize function with nearest neighbor interpolation
        pil_image = Image.fromarray(image)
        resized_pil = pil_image.resize((new_width, new_height), Image.Resampling.NEAREST)
        resized_image = np.array(resized_pil)

        return resized_image


_attrCategory = "Rec"
_recModeAttr = "Mode"
_framesAttr = "Frames"
_timeAttr = "Time"


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
