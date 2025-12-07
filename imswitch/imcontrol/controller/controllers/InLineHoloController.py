import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import time
import traceback
import threading
import queue

try:
    from scipy import fft as scipy_fft
    hasSciPyFFT = False
except:
    hasSciPyFFT = False


try:
    import cv2
    hasCV2 = True
except:
    hasCV2 = False

from imswitch.imcommon.model import dirtools, initLogger, APIExport
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex
from imswitch.imcontrol.view import guitools
from ..basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS


# =========================
# Dataclasses (API-stable)
# =========================
@dataclass
class InLineHoloParams:
    """Inline hologram processing parameters"""
    pixelsize: float = 3.45e-6  # meters
    wavelength: float = 488e-9  # meters
    na: float = 0.3
    dz: float = 0.0  # propagation distance in meters
    roi_center: Optional[List[int]] = None  # [x, y] in pixels
    roi_size: Optional[int] = 256  # square ROI size
    color_channel: str = "green"  # "red", "green", "blue"
    flip_x: bool = False
    flip_y: bool = False
    rotation: int = 0  # 0, 90, 180, 270
    update_freq: float = 10.0  # Hz (processing framerate)
    binning: int = 1  # binning factor (1, 2, 4, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixelsize": self.pixelsize,
            "wavelength": self.wavelength,
            "na": self.na,
            "dz": self.dz,
            "roi_center": self.roi_center,
            "roi_size": self.roi_size,
            "color_channel": self.color_channel,
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "rotation": self.rotation,
            "update_freq": self.update_freq,
            "binning": self.binning,
        }


@dataclass
class InLineHoloState:
    """Inline hologram processing state"""
    is_processing: bool = False
    is_paused: bool = False
    is_streaming: bool = False
    last_process_time: float = 0.0
    frame_count: int = 0
    processed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_processing": self.is_processing,
            "is_paused": self.is_paused,
            "is_streaming": self.is_streaming,
            "last_process_time": self.last_process_time,
            "frame_count": self.frame_count,
            "processed_count": self.processed_count,
        }


class InLineHoloController(LiveUpdatedController):
    """
    Inline hologram processing controller with backend processing and API control.
    
    Features:
    - Fresnel propagation for inline holograms
    - Frame queue with configurable processing rate (via update_freq)
    - Rate-limited frame reception via update() callback
    - Pause/resume mechanism
    - Binning support with automatic pixel size adjustment
    - API control via RESTful endpoints
    
    Architecture:
    - update() is called for every frame via sigUpdateImage
    - Rate limiting is done via counter (like FFTController)
    - Frames are queued and processed by background thread
    """

    sigHoloImageComputed = Signal(np.ndarray, str)  # (image, name)
    sigHoloStateChanged = Signal(object)  # state_dict
    sigImageReceived = Signal()  # Signal to trigger processing worker

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # Get camera from setup or use first available detector
        if hasattr(self._setupInfo, 'holo') and self._setupInfo.holo is not None:
            self.camera = getattr(self._setupInfo.holo, 'camera', None)
        else:
            self.camera = None
            
        # If no camera specified, use first available detector
        if self.camera is None:
            try:
                all_detectors = self._master.detectorsManager.getAllDeviceNames()
                if all_detectors:
                    self.camera = all_detectors[0]
                    self._logger.info(f"Using first available detector: {self.camera}")
                else:
                    self._logger.error("No detectors available")
                    return
            except Exception as e:
                self._logger.error(f"Failed to get detector list: {e}")
                return
        
        # Initialize parameters from setup or defaults
        if hasattr(self._setupInfo, 'holo') and self._setupInfo.holo is not None:
            self._params = InLineHoloParams(
                pixelsize=getattr(self._setupInfo.holo, "pixelsize", 3.45e-6),
                wavelength=getattr(self._setupInfo.holo, "wavelength", 488e-9),
                na=getattr(self._setupInfo.holo, "na", 0.3),
                roi_center=getattr(self._setupInfo.holo, "roi_center", None),
                roi_size=getattr(self._setupInfo.holo, "roi_size", 256),
                update_freq=getattr(self._setupInfo.holo, "update_freq", 10.0),
                binning=getattr(self._setupInfo.holo, "binning", 1),
            )
        else:
            self._params = InLineHoloParams()
        
        self._state = InLineHoloState()
        self._processing_lock = threading.Lock()
        
        # Rate limiting counter (like FFTController pattern)
        # updateRate = number of frames to skip between processing
        # Calculated from update_freq relative to assumed ~30fps camera rate
        self._update_rate = 3  # Default: process every 3rd frame
        self._frame_counter = 0
        
        # Store last frame for pause mode and on-demand processing
        self._last_frame = None
        
        # MJPEG streaming
        self._mjpeg_queue = queue.Queue(maxsize=10)
        self._jpeg_quality = 85
        
        # Frame queue for decoupled processing (small queue, drop frames if full)
        self._frame_queue = queue.Queue(maxsize=3)
        
        # Processing worker and thread
        self._processing_worker = self.HoloProcessingWorker(self)
        self._processing_worker.sigHoloProcessed.connect(self._on_holo_processed)
        self._processing_thread = Thread()
        self._processing_worker.moveToThread(self._processing_thread)
        self.sigImageReceived.connect(self._processing_worker.processHologram)
        self._processing_thread.start()
        
        # Connect to CommunicationChannel signal for frame updates
        # This is the standard pattern used by HistogrammController, FFTController, etc.
        self._commChannel.sigUpdateImage.connect(self.update)
        
        # Legacy GUI setup
        if not IS_HEADLESS:
            self._setup_legacy_gui()
            
        self._logger.info("InLineHoloController initialized successfully")

    def __del__(self):
        """Cleanup on deletion"""
        # Stop processing thread
        self._processing_thread.quit()
        self._processing_thread.wait()
        
        if hasattr(super(), '__del__'):
            super().__del__()

    def update(self, detectorName, image, init, scale, isCurrentDetector):
        """
        Periodic update called for every frame via sigUpdateImage.
        
        This follows the FFTController pattern:
        - Rate limiting via counter
        - Frames are queued for background processing
        - Dropped if queue is full (to prevent memory buildup)
        
        Args:
            detectorName: Name of the detector
            image: The frame data
            init: Whether this is an initialization frame
            scale: Scale information
            isCurrentDetector: Whether this is the currently selected detector
        """
        # Only process frames from our target camera
        if detectorName != self.camera:
            return
        
        # Skip if processing is not enabled
        if not self._state.is_processing:
            return
        
        # Skip if image is None
        if image is None:
            return
        
        # Store last frame (always, for pause mode)
        self._last_frame = image
        self._state.frame_count += 1
        
        # Rate limiting: process every N-th frame
        if self._frame_counter >= self._update_rate:
            self._frame_counter = 0
            
            # Queue frame for processing (or reprocess last frame if paused)
            frame_to_process = self._last_frame if self._state.is_paused else image
            
            # Prepare worker and emit signal to trigger processing
            self._processing_worker.prepareForNewImage(frame_to_process)
            self.sigImageReceived.emit()
        else:
            self._frame_counter += 1

    def _on_holo_processed(self, result):
        """
        Callback when hologram processing is complete.
        
        Args:
            result: The processed hologram image
        """
        if result is not None:
            self.sigHoloImageComputed.emit(result, "inline_holo")
            self._state.processed_count += 1
            self._state.last_process_time = time.time()
            
            # Add to MJPEG stream if active
            if self._state.is_streaming:
                self._add_to_mjpeg_stream(result)

    def set_update_rate(self, update_freq: float):
        """
        Set the processing rate.
        
        Args:
            update_freq: Desired processing frequency in Hz
        """
        # Assume camera runs at ~30fps, calculate skip rate
        # update_rate = frames to skip = (camera_fps / update_freq) - 1
        # For update_freq=10Hz at 30fps: skip every 2 frames (process every 3rd)
        # For update_freq=30Hz at 30fps: skip 0 frames (process every frame)
        assumed_camera_fps = 30.0
        if update_freq <= 0:
            update_freq = 1.0
        skip_rate = max(0, int(assumed_camera_fps / update_freq) - 1)
        self._update_rate = skip_rate
        self._frame_counter = 0
        self._params.update_freq = update_freq
        self._logger.info(f"Set update rate: {update_freq} Hz (skip every {skip_rate} frames)")
    
    # =========================
    # Hologram Processing Core
    # =========================
    @staticmethod
    def _abssqr(x):
        """Calculate intensity (what a detector sees)"""
        return np.real(x * np.conj(x))


    @staticmethod
    def _FT(x):
        """Forward Fourier transform with proper frequency shift"""
        if hasSciPyFFT:
            return scipy_fft.fftshift(
                scipy_fft.fft2(x, workers=4)
            )
        return np.fft.fftshift(np.fft.fft2(x))
    
    @staticmethod
    def _iFT(x):
        """Inverse Fourier transform with proper frequency shift"""
        if hasSciPyFFT:
            return scipy_fft.ifft2(
                scipy_fft.ifftshift(x), workers=4
            )
        return np.fft.ifft2(np.fft.ifftshift(x))

    def _fresnel_propagator(self, E0, dz):
        """
        Freespace propagation using Fresnel kernel
        
        Args:
            E0: Initial complex field in x-y source plane
            dz: Distance from sensor to object in meters
        
        Returns:
            Ef: Propagated output field
        """
        # Use effective pixel size (adjusted for binning)
        ps = self._params.pixelsize * self._params.binning
        lambda0 = self._params.wavelength
        
        nx = E0.shape[1]  # Image width in pixels
        ny = E0.shape[0]  # Image height in pixels
        grid_size_x = ps * nx  # Grid size in x-direction
        grid_size_y = ps * ny  # Grid size in y-direction
        
        # 1-D frequency grids
        fx = np.linspace(-(nx-1)/2*(1/grid_size_x), (nx-1)/2*(1/grid_size_x), nx)
        fy = np.linspace(-(ny-1)/2*(1/grid_size_y), (ny-1)/2*(1/grid_size_y), ny)
        # 1-D Fresnel factors
        phase = 1j * np.pi * lambda0 * dz
        hfx = np.exp(phase * fx**2)  # shape (n,)
        hfy = np.exp(phase * fy**2)  # shape (n,)

        E0fft = self._FT(E0)  # shape (n, n)

        # Broadcasted multiply without forming a 2-D exp
        G = E0fft * hfx  # broadcasts along columns
        G *= hfy[:, None]  # broadcasts along rows

        Ef = self._iFT(G)
        
        return Ef

    def _apply_binning(self, image):
        """Apply binning to image if binning > 1"""
        if self._params.binning <= 1:
            return image
        
        # subsakmple
        if len(image.shape) == 2:
            # Grayscale
            return image[::2, ::2]
        else:
            # Color
            return image[::2, ::2, :]

    def _extract_roi(self, image):
        """Extract ROI from image based on current parameters"""
        h, w = image.shape[:2]
        
        # Determine ROI center
        if self._params.roi_center is not None and self._params.roi_center[0] is not None and self._params.roi_center[1] is not None:
            cx, cy = self._params.roi_center
        else:
            cx, cy = w // 2, h // 2
        
        # Calculate ROI bounds
        roi_size = np.min([self._params.roi_size, np.max([h, w])])
        half_size = roi_size // 2
        
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        
        return image[y1:y2, x1:x2]

    def _extract_color_channel(self, image):
        """Extract specified color channel from RGB image"""
        if len(image.shape) == 2:
            return image  # Already grayscale
        
        channel_map = {"red": 0, "green": 1, "blue": 2}
        channel_idx = channel_map.get(self._params.color_channel, 1)
        
        return image[:, :, channel_idx]

    def _apply_transforms(self, image):
        """Apply flip and rotation transformations"""
        if self._params.flip_x:
            image = np.fliplr(image)
        if self._params.flip_y:
            image = np.flipud(image)
        
        # Apply rotation (counter-clockwise)
        if self._params.rotation == 90:
            image = np.rot90(image, k=1)
        elif self._params.rotation == 180:
            image = np.rot90(image, k=2)
        elif self._params.rotation == 270:
            image = np.rot90(image, k=3)
        
        return image

    def _process_inline(self, image):
        """Process inline hologram"""
        # Apply binning first
        binned = self._apply_binning(image)
        
        # Extract ROI and color channel
        roi = self._extract_roi(binned)
        gray = self._extract_color_channel(roi)
        gray = self._apply_transforms(gray)
        
        # Convert to complex field (E-field from intensity)
        E0 = np.sqrt(gray.astype(float))
        
        # Propagate
        Ef = self._fresnel_propagator(E0, self._params.dz)
        
        # Return intensity
        return self._abssqr(Ef)

    def _process_frame(self, image):
        """Process a single frame and return result"""
        try:
            result = self._process_inline(image)
            return result
        except Exception as e:
            self._logger.error(f"Error processing hologram: {e}")
            self._logger.debug(traceback.format_exc())
        return None

    def _get_latest_frame(self):
        """
        Get the last received frame (for manual/on-demand processing).
        
        This returns the most recently received frame from the signal callback.
        For continuous processing, use start_processing() which handles frames
        automatically via the sigUpdateImage signal.
        
        Returns:
            The last received frame, or None if no frame has been received yet.
        """
        return self._last_frame

    def _add_to_mjpeg_stream(self, image):
        """
        Encode and add reconstructed hologram to MJPEG stream.
        
        Args:
            image: Processed hologram result (float array)
        """
        if not hasCV2:
            return
        
        try:
            # Normalize to uint8
            frame = np.array(image)
            if frame.dtype != np.uint8:
                vmin = float(np.min(frame))
                vmax = float(np.max(frame))
                if vmax > vmin:
                    frame = ((frame - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            
            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            success, encoded = cv2.imencode('.jpg', frame, encode_params)
            
            if success:
                jpeg_bytes = encoded.tobytes()
                # Build MJPEG frame with proper headers
                header = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                )
                content_length = f'Content-Length: {len(jpeg_bytes)}\r\n\r\n'.encode('ascii')
                mjpeg_frame = header + content_length + jpeg_bytes + b'\r\n'
                
                # Put in queue, drop frame if full
                try:
                    self._mjpeg_queue.put_nowait(mjpeg_frame)
                except queue.Full:
                    pass  # Drop frame if queue is full
        except Exception as e:
            self._logger.debug(f"Error encoding MJPEG frame: {e}")

    def _reprocess_last_frame(self):
        """
        Reprocess the last received frame (used for manual reprocessing).
        Called when parameters change while paused to show updated results.
        """
        if self._last_frame is not None:
            self._processing_worker.prepareForNewImage(self._last_frame)
            self.sigImageReceived.emit()

    # =========================
    # Processing Worker (like FFTController pattern)
    # =========================
    class HoloProcessingWorker(Worker):
        """
        Worker that processes hologram frames in a separate thread.
        
        This follows the FFTController pattern where:
        - prepareForNewImage() stores the image to process
        - processHologram() is triggered via signal and does the actual work
        - sigHoloProcessed emits the result when done
        """
        sigHoloProcessed = Signal(np.ndarray)
        
        def __init__(self, controller):
            super().__init__()
            self._controller = controller
            self._image = None
            self._numQueuedImages = 0
            self._numQueuedImagesMutex = Mutex()
        
        def prepareForNewImage(self, image):
            """Must always be called before the worker receives a new image."""
            self._image = image
            self._numQueuedImagesMutex.lock()
            self._numQueuedImages += 1
            self._numQueuedImagesMutex.unlock()
        
        def processHologram(self):
            """Process the hologram image."""
            try:
                if self._numQueuedImages > 1:
                    # Skip this frame to catch up
                    return
                
                if self._image is None:
                    return
                
                # Use the controller's processing method
                result = self._controller._process_frame(self._image)
                if result is not None:
                    self.sigHoloProcessed.emit(result)
            finally:
                self._numQueuedImagesMutex.lock()
                self._numQueuedImages -= 1
                self._numQueuedImagesMutex.unlock()

    # =========================
    # API: Parameter Control
    # =========================
    @APIExport(runOnUIThread=True)
    def get_parameters_inlineholo(self) -> Dict[str, Any]:
        """
        Get current hologram processing parameters
        
        Returns:
            Dictionary with all parameters
            
        Example:
            {
                "pixelsize": 3.45e-6,
                "wavelength": 488e-9,
                "na": 0.3,
                "dz": 0.005,
                "roi_center": [512, 512],
                "roi_size": 256,
                "color_channel": "green",
                "flip_x": false,
                "flip_y": false,
                "rotation": 0,
                "update_freq": 10.0,
                "binning": 1
            }
        """
        return self._params.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def set_parameters_inlineholo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update hologram processing parameters
        
        Args:
            params: Dictionary with parameter updates
                - pixelsize: float (meters)
                - wavelength: float (meters)
                - na: float
                - dz: float (meters)
                - roi_center: [x, y]
                - roi_size: int
                - color_channel: str ("red", "green", "blue")
                - flip_x: bool
                - flip_y: bool
                - rotation: int (0, 90, 180, 270)
                - update_freq: float (Hz)
                - binning: int (1, 2, 4, etc.)
        
        Returns:
            Updated parameters dictionary
            
        Example request:
            {"dz": 0.005, "wavelength": 488e-9, "binning": 2}
        """
        with self._processing_lock:
            for key, value in params.items():
                if hasattr(self._params, key):
                    setattr(self._params, key, value)
        
        self._emit_state_changed()
        return self._params.to_dict()

    @APIExport(runOnUIThread=True)
    def set_pixelsize_inlineholo(self, pixelsize: float) -> Dict[str, Any]:
        """Set pixel size in meters"""
        return self.set_parameters_inlineholo({"pixelsize": pixelsize})

    @APIExport(runOnUIThread=True)
    def set_wavelength_inlineholo(self, wavelength: float) -> Dict[str, Any]:
        """Set wavelength in meters"""
        return self.set_parameters_inlineholo({"wavelength": wavelength})

    @APIExport(runOnUIThread=True)
    def set_dz_inlineholo(self, dz: float) -> Dict[str, Any]:
        """Set propagation distance in meters"""
        return self.set_parameters_inlineholo({"dz": dz})

    @APIExport(runOnUIThread=True)
    def set_roi_inlineholo(self, center_x: int=None, center_y: int=None, size: int=256) -> Dict[str, Any]:
        """
        Set ROI center and size
        example request:
            {"center": [512, 512], "size": 256}
        """ 
        center = [center_x, center_y] if center_x is not None and center_y is not None else None
        return self.set_parameters_inlineholo({"roi_center": center, "roi_size": size})
    
    @APIExport(runOnUIThread=True)
    def set_binning_inlineholo(self, binning: int) -> Dict[str, Any]:
        """
        Set binning factor (1, 2, 4, etc.)
        Note: Pixel size in reconstruction kernel is automatically adjusted
        """
        return self.set_parameters_inlineholo({"binning": binning})

    @APIExport(runOnUIThread=True)
    def set_update_freq_inlineholo(self, update_freq: float) -> Dict[str, Any]:
        """
        Set the processing frame rate in Hz.
        
        This controls how many frames are skipped between processing.
        Higher values = faster processing but more CPU usage.
        Lower values = slower processing but lower CPU usage.
        
        Args:
            update_freq: Processing rate in Hz (default: 10.0)
            
        Returns:
            Updated parameters dictionary
        """
        self.set_update_rate(update_freq)
        return self._params.to_dict()

    # =========================
    # API: Processing Control
    # =========================
    @APIExport(runOnUIThread=True)
    def get_state_inlineholo(self) -> Dict[str, Any]:
        """Get current processing state"""
        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def start_processing_inlineholo(self) -> Dict[str, Any]:
        """
        Start hologram processing.
        
        Processing uses the update() callback pattern (like FFTController):
        - Frames are received via sigUpdateImage at camera rate
        - Rate limiting is done via counter (skip N frames)
        - Frames are processed by worker thread
        
        Returns:
            Current state dictionary
        """
        with self._processing_lock:
            self._state.is_processing = True
            self._state.is_paused = False
            self._state.frame_count = 0
            self._state.processed_count = 0
        
        # Reset frame counter
        self._frame_counter = 0
        
        # Ensure camera is running
        self._ensure_camera_running()
        
        # Set update rate from params
        self.set_update_rate(self._params.update_freq)
        
        self._logger.info(f"Started inline hologram processing ({self._params.update_freq} Hz)")
        self._emit_state_changed()
        
        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def stop_processing_inlineholo(self) -> Dict[str, Any]:
        """
        Stop hologram processing.
        
        Returns:
            Current state dictionary
        """
        with self._processing_lock:
            self._state.is_processing = False
            self._state.is_paused = False
        
        self._logger.info("Stopped hologram processing")
        self._emit_state_changed()
        
        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def pause_processing_inlineholo(self) -> Dict[str, Any]:
        """
        Pause processing - will continuously process last frame at update rate
        
        Returns:
            Current state dictionary
        """
        with self._processing_lock:
            if self._state.is_processing:
                self._state.is_paused = True
        
        self._logger.info("Paused hologram processing (processing last frame)")
        self._emit_state_changed()
        
        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def resume_processing_inlineholo(self) -> Dict[str, Any]:
        """
        Resume processing - will process incoming frames continuously
        
        Returns:
            Current state dictionary
        """
        with self._processing_lock:
            if self._state.is_processing:
                self._state.is_paused = False
        
        self._logger.info("Resumed hologram processing")
        self._emit_state_changed()
        
        return self._state.to_dict()

    @APIExport(runOnUIThread=False)
    def mjpeg_stream_inlineholo(self, startStream: bool = True, jpeg_quality: int = 85):
        """
        HTTP endpoint for MJPEG streaming of reconstructed holograms.
        
        Args:
            startStream: Whether to start streaming (True) or stop (False)
            jpeg_quality: JPEG compression quality (0-100, default 85)
        
        Returns:
            StreamingResponse with MJPEG data or status message
            
        Example:
            GET /holocontroller/mjpeg_stream_inline?startStream=true&jpeg_quality=90
        """
        try:
            from fastapi.responses import StreamingResponse
        except ImportError:
            return {"status": "error", "message": "FastAPI not available"}
        
        if not hasCV2:
            return {"status": "error", "message": "opencv-python required for MJPEG streaming"}
        
        if not startStream:
            # Stop streaming
            with self._processing_lock:
                self._state.is_streaming = False
            # Clear queue
            while not self._mjpeg_queue.empty():
                try:
                    self._mjpeg_queue.get_nowait()
                except queue.Empty:
                    break
            self._logger.info("Stopped MJPEG stream")
            self._emit_state_changed()
            return {"status": "success", "message": "stream stopped"}
        
        # Update JPEG quality
        self._jpeg_quality = max(0, min(100, jpeg_quality))
        
        # Start streaming
        with self._processing_lock:
            self._state.is_streaming = True
        
        # Ensure processing is running
        if not self._state.is_processing:
            self.start_processing_inlineholo()
        
        self._logger.info(f"Started MJPEG stream (quality={self._jpeg_quality})")
        self._emit_state_changed()
        
        # Create generator for streaming response
        def frame_generator():
            """Generator that yields MJPEG frames."""
            try:
                while self._state.is_streaming:
                    try:
                        frame = self._mjpeg_queue.get(timeout=1.0)
                        if frame:
                            yield frame
                    except queue.Empty:
                        continue
            except GeneratorExit:
                self._logger.info("MJPEG stream connection closed by client")
                with self._processing_lock:
                    self._state.is_streaming = False
                self._emit_state_changed()
            except Exception as e:
                self._logger.error(f"Error in MJPEG frame generator: {e}")
        
        # Return streaming response with proper headers
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        
        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace;boundary=frame",
            headers=headers
        )

    #@APIExport(runOnUIThread=True)
    def process_single_frame(self, image: np.ndarray = None) -> Dict[str, Any]:
        """
        Process a single frame manually
        
        Args:
            image: Optional numpy array. If None, captures from camera
        
        Returns:
            Result info dictionary
        """
        if image is None:
            # Capture from camera
            image = self._capture_camera_frame()
        
        if image is None:
            return {"success": False, "error": "No image available"}
        
        result = self._process_frame(image)
        
        return {
            "success": result is not None,
            "frame_shape": result.shape if result is not None else None
        }

    def _ensure_camera_running(self):
        """Ensure camera is running, start if necessary"""
        try:
            detector = self._master.detectorsManager[self.camera]
            if not detector._running:
                self._logger.info(f"Starting camera {self.camera}")
                detector.startAcquisition()
        except Exception as e:
            self._logger.error(f"Failed to start camera: {e}")

    def _capture_camera_frame(self):
        """Capture a single frame from camera"""
        try:
            detector = self._master.detectorsManager[self.camera]
            return detector.getLatestFrame()
        except Exception as e:
            self._logger.error(f"Failed to capture frame: {e}")
            return None

    def _emit_state_changed(self):
        """Emit state changed signal"""
        self.sigHoloStateChanged.emit(self._state.to_dict())

    # =========================
    # Legacy GUI compatibility (if not headless)
    # =========================
    def setShowInLineHolo(self, enabled):
        """Legacy: Show or hide inline hologram processing"""
        if enabled:
            self.start_processing()
        else:
            self.stop_processing()

    def changeRate(self, updateRate):
        """Legacy: Change update rate"""
        if updateRate == "" or updateRate <= 0:
            updateRate = 1
        self.set_parameters_inlineholo({"update_freq": updateRate})

    def inLineValueChanged(self, magnitude):
        """Legacy: Change inline propagation distance"""
        dz = magnitude * 1e-3  # Convert to meters
        self.set_dz(dz)

    def displayImage(self, im, name):
        """Legacy: Display image in napari widget"""
        if IS_HEADLESS:
            return
        
        if im.dtype == complex or np.iscomplexobj(im):
            self._widget.setImage(np.abs(im), name + "_abs")
            self._widget.setImage(np.angle(im), name + "_angle")
        else:
            self._widget.setImage(np.abs(im), name)



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
