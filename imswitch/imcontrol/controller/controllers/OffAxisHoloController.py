import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import time
import traceback
import threading
import queue

# Try to import skimage for proper 2D phase unwrapping
try:
    from skimage.restoration import unwrap_phase as skimage_unwrap_phase
    HAS_SKIMAGE_UNWRAP = True
except ImportError:
    HAS_SKIMAGE_UNWRAP = False

# Try to import scipy for window functions
try:
    from scipy import signal as scipy_signal
    HAS_SCIPY_SIGNAL = True
except ImportError:
    HAS_SCIPY_SIGNAL = False

try:
    import NanoImagingPack as nip
    isNIP = True
except:
    print("OffAxisHoloController: NanoImagingPack not available, using fallback processing")
    isNIP = False

try:
    import cv2
    hasCV2 = True
except:
    hasCV2 = False

from imswitch.imcommon.model import initLogger, APIExport
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex
from ..basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS


# =========================
# Dataclasses (API-stable)
# =========================
@dataclass
class OffAxisHoloParams:
    """Off-axis hologram processing parameters"""
    pixelsize: float = 3.45e-6  # meters
    wavelength: float = 488e-9  # meters
    na: float = 0.3
    dz: float = 0.0  # propagation distance in meters (digital refocus)
    roi_center: Optional[List[int]] = None  # [x, y] in pixels (sensor ROI)
    roi_size: Optional[int] = 512  # square ROI size for sensor crop
    color_channel: str = "green"  # "red", "green", "blue"
    flip_x: bool = False
    flip_y: bool = False
    rotation: int = 0  # 0, 90, 180, 270
    update_freq: float = 10.0  # Hz (processing framerate)
    binning: int = 1  # binning factor (1, 2, 4, etc.)
    
    # Off-axis specific: sideband selection in FFT space
    cc_center: Optional[List[int]] = None  # Cross-correlation (sideband) center [x, y] in FFT pixels
    cc_size: Optional[List[int]] = None  # Crop size [width, height] or [size] for square
    cc_radius: int = 100  # Legacy: square crop radius (half-size)
    
    # Apodization (edge damping)
    apodization_enabled: bool = False
    apodization_type: str = "tukey"  # "tukey", "hann", "hamming", "blackman"
    apodization_alpha: float = 0.1  # tukey parameter (0=rect, 1=hann)
    
    # Preview downsampling for MJPEG streams
    preview_max_size: int = 512  # max dimension for streamed previews

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
            "cc_center": self.cc_center,
            "cc_size": self.cc_size,
            "cc_radius": self.cc_radius,
            "apodization_enabled": self.apodization_enabled,
            "apodization_type": self.apodization_type,
            "apodization_alpha": self.apodization_alpha,
            "preview_max_size": self.preview_max_size,
        }


@dataclass
class OffAxisHoloState:
    """Off-axis hologram processing state"""
    is_processing: bool = False
    is_paused: bool = False
    is_streaming_fft: bool = False
    is_streaming_magnitude: bool = False
    is_streaming_phase: bool = False
    last_process_time: float = 0.0
    frame_count: int = 0
    processed_count: int = 0
    fft_shape: Optional[Tuple[int, int]] = None  # shape of FFT image

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_processing": self.is_processing,
            "is_paused": self.is_paused,
            "is_streaming_fft": self.is_streaming_fft,
            "is_streaming_magnitude": self.is_streaming_magnitude,
            "is_streaming_phase": self.is_streaming_phase,
            "last_process_time": self.last_process_time,
            "frame_count": self.frame_count,
            "processed_count": self.processed_count,
            "fft_shape": self.fft_shape,
        }


class OffAxisHoloController(LiveUpdatedController):
    """
    Off-axis hologram processing controller with backend processing and API control.
    
    Features:
    - Cross-correlation based off-axis holography
    - FFT visualization with sideband selection
    - Phase unwrapping (2D)
    - Apodization/edge damping for cropped region
    - Digital refocus (Fresnel propagation)
    - Frame queue with configurable processing rate
    - Multiple MJPEG streams: FFT magnitude, reconstructed magnitude, unwrapped phase
    - Pause/resume mechanism
    - Binning support with automatic pixel size adjustment
    - API control via RESTful endpoints
    
    Architecture:
    - update() is called for every frame via sigUpdateImage
    - Rate limiting is done via counter (like FFTController)
    - Frames are queued and processed by background worker thread
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
            self._params = OffAxisHoloParams(
                pixelsize=getattr(self._setupInfo.holo, "pixelsize", 3.45e-6),
                wavelength=getattr(self._setupInfo.holo, "wavelength", 488e-9),
                na=getattr(self._setupInfo.holo, "na", 0.3),
                roi_center=getattr(self._setupInfo.holo, "roi_center", None),
                roi_size=getattr(self._setupInfo.holo, "roi_size", 512),
                update_freq=getattr(self._setupInfo.holo, "update_freq", 10.0),
                binning=getattr(self._setupInfo.holo, "binning", 1),
                cc_center=getattr(self._setupInfo.holo, "cc_center", None),
                cc_radius=getattr(self._setupInfo.holo, "cc_radius", 100),
            )
        else:
            self._params = OffAxisHoloParams()

        self._state = OffAxisHoloState()
        self._processing_lock = threading.Lock()

        # Rate limiting counter (like FFTController pattern)
        self._update_rate = 8  # Default: process every 8th frame
        self._frame_counter = 0

        # Store last frame for pause mode
        self._last_frame = None
        
        # Store last processed results for streaming
        self._last_fft_magnitude = None
        self._last_crop_magnitude = None
        self._last_unwrapped_phase = None

        # MJPEG streaming queues (separate for each stream type)
        self._mjpeg_queue_fft = queue.Queue(maxsize=5)
        self._mjpeg_queue_magnitude = queue.Queue(maxsize=5)
        self._mjpeg_queue_phase = queue.Queue(maxsize=5)
        self._jpeg_quality = 85

        # Processing worker and thread
        self._processing_worker = self.OffAxisProcessingWorker(self)
        self._processing_worker.sigHoloProcessed.connect(self._on_holo_processed)
        self._processing_thread = Thread()
        self._processing_worker.moveToThread(self._processing_thread)
        self.sigImageReceived.connect(self._processing_worker.processHologram)
        self._processing_thread.start()

        # Connect to CommunicationChannel signal for frame updates
        self._commChannel.sigUpdateImage.connect(self.update)

        # Legacy GUI setup
        if not IS_HEADLESS:
            self._setup_legacy_gui()

        self._logger.info("OffAxisHoloController initialized successfully")
        self._logger.info(f"  - Phase unwrap: {'skimage' if HAS_SKIMAGE_UNWRAP else 'numpy fallback'}")
        self._logger.info(f"  - Window functions: {'scipy' if HAS_SCIPY_SIGNAL else 'numpy fallback'}")

    def __del__(self):
        """Cleanup on deletion"""
        # Stop processing thread
        if hasattr(self, '_processing_thread'):
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
        if self._frame_counter >= max(3, self._update_rate):
            self._frame_counter = 0

            # Queue frame for processing (or reprocess last frame if paused)
            frame_to_process = self._last_frame if self._state.is_paused else image

            # Prepare worker and emit signal to trigger processing
            self._processing_worker.prepareForNewImage(frame_to_process)
            self.sigImageReceived.emit()
        else:
            self._frame_counter += 1

    def _on_holo_processed(self, results):
        """
        Callback when hologram processing is complete.
        
        Args:
            results: dict with 'fft_magnitude', 'crop_magnitude', 'unwrapped_phase'
        """
        if results is not None:
            self._state.processed_count += 1
            self._state.last_process_time = time.time()

            # Store results
            self._last_fft_magnitude = results.get('fft_magnitude')
            self._last_crop_magnitude = results.get('crop_magnitude')
            self._last_unwrapped_phase = results.get('unwrapped_phase')

            # Add to MJPEG streams if active
            if self._state.is_streaming_fft and self._last_fft_magnitude is not None:
                self._add_to_mjpeg_stream(self._last_fft_magnitude, self._mjpeg_queue_fft, colormap='viridis')
            if self._state.is_streaming_magnitude and self._last_crop_magnitude is not None:
                self._add_to_mjpeg_stream(self._last_crop_magnitude, self._mjpeg_queue_magnitude)
            if self._state.is_streaming_phase and self._last_unwrapped_phase is not None:
                self._add_to_mjpeg_stream(self._last_unwrapped_phase, self._mjpeg_queue_phase, colormap='twilight')

            # Emit signal for legacy GUI
            if self._last_crop_magnitude is not None:
                self.sigHoloImageComputed.emit(self._last_crop_magnitude, "offaxis_magnitude")

    def set_update_rate(self, update_freq: float):
        """Set the processing rate."""
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
        return np.fft.fftshift(np.fft.fft2(x))

    @staticmethod
    def _iFT(x):
        """Inverse Fourier transform with proper frequency shift"""
        return np.fft.ifft2(np.fft.ifftshift(x))

    def _create_apodization_window(self, shape):
        """
        Create a 2D apodization window for edge damping.
        
        Args:
            shape: tuple (height, width) of the window
            
        Returns:
            2D numpy array with window values
        """
        h, w = shape
        
        if HAS_SCIPY_SIGNAL:
            if self._params.apodization_type == "tukey":
                win_y = scipy_signal.windows.tukey(h, alpha=self._params.apodization_alpha)
                win_x = scipy_signal.windows.tukey(w, alpha=self._params.apodization_alpha)
            elif self._params.apodization_type == "hann":
                win_y = scipy_signal.windows.hann(h)
                win_x = scipy_signal.windows.hann(w)
            elif self._params.apodization_type == "hamming":
                win_y = scipy_signal.windows.hamming(h)
                win_x = scipy_signal.windows.hamming(w)
            elif self._params.apodization_type == "blackman":
                win_y = scipy_signal.windows.blackman(h)
                win_x = scipy_signal.windows.blackman(w)
            else:
                win_y = scipy_signal.windows.tukey(h, alpha=self._params.apodization_alpha)
                win_x = scipy_signal.windows.tukey(w, alpha=self._params.apodization_alpha)
        else:
            # Fallback: simple Hann window using numpy
            win_y = np.hanning(h)
            win_x = np.hanning(w)
        
        # Create 2D window via outer product
        return np.outer(win_y, win_x)

    def _unwrap_phase(self, phase):
        """
        Unwrap 2D phase image.
        
        Args:
            phase: 2D array of wrapped phase values
            
        Returns:
            2D array of unwrapped phase values
        """
        if HAS_SKIMAGE_UNWRAP:
            return skimage_unwrap_phase(phase)
        else:
            # Fallback: simple row-then-column unwrap
            unwrapped = np.unwrap(phase, axis=0)
            unwrapped = np.unwrap(unwrapped, axis=1)
            return unwrapped

    def _fresnel_propagator(self, E0, dz):
        """
        Freespace propagation using Fresnel kernel (digital refocus)
        
        Args:
            E0: Initial complex field in x-y source plane
            dz: Distance from sensor to object in meters
        
        Returns:
            Ef: Propagated output field
        """
        if dz == 0:
            return E0
            
        # Use effective pixel size (adjusted for binning)
        ps = self._params.pixelsize * self._params.binning
        lambda0 = self._params.wavelength

        ny, nx = E0.shape[:2]
        grid_size_x = ps * nx
        grid_size_y = ps * ny

        # 1-D frequency grids
        fx = np.linspace(-(nx-1)/2*(1/grid_size_x), (nx-1)/2*(1/grid_size_x), nx)
        fy = np.linspace(-(ny-1)/2*(1/grid_size_y), (ny-1)/2*(1/grid_size_y), ny)

        # 1-D Fresnel factors
        phase = 1j * np.pi * lambda0 * dz
        hfx = np.exp(phase * fx**2)
        hfy = np.exp(phase * fy**2)

        E0fft = self._FT(E0)

        # Broadcasted multiply without forming a 2-D exp
        G = E0fft * hfx
        G *= hfy[:, None]

        Ef = self._iFT(G)

        return Ef

    def _apply_binning(self, image):
        """Apply binning to image if binning > 1"""
        if self._params.binning <= 1:
            return image

        b = self._params.binning
        h, w = image.shape[:2]

        # Simple subsampling for speed
        if len(image.shape) == 2:
            return image[::b, ::b]
        else:
            return image[::b, ::b, :]

    def _extract_roi(self, image):
        """Extract ROI from image based on current parameters"""
        h, w = image.shape[:2]

        # Determine ROI center
        if self._params.roi_center is not None and self._params.roi_center[0] is not None:
            cx, cy = self._params.roi_center
        else:
            cx, cy = w // 2, h // 2

        # Calculate ROI bounds
        roi_size = min(self._params.roi_size or 512, min(h, w))
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

    def _process_offaxis(self, image):
        """
        Process off-axis hologram with full pipeline.
        
        Returns:
            dict with 'fft_magnitude', 'crop_magnitude', 'unwrapped_phase'
            or None if processing fails
        """
        try:
            # Apply binning first
            binned = self._apply_binning(image)

            # Extract ROI and color channel
            roi = self._extract_roi(binned)
            gray = self._extract_color_channel(roi)
            gray = self._apply_transforms(gray).astype(np.float64)

            # Store FFT shape for coordinate calculations
            self._state.fft_shape = gray.shape

            # === Step 1: Compute FFT ===
            F = self._FT(gray)
            
            # FFT magnitude (log scale for visualization)
            fft_mag = np.log1p(np.abs(F))
            
            # Downsample for streaming if needed
            fft_mag_preview = self._downsample_for_preview(fft_mag)

            # === Step 2: Crop sideband ===
            if self._params.cc_center is None:
                # No sideband center set - return just FFT
                return {
                    'fft_magnitude': fft_mag_preview,
                    'crop_magnitude': None,
                    'unwrapped_phase': None,
                }
            
            # Get crop parameters
            cx, cy = self._params.cc_center
            if self._params.cc_size is not None:
                if isinstance(self._params.cc_size, (list, tuple)) and len(self._params.cc_size) >= 2:
                    crop_w, crop_h = self._params.cc_size[0], self._params.cc_size[1]
                else:
                    crop_w = crop_h = self._params.cc_size[0] if isinstance(self._params.cc_size, (list, tuple)) else self._params.cc_size
            else:
                crop_w = crop_h = self._params.cc_radius * 2
            
            half_w = crop_w // 2
            half_h = crop_h // 2
            
            # Ensure bounds are within FFT
            fft_h, fft_w = F.shape
            y1 = max(0, cy - half_h)
            y2 = min(fft_h, cy + half_h)
            x1 = max(0, cx - half_w)
            x2 = min(fft_w, cx + half_w)
            
            # Crop the sideband
            C = F[y1:y2, x1:x2].copy()
            
            # === Step 3: Apply apodization if enabled ===
            if self._params.apodization_enabled and C.shape[0] > 0 and C.shape[1] > 0:
                window = self._create_apodization_window(C.shape)
                C = C * window
            
            # === Step 4: Shift to center and inverse FFT ===
            # Create zero-padded array with sideband at center
            padded = np.zeros_like(F)
            pad_h, pad_w = padded.shape
            
            # Calculate where to place the cropped region (centered)
            ph = C.shape[0]
            pw = C.shape[1]
            py1 = (pad_h - ph) // 2
            py2 = py1 + ph
            px1 = (pad_w - pw) // 2
            px2 = px1 + pw
            
            padded[py1:py2, px1:px2] = C
            
            # Inverse FFT to get complex field
            E = self._iFT(padded)
            
            # === Step 5: Digital refocus if dz != 0 ===
            if self._params.dz != 0:
                E = self._fresnel_propagator(E, self._params.dz)
            
            # === Step 6: Extract magnitude and phase ===
            magnitude = np.abs(E)
            phase = np.angle(E)
            
            # === Step 7: Unwrap phase ===
            if 0:
                unwrapped_phase = self._unwrap_phase(phase)
            else:
                unwrapped_phase = phase  # Skip unwrapping for now (can be slow)
            
            # Downsample for preview
            magnitude_preview = self._downsample_for_preview(magnitude)
            phase_preview = self._downsample_for_preview(unwrapped_phase)
            
            return {
                'fft_magnitude': fft_mag_preview,
                'crop_magnitude': magnitude_preview,
                'unwrapped_phase': phase_preview,
            }
            
        except Exception as e:
            self._logger.error(f"Error in off-axis processing: {e}")
            self._logger.debug(traceback.format_exc())
            return None

    def _downsample_for_preview(self, image):
        """Downsample image for MJPEG streaming preview"""
        if image is None:
            return None
            
        max_size = self._params.preview_max_size
        h, w = image.shape[:2]
        
        if max(h, w) <= max_size:
            return image
        
        scale = max_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        if hasCV2:
            return cv2.resize(image.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Simple subsampling fallback
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            return image[::step_h, ::step_w]

    def _process_frame(self, image):
        """Process a single frame and return results dict"""
        try:
            return self._process_offaxis(image)
        except Exception as e:
            self._logger.error(f"Error processing hologram: {e}")
            self._logger.debug(traceback.format_exc())
        return None

    def _get_latest_frame(self):
        """
        Fetch latest frame from camera detector.
        Returns None if camera not available or no frame ready.
        """
        try:
            detector = self._master.detectorsManager[self.camera]
            return detector.getLatestFrame()
        except Exception as e:
            self._logger.debug(f"Failed to get frame: {e}")
            return None

    def _add_to_mjpeg_stream(self, image, target_queue, colormap=None):
        """
        Encode and add image to specified MJPEG stream queue.
        
        Args:
            image: Image array (float or uint8)
            target_queue: Queue to add frame to
            colormap: Optional colormap name ('viridis', 'twilight', etc.)
        """
        if not hasCV2 or image is None:
            return

        try:
            # Normalize to uint8
            frame = np.array(image)
            if frame.dtype != np.uint8:
                vmin = float(np.nanmin(frame))
                vmax = float(np.nanmax(frame))
                if vmax > vmin:
                    frame = ((frame - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                else:
                    frame = np.zeros(frame.shape, dtype=np.uint8)

            # Apply colormap if specified
            if colormap and hasCV2:
                if colormap == 'viridis':
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
                elif colormap == 'twilight':
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_TWILIGHT)
                elif colormap == 'jet':
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                elif colormap == 'hot':
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

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
                    target_queue.put_nowait(mjpeg_frame)
                except queue.Full:
                    pass  # Drop frame if queue is full
        except Exception as e:
            self._logger.debug(f"Error encoding MJPEG frame: {e}")

    # =========================
    # Processing Worker (like FFTController pattern)
    # =========================
    class OffAxisProcessingWorker(Worker):
        """
        Worker that processes off-axis hologram frames in a separate thread.
        """
        sigHoloProcessed = Signal(object)  # dict with results

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
    def get_parameters_offaxisholo(self) -> Dict[str, Any]:
        """
        Get current off-axis hologram processing parameters
        
        Returns:
            Dictionary with all parameters including:
            - pixelsize, wavelength, na, dz (optical params)
            - roi_center, roi_size (sensor ROI)
            - cc_center, cc_size, cc_radius (FFT sideband selection)
            - apodization_enabled, apodization_type, apodization_alpha
            - binning, update_freq, preview_max_size
        """
        return self._params.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def set_parameters_offaxisholo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update off-axis hologram processing parameters
        
        Args:
            params: Dictionary with parameter updates
        
        Returns:
            Updated parameters dictionary
        """
        with self._processing_lock:
            for key, value in params.items():
                if hasattr(self._params, key):
                    setattr(self._params, key, value)
                    
        # Update rate if update_freq changed
        if 'update_freq' in params:
            self.set_update_rate(params['update_freq'])

        self._emit_state_changed()
        return self._params.to_dict()

    @APIExport(runOnUIThread=True)
    def set_pixelsize_offaxisholo(self, pixelsize: float) -> Dict[str, Any]:
        """Set pixel size in meters"""
        return self.set_parameters_offaxisholo({"pixelsize": pixelsize})

    @APIExport(runOnUIThread=True)
    def set_wavelength_offaxisholo(self, wavelength: float) -> Dict[str, Any]:
        """Set wavelength in meters"""
        return self.set_parameters_offaxisholo({"wavelength": wavelength})

    @APIExport(runOnUIThread=True)
    def set_dz_offaxisholo(self, dz: float) -> Dict[str, Any]:
        """Set digital refocus distance in meters"""
        return self.set_parameters_offaxisholo({"dz": dz})

    @APIExport(runOnUIThread=True)
    def set_roi_offaxisholo(self, center_x: int = None, center_y: int = None, size: int = 512) -> Dict[str, Any]:
        """Set sensor ROI center and size"""
        center = [center_x, center_y] if center_x is not None and center_y is not None else None
        return self.set_parameters_offaxisholo({"roi_center": center, "roi_size": size})

    @APIExport(runOnUIThread=True)
    def set_binning_offaxisholo(self, binning: int) -> Dict[str, Any]:
        """Set binning factor (1, 2, 4, etc.)"""
        return self.set_parameters_offaxisholo({"binning": binning})

    @APIExport(runOnUIThread=True)
    def set_cc_roi_offaxisholo(self, center_x: int, center_y: int, size_x: int = None, size_y: int = None) -> Dict[str, Any]:
        """
        Set cross-correlation (sideband) center and size in FFT space
        
        Args:
            center_x: X coordinate of sideband center in FFT pixels
            center_y: Y coordinate of sideband center in FFT pixels  
            size_x: Width of crop region (defaults to 2*cc_radius if None)
            size_y: Height of crop region (defaults to size_x if None)
        
        Returns:
            Updated parameters dictionary
        """
        params = {"cc_center": [center_x, center_y]}
        if size_x is not None:
            if size_y is None:
                size_y = size_x
            params["cc_size"] = [size_x, size_y]
            params["cc_radius"] = max(size_x, size_y) // 2
        return self.set_parameters_offaxisholo(params)

    @APIExport(runOnUIThread=True)
    def set_apodization_offaxisholo(self, enabled: bool, window_type: str = "tukey", alpha: float = 0.1) -> Dict[str, Any]:
        """
        Set apodization (edge damping) parameters
        
        Args:
            enabled: Enable/disable apodization
            window_type: "tukey", "hann", "hamming", "blackman"
            alpha: Tukey window parameter (0=rect, 1=hann)
        """
        return self.set_parameters_offaxisholo({
            "apodization_enabled": enabled,
            "apodization_type": window_type,
            "apodization_alpha": alpha
        })

    @APIExport(runOnUIThread=True)
    def set_update_freq_offaxisholo(self, update_freq: float) -> Dict[str, Any]:
        """Set processing frequency in Hz"""
        self.set_update_rate(update_freq)
        return self._params.to_dict()

    # =========================
    # API: Processing Control
    # =========================
    @APIExport(runOnUIThread=True)
    def get_state_offaxisholo(self) -> Dict[str, Any]:
        """Get current processing state"""
        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def start_processing_offaxisholo(self) -> Dict[str, Any]:
        """
        Start off-axis hologram processing
        
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

        self._logger.info(f"Started off-axis hologram processing ({self._params.update_freq} Hz)")
        self._emit_state_changed()

        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def stop_processing_offaxisholo(self) -> Dict[str, Any]:
        """Stop off-axis hologram processing"""
        with self._processing_lock:
            self._state.is_processing = False
            self._state.is_paused = False
            self._state.is_streaming_fft = False
            self._state.is_streaming_magnitude = False
            self._state.is_streaming_phase = False

        self._logger.info("Stopped off-axis hologram processing")
        self._emit_state_changed()

        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def pause_processing_offaxisholo(self) -> Dict[str, Any]:
        """Pause processing - will continuously process last frame"""
        with self._processing_lock:
            if self._state.is_processing:
                self._state.is_paused = True

        self._logger.info("Paused off-axis hologram processing")
        self._emit_state_changed()

        return self._state.to_dict()

    @APIExport(runOnUIThread=True)
    def resume_processing_offaxisholo(self) -> Dict[str, Any]:
        """Resume processing from paused state"""
        with self._processing_lock:
            if self._state.is_processing:
                self._state.is_paused = False

        self._logger.info("Resumed off-axis hologram processing")
        self._emit_state_changed()

        return self._state.to_dict()

    # =========================
    # API: MJPEG Streaming
    # =========================
    def _create_mjpeg_response(self, target_queue, stream_flag_name):
        """Helper to create MJPEG streaming response"""
        try:
            from fastapi.responses import StreamingResponse
        except ImportError:
            return {"status": "error", "message": "FastAPI not available"}

        if not hasCV2:
            return {"status": "error", "message": "opencv-python required for MJPEG streaming"}

        def frame_generator():
            """Generator that yields MJPEG frames."""
            try:
                while getattr(self._state, stream_flag_name):
                    try:
                        frame = target_queue.get(timeout=1.0)
                        if frame:
                            yield frame
                    except queue.Empty:
                        continue
            except GeneratorExit:
                self._logger.info(f"MJPEG stream {stream_flag_name} connection closed")
                with self._processing_lock:
                    setattr(self._state, stream_flag_name, False)
                self._emit_state_changed()
            except Exception as e:
                self._logger.error(f"Error in MJPEG frame generator: {e}")

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

    @APIExport(runOnUIThread=False)
    def mjpeg_stream_offaxisholo_fft(self, startStream: bool = True, jpeg_quality: int = 85):
        """
        MJPEG stream of FFT magnitude (log scale, viridis colormap)
        
        Args:
            startStream: Start or stop the stream
            jpeg_quality: JPEG compression quality (0-100)
        """
        if not startStream:
            with self._processing_lock:
                self._state.is_streaming_fft = False
            while not self._mjpeg_queue_fft.empty():
                try:
                    self._mjpeg_queue_fft.get_nowait()
                except queue.Empty:
                    break
            return {"status": "success", "message": "FFT stream stopped"}

        self._jpeg_quality = max(0, min(100, jpeg_quality))
        
        with self._processing_lock:
            self._state.is_streaming_fft = True

        if not self._state.is_processing:
            self.start_processing_offaxisholo()

        self._logger.info(f"Started FFT MJPEG stream (quality={self._jpeg_quality})")
        self._emit_state_changed()

        return self._create_mjpeg_response(self._mjpeg_queue_fft, 'is_streaming_fft')

    @APIExport(runOnUIThread=False)
    def mjpeg_stream_offaxisholo_mag(self, startStream: bool = True, jpeg_quality: int = 85):
        """
        MJPEG stream of reconstructed magnitude
        
        Args:
            startStream: Start or stop the stream
            jpeg_quality: JPEG compression quality (0-100)
        """
        if not startStream:
            with self._processing_lock:
                self._state.is_streaming_magnitude = False
            while not self._mjpeg_queue_magnitude.empty():
                try:
                    self._mjpeg_queue_magnitude.get_nowait()
                except queue.Empty:
                    break
            return {"status": "success", "message": "Magnitude stream stopped"}

        self._jpeg_quality = max(0, min(100, jpeg_quality))
        
        with self._processing_lock:
            self._state.is_streaming_magnitude = True

        if not self._state.is_processing:
            self.start_processing_offaxisholo()

        self._logger.info(f"Started magnitude MJPEG stream (quality={self._jpeg_quality})")
        self._emit_state_changed()

        return self._create_mjpeg_response(self._mjpeg_queue_magnitude, 'is_streaming_magnitude')

    @APIExport(runOnUIThread=False)
    def mjpeg_stream_offaxisholo_phase(self, startStream: bool = True, jpeg_quality: int = 85):
        """
        MJPEG stream of unwrapped phase (twilight colormap)
        
        Args:
            startStream: Start or stop the stream
            jpeg_quality: JPEG compression quality (0-100)
        """
        if not startStream:
            with self._processing_lock:
                self._state.is_streaming_phase = False
            while not self._mjpeg_queue_phase.empty():
                try:
                    self._mjpeg_queue_phase.get_nowait()
                except queue.Empty:
                    break
            return {"status": "success", "message": "Phase stream stopped"}

        self._jpeg_quality = max(0, min(100, jpeg_quality))
        
        with self._processing_lock:
            self._state.is_streaming_phase = True

        if not self._state.is_processing:
            self.start_processing_offaxisholo()

        self._logger.info(f"Started phase MJPEG stream (quality={self._jpeg_quality})")
        self._emit_state_changed()

        return self._create_mjpeg_response(self._mjpeg_queue_phase, 'is_streaming_phase')

    # =========================
    # Helper methods
    # =========================
    def _ensure_camera_running(self):
        """Ensure camera is running, start if necessary"""
        try:
            detector = self._master.detectorsManager[self.camera]
            if hasattr(detector, '_running') and not detector._running:
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
            "has_fft": result.get('fft_magnitude') is not None if result else False,
            "has_magnitude": result.get('crop_magnitude') is not None if result else False,
            "has_phase": result.get('unwrapped_phase') is not None if result else False,
        }

    # =========================
    # Legacy GUI compatibility (if not headless)
    # =========================
    def setShowOffAxisHolo(self, enabled):
        """Legacy: Show or hide off-axis hologram processing"""
        if enabled:
            self.start_processing_offaxisholo()
            if not IS_HEADLESS and hasattr(self, '_widget'):
                try:
                    self._widget.createPointsLayer()
                except:
                    pass
        else:
            self.stop_processing_offaxisholo()

    def changeRate(self, updateRate):
        """Legacy: Change update rate"""
        if updateRate == "" or updateRate <= 0:
            updateRate = 1
        self.set_parameters_offaxisholo({"update_freq": updateRate})

    def offAxisValueChanged(self, magnitude):
        """Legacy: Change off-axis propagation distance"""
        dz = magnitude * 1e-4  # Convert to meters
        self.set_dz_offaxisholo(dz)

    def selectCCCenter(self):
        """Legacy: Select cross-correlation center from GUI"""
        if IS_HEADLESS or not hasattr(self, '_widget'):
            return

        try:
            cc_center = self._widget.getCCCenterFromNapari()
            cc_radius = self._widget.getCCRadius()

            if cc_radius is None or cc_radius < 50:
                cc_radius = 100

            self.set_cc_roi_offaxisholo(cc_center[0], cc_center[1], cc_radius * 2, cc_radius * 2)
        except Exception as e:
            self._logger.warning(f"Failed to get CC center from GUI: {e}")

    def updateCCCenter(self):
        """Legacy: Update CC center from text fields"""
        if IS_HEADLESS or not hasattr(self, '_widget'):
            return

        try:
            centerX = int(self._widget.textEditCCCenterX.text())
            centerY = int(self._widget.textEditCCCenterY.text())
            self.set_cc_roi_offaxisholo(centerX, centerY, self._params.cc_radius * 2, self._params.cc_radius * 2)
        except Exception as e:
            self._logger.warning(f"Failed to update CC center: {e}")

    def updateCCRadius(self):
        """Legacy: Update CC radius from text field"""
        if IS_HEADLESS or not hasattr(self, '_widget'):
            return

        try:
            radius = int(self._widget.textEditCCRadius.text())
            self.set_cc_params(self._params.cc_center, radius)
        except Exception as e:
            self._logger.warning(f"Failed to update CC radius: {e}")

    def displayImage(self, im, name):
        """Legacy: Display image in napari widget"""
        if IS_HEADLESS:
            return

        if im.dtype == complex or np.iscomplexobj(im):
            self._widget.setImage(np.abs(im), name + "_abs")
            self._widget.setImage(np.angle(im), name + "_angle")
        else:
            self._widget.setImage(np.abs(im), name)

    @APIExport(runOnUIThread=True)
    def displayImageNapari(self, im, name):
        """API: Display image in napari"""
        self.displayImage(np.array(im), name)

    def _setup_legacy_gui(self):
        """Setup legacy GUI connections"""
        if IS_HEADLESS or not hasattr(self, '_widget'):
            return

        try:
            self._widget.sigShowOffAxisToggled.connect(self.setShowOffAxisHolo)
            self._widget.sigUpdateRateChanged.connect(self.changeRate)
            self._widget.sigOffAxisSliderValueChanged.connect(self.offAxisValueChanged)

            if hasattr(self._widget, 'btnSelectCCCenter'):
                self._widget.btnSelectCCCenter.clicked.connect(self.selectCCCenter)
            if hasattr(self._widget, 'textEditCCCenterX'):
                self._widget.textEditCCCenterX.textChanged.connect(self.updateCCCenter)
            if hasattr(self._widget, 'textEditCCCenterY'):
                self._widget.textEditCCCenterY.textChanged.connect(self.updateCCCenter)
            if hasattr(self._widget, 'textEditCCRadius'):
                self._widget.textEditCCRadius.textChanged.connect(self.updateCCRadius)

            # Set initial values
            if hasattr(self._widget, 'getUpdateRate'):
                self.changeRate(self._widget.getUpdateRate())
            if hasattr(self._widget, 'getShowOffAxisHoloChecked'):
                self.setShowOffAxisHolo(self._widget.getShowOffAxisHoloChecked())

            # Connect signal for displaying processed images
            self.sigHoloImageComputed.connect(self.displayImage)

        except Exception as e:
            self._logger.warning(f"Could not setup all GUI connections: {e}")


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
