"""
CompositeController - Multi-illumination composite acquisition controller

This controller enables sequential acquisition of images under different illumination
states (e.g., laser 488, laser 635, LED), fuses them into a single composite RGB JPEG,
and streams the result to the frontend.

Architecture follows the InLineHoloController pattern:
- Background worker loop for non-blocking acquisition
- Frame queue for decoupled processing
- MJPEG streaming for live preview
- RESTful API for control

Author: ImSwitch developers
"""

import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time
import traceback
import threading
import queue

try:
    import cv2
    hasCV2 = True
except ImportError:
    hasCV2 = False

from imswitch.imcommon.model import initLogger, APIExport
from imswitch.imcommon.framework import Signal
from ..basecontrollers import ImConWidgetController


# =========================
# Dataclasses (API-stable)
# =========================

@dataclass
class IlluminationStep:
    """Single illumination step configuration"""
    illumination: str  # Name of illumination source (e.g., "laser488", "LED")
    intensity: float = 0.5  # Intensity value (0.0-1.0 normalized or device-specific)
    exposure_ms: Optional[float] = None  # Optional exposure override in ms
    settle_ms: float = 10.0  # Settle time after setting illumination
    enabled: bool = True  # Whether this step is active

    def to_dict(self) -> Dict[str, Any]:
        return {
            "illumination": self.illumination,
            "intensity": self.intensity,
            "exposure_ms": self.exposure_ms,
            "settle_ms": self.settle_ms,
            "enabled": self.enabled,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'IlluminationStep':
        return IlluminationStep(
            illumination=d.get("illumination", ""),
            intensity=d.get("intensity", 0.5),
            exposure_ms=d.get("exposure_ms"),
            settle_ms=d.get("settle_ms", 10.0),
            enabled=d.get("enabled", True),
        )


@dataclass
class CompositeParams:
    """Composite acquisition parameters"""
    steps: List[IlluminationStep] = field(default_factory=list)
    mapping: Dict[str, str] = field(default_factory=lambda: {"R": "", "G": "", "B": ""})
    fps_target: float = 5.0  # Target frames per second
    jpeg_quality: int = 85  # JPEG compression quality (0-100)
    normalize_channels: bool = True  # Normalize each channel before fusion
    auto_exposure: bool = False  # Use per-step exposure overrides

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "mapping": self.mapping,
            "fps_target": self.fps_target,
            "jpeg_quality": self.jpeg_quality,
            "normalize_channels": self.normalize_channels,
            "auto_exposure": self.auto_exposure,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'CompositeParams':
        steps = [IlluminationStep.from_dict(s) for s in d.get("steps", [])]
        return CompositeParams(
            steps=steps,
            mapping=d.get("mapping", {"R": "", "G": "", "B": ""}),
            fps_target=d.get("fps_target", 5.0),
            jpeg_quality=d.get("jpeg_quality", 85),
            normalize_channels=d.get("normalize_channels", True),
            auto_exposure=d.get("auto_exposure", False),
        )


@dataclass
class CompositeState:
    """Composite acquisition state"""
    is_running: bool = False
    is_streaming: bool = False
    current_step: int = 0
    cycle_count: int = 0
    last_cycle_time_ms: float = 0.0
    average_fps: float = 0.0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "is_streaming": self.is_streaming,
            "current_step": self.current_step,
            "cycle_count": self.cycle_count,
            "last_cycle_time_ms": self.last_cycle_time_ms,
            "average_fps": self.average_fps,
            "error_message": self.error_message,
        }


class CompositeController(ImConWidgetController):
    """
    Multi-illumination composite acquisition controller.
    
    Features:
    - Sequential acquisition with configurable illumination steps
    - RGB channel fusion from grayscale frames
    - MJPEG streaming of composite image
    - Non-blocking acquisition via worker thread
    - RESTful API control
    
    Architecture:
    - Worker thread handles acquisition loop
    - Each cycle: iterate steps → acquire frame → store → fuse → encode → publish
    - MJPEG queue for streaming to frontend
    """

    sigCompositeImageReady = Signal(np.ndarray, dict)  # (image, metadata)
    sigCompositeStateChanged = Signal(object)  # state_dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # Get camera from setup or use first available detector
        self.camera = None
        try:
            all_detectors = self._master.detectorsManager.getAllDeviceNames()
            if all_detectors:
                self.camera = all_detectors[0]
                self._logger.info(f"CompositeController using detector: {self.camera}")
            else:
                self._logger.error("No detectors available for CompositeController")
        except Exception as e:
            self._logger.error(f"Failed to get detector list: {e}")

        # Initialize parameters with default 3-channel setup
        self._params = CompositeParams()
        self._state = CompositeState()
        self._processing_lock = threading.Lock()

        # Frame storage for current acquisition cycle
        self._channel_frames: Dict[str, np.ndarray] = {}

        # MJPEG streaming queue
        self._mjpeg_queue = queue.Queue(maxsize=10)

        # Worker thread for acquisition
        self._acquisition_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Timing for FPS calculation
        self._cycle_times: List[float] = []
        self._max_cycle_times = 30  # Rolling average window

        # Illumination managers cache
        self._lasers_manager = None
        self._leds_manager = None
        self._init_illumination_managers()

        self._logger.info("CompositeController initialized successfully")

    def _init_illumination_managers(self):
        """Initialize references to illumination managers"""
        try:
            if hasattr(self._master, 'lasersManager'):
                self._lasers_manager = self._master.lasersManager
                self._logger.info(f"Found lasers: {self._lasers_manager.getAllDeviceNames()}")
        except Exception as e:
            self._logger.debug(f"No lasersManager available: {e}")

        try:
            if hasattr(self._master, 'LEDsManager'):
                self._leds_manager = self._master.LEDsManager
                self._logger.info(f"Found LEDs: {self._leds_manager.getAllDeviceNames()}")
        except Exception as e:
            self._logger.debug(f"No LEDsManager available: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_composite()
        if hasattr(super(), '__del__'):
            super().__del__()

    # =========================
    # Illumination Control
    # =========================

    def _get_all_illumination_sources(self) -> List[str]:
        """Get list of all available illumination sources"""
        sources = []

        if self._lasers_manager:
            try:
                sources.extend(self._lasers_manager.getAllDeviceNames())
            except Exception as e:
                self._logger.debug(f"Error getting laser names: {e}")

        if self._leds_manager:
            try:
                sources.extend(self._leds_manager.getAllDeviceNames())
            except Exception as e:
                self._logger.debug(f"Error getting LED names: {e}")

        return sources

    def _set_illumination(self, source_name: str, intensity: float, enabled: bool = True):
        """
        Set illumination source state and intensity.
        
        Args:
            source_name: Name of illumination source
            intensity: Intensity value (device-specific units)
            enabled: Whether to enable (True) or disable (False)
        """
        # Try lasers first
        if self._lasers_manager:
            try:
                laser_names = self._lasers_manager.getAllDeviceNames()
                if source_name in laser_names:
                    laser = self._lasers_manager[source_name]
                    if enabled:
                        laser.setValue(intensity*laser.valueRangeMax)
                        laser.setEnabled(True)
                    else:
                        laser.setEnabled(False)
                        laser.setValue(0)
                    return
            except Exception as e:
                self._logger.debug(f"Error setting laser {source_name}: {e}")

        # Try LEDs
        if self._leds_manager:
            try:
                led_names = self._leds_manager.getAllDeviceNames()
                if source_name in led_names:
                    led = self._leds_manager[source_name]
                    if enabled:
                        led.setValue(intensity*255)
                        led.setEnabled(True)
                    else:
                        led.setEnabled(False)
                        led.setValue(0)
                    return
            except Exception as e:
                self._logger.debug(f"Error setting LED {source_name}: {e}")

        self._logger.warning(f"Illumination source not found: {source_name}")

    def _turn_off_all_illumination(self):
        """Turn off all illumination sources"""
        sources = self._get_all_illumination_sources()
        for source in sources:
            try:
                self._set_illumination(source, 0, enabled=False)
            except Exception as e:
                self._logger.debug(f"Error turning off {source}: {e}")

    # =========================
    # Camera Control
    # =========================

    def _ensure_camera_running(self):
        """Ensure camera is running"""
        if not self.camera:
            return False
        try:
            detector = self._master.detectorsManager[self.camera]
            if not detector._running:
                self._logger.info(f"Starting camera {self.camera}")
                detector.startAcquisition()
            return True
        except Exception as e:
            self._logger.error(f"Failed to start camera: {e}")
            return False

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera"""
        if not self.camera:
            return None
        try:
            detector = self._master.detectorsManager[self.camera]
            frame = detector.getLatestFrame()
            return frame
        except Exception as e:
            self._logger.error(f"Failed to capture frame: {e}")
            return None

    def _set_exposure(self, exposure_ms: float):
        """Set camera exposure time"""
        if not self.camera:
            return
        try:
            detector = self._master.detectorsManager[self.camera]
            if hasattr(detector, 'setExposureTime'):
                detector.setExposureTime(exposure_ms / 1000.0)  # Convert to seconds if needed
        except Exception as e:
            self._logger.debug(f"Could not set exposure: {e}")

    # =========================
    # Image Processing & Fusion
    # =========================

    def _extract_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Extract grayscale from image (handles RGB and grayscale input)"""
        if image is None:
            return None

        if len(image.shape) == 3 and image.shape[2] >= 3:
            # RGB image - convert to grayscale using luminance
            gray = (0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2])
            return gray.astype(np.float32)
        elif len(image.shape) == 2:
            return image.astype(np.float32)
        else:
            return image.astype(np.float32)

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize channel to 0-255 range"""
        if channel is None:
            return None

        vmin = np.min(channel)
        vmax = np.max(channel)

        if vmax > vmin:
            normalized = (channel - vmin) / (vmax - vmin) * 255.0
        else:
            normalized = np.zeros_like(channel)

        return normalized.astype(np.uint8)

    def _fuse_channels(self, frames: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Fuse grayscale channel frames into RGB composite.
        
        Args:
            frames: Dictionary mapping illumination source names to grayscale frames
            
        Returns:
            RGB composite image as numpy array (H, W, 3) uint8
        """
        if not frames:
            return None

        # Get image dimensions from first frame
        first_frame = next(iter(frames.values()))
        if first_frame is None:
            return None

        height, width = first_frame.shape[:2]

        # Create empty RGB channels
        r_channel = np.zeros((height, width), dtype=np.float32)
        g_channel = np.zeros((height, width), dtype=np.float32)
        b_channel = np.zeros((height, width), dtype=np.float32)

        # Map frames to RGB channels based on mapping configuration
        mapping = self._params.mapping

        if mapping.get("R") and mapping["R"] in frames:
            r_channel = self._extract_grayscale(frames[mapping["R"]])

        if mapping.get("G") and mapping["G"] in frames:
            g_channel = self._extract_grayscale(frames[mapping["G"]])

        if mapping.get("B") and mapping["B"] in frames:
            b_channel = self._extract_grayscale(frames[mapping["B"]])

        # Normalize channels if enabled
        if self._params.normalize_channels:
            if r_channel is not None:
                r_channel = self._normalize_channel(r_channel)
            if g_channel is not None:
                g_channel = self._normalize_channel(g_channel)
            if b_channel is not None:
                b_channel = self._normalize_channel(b_channel)
        else:
            # Simple clip to 0-255
            r_channel = np.clip(r_channel, 0, 255).astype(np.uint8)
            g_channel = np.clip(g_channel, 0, 255).astype(np.uint8)
            b_channel = np.clip(b_channel, 0, 255).astype(np.uint8)

        # Stack channels into RGB image
        composite = np.stack([r_channel, g_channel, b_channel], axis=2).astype(np.uint8)

        return composite

    def _encode_jpeg(self, image: np.ndarray) -> Optional[bytes]:
        """Encode image as JPEG"""
        if not hasCV2 or image is None:
            return None

        try:
            # Ensure image is uint8
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._params.jpeg_quality]
            success, encoded = cv2.imencode('.jpg', image_bgr, encode_params)

            if success:
                return encoded.tobytes()
        except Exception as e:
            self._logger.error(f"JPEG encoding error: {e}")

        return None

    # =========================
    # Acquisition Loop
    # =========================

    def _acquisition_worker(self):
        """
        Main acquisition worker loop.
        
        Executes continuously while running:
        1. For each enabled step: set illumination → wait settle → capture frame
        2. After all steps: fuse frames → encode JPEG → add to stream
        3. Calculate timing statistics
        """
        self._logger.info("Composite acquisition worker started")

        while not self._stop_event.is_set():
            cycle_start = time.time()

            try:
                # Clear channel frames for new cycle
                self._channel_frames.clear()

                # Get enabled steps
                enabled_steps = [s for s in self._params.steps if s.enabled]

                if not enabled_steps:
                    self._logger.warning("No enabled illumination steps")
                    time.sleep(0.1)
                    continue

                # Execute each illumination step
                for i, step in enumerate(enabled_steps):
                    if self._stop_event.is_set():
                        break

                    with self._processing_lock:
                        self._state.current_step = i

                    # Turn off all illumination first
                    self._turn_off_all_illumination()

                    # Set exposure if specified
                    if step.exposure_ms and self._params.auto_exposure:
                        self._set_exposure(step.exposure_ms)

                    # Set illumination for this step
                    self._set_illumination(step.illumination, step.intensity, enabled=True)

                    # Wait for settle time
                    if step.settle_ms > 0:
                        time.sleep(step.settle_ms / 1000.0)

                    # Capture frame
                    frame = self._capture_frame()

                    if frame is not None:
                        self._channel_frames[step.illumination] = frame.copy()
                    else:
                        self._logger.warning(f"No frame captured for step {step.illumination}")

                # Turn off illumination after acquisition
                self._turn_off_all_illumination()

                if self._stop_event.is_set():
                    break

                # Fuse channels into composite
                composite = self._fuse_channels(self._channel_frames)

                if composite is not None:
                    # Build metadata
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "mapping": self._params.mapping,
                        "steps": [s.to_dict() for s in enabled_steps],
                        "cycle_count": self._state.cycle_count,
                    }

                    # Emit signal for other components
                    self.sigCompositeImageReady.emit(composite, metadata)

                    # Add to MJPEG stream if streaming
                    if self._state.is_streaming:
                        self._add_to_mjpeg_stream(composite)

                    with self._processing_lock:
                        self._state.cycle_count += 1

                # Calculate timing
                cycle_end = time.time()
                cycle_time_ms = (cycle_end - cycle_start) * 1000.0

                with self._processing_lock:
                    self._state.last_cycle_time_ms = cycle_time_ms

                    # Update rolling average FPS
                    self._cycle_times.append(cycle_time_ms)
                    if len(self._cycle_times) > self._max_cycle_times:
                        self._cycle_times.pop(0)

                    if self._cycle_times:
                        avg_time = sum(self._cycle_times) / len(self._cycle_times)
                        self._state.average_fps = 1000.0 / avg_time if avg_time > 0 else 0.0

                # Rate limiting - wait to achieve target FPS
                target_cycle_time = 1000.0 / self._params.fps_target
                if cycle_time_ms < target_cycle_time:
                    sleep_time = (target_cycle_time - cycle_time_ms) / 1000.0
                    time.sleep(sleep_time)

            except Exception as e:
                self._logger.error(f"Error in acquisition cycle: {e}")
                self._logger.debug(traceback.format_exc())
                with self._processing_lock:
                    self._state.error_message = str(e)
                time.sleep(0.1)  # Prevent tight loop on errors

        # Cleanup on exit
        self._turn_off_all_illumination()
        self._logger.info("Composite acquisition worker stopped")

    def _add_to_mjpeg_stream(self, image: np.ndarray):
        """Add composite image to MJPEG streaming queue"""
        jpeg_bytes = self._encode_jpeg(image)

        if jpeg_bytes:
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

    # =========================
    # API: Control
    # =========================

    @APIExport(runOnUIThread=True)
    def get_illumination_sources_composite(self) -> List[str]:
        """
        Get list of all available illumination sources.
        
        Returns:
            List of illumination source names (lasers + LEDs)
            
        Example response:
            ["laser488", "laser635", "LED_white", "LED_blue"]
        """
        return self._get_all_illumination_sources()

    @APIExport(runOnUIThread=True)
    def get_parameters_composite(self) -> Dict[str, Any]:
        """
        Get current composite acquisition parameters.
        
        Returns:
            Dictionary with all parameters including steps and mapping
            
        Example response:
            {
                "steps": [
                    {"illumination": "laser488", "intensity": 0.3, "exposure_ms": 50, "settle_ms": 10, "enabled": true},
                    {"illumination": "laser635", "intensity": 0.2, "exposure_ms": 80, "settle_ms": 10, "enabled": true}
                ],
                "mapping": {"R": "laser635", "G": "laser488", "B": ""},
                "fps_target": 5.0,
                "jpeg_quality": 85,
                "normalize_channels": true,
                "auto_exposure": false
            }
        """
        return self._params.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def set_parameters_composite(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update composite acquisition parameters.
        Accepts JSON body or JSON-encoded query parameter.
        
        Args:
            params: Dictionary with parameter updates or JSON string (partial update supported)
            
        Returns:
            Updated parameters dictionary
            
        Example request:
            {
                "steps": [
                    {"illumination": "laser488", "intensity": 0.5, "settle_ms": 20},
                    {"illumination": "LED_white", "intensity": 1.0}
                ],
                "mapping": {"R": "", "G": "laser488", "B": "LED_white"},
                "fps_target": 3.0
            }
        """
        import json
        
        # Handle JSON-string from query parameter
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception as e:
                self._logger.warning(f"Failed to parse params JSON string: {e}")
                params = {}
        
        if params is None:
            params = {}
        
        self._logger.info(f"set_parameters_composite received: {params}")
        
        with self._processing_lock:
            # Update steps if provided
            if "steps" in params:
                self._params.steps = [IlluminationStep.from_dict(s) for s in params["steps"]]
                self._logger.info(f"Updated steps: {len(self._params.steps)} steps")

            # Update mapping if provided
            if "mapping" in params:
                self._params.mapping.update(params["mapping"])
                self._logger.info(f"Updated mapping: {self._params.mapping}")

            # Update scalar parameters
            if "fps_target" in params:
                self._params.fps_target = max(0.1, min(30.0, float(params["fps_target"])))

            if "jpeg_quality" in params:
                self._params.jpeg_quality = max(1, min(100, int(params["jpeg_quality"])))

            if "normalize_channels" in params:
                self._params.normalize_channels = bool(params["normalize_channels"])

            if "auto_exposure" in params:
                self._params.auto_exposure = bool(params["auto_exposure"])

        self._emit_state_changed()
        return self._params.to_dict()

    @APIExport(runOnUIThread=True)
    def get_state_composite(self) -> Dict[str, Any]:
        """
        Get current composite acquisition state.
        
        Returns:
            State dictionary with running status and statistics
            
        Example response:
            {
                "is_running": true,
                "is_streaming": true,
                "current_step": 1,
                "cycle_count": 42,
                "last_cycle_time_ms": 180.5,
                "average_fps": 4.8,
                "error_message": ""
            }
        """
        return self._state.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def start_composite(self) -> Dict[str, Any]:
        """
        Start composite acquisition.
        
        Begins the acquisition loop that cycles through illumination steps,
        captures frames, fuses them into composite, and streams results.
        
        Returns:
            Current state dictionary
        """
        if self._state.is_running:
            return self._state.to_dict()

        # Ensure camera is ready
        if not self._ensure_camera_running():
            self._state.error_message = "Failed to start camera"
            return self._state.to_dict()

        # Reset state
        with self._processing_lock:
            self._state.is_running = True
            self._state.cycle_count = 0
            self._state.error_message = ""
            self._cycle_times.clear()

        # Start acquisition thread
        self._stop_event.clear()
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_worker,
            name="CompositeAcquisitionWorker",
            daemon=True
        )
        self._acquisition_thread.start()

        self._logger.info("Started composite acquisition")
        self._emit_state_changed()

        return self._state.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def stop_composite(self) -> Dict[str, Any]:
        """
        Stop composite acquisition.
        
        Stops the acquisition loop and turns off all illumination.
        
        Returns:
            Current state dictionary
        """
        if not self._state.is_running:
            return self._state.to_dict()

        # Signal worker to stop
        self._stop_event.set()

        # Wait for thread to finish
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=2.0)

        # Update state
        with self._processing_lock:
            self._state.is_running = False
            self._state.is_streaming = False

        # Ensure illumination is off
        self._turn_off_all_illumination()

        # Clear MJPEG queue
        while not self._mjpeg_queue.empty():
            try:
                self._mjpeg_queue.get_nowait()
            except queue.Empty:
                break

        self._logger.info("Stopped composite acquisition")
        self._emit_state_changed()

        return self._state.to_dict()

    @APIExport(runOnUIThread=False)
    def mjpeg_stream_composite(self, startStream: bool = True, jpeg_quality: int = None):
        """
        HTTP endpoint for MJPEG streaming of composite images.
        
        Args:
            startStream: Whether to start streaming (True) or stop (False)
            jpeg_quality: Optional JPEG compression quality override (0-100)
        
        Returns:
            StreamingResponse with MJPEG data or status message
            
        Example:
            GET /CompositeController/mjpeg_stream_composite?startStream=true
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
            self._logger.info("Stopped composite MJPEG stream")
            self._emit_state_changed()
            return {"status": "success", "message": "stream stopped"}

        # Update JPEG quality if provided
        if jpeg_quality is not None:
            self._params.jpeg_quality = max(1, min(100, jpeg_quality))

        # Start streaming
        with self._processing_lock:
            self._state.is_streaming = True

        # Ensure acquisition is running
        if not self._state.is_running:
            self.start_composite()

        self._logger.info(f"Started composite MJPEG stream (quality={self._params.jpeg_quality})")
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
                self._logger.info("Composite MJPEG stream connection closed by client")
                with self._processing_lock:
                    self._state.is_streaming = False
                self._emit_state_changed()
            except Exception as e:
                self._logger.error(f"Error in composite MJPEG frame generator: {e}")

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

    @APIExport(runOnUIThread=True, requestType="POST")
    def add_step_composite(self, illumination: str, intensity: float = 0.5,
                           exposure_ms: float = None, settle_ms: float = 10.0) -> Dict[str, Any]:
        """
        Add a new illumination step to the sequence.
        
        Args:
            illumination: Name of illumination source
            intensity: Intensity value
            exposure_ms: Optional exposure override
            settle_ms: Settle time in ms
            
        Returns:
            Updated parameters dictionary
        """
        step = IlluminationStep(
            illumination=illumination,
            intensity=intensity,
            exposure_ms=exposure_ms,
            settle_ms=settle_ms,
            enabled=True
        )

        with self._processing_lock:
            self._params.steps.append(step)

        self._emit_state_changed()
        return self._params.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def remove_step_composite(self, index: int) -> Dict[str, Any]:
        """
        Remove an illumination step by index.
        
        Args:
            index: Index of step to remove (0-based)
            
        Returns:
            Updated parameters dictionary
        """
        with self._processing_lock:
            if 0 <= index < len(self._params.steps):
                self._params.steps.pop(index)

        self._emit_state_changed()
        return self._params.to_dict()

    @APIExport(runOnUIThread=True, requestType="POST")
    def set_mapping_composite(self, r_source: str = "", g_source: str = "",
                              b_source: str = "") -> Dict[str, Any]:
        """
        Set RGB channel mapping.
        
        Args:
            r_source: Illumination source to map to Red channel
            g_source: Illumination source to map to Green channel
            b_source: Illumination source to map to Blue channel
            
        Returns:
            Updated parameters dictionary
        """
        with self._processing_lock:
            self._params.mapping = {
                "R": r_source,
                "G": g_source,
                "B": b_source
            }

        self._emit_state_changed()
        return self._params.to_dict()

    @APIExport(runOnUIThread=True)
    def capture_single_composite(self) -> Dict[str, Any]:
        """
        Capture a single composite image (one-shot mode).
        
        Executes one acquisition cycle and returns the composite image
        encoded as base64 JPEG.
        
        Returns:
            Dictionary with status and base64-encoded JPEG image
        """
        import base64

        if not self._ensure_camera_running():
            return {"status": "error", "message": "Camera not available"}

        try:
            # Clear channel frames
            self._channel_frames.clear()

            # Get enabled steps
            enabled_steps = [s for s in self._params.steps if s.enabled]

            if not enabled_steps:
                return {"status": "error", "message": "No enabled illumination steps"}

            # Execute each step
            for step in enabled_steps:
                self._turn_off_all_illumination()

                if step.exposure_ms and self._params.auto_exposure:
                    self._set_exposure(step.exposure_ms)

                self._set_illumination(step.illumination, step.intensity, enabled=True)

                if step.settle_ms > 0:
                    time.sleep(step.settle_ms / 1000.0)

                frame = self._capture_frame()

                if frame is not None:
                    self._channel_frames[step.illumination] = frame.copy()

            # Turn off illumination
            self._turn_off_all_illumination()

            # Fuse channels
            composite = self._fuse_channels(self._channel_frames)

            if composite is None:
                return {"status": "error", "message": "Failed to create composite"}

            # Encode as JPEG
            jpeg_bytes = self._encode_jpeg(composite)

            if jpeg_bytes is None:
                return {"status": "error", "message": "Failed to encode JPEG"}

            # Build metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "mapping": self._params.mapping,
                "steps": [s.to_dict() for s in enabled_steps],
                "image_shape": list(composite.shape),
            }

            return {
                "status": "success",
                "image_base64": base64.b64encode(jpeg_bytes).decode('ascii'),
                "metadata": metadata
            }

        except Exception as e:
            self._logger.error(f"Error in single capture: {e}")
            return {"status": "error", "message": str(e)}

    def _emit_state_changed(self):
        """Emit state changed signal"""
        self.sigCompositeStateChanged.emit(self._state.to_dict())


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
