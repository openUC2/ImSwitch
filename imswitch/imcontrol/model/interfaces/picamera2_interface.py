"""
Raspberry Pi Camera (Picamera2) Interface
Provides a callback-based interface similar to HIK camera driver for scientific imaging.
Supports hardware encoding options while maintaining scientific camera features.
"""

from logging import raiseExceptions
import collections
import numpy as np
import time
import cv2
from imswitch.imcommon.model import initLogger
from typing import List, Optional, Union, Callable
import threading

# Try to import picamera2, fall back to mock if not available
try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder, MJPEGEncoder
    from picamera2.outputs import FileOutput
    import libcamera
    PICAMERA2_AVAILABLE = True
except ImportError as e:
    print(f"Picamera2 not available: {e}")
    PICAMERA2_AVAILABLE = False


class CameraPicamera2:
    """
    Raspberry Pi camera wrapper using Picamera2 with callback-based frame delivery.
    Provides interface compatible with HIK camera driver for seamless integration.
    """
    
    # Class-level tracking of opened cameras
    _opened_cameras = set()

    def __init__(
        self, 
        cameraNo: int = 0,
        exposure_time: float = 10000,  # microseconds
        gain: float = 1.0,
        frame_rate: int = 30,
        blacklevel: int = 0,
        isRGB: bool = True,
        binning: int = 1,
        flipImage: tuple = (False, False),
        resolution: tuple = (640, 480),
        use_video_mode: bool = True
    ):
        """
        Initialize Raspberry Pi camera.
        
        Args:
            cameraNo: Camera index (0 for first camera)
            exposure_time: Exposure time in microseconds
            gain: Analogue gain (1.0-16.0)
            frame_rate: Target frame rate
            blacklevel: Black level offset (currently not used on RPi)
            isRGB: True for RGB, False for mono
            binning: Binning factor (simulated via resolution reduction)
            flipImage: (flipY, flipX) tuple
            resolution: (width, height) tuple
            use_video_mode: Use video mode for continuous streaming (recommended)
        """
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=False)

        self.model = "CameraPicamera2"
        self.shape = (0, 0)
        self.is_connected = False
        self.is_streaming = False
        
        # Camera settings
        self.exposure_time = exposure_time  # microseconds
        self.gain = gain
        self.frame_rate = frame_rate
        self.cameraNo = cameraNo
        self.flipImage = flipImage  # (flipY, flipX)
        self.isRGB = isRGB
        self.binning = binning
        self.use_video_mode = use_video_mode
        
        # Buffer management
        self.NBuffer = 5
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)
        self.flatfieldImage = None
        self.isFlatfielding = False
        
        self.camera = None
        self.DEBUG = False
        
        # Frame tracking
        self.lastFrameFromBuffer = None
        self.lastFrameId = -1
        self.frameNumber = -1
        self.timestamp = 0
        
        # Thread management for frame grabbing
        self._grab_thread = None
        self._stop_event = threading.Event()
        
        # Resolution
        self.SensorWidth = resolution[0]
        self.SensorHeight = resolution[1]
        self.frame = np.zeros((self.SensorHeight, self.SensorWidth, 3 if isRGB else 1), dtype=np.uint8)
        
        # Auto exposure/white balance
        self.exposure_auto = False
        self.awb_auto = True
        
        # Trigger mode
        self.trigger_source = "Continuous"
        
        # Open camera
        self._open_camera(self.cameraNo)
        
        self.__logger.info(f"Camera initialized: model={self.model}, RGB={self.isRGB}, resolution={self.SensorWidth}x{self.SensorHeight}")

    def _release_camera_from_other_processes(self, camera_index: int):
        """
        Attempt to release camera from other processes.
        This uses fuser to find and optionally kill processes holding the camera.
        """
        import subprocess
        import os
        
        # Camera device paths that might be locked
        device_paths = [
            f"/dev/video{camera_index}",
            "/dev/media0",
            "/dev/media1",
            "/dev/media2",
        ]
        
        for device_path in device_paths:
            if not os.path.exists(device_path):
                continue
            
            try:
                # Find processes using the device
                result = subprocess.run(
                    ["fuser", device_path],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split()
                    self.__logger.warning(f"Device {device_path} is being used by PIDs: {pids}")
                    
                    # Get process info
                    for pid in pids:
                        try:
                            ps_result = subprocess.run(
                                ["ps", "-p", pid, "-o", "comm="],
                                capture_output=True,
                                text=True,
                                timeout=1
                            )
                            process_name = ps_result.stdout.strip()
                            self.__logger.warning(f"  PID {pid}: {process_name}")
                            
                            # Kill the process if it's not our own process
                            current_pid = os.getpid()
                            if int(pid) != current_pid:
                                self.__logger.warning(f"Attempting to kill PID {pid} ({process_name})")
                                subprocess.run(["kill", "-9", pid], timeout=1)
                                import time
                                time.sleep(0.5)  # Give it time to release
                        except Exception as e:
                            self.__logger.debug(f"Could not process PID {pid}: {e}")
                
            except subprocess.TimeoutExpired:
                self.__logger.debug(f"Timeout checking {device_path}")
            except FileNotFoundError:
                self.__logger.debug("fuser command not found, skipping process check")
                break
            except Exception as e:
                self.__logger.debug(f"Error checking {device_path}: {e}")
        
        # Alternative: try to close all Picamera2 instances
        try:
            from picamera2 import Picamera2
            # This will close all global instances
            Picamera2.close_all_cameras()
            import time
            time.sleep(0.5)
        except Exception as e:
            self.__logger.debug(f"Could not close all cameras: {e}")

    def _open_camera(self, camera_index: int):
        """Open and configure the camera"""
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("Picamera2 not available")
        
        # Initialize camera to None first to avoid issues in close()
        self.camera = None
        
        try:
            # Try to release camera from other processes
            self._release_camera_from_other_processes(camera_index)
            
            # Create camera instance
            self.camera = Picamera2() # sudo fuser -k /dev/video0 /dev/media0 /dev/media1 /dev/media2
            
            # Get sensor resolution
            sensor_modes = self.camera.sensor_modes
            if sensor_modes:
                max_mode = max(sensor_modes, key=lambda m: m['size'][0] * m['size'][1])
                self.SensorWidth, self.SensorHeight = max_mode['size']
                self.__logger.info(f"Max sensor resolution: {self.SensorWidth}x{self.SensorHeight}")
            
            # Configure camera based on mode
            if self.isRGB:
                # RGB mode - use RGB888 format (3 channels, 8-bit each)
                config = self.camera.create_video_configuration(
                    main={"size": (self.SensorWidth, self.SensorHeight), "format": "RGB888"},
                    buffer_count=1,
                    controls={
                        "FrameRate": self.frame_rate,
                    }
                )
            else:
                # Mono mode - use Y8 format (1 channel, 8-bit)
                # Note: Not all cameras support Y8, may fall back to RGB888 and convert
                try:
                    config = self.camera.create_video_configuration(
                        main={"size": (self.SensorWidth, self.SensorHeight), "format": "Y8"},
                        buffer_count=1,
                        controls={
                            "FrameRate": self.frame_rate,
                        }
                    )
                except Exception as e:
                    self.__logger.warning(f"Y8 format not supported, using RGB888: {e}")
                    config = self.camera.create_video_configuration(
                        main={"size": (self.SensorWidth, self.SensorHeight), "format": "RGB888"},
                        buffer_count=1,
                        controls={
                            "FrameRate": self.frame_rate,
                        }
                    )
            
            self.camera.configure(config)
            
            # Set initial controls
            self._apply_camera_controls()
            
            self.is_connected = True
            
            # Track this camera
            self._serial_checksum = camera_index  # Use index as identifier
            CameraPicamera2._opened_cameras.add(self._serial_checksum)
            
        except Exception as e:
            self.__logger.error(f"Failed to open camera {camera_index}: {e}")
            # Clean up partial initialization
            if self.camera is not None:
                try:
                    # Don't call camera.close() as it might fail with AttributeError
                    # Just release the underlying camera object if possible
                    if hasattr(self.camera, 'camera') and self.camera.camera is not None:
                        try:
                            self.camera.camera.release()
                        except:
                            pass
                    self.camera = None
                except:
                    pass
            raise

    def _apply_camera_controls(self):
        """Apply camera control settings"""
        if self.camera is None:
            return
        
        controls = {}
        
        # Exposure control
        if not self.exposure_auto:
            controls["ExposureTime"] = int(self.exposure_time)
            controls["AeEnable"] = False
        else:
            controls["AeEnable"] = True
        
        # Gain control
        controls["AnalogueGain"] = float(self.gain)
        
        # Auto white balance
        controls["AwbEnable"] = self.awb_auto
        
        try:
            self.camera.set_controls(controls)
            self.__logger.debug(f"Applied camera controls: {controls}")
        except Exception as e:
            self.__logger.error(f"Failed to apply controls: {e}")

    def _grab_frames(self):
        """Thread function to continuously grab frames"""
        self.__logger.debug("Frame grabbing thread started")
        
        while not self._stop_event.is_set():
            try:
                # Capture frame
                request = self.camera.capture_request()
                
                try:
                    # Get frame data
                    array = request.make_array("main")
                    
                    # Get metadata
                    metadata = request.get_metadata()
                    frame_id = metadata.get("FrameId", self.frameNumber + 1)
                    timestamp = metadata.get("SensorTimestamp", int(time.time() * 1e6))
                    
                    # Process frame
                    frame = self._process_frame(array)
                    
                    # Update frame number and timestamp
                    self.frameNumber = frame_id
                    self.timestamp = timestamp
                    
                    # Add to buffer
                    self.frame_buffer.append(frame)
                    self.frameid_buffer.append(frame_id)
                    
                    if self.DEBUG:
                        self.__logger.debug(f"Frame {frame_id} captured, buffer size: {len(self.frame_buffer)}")
                
                finally:
                    # Always release the request
                    request.release()
                
            except Exception as e:
                if not self._stop_event.is_set():
                    self.__logger.error(f"Error capturing frame: {e}")
                    time.sleep(0.01)  # Brief pause on error
        
        self.__logger.debug("Frame grabbing thread stopped")

    def _process_frame(self, array: np.ndarray) -> np.ndarray:
        """
        Process captured frame (apply flipping, color conversion, flatfielding).
        
        Args:
            array: Raw frame from camera
            
        Returns:
            Processed frame
        """
        # Convert to mono if needed
        if not self.isRGB and len(array.shape) == 3:
            # Convert RGB to grayscale
            array = np.dot(array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        # Apply flipping
        if self.flipImage[0]:  # Flip Y
            array = np.flipud(array)
        if self.flipImage[1]:  # Flip X
            array = np.fliplr(array)
        
        # Apply flatfielding if enabled
        if self.isFlatfielding and self.flatfieldImage is not None:
            try:
                # Normalize by flatfield
                array = np.clip(array.astype(np.float32) / (self.flatfieldImage.astype(np.float32) + 1e-6) * 255, 0, 255).astype(np.uint8)
            except Exception as e:
                self.__logger.error(f"Flatfield correction failed: {e}")
        
        return array

    def reconnectCamera(self):
        """Reconnect the camera after disconnection"""
        self.__logger.info("Reconnecting camera...")
        
        old_checksum = getattr(self, '_serial_checksum', None)
        
        # Close existing camera
        if self.camera is not None:
            if self.is_streaming:
                self.stop_live()
            
            try:
                self.camera.close()
            except Exception as e:
                self.__logger.error(f"Error closing camera: {e}")
            
            # Remove from opened cameras
            if old_checksum in CameraPicamera2._opened_cameras:
                CameraPicamera2._opened_cameras.remove(old_checksum)
        
        # Reopen camera
        try:
            self._open_camera(self.cameraNo)
            self.__logger.info("Camera reconnected successfully")
        except Exception as e:
            self.__logger.error(f"Failed to reconnect camera: {e}")
            raise

    def getTriggerTypes(self) -> List[str]:
        """Return list of available trigger types"""
        return [
            "Continuous (Free Run)",
            "Software Trigger",
            "External Trigger (GPIO)"
        ]

    def getTriggerSource(self) -> str:
        """Return current trigger source"""
        return self.trigger_source

    def setTriggerSource(self, trigger_source: str):
        """
        Set trigger source.
        
        Args:
            trigger_source: One of "Continuous", "Software Trigger", "External Trigger"
        """
        self.trigger_source = trigger_source
        self.__logger.info(f"Trigger source set to: {trigger_source}")
        
        # Note: External trigger would require GPIO configuration
        # Software trigger is handled in getLast() method

    def get_camera_parameters(self):
        """Get current camera parameters"""
        param_dict = {
            "model_name": "Raspberry Pi Camera",
            "isRGB": self.isRGB,
            "width": self.SensorWidth,
            "height": self.SensorHeight,
            "exposure_time": self.exposure_time,
            "exposure_auto": self.exposure_auto,
            "gain": self.gain,
            "frame_rate": self.frame_rate,
            "trigger_source": self.trigger_source,
        }
        
        # Get current metadata if available
        if self.camera is not None and self.is_streaming:
            try:
                metadata = self.camera.capture_metadata()
                param_dict["exposure_time"] = metadata.get("ExposureTime", self.exposure_time)
                param_dict["gain"] = metadata.get("AnalogueGain", self.gain)
            except Exception as e:
                self.__logger.debug(f"Could not read metadata: {e}")
        
        return param_dict

    def get_gain(self):
        """Get current/min/max gain values"""
        # Typical RPi camera gain range
        return (self.gain, 1.0, 16.0)

    def get_exposuretime(self):
        """Get current/min/max exposure time"""
        # Typical RPi camera exposure range (microseconds)
        return (self.exposure_time, 100, 1000000)

    def start_live(self):
        """Start live streaming"""
        if self.is_streaming:
            self.__logger.warning("Camera already streaming")
            return
        
        self.flushBuffer()
        
        if self.camera is None:
            self.__logger.error("Camera not initialized")
            self.reconnectCamera() # TODO: This will probably not work if camera wasn't connected on boot 
        
        try:
            # Start camera
            self.camera.start()
            
            # Start frame grabbing thread
            self._stop_event.clear()
            self._grab_thread = threading.Thread(target=self._grab_frames, daemon=True)
            self._grab_thread.start()
            
            self.is_streaming = True
            self.__logger.info("Camera streaming started")
            
        except Exception as e:
            self.__logger.error(f"Failed to start streaming: {e}")
            raise

    def stop_live(self):
        """Stop live streaming"""
        if not self.is_streaming:
            self.__logger.warning("Camera not streaming")
            return
        
        # Stop grabbing thread
        self._stop_event.set()
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None
        
        # Stop camera
        try:
            self.camera.stop()
        except Exception as e:
            self.__logger.error(f"Error stopping camera: {e}")
        
        self.is_streaming = False
        self.__logger.info("Camera streaming stopped")

    def suspend_live(self):
        """Suspend live streaming (alias for stop_live)"""
        self.stop_live()

    def prepare_live(self):
        """Prepare for live streaming (no-op)"""
        pass

    def close(self):
        """Close camera and clean up resources"""
        if self.is_streaming:
            self.stop_live()
        
        # Remove from opened cameras
        if hasattr(self, '_serial_checksum') and self._serial_checksum in CameraPicamera2._opened_cameras:
            CameraPicamera2._opened_cameras.remove(self._serial_checksum)
        
        # Close camera
        if self.camera is not None:
            try:
                # Safely close the camera
                # Check if camera is properly initialized before closing
                if hasattr(self.camera, '_preview'):
                    self.camera.close()
                else:
                    # Camera didn't initialize properly, try to clean up manually
                    self.__logger.warning("Camera not fully initialized, attempting manual cleanup")
                    try:
                        if hasattr(self.camera, 'camera') and self.camera.camera is not None:
                            self.camera.camera.release()
                    except Exception as cleanup_error:
                        self.__logger.debug(f"Manual cleanup failed: {cleanup_error}")
            except AttributeError as e:
                self.__logger.warning(f"Camera close skipped due to incomplete initialization: {e}")
            except Exception as e:
                self.__logger.error(f"Error closing camera: {e}")
            
            self.camera = None
        
        self.is_connected = False
        self.__logger.info("Camera closed")

    def set_exposure_time(self, exposure_time: float):
        """
        Set exposure time in milliseconds.
        
        Args:
            exposure_time: Exposure time in milliseconds
        """
        self.exposure_time = exposure_time * 1000  # Convert to microseconds
        if self.camera is not None and not self.exposure_auto:
            self.camera.set_controls({"ExposureTime": int(self.exposure_time)})

    def set_exposure_mode(self, exposure_mode: str = "manual"):
        """
        Set exposure mode.
        
        Args:
            exposure_mode: "manual", "auto", or "once"
        """
        exposure_mode = exposure_mode.lower()
        
        if exposure_mode == "manual":
            self.exposure_auto = False
            if self.camera is not None:
                self.camera.set_controls({
                    "AeEnable": False,
                    "ExposureTime": int(self.exposure_time)
                })
        elif exposure_mode == "auto":
            self.exposure_auto = True
            if self.camera is not None:
                self.camera.set_controls({"AeEnable": True})
        elif exposure_mode == "once":
            # One-shot auto exposure, then lock
            if self.camera is not None:
                self.camera.set_controls({"AeEnable": True})
                time.sleep(0.5)  # Allow AE to settle
                metadata = self.camera.capture_metadata()
                self.exposure_time = metadata.get("ExposureTime", self.exposure_time)
                self.camera.set_controls({
                    "AeEnable": False,
                    "ExposureTime": int(self.exposure_time)
                })
                self.exposure_auto = False
        else:
            self.__logger.warning(f"Unknown exposure mode: {exposure_mode}")

    def set_camera_mode(self, isAutomatic: bool):
        """Set camera to automatic or manual mode"""
        self.set_exposure_mode("auto" if isAutomatic else "manual")

    def set_gain(self, gain: float):
        """Set analogue gain (1.0-16.0)"""
        self.gain = max(1.0, min(16.0, gain))
        if self.camera is not None:
            self.camera.set_controls({"AnalogueGain": self.gain})

    def set_frame_rate(self, frame_rate: int):
        """Set target frame rate"""
        self.frame_rate = frame_rate
        # Note: Changing frame rate requires reconfiguration
        # For now, just store the value
        self.__logger.info(f"Frame rate set to {frame_rate} (requires restart)")

    def set_flatfielding(self, is_flatfielding: bool):
        """Enable/disable flatfielding"""
        self.isFlatfielding = is_flatfielding

    def setFlatfieldImage(self, flatfieldImage: np.ndarray, isFlatfieldEnabled: bool = True):
        """Set flatfield correction image"""
        self.flatfieldImage = flatfieldImage
        self.isFlatfielding = isFlatfieldEnabled

    def set_blacklevel(self, blacklevel: int):
        """Set black level (not implemented for RPi camera)"""
        self.__logger.warning("Black level adjustment not supported on RPi camera")

    def set_pixel_format(self, format: str):
        """Set pixel format (requires restart)"""
        self.__logger.warning("Pixel format change requires camera restart")

    def setBinning(self, binning: int = 1):
        """
        Set binning factor (simulated via resolution).
        
        Args:
            binning: Binning factor (1, 2, 4)
        """
        self.binning = binning
        # Binning is simulated by reducing resolution
        self.__logger.info(f"Binning set to {binning} (requires restart)")

    def getLast(
        self,
        returnFrameNumber: bool = False,
        timeout: float = 1.0,
        auto_trigger: bool = True
    ):
        """
        Get the latest frame from the buffer.
        
        Args:
            returnFrameNumber: If True, return (frame, frame_id) tuple
            timeout: Seconds to wait for a frame
            auto_trigger: Automatically trigger in software trigger mode
            
        Returns:
            Frame or (frame, frame_id) tuple
        """
        # Handle software trigger
        if auto_trigger and self.trigger_source.lower() in ("software", "software trigger"):
            # In software trigger mode, capture a single frame
            if not self.is_streaming:
                # Start camera temporarily
                self.camera.start()
                request = self.camera.capture_request()
                try:
                    array = request.make_array("main")
                    frame = self._process_frame(array)
                    metadata = request.get_metadata()
                    frame_id = metadata.get("FrameId", self.frameNumber + 1)
                    self.frameNumber = frame_id
                finally:
                    request.release()
                    self.camera.stop()
                
                if returnFrameNumber:
                    return frame, frame_id
                return frame
        
        # Wait for frame in buffer
        t0 = time.time()
        while not self.frame_buffer:
            if time.time() - t0 > timeout:
                self.__logger.warning(f"Timeout waiting for frame ({timeout}s)")
                # Return last known frame or zeros
                if self.lastFrameFromBuffer is not None:
                    if returnFrameNumber:
                        return self.lastFrameFromBuffer, self.lastFrameId
                    return self.lastFrameFromBuffer
                else:
                    empty_frame = np.zeros((self.SensorHeight, self.SensorWidth, 3 if self.isRGB else 1), dtype=np.uint8)
                    if returnFrameNumber:
                        return empty_frame, -1
                    return empty_frame
            time.sleep(0.001)
        
        # Get latest frame
        latest_frame = self.frame_buffer.pop()
        latest_frame_id = self.frameid_buffer.pop()
        
        # Store as last frame
        self.lastFrameFromBuffer = latest_frame
        self.lastFrameId = latest_frame_id
        
        if returnFrameNumber:
            return latest_frame, latest_frame_id
        return latest_frame

    def flushBuffer(self):
        """Clear frame buffers"""
        self.frameid_buffer.clear()
        self.frame_buffer.clear()

    def getLastChunk(self):
        """Get metadata chunk (not implemented)"""
        return None

    def setROI(self, hpos=None, vpos=None, hsize=None, vsize=None):
        """
        Set region of interest.
        
        Note: ROI on RPi camera requires reconfiguration or post-processing.
        For now, we'll handle this via cropping in post-processing.
        """
        self.__logger.warning("ROI setting requires camera restart - not implemented yet")

    def setPropertyValue(self, property_name: str, property_value):
        """Set camera property by name and apply to camera hardware."""
        property_map = {
            "exposure": self.set_exposure_time,
            "gain": self.set_gain,
            "frame_rate": self.set_frame_rate,
            "blacklevel": self.set_blacklevel,
            "exposure_mode": self.set_exposure_mode,
            "flat_fielding": self.set_flatfielding,
            "trigger_source": self.setTriggerSource,
            "mode": lambda v: self.set_camera_mode(isAutomatic=v),
        }
        
        if property_name in property_map:
            try:
                property_map[property_name](property_value)
                self.__logger.debug(f"Property '{property_name}' set to {property_value}")
                return property_value
            except Exception as e:
                self.__logger.error(f"Failed to set {property_name}: {e}")
                return None
        else:
            self.__logger.warning(f"Unknown property: {property_name}")
            return None 

    def getPropertyValue(self, property_name: str):
        """Get camera property by name, reading actual values from camera when possible."""
        # Try to read actual values from camera metadata if streaming
        if self.camera is not None and self.is_streaming:
            try:
                metadata = self.camera.capture_metadata()
                if property_name == "exposure":
                    # Return exposure time in milliseconds
                    return metadata.get("ExposureTime", self.exposure_time) / 1000
                elif property_name == "gain":
                    return metadata.get("AnalogueGain", self.gain)
                elif property_name == "frame_rate":
                    # FrameDuration is in microseconds, convert to fps
                    frame_duration = metadata.get("FrameDuration", 0)
                    if frame_duration > 0:
                        return 1000000 / frame_duration
                    return self.frame_rate
            except Exception as e:
                self.__logger.debug(f"Could not read metadata for {property_name}: {e}")
                # Fall back to stored values
        
        # Property map for stored values (fallback or non-metadata properties)
        property_map = {
            "exposure": lambda: self.exposure_time / 1000,  # Convert to ms
            "gain": lambda: self.gain,
            "frame_rate": lambda: self.frame_rate,
            "blacklevel": lambda: 0,
            "exposure_mode": lambda: "auto" if self.exposure_auto else "manual",
            "flat_fielding": lambda: self.isFlatfielding,
            "frame_number": lambda: self.frameNumber,
            "trigger_source": lambda: self.trigger_source,
            "mode": lambda: self.exposure_auto,
        }
        
        if property_name in property_map:
            try:
                return property_map[property_name]()
            except Exception as e:
                self.__logger.error(f"Failed to get {property_name}: {e}")
                return None
        else:
            self.__logger.warning(f"Unknown property: {property_name}")
            return None

    def send_trigger(self):
        """Send software trigger (handled in getLast)"""
        self.__logger.debug("Software trigger requested")
        return True

    def openPropertiesGUI(self):
        """Open properties GUI (not implemented)"""
        self.__logger.warning("Properties GUI not available")

    def recordFlatfieldImage(self, nFrames: int = 10, nGauss: int = 5, nMedian: int = 5):
        """Record flatfield image by averaging multiple frames"""
        if not self.is_streaming:
            self.__logger.error("Camera must be streaming to record flatfield")
            return
        
        self.__logger.info(f"Recording flatfield image ({nFrames} frames)...")
        
        frames = []
        for i in range(nFrames):
            frame = self.getLast(timeout=2.0)
            if frame is not None:
                frames.append(frame.astype(np.float32))
            time.sleep(0.1)
        
        if len(frames) < nFrames:
            self.__logger.warning(f"Only captured {len(frames)}/{nFrames} frames")
        
        # Average frames
        flatfield = np.mean(frames, axis=0).astype(np.uint8)
        
        # Apply smoothing if requested
        if nGauss > 0:
            from skimage.filters import gaussian
            flatfield = gaussian(flatfield, sigma=nGauss, preserve_range=True).astype(np.uint8)
        
        if nMedian > 0:
            from skimage.filters import median
            from skimage.morphology import disk
            flatfield = median(flatfield, disk(nMedian)).astype(np.uint8)
        
        self.setFlatfieldImage(flatfield, True)
        self.__logger.info("Flatfield image recorded")

    def getFrameNumber(self):
        """Get current frame number"""
        return self.frameNumber

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit"""
        self.close()


# Mock camera for testing when picamera2 is not available
class MockCameraPicamera2:
    """Mock camera for testing without real hardware"""
    
    def __init__(self, *args, **kwargs):
        self.model = "MockPicamera2"
        self.SensorWidth = kwargs.get('resolution', (640, 480))[0]
        self.SensorHeight = kwargs.get('resolution', (640, 480))[1]
        self.isRGB = kwargs.get('isRGB', True)
        self.is_connected = True
        self.is_streaming = False
        
        self.frame_buffer = collections.deque(maxlen=5)
        self.frameid_buffer = collections.deque(maxlen=5)
        self.frameNumber = 0
        
        self.exposure_time = kwargs.get('exposure_time', 10000)
        self.gain = kwargs.get('gain', 1.0)
        self.frame_rate = kwargs.get('frame_rate', 30)
        self.trigger_source = "Continuous"
        
        self.flatfieldImage = None
        self.isFlatfielding = False
        
        self._grab_thread = None
        self._stop_event = threading.Event()
        
        # get last frame lock
        self._get_last_frame_lock = threading.Lock()
        
        print(f"Mock Picamera2 initialized: {self.SensorWidth}x{self.SensorHeight}, RGB={self.isRGB}")
    
    def _generate_frame(self):
        """Generate a mock frame with pattern"""
        if self.isRGB:
            frame = np.random.randint(0, 255, (self.SensorHeight, self.SensorWidth, 3), dtype=np.uint8)
            # Add pattern
            cv2.rectangle(frame, (self.SensorWidth//2-25, self.SensorHeight//2-25), 
                         (self.SensorWidth//2+25, self.SensorHeight//2+25), (255, 255, 255), -1)
        else:
            frame = np.random.randint(0, 255, (self.SensorHeight, self.SensorWidth), dtype=np.uint8)
            cv2.rectangle(frame, (self.SensorWidth//2-25, self.SensorHeight//2-25), 
                         (self.SensorWidth//2+25, self.SensorHeight//2+25), 255, -1)
        
        # Add frame number
        import cv2
        cv2.putText(frame, f"Frame {self.frameNumber}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.isRGB else 255, 2)
        return frame
    
    def _grab_frames(self):
        """Mock frame grabbing"""
        while not self._stop_event.is_set():
            frame = self._generate_frame()
            self.frameNumber += 1
            self.frame_buffer.append(frame)
            self.frameid_buffer.append(self.frameNumber)
            time.sleep(1.0 / self.frame_rate)
    
    def start_live(self):
        """Start mock streaming"""
        if not self.is_streaming:
            self._stop_event.clear()
            self._grab_thread = threading.Thread(target=self._grab_frames, daemon=True)
            self._grab_thread.start()
            self.is_streaming = True
    
    def stop_live(self):
        """Stop mock streaming"""
        if self.is_streaming:
            self._stop_event.set()
            if self._grab_thread:
                self._grab_thread.join(timeout=1.0)
            self.is_streaming = False
    
    def suspend_live(self):
        self.stop_live()
    
    def prepare_live(self):
        pass
    
    def close(self):
        self.stop_live()
    
    def getLast(self, returnFrameNumber=False, timeout=1.0, auto_trigger=True):
        with self._get_last_frame_lock:
            """Get latest mock frame"""
            if not self.frame_buffer:
                frame = self._generate_frame()
                frame_id = self.frameNumber
            else:
                frame = self.frame_buffer.pop()
                frame_id = self.frameid_buffer.pop()
            
            if returnFrameNumber:
                return frame, frame_id
            return frame
    
    def flushBuffer(self):
        self.frame_buffer.clear()
        self.frameid_buffer.clear()
    
    # Stub methods
    def getTriggerTypes(self): return ["Continuous", "Software Trigger"]
    def getTriggerSource(self): return self.trigger_source
    def setTriggerSource(self, source): self.trigger_source = source
    def get_camera_parameters(self): return {"model_name": "Mock", "isRGB": self.isRGB}
    def get_gain(self): return (self.gain, 1.0, 16.0)
    def get_exposuretime(self): return (self.exposure_time, 100, 1000000)
    def set_exposure_time(self, t): self.exposure_time = t * 1000
    def set_exposure_mode(self, mode): pass
    def set_camera_mode(self, auto): pass
    def set_gain(self, g): self.gain = g
    def set_frame_rate(self, fr): self.frame_rate = fr
    def set_flatfielding(self, enabled): self.isFlatfielding = enabled
    def setFlatfieldImage(self, img, enabled=True): 
        self.flatfieldImage = img
        self.isFlatfielding = enabled
    def set_blacklevel(self, level): pass
    def set_pixel_format(self, fmt): pass
    def setBinning(self, binning): pass
    def getLastChunk(self): return None
    def setROI(self, *args, **kwargs): pass
    
    def setPropertyValue(self, property_name, property_value):
        """Set camera property by name and apply to mock camera."""
        property_map = {
            "exposure": self.set_exposure_time,
            "gain": self.set_gain,
            "frame_rate": self.set_frame_rate,
            "blacklevel": self.set_blacklevel,
            "exposure_mode": self.set_exposure_mode,
            "flat_fielding": self.set_flatfielding,
            "trigger_source": self.setTriggerSource,
            "mode": lambda v: self.set_camera_mode(auto=v),
        }
        if property_name in property_map:
            try:
                property_map[property_name](property_value)
                return property_value
            except Exception:
                return None
        return None
    
    def getPropertyValue(self, property_name):
        """Get camera property by name."""
        property_map = {
            "exposure": lambda: self.exposure_time / 1000,  # Convert to ms
            "gain": lambda: self.gain,
            "frame_rate": lambda: self.frame_rate,
            "blacklevel": lambda: 0,
            "exposure_mode": lambda: "manual",
            "flat_fielding": lambda: self.isFlatfielding,
            "frame_number": lambda: self.frameNumber,
            "trigger_source": lambda: self.trigger_source,
            "mode": lambda: False,
        }
        if property_name in property_map:
            try:
                return property_map[property_name]()
            except Exception:
                return None
        return None
    
    def send_trigger(self): return True
    def openPropertiesGUI(self): pass
    def recordFlatfieldImage(self, *args, **kwargs): pass
    def getFrameNumber(self): return self.frameNumber
    def reconnectCamera(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
