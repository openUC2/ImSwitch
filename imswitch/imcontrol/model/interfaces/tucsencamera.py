import collections
import time
import threading
import numpy as np
from typing import List, Optional, Union
from ctypes import *
from enum import Enum
import sys

from imswitch.imcommon.model import initLogger

# Platform-specific imports
if sys.platform.startswith('linux'):
    try:
        from imswitch.imcontrol.model.interfaces.tucam.TUCam import *
        TUCSEN_SDK_AVAILABLE = True
        TUCSEN_PLATFORM = "linux"
    except Exception as e:
        print(f"Could not import Tucsen camera libraries for Linux: {e}")
        TUCSEN_SDK_AVAILABLE = False
        TUCSEN_PLATFORM = None
elif sys.platform.startswith('win'):
    try:
        from imswitch.imcontrol.model.interfaces.tucam_win.TUCam import *
        TUCSEN_SDK_AVAILABLE = True
        TUCSEN_PLATFORM = "windows"
    except Exception as e:
        print(f"Could not import Tucsen camera libraries for Windows: {e}")
        TUCSEN_SDK_AVAILABLE = False
        TUCSEN_PLATFORM = None
else:
    print(f"Tucsen camera interface not supported on {sys.platform}")
    TUCSEN_SDK_AVAILABLE = False
    TUCSEN_PLATFORM = None

# Tucsen camera modes
class TucsenMode(Enum):
    HDR = 0
    CMS = 1
    HIGH_SPEED = 2

class CameraTucsen:
    """Minimal wrapper for Tucsen cameras compatible with ImSwitch architecture."""

    @staticmethod
    def force_cleanup():
        """Force cleanup of any existing Tucsen camera connections."""
        try:
            if TUCSEN_PLATFORM == "windows":
                # Try to uninitialize the API to free any stuck connections
                TUCAM_Api_Uninit()
                time.sleep(0.2)  # Give more time for Windows to release resources
        except Exception as e:
            print(f"Force cleanup warning: {e}")

    def _is_success(self, ret):
        """Helper to check if a return value indicates success."""
        if hasattr(ret, 'value'):
            return ret == TUCAMRET.TUCAMRET_SUCCESS
        else:
            return ret == TUCAMRET.TUCAMRET_SUCCESS.value

    def __init__(self, cameraNo=None, exposure_time=10000, gain=0, frame_rate=-1, blacklevel=100, isRGB=False, binning=1):
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=False)

        self.model = "CameraTucsen"
        self.shape = (0, 0)
        self.is_connected = False
        self.is_streaming = False
        self.downsamplepreview = 1

        self.blacklevel = blacklevel
        self.exposure_time = exposure_time
        self.gain = gain
        self.preview_width = 600
        self.preview_height = 600
        self.frame_rate = frame_rate
        self.cameraNo = cameraNo if cameraNo is not None else 0

        self.NBuffer = 5
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)
        self.flatfieldImage = None
        self.camera_handle = None

        # Binning
        self.binning = binning

        self.SensorHeight = 0
        self.SensorWidth = 0
        self.frame = np.zeros((self.SensorHeight, self.SensorWidth))

        self.lastFrameFromBuffer = None
        self.lastFrameId = -1
        self.frameNumber = -1
        self.g_bExit = False

        # Threading for frame acquisition
        self._read_thread_lock = threading.Lock()
        self._read_thread: Optional[threading.Thread] = None
        self._read_thread_keep_running = threading.Event()
        self._read_thread_keep_running.clear()
        self._read_thread_wait_period_s = 1.0
        self._read_thread_running = threading.Event()
        self._read_thread_running.clear()

        self._frame_lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._last_trigger_timestamp = 0
        self._trigger_sent = threading.Event()
        self._is_streaming = threading.Event()

        # Check if SDK is available
        if not TUCSEN_SDK_AVAILABLE:
            raise Exception("Tucsen SDK not available")

        # Platform-specific initialization objects
        if TUCSEN_PLATFORM == "windows":
            self.TUCAMINIT = None
            self.TUCAMOPEN = None

        # Initialize camera
        self._open_camera(self.cameraNo)

        self.isFlatfielding = False
        self.isRGB = bool(isRGB)

        # Trigger settings
        self.trigger_source = "Continuous"

        # Image buffer
        self._m_frame = None

    def _open_camera(self, camera_index: int):
        """Initialize and open the Tucsen camera."""
        try:
            if TUCSEN_PLATFORM == "linux":
                self._open_camera_linux(camera_index)
            elif TUCSEN_PLATFORM == "windows":
                self._open_camera_windows(camera_index)
            else:
                raise Exception("Unsupported platform for Tucsen camera")
                
        except Exception as e:
            self.__logger.error(f"Failed to open Tucsen camera: {e}")
            self.is_connected = False
            raise

    def _open_camera_linux(self, camera_index: int):
        """Linux-specific camera initialization."""
        # Initialize the Tucsen API
        ret = TUCAM_Api_Init()
        if ret != TUCAMRET.TUCAMRET_SUCCESS.value:
            raise Exception(f"Failed to initialize Tucsen API: {ret}")

        # Open camera by index
        opCam = TUCAM_OPEN()
        opCam.uiIdxOpen = camera_index
        
        ret = TUCAM_Dev_Open(byref(opCam))
        if ret != TUCAMRET.TUCAMRET_SUCCESS.value:
            raise Exception(f"Failed to open Tucsen camera {camera_index}: {ret}")
            
        self.camera_handle = opCam.hIdxTUCam

        # Get sensor dimensions
        self._get_sensor_info()
        
        self.is_connected = True
        self.__logger.info(f"Successfully opened Tucsen camera {camera_index} (Linux)")
        self.__logger.info(f"Sensor size: {self.SensorWidth} x {self.SensorHeight}")

    def _open_camera_windows(self, camera_index: int):
        """Windows-specific camera initialization."""
        # Try to cleanup any previous sessions first
        try:
            TUCAM_Api_Uninit()
            time.sleep(0.1)  # Give time for cleanup
        except Exception as e:
            self.__logger.error(f"Error during camera cleanup: {e}")
            pass  # Ignore errors from uninit if nothing was initialized    
        # Initialize the Tucsen API for Windows - exact pattern from working example
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)  # Initialize with defaults first
        
        # Initialize API
        ret = TUCAM_Api_Init(pointer(self.TUCAMINIT), 5000)
        self.__logger.info(f"API Init result: {ret}")
        self.__logger.info(f"Camera count: {self.TUCAMINIT.uiCamCount}")
        self.__logger.info(f"Config path: {self.TUCAMINIT.pstrConfigPath.decode() if self.TUCAMINIT.pstrConfigPath else 'None'}")
        
        if self.TUCAMINIT.uiCamCount == 0:
            raise Exception("No Tucsen cameras found")
            
        if camera_index >= self.TUCAMINIT.uiCamCount:
            raise Exception(f"Camera index {camera_index} not available. Found {self.TUCAMINIT.uiCamCount} cameras")

        self.__logger.info(f'Found {self.TUCAMINIT.uiCamCount} Tucsen camera(s)')

        # Open specific camera - exact pattern from working example
        self.TUCAMOPEN = TUCAM_OPEN(camera_index, 0)
        self.__logger.info(f"Attempting to open camera {camera_index}...")
        
        # Call TUCAM_Dev_Open directly like the working example
        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))
        
        # Check result exactly like the working example
        if self.TUCAMOPEN.hIdxTUCam == 0:
            self.__logger.error("Open the camera failure!")
            raise Exception(f"Failed to open Tucsen camera {camera_index}")
        else:
            self.__logger.info("Open the camera success!")
            
        self.camera_handle = self.TUCAMOPEN.hIdxTUCam

        # Get sensor dimensions
        self._get_sensor_info()
        
        self.is_connected = True
        self.__logger.info(f"Successfully opened Tucsen camera {camera_index} (Windows)")
        self.__logger.info(f"Sensor size: {self.SensorWidth} x {self.SensorHeight}")

    def _get_sensor_info(self):
        """Get sensor information from the camera."""
        try:
            if TUCSEN_PLATFORM == "windows":
                # Try to get actual sensor info by allocating a frame temporarily
                temp_frame = TUCAM_FRAME()
                temp_frame.pBuffer = 0
                temp_frame.ucFormatGet = TUFRM_FORMATS.TUFRM_FMT_USUAl.value
                temp_frame.uiRsdSize = 1
                
                ret = TUCAM_Buf_Alloc(self.camera_handle, pointer(temp_frame))
                if self._is_success(ret):
                    # Get actual dimensions from the allocated frame
                    self.SensorWidth = temp_frame.usWidth
                    self.SensorHeight = temp_frame.usHeight
                    self.__logger.info(f"Got actual sensor dimensions from frame: {self.SensorWidth} x {self.SensorHeight}")
                    
                    # Release the temporary frame
                    TUCAM_Buf_Release(self.camera_handle)
                else:
                    self.__logger.warning("Could not allocate temporary frame for sensor info, using defaults")
                    self.SensorWidth = 2048
                    self.SensorHeight = 2048
            else:
                # Linux implementation would go here
                self.SensorWidth = 2048
                self.SensorHeight = 2048
                
            self.shape = (self.SensorHeight, self.SensorWidth)
            self.__logger.info(f"Sensor dimensions: {self.SensorWidth} x {self.SensorHeight}")
            
        except Exception as e:
            self.__logger.warning(f"Could not get sensor info, using defaults: {e}")
            self.SensorWidth = 2048
            self.SensorHeight = 2048
            self.shape = (self.SensorHeight, self.SensorWidth)

    def get_camera_parameters(self):
        """Get current camera parameters."""
        param_dict = {}
        
        param_dict["model"] = self.model
        param_dict["isRGB"] = self.isRGB
        param_dict["width"] = self.SensorWidth
        param_dict["height"] = self.SensorHeight
        param_dict["exposure_time"] = self.exposure_time
        param_dict["gain"] = self.gain
        param_dict["blacklevel"] = self.blacklevel
        param_dict["binning"] = self.binning
        
        return param_dict

    def get_gain(self):
        """Get current gain settings."""
        try:
            # Placeholder - implement actual Tucsen API call
            return (self.gain, 0.0, 100.0)  # current, min, max
        except Exception as e:
            self.__logger.error(f"Failed to get gain: {e}")
            return (None, None, None)

    def get_exposuretime(self):
        """Get current exposure time settings."""
        try:
            # Placeholder - implement actual Tucsen API call
            return (self.exposure_time, 0.1, 10000.0)  # current, min, max
        except Exception as e:
            self.__logger.error(f"Failed to get exposure time: {e}")
            return (None, None, None)

    def start_live(self):
        """Start live acquisition."""
        if self.is_streaming:
            self.__logger.warning("Camera is already streaming")
            return

        try:
            self.flushBuffer()
            
            # Allocate buffer - both platforms need this
            if self._m_frame is None:
                self._allocate_buffer()
            
            # Set basic camera properties before starting capture
            self.__logger.info("Setting camera properties...")
            self.set_exposure_time(self.exposure_time)
            self.set_gain(self.gain)
            self.set_blacklevel(self.blacklevel)
            
            # Start capture - platform specific
            if TUCSEN_PLATFORM == "linux":
                ret = TUCAM_Cap_Start(self.camera_handle, TUCAM_CAPTURE_MODES.TUCCM_SEQUENCE.value)
                if not self._is_success(ret):
                    raise Exception(f"Failed to start capture: {ret}")
            elif TUCSEN_PLATFORM == "windows":
                # Windows implementation - follow working example pattern
                self.__logger.info("Starting capture...")
                ret = TUCAM_Cap_Start(self.camera_handle, TUCAM_CAPTURE_MODES.TUCCM_SEQUENCE.value)
                self.__logger.info(f"Capture start result: {ret}")
                if not self._is_success(ret):
                    raise Exception(f"Failed to start capture: {ret}")
                self.__logger.info("Capture started successfully")

            self._ensure_read_thread_running()
            self._trigger_sent.clear()
            self._is_streaming.set()
            self.is_streaming = True
            
            self.__logger.info("Tucsen camera started streaming")
            
        except Exception as e:
            self.__logger.error(f"Failed to start live: {e}")
            raise

    def _allocate_buffer(self):
        """Allocate image buffer."""
        try:
            self._m_frame = TUCAM_FRAME()
            self._m_frame.pBuffer = 0
            self._m_frame.ucFormatGet = TUFRM_FORMATS.TUFRM_FMT_USUAl.value
            self._m_frame.uiRsdSize = 1

            ret = TUCAM_Buf_Alloc(self.camera_handle, pointer(self._m_frame))
            self.__logger.info(f"Buffer allocation return value: {ret}")
            self.__logger.info(f"Return type: {type(ret)}")
            
            # Check if allocation was successful using helper method
            if not self._is_success(ret):
                raise Exception(f"Failed to allocate buffer: {ret}")
            
            self.__logger.info(f"Buffer allocated successfully. Frame size: {self._m_frame.uiImgSize}")
                
        except Exception as e:
            self.__logger.error(f"Failed to allocate buffer: {e}")
            raise

    def stop_live(self):
        """Stop live acquisition."""
        if not self.is_streaming:
            self.__logger.warning("Camera is not streaming")
            return

        try:
            self._cleanup_read_thread()
            
            # Stop capture and release buffer
            TUCAM_Buf_AbortWait(self.camera_handle)
            ret = TUCAM_Cap_Stop(self.camera_handle)
            if not self._is_success(ret):
                self.__logger.warning(f"Failed to stop capture cleanly: {ret}")

            # Release buffer
            if self._m_frame is not None:
                TUCAM_Buf_Release(self.camera_handle)
                self._m_frame = None

            self._trigger_sent.clear()
            self._is_streaming.clear()
            self.is_streaming = False
            
            self.__logger.info("Tucsen camera stopped streaming")
            
        except Exception as e:
            self.__logger.error(f"Failed to stop live: {e}")

    def suspend_live(self):
        """Suspend live acquisition."""
        self.stop_live()

    def prepare_live(self):
        """Prepare for live acquisition."""
        pass

    def close(self):
        """Close camera and cleanup."""
        try:
            if self.is_streaming:
                self.stop_live()
                
            if TUCSEN_PLATFORM == "linux":
                self._close_camera_linux()
            elif TUCSEN_PLATFORM == "windows":
                self._close_camera_windows()
                
        except Exception as e:
            self.__logger.error(f"Failed to close camera: {e}")

    def _close_camera_linux(self):
        """Linux-specific camera cleanup."""
        if self.camera_handle:
            ret = TUCAM_Dev_Close(self.camera_handle)
            if ret != TUCAMRET.TUCAMRET_SUCCESS.value:
                self.__logger.warning(f"Failed to close device cleanly: {ret}")
                
        TUCAM_Api_Uninit()
        self.is_connected = False
        self.__logger.info("Tucsen camera closed successfully (Linux)")

    def _close_camera_windows(self):
        """Windows-specific camera cleanup."""
        try:
            # Stop any ongoing operations first
            if self.is_streaming:
                try:
                    TUCAM_Buf_AbortWait(self.camera_handle)
                    TUCAM_Cap_Stop(self.camera_handle)
                except:
                    pass
                    
            # Release buffer if allocated
            if self._m_frame is not None:
                try:
                    TUCAM_Buf_Release(self.camera_handle)
                except:
                    pass
                self._m_frame = None
                
            # Close the camera device
            if self.camera_handle and self.camera_handle != 0:
                try:
                    ret = TUCAM_Dev_Close(self.camera_handle)
                    if not self._is_success(ret):
                        self.__logger.warning(f"Failed to close device cleanly: {ret}")
                except Exception as e:
                    self.__logger.warning(f"Error closing camera device: {e}")
                    
            # Wait a bit before uninitializing
            time.sleep(0.1)
            
            # Uninitialize the API
            try:
                TUCAM_Api_Uninit()
            except Exception as e:
                self.__logger.warning(f"Error uninitializing API: {e}")
                
        except Exception as e:
            self.__logger.error(f"Error in Windows camera cleanup: {e}")
        finally:
            self.camera_handle = None
            self.is_connected = False
            self.__logger.info("Tucsen camera closed successfully (Windows)")

    def _ensure_read_thread_running(self):
        """Ensure the frame reading thread is running."""
        with self._read_thread_lock:
            if self._read_thread is None or not self._read_thread.is_alive():
                self._read_thread_keep_running.set()
                self._read_thread = threading.Thread(target=self._wait_for_frame, daemon=True)
                self._read_thread.start()

    def _cleanup_read_thread(self):
        """Clean up the frame reading thread."""
        self.__logger.debug("Cleaning up read thread.")
        with self._read_thread_lock:
            if self._read_thread is not None:
                self._read_thread_keep_running.clear()
                self._read_thread.join(timeout=2.0)
                self._read_thread = None

    def _wait_for_frame(self):
        """Thread function to wait for and process frames."""
        self.__logger.info("Starting Tucsen read thread.")
        self._read_thread_running.set()
        
        frame_count = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 10
        
        while self._read_thread_keep_running.is_set():
            try:
                self.__logger.debug(f"Thread loop iteration {frame_count}, streaming: {self._is_streaming.is_set()}")
                
                if not self._is_streaming.is_set():
                    self.__logger.debug("Not streaming, sleeping...")
                    time.sleep(0.01)
                    consecutive_timeouts = 0  # Reset timeout counter when not streaming
                    continue
                
                if self._m_frame is None:
                    self.__logger.warning("Frame buffer not allocated")
                    time.sleep(0.01)
                    continue
                    
                self.__logger.debug(f"About to wait for frame {frame_count}...")
                
                # Wait for frame - follow the working example pattern exactly
                try:
                    self.__logger.debug(f"Waiting for frame... Handle: {self.camera_handle}")
                    # Don't check return value, just call the function like in working examples
                    TUCAM_Buf_WaitForFrame(self.camera_handle, pointer(self._m_frame), 1000)
                    
                    # If we get here without exception, frame was received
                    self.__logger.info(f"Frame received! Size: {self._m_frame.uiImgSize}, Width: {self._m_frame.usWidth}, Height: {self._m_frame.usHeight}")
                    consecutive_timeouts = 0  # Reset timeout counter on successful frame
                    
                    # Convert frame to numpy array
                    frame_np = self._convert_frame_to_numpy(self._m_frame)
                    
                    if frame_np is not None:
                        # Store frame in buffer
                        with self._frame_lock:
                            self.frame_buffer.append(frame_np)
                            self.frameid_buffer.append(self.frameNumber)
                            self.frameNumber += 1
                            self._current_frame = frame_np
                        time.sleep(.01) # Slight delay to avoid overwhelming the CPU
                        self.__logger.info(f"Frame {self.frameNumber} captured successfully. Size: {frame_np.shape}")
                        frame_count += 1
                        
                except Exception as frame_error:
                    # Handle specific timeout case
                    error_str = str(frame_error)
                    if "-2147483128" in error_str or "TUCAMRET_TIMEOUT" in error_str:
                        consecutive_timeouts += 1
                        if consecutive_timeouts <= 3:  # Only log first few timeouts to avoid spam
                            self.__logger.debug(f"Frame timeout #{consecutive_timeouts} (this is normal during startup)")
                        elif consecutive_timeouts == max_consecutive_timeouts:
                            self.__logger.warning(f"Too many consecutive timeouts ({max_consecutive_timeouts}), camera may not be producing frames")
                    else:
                        # Log other errors normally
                        self.__logger.warning(f"Frame wait failed: {frame_error}")
                        consecutive_timeouts = 0
                    
                    time.sleep(0.001)
                    
            except Exception as e:
                self.__logger.error(f"Error in frame reading thread: {e}")
                time.sleep(0.01)
                
        self._read_thread_running.clear()
        self.__logger.info("Tucsen read thread stopped.")

    def _convert_frame_to_numpy(self, frame: TUCAM_FRAME) -> np.ndarray:
        """Convert Tucsen frame to numpy array."""
        try:
            if frame.uiImgSize == 0 or frame.pBuffer == 0:
                self.__logger.warning("Invalid frame data")
                return None
                
            # Create buffer to hold frame data
            buf = create_string_buffer(frame.uiImgSize)
            pointer_data = c_void_p(frame.pBuffer + frame.usHeader)
            memmove(buf, pointer_data, frame.uiImgSize)

            # Convert to bytes and then to numpy array
            data = bytes(buf)
            
            # Determine dtype based on element size
            if frame.ucElemBytes == 1:
                dtype = np.uint8
            elif frame.ucElemBytes == 2:
                dtype = np.uint16
            else:
                self.__logger.warning(f"Unsupported element size: {frame.ucElemBytes}")
                return None
                
            # Convert to numpy array
            image_np = np.frombuffer(data, dtype=dtype)
            
            # Handle color vs mono images
            if frame.ucChannels == 1:
                # Mono image
                image_np = image_np.reshape((frame.usHeight, frame.usWidth))
            elif frame.ucChannels == 3:
                # Color image (RGB)
                image_np = image_np.reshape((frame.usHeight, frame.usWidth, 3))
            else:
                self.__logger.warning(f"Unsupported channel count: {frame.ucChannels}")
                return None

            return image_np
            
        except Exception as e:
            self.__logger.error(f"Failed to convert frame: {e}")
            return None

    def set_exposure_time(self, exposure_time):
        """Set camera exposure time in milliseconds."""
        try:
            self.exposure_time = exposure_time
            # Convert ms to microseconds for Tucsen API
            exposure_us = exposure_time * 1000
            
            # First set capability if needed, then set property value
            ret = TUCAM_Prop_SetValue(self.camera_handle, TUCAM_IDPROP.TUIDP_EXPOSURETM.value, 
                                    c_double(exposure_us), 0)
            if not self._is_success(ret):
                self.__logger.warning(f"Failed to set exposure time: {ret}")
            else:
                self.__logger.debug(f"Exposure time set to {exposure_time}ms ({exposure_us}us)")
                
        except Exception as e:
            self.__logger.error(f"Failed to set exposure time: {e}")

    def set_gain(self, gain):
        """Set camera gain."""
        try:
            self.gain = gain
            ret = TUCAM_Prop_SetValue(self.camera_handle, TUCAM_IDPROP.TUIDP_GLOBALGAIN.value, 
                                    c_double(float(gain)), 0)
            if not self._is_success(ret):
                self.__logger.warning(f"Failed to set gain: {ret}")
            else:
                self.__logger.debug(f"Gain set to {gain}")
                
        except Exception as e:
            self.__logger.error(f"Failed to set gain: {e}")

    def set_blacklevel(self, blacklevel):
        """Set camera black level."""
        try:
            self.blacklevel = blacklevel
            ret = TUCAM_Prop_SetValue(self.camera_handle, TUCAM_IDPROP.TUIDP_BLACKLEVEL.value, 
                                    c_double(float(blacklevel)), 0)
            if not self._is_success(ret):
                self.__logger.warning(f"Failed to set black level: {ret}")
            else:
                self.__logger.debug(f"Black level set to {blacklevel}")
                
        except Exception as e:
            self.__logger.error(f"Failed to set black level: {e}")

    def setBinning(self, binning=1):
        """Set camera binning."""
        try:
            self.binning = binning
            # Implement binning setting based on Tucsen API
            # This would depend on the specific camera model
            
        except Exception as e:
            self.__logger.error(f"Failed to set binning: {e}")

    def getLast(self, returnFrameNumber: bool = False, timeout: float = 1.0, auto_trigger: bool = True):
        """Get the latest frame from the buffer."""
        # Handle software trigger if necessary
        if auto_trigger and self.trigger_source.lower() in ("software", "software trigger"):
            self.send_trigger()

        # Wait for frame
        t0 = time.time()
        while not self.frame_buffer:
            if time.time() - t0 > timeout:
                self.__logger.warning("Timeout waiting for frame")
                return None if not returnFrameNumber else (None, None)
            time.sleep(0.001)

        with self._frame_lock:
            frame = self.frame_buffer[-1] if self.frame_buffer else None
            frame_id = self.frameid_buffer[-1] if self.frameid_buffer else -1

        if returnFrameNumber:
            return (frame, frame_id)
        return frame

    def flushBuffer(self):
        """Clear the frame buffer."""
        with self._frame_lock:
            self.frameid_buffer.clear()
            self.frame_buffer.clear()

    def getLastChunk(self):
        """Get all frames from buffer and clear it."""
        with self._frame_lock:
            frames = list(self.frame_buffer)
            ids = list(self.frameid_buffer)
            self.flushBuffer()

        self.lastFrameFromBuffer = frames[-1] if frames else None
        return frames, ids

    def setPropertyValue(self, property_name, property_value):
        """Set a camera property value."""
        try:
            if property_name == "exposure_time":
                self.set_exposure_time(property_value)
            elif property_name == "gain":
                self.set_gain(property_value)
            elif property_name == "blacklevel":
                self.set_blacklevel(property_value)
            elif property_name == "binning":
                self.setBinning(property_value)
            elif property_name == "frame_rate":
                self.frame_rate = property_value
                self.__logger.debug(f"Frame rate set to {property_value} (stored only)")
            else:
                self.__logger.warning(f"Unknown property: {property_name}")
                
        except Exception as e:
            self.__logger.error(f"Failed to set property {property_name}: {e}")

    def getPropertyValue(self, property_name):
        """Get a camera property value."""
        try:
            if property_name == "exposure_time":
                return self.exposure_time
            elif property_name == "gain":
                return self.gain
            elif property_name == "blacklevel":
                return self.blacklevel
            elif property_name == "binning":
                return self.binning
            elif property_name == "frame_rate":
                return self.frame_rate
            elif property_name == "image_width":
                return self.SensorWidth
            elif property_name == "image_height":
                return self.SensorHeight
            else:
                self.__logger.warning(f"Unknown property: {property_name}")
                return None
                
        except Exception as e:
            self.__logger.error(f"Failed to get property {property_name}: {e}")
            return None

    def getTriggerTypes(self) -> List[str]:
        """Return available trigger types."""
        return [
            "Continuous",
            "Software Trigger",
            "External Trigger"
        ]

    def getTriggerSource(self) -> str:
        """Get current trigger source."""
        return self.trigger_source

    def setTriggerSource(self, trigger_source):
        """Set trigger source."""
        try:
            self.trigger_source = trigger_source
            
            if trigger_source.lower() in ("continuous", "free run"):
                # Set continuous mode
                ret = TUCAM_Capa_SetValue(self.camera_handle, 
                                        TUCAM_IDCAPA.TUIDC_TRIGGERMODES.value, 
                                        0)  # Placeholder value
            elif trigger_source.lower() in ("software", "software trigger"):
                # Set software trigger mode
                ret = TUCAM_Capa_SetValue(self.camera_handle, 
                                        TUCAM_IDCAPA.TUIDC_TRIGGERMODES.value, 
                                        1)  # Placeholder value
            elif trigger_source.lower() in ("external", "external trigger"):
                # Set external trigger mode
                ret = TUCAM_Capa_SetValue(self.camera_handle, 
                                        TUCAM_IDCAPA.TUIDC_TRIGGERMODES.value, 
                                        2)  # Placeholder value
            
        except Exception as e:
            self.__logger.error(f"Failed to set trigger source: {e}")

    def send_trigger(self):
        """Send software trigger."""
        try:
            ret = TUCAM_Cap_DoSoftwareTrigger(self.camera_handle)
            if self._is_success(ret):
                return True
            else:
                self.__logger.warning(f"Failed to send software trigger: {ret}")
                return False
                
        except Exception as e:
            self.__logger.error(f"Failed to send trigger: {e}")
            return False

    def openPropertiesGUI(self):
        """Open camera properties GUI (placeholder)."""
        self.__logger.info("Properties GUI not implemented for Tucsen camera")

    def getFrameNumber(self):
        """Get current frame number."""
        return self.frameNumber

    def force_reconnect(self):
        """Force reconnect to camera - useful when camera gets stuck."""
        self.__logger.info("Forcing camera reconnection...")
        try:
            # Close current connection
            self.close()
            time.sleep(0.5)  # Wait for cleanup
            
            # Force cleanup any remaining resources
            self.force_cleanup()
            time.sleep(0.5)
            
            # Reinitialize
            self._open_camera(self.cameraNo)
            self.__logger.info("Camera reconnection successful")
            
        except Exception as e:
            self.__logger.error(f"Failed to force reconnect: {e}")
            raise

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
