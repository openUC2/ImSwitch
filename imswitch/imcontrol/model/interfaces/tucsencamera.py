import collections
import time
import numpy as np
from typing import List, Optional
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

class TucsenMode(Enum):
    HDR = 0
    CMS = 1
    HIGH_SPEED = 2


# ----------------------------------------------------------------------------
#  Callback signature for Tucsen
# ----------------------------------------------------------------------------
TUCSEN_CALLBACK_SIG = CFUNCTYPE(
    None,
    c_void_p  # user data pointer
)


class CameraTucsen:
    """Callback-based Tucsen wrapper compatible with ImSwitch - no threading needed."""

    @staticmethod
    def force_cleanup():
        """Emergency cleanup that can be called even without a proper instance"""
        print("Performing emergency cleanup...")
        try:
            # Try to uninit any existing API session
            TUCAM_Api_Uninit()
            time.sleep(0.5)
        except Exception:
            pass
        
        try:
            # Try to init and immediately uninit to reset state
            dummy_init = TUCAM_INIT(0, b'./')
            TUCAM_Api_Init(pointer(dummy_init), 1000)
            time.sleep(0.1)
            TUCAM_Api_Uninit()
            time.sleep(0.5)
        except Exception:
            pass
        
        print("Emergency cleanup completed")

    @staticmethod
    def _rc(ret) -> int:
        try:
            return int(ret.value) if hasattr(ret, "value") else int(ret)
        except Exception:
            return 0

    @staticmethod
    def _ok(ret) -> bool:
        try:
            success = TUCAMRET.TUCAMRET_SUCCESS.value
        except Exception:
            success = 0
        return CameraTucsen._rc(ret) == success

    def __init__(self, cameraNo=None, exposure_time=10, gain=0, frame_rate=-1, blacklevel=100, isRGB=False, binning=1):
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=False)

        self.model = "CameraTucsen"
        self.shape = (0, 0)
        self.is_connected = False
        self.is_streaming = False

        self.blacklevel = blacklevel
        self.exposure_time = exposure_time
        self.gain = gain
        self.frame_rate = frame_rate
        self.cameraNo = cameraNo if cameraNo is not None else 0
        self.isRGB = bool(isRGB)
        self.binning = binning

        self.NBuffer = 5
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)

        self.SensorHeight = 0
        self.SensorWidth = 0

        self.lastFrameFromBuffer = None
        self.lastFrameId = -1
        self.frameNumber = -1

        # Callback-based approach - no threading
        self._current_frame: Optional[np.ndarray] = None
        self.camera_handle = None

        if not TUCSEN_SDK_AVAILABLE:
            raise Exception("Tucsen SDK not available")

        if TUCSEN_PLATFORM == "windows":
            # Use the tucam_win directory as config path to avoid conflicts with other camera SDKs
            import os
            tucam_dir = os.path.dirname(os.path.abspath(__file__))
            tucam_win_dir = os.path.join(tucam_dir, 'tucam_win')
            self.Path = tucam_win_dir
            self.TUCAMINIT = None
            self.TUCAMOPEN = None

        self.force_cleanup()
       
        self._open_camera(self.cameraNo)
        self.trigger_source = "Continuous"
        self.isFlatfielding = False
        self.flatfieldImage = None

        # Setup callback like HIK camera
        self._setup_callback()

    def _setup_callback(self):
        """Setup callback function for frame capture like HIK camera"""
        self._sdk_cb = self._wrap_callback(self._on_frame_callback)
        try:
            ret = TUCAM_Buf_DataCallBack(self.camera_handle, self._sdk_cb, None)
            if ret != TUCAMRET.TUCAMRET_SUCCESS:
                self.__logger.warning(f"Callback registration returned: {ret}")
            else:
                self.__logger.info("Callback registered successfully")
        except Exception as e:
            self.__logger.error(f"Failed to register callback: {e}")

    def _wrap_callback(self, user_cb):
        """Wrap user callback for SDK"""
        def _callback():
            try:
                # Get frame data using TUCAM_Buf_GetData like the example
                m_rawHeader = TUCAM_RAWIMG_HEADER()
                result = TUCAM_Buf_GetData(self.camera_handle, pointer(m_rawHeader))
                
                if result == TUCAMRET.TUCAMRET_SUCCESS:
                    # Convert to numpy array
                    frame = self._convert_raw_to_numpy(m_rawHeader)
                    if frame is not None:
                        user_cb(frame, m_rawHeader.uiIndex, m_rawHeader.dblTimeStamp)
                else:
                    pass  # No frame available
                    
            except Exception as e:
                self.__logger.error(f"Callback error: {e}")
        
        return BUFFER_CALLBACK(_callback)

    def _convert_raw_to_numpy(self, rawHeader):
        """Convert TUCAM_RAWIMG_HEADER to numpy array"""
        try:
            width = rawHeader.usWidth
            height = rawHeader.usHeight
            channels = rawHeader.ucChannels
            elem_bytes = rawHeader.ucElemBytes
            img_size = rawHeader.uiImgSize
            
            if rawHeader.pImgData == 0 or img_size == 0:
                return None
                
            # Create buffer from raw data
            if elem_bytes == 1:
                dtype = np.uint8
            elif elem_bytes == 2:
                dtype = np.uint16
            else:
                dtype = np.uint8
                
            # Extract data from pointer
            data_ptr = cast(rawHeader.pImgData, POINTER(c_ubyte * img_size))
            buf = np.frombuffer(data_ptr.contents, dtype=dtype)
            
            # Reshape based on channels
            if channels == 1:
                frame = buf.reshape(height, width)
            elif channels == 3:
                frame = buf.reshape(height, width, 3)
            else:
                frame = buf.reshape(height, width)
                
            return frame.copy()  # Make a copy to avoid memory issues
            
        except Exception as e:
            self.__logger.error(f"Frame conversion error: {e}")
            return None

    def _on_frame_callback(self, frame: np.ndarray, frame_id: int, timestamp: float):
        """Handle new frame from callback - similar to HIK camera"""
        try:
            self.frame_buffer.append(frame)
            self.frameid_buffer.append(frame_id)
            self.frameNumber = frame_id
            self._current_frame = frame
            

        except Exception as e:
            self.__logger.error(f"Frame handling error: {e}")

    # -------- Open/close ----------------------------------------------------
    def _open_camera(self, camera_index: int):
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
        ret = TUCAM_Api_Init()
        # Do not hard-fail on non-zero here; Linux bindings may return 0/None
        opCam = TUCAM_OPEN()
        opCam.uiIdxOpen = camera_index
        ret = TUCAM_Dev_Open(byref(opCam))
        if not self._ok(ret):
            raise Exception(f"Failed to open Tucsen camera {camera_index}: {ret}")
        self.camera_handle = opCam.hIdxTUCam
        self._get_sensor_info()
        self.is_connected = True
        self.__logger.info(f"Opened Tucsen camera {camera_index} (Linux)")

    def _open_camera_windows(self, camera_index: int):
        """Windows-specific camera initialization following working pattern exactly."""
        # Cleanup any existing session before opening (like working code)
        self.__logger.info("Cleaning up any existing camera session...")
        try:
            self._shutdown_capture()
        except:
            pass  # May not have camera_handle yet
            
        # Try to close any existing handle  
        if hasattr(self, 'TUCAMOPEN') and getattr(self.TUCAMOPEN, 'hIdxTUCam', 0) != 0:
            try:
                TUCAM_Dev_Close(self.TUCAMOPEN.hIdxTUCam)
                self.__logger.info("Closed existing camera handle")
            except Exception as e:
                self.__logger.info(f"Error closing existing handle: {e}")
                
        # Initialize exactly like working code
        self.TUCAMINIT = TUCAM_INIT(0, self.Path.encode('utf-8'))
        self.TUCAMOPEN = TUCAM_OPEN(0, 0)
        ret = TUCAM_Api_Init(pointer(self.TUCAMINIT), 2000)  # Match working code timeout
        self.__logger.info(f"API Init result: {ret}")
        self.__logger.info(f"Camera count: {self.TUCAMINIT.uiCamCount}")
        
        if self.TUCAMINIT.uiCamCount == 0:
            raise Exception("No Tucsen cameras found")
        if camera_index >= self.TUCAMINIT.uiCamCount:
            raise Exception(f"Camera index {camera_index} not available. Found {self.TUCAMINIT.uiCamCount} cameras")
        
        # Open camera exactly like working code
        self.TUCAMOPEN = TUCAM_OPEN(camera_index, 0)
        self.__logger.info(f"Opening camera index {camera_index}")
        TUCAM_Dev_Open(pointer(self.TUCAMOPEN))
        
        if self.TUCAMOPEN.hIdxTUCam == 0:
            raise Exception(f"Failed to open Tucsen camera {camera_index}")
        else:
            self.__logger.info("Open the camera success!")
            
        self.camera_handle = self.TUCAMOPEN.hIdxTUCam
        self.__logger.info(f"Camera handle set to: {self.camera_handle}")
        
        # Skip immediate test - it interferes with threaded capture
        # The camera is working (was verified), go straight to normal operation
        
        self._get_sensor_info()
        self.is_connected = True
        self.__logger.info(f"Opened Tucsen camera {camera_index} (Windows)")

    def _test_immediate_capture(self):
        """Test capture immediately after camera opens to verify handle works."""
        self.__logger.info("Testing immediate capture to verify camera handle...")
        
        try:
            # Check current trigger mode first
            m_tgr = TUCAM_TRIGGER_ATTR()
            try:
                ret = TUCAM_Cap_GetTrigger(self.camera_handle, pointer(m_tgr))
                self.__logger.info(f"Current trigger mode: {m_tgr.nTgrMode}")
                self.__logger.info(f"Trigger buffer frames: {m_tgr.nBufFrames}")
            except Exception as e:
                self.__logger.error(f"Failed to get trigger mode: {e}")
            
            # Force set to continuous sequence mode like working example
            m_tgr.nTgrMode = TUCAM_CAPTURE_MODES.TUCCM_SEQUENCE.value
            m_tgr.nBufFrames = 10  # Like working examples
            try:
                ret = TUCAM_Cap_SetTrigger(self.camera_handle, m_tgr)
                self.__logger.info(f"Set trigger mode result: {ret}")
            except Exception as e:
                self.__logger.error(f"Failed to set trigger mode: {e}")
            
            # Follow exact working pattern from the standalone code
            m_frame = TUCAM_FRAME()
            m_frformat = TUFRM_FORMATS
            m_capmode = TUCAM_CAPTURE_MODES

            m_frame.pBuffer = 0
            m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
            m_frame.uiRsdSize = 1

            # Allocate buffer using the handle we just got
            self.__logger.info(f"Allocating buffer with handle: {self.camera_handle}")
            ret = TUCAM_Buf_Alloc(self.camera_handle, pointer(m_frame))
            self.__logger.info(f"Immediate test - Buffer allocation result: {ret}")
            
            # Start capture
            self.__logger.info("Starting capture for immediate test...")
            ret = TUCAM_Cap_Start(self.camera_handle, m_tgr.nTgrMode)
            self.__logger.info(f"Immediate test - Capture start result: {ret}")

            # Try to get one frame
            self.__logger.info("Waiting for one frame in immediate test...")
            try:
                result = TUCAM_Buf_WaitForFrame(self.camera_handle, pointer(m_frame), 2000)
                self.__logger.info(f"IMMEDIATE TEST SUCCESS! Frame captured: {result}")
                self.__logger.info(
                    f"Frame: width:{m_frame.usWidth}, height:{m_frame.usHeight}, "
                    f"channels:{m_frame.ucChannels}, elembytes:{m_frame.ucElemBytes}, "
                    f"image size:{m_frame.uiImgSize}"
                )
            except Exception as frame_error:
                self.__logger.error(f"IMMEDIATE TEST FAILED - Frame wait error: {frame_error}")

            # Cleanup immediately
            self.__logger.info("Cleaning up immediate test...")
            try:
                TUCAM_Buf_AbortWait(self.camera_handle)
            except:
                pass
            try:
                TUCAM_Cap_Stop(self.camera_handle)
            except:
                pass
            try:
                TUCAM_Buf_Release(self.camera_handle)
            except:
                pass
            
            self.__logger.info("Immediate test completed")
            
        except Exception as e:
            self.__logger.error(f"Immediate test failed: {e}")
            import traceback
            traceback.print_exc()

    def _get_sensor_info(self):
        """Get sensor info without allocating buffers (follow working code pattern)."""
        try:
            # Don't allocate buffers during initialization like the working code
            # Just use default values that will be updated when frames are captured
            self.SensorWidth = 3000  # From working code output
            self.SensorHeight = 3000  # From working code output
            self.__logger.info(f"Using default sensor dimensions: {self.SensorWidth} x {self.SensorHeight}")
        except Exception as e:
            self.__logger.warning(f"Sensor info fallback: {e}")
            self.SensorWidth = 3000
            self.SensorHeight = 3000
        self.shape = (self.SensorHeight, self.SensorWidth)

    def openPropertiesGUI(self): 
        """Open camera properties GUI (placeholder)."""
        self.__logger.info("Properties GUI not implemented for Tucsen camera")

    def setPropertyValue(self, property_name, property_value):
        """Unified setter used by TucsenCamManager."""
        try:
            key = str(property_name).strip().lower().replace(" ", "_")
            if key in ("exposure", "exposure_time"):
                self.set_exposure_time(float(property_value))
                return self.exposure_time
            elif key == "gain":
                self.set_gain(float(property_value))
                return self.gain
            elif key == "blacklevel":
                self.set_blacklevel(float(property_value))
                return self.blacklevel
            elif key == "binning":
                self.setBinning(int(property_value))
                return self.binning
            elif key == "frame_rate":
                self.frame_rate = float(property_value)
                return self.frame_rate
            elif key in ("trigger_source", "trigger"):
                self.setTriggerSource(str(property_value))
                return self.trigger_source
            elif key in ("image_width", "width"):
                # read-only; return current
                return int(self.SensorWidth)
            elif key in ("image_height", "height"):
                # read-only; return current
                return int(self.SensorHeight)
            elif key in ("flat_fielding", "flatfielding"):
                self.isFlatfielding = bool(property_value)
                return self.isFlatfielding
            else:
                self.__logger.warning(f"Unknown property '{property_name}'")
                return self.getPropertyValue(property_name)
        except Exception as e:
            self.__logger.error(f"setPropertyValue('{property_name}', {property_value}) failed: {e}")
            return None


    def getPropertyValue(self, property_name):
        """Unified getter used by TucsenCamManager."""
        try:
            key = str(property_name).strip().lower().replace(" ", "_")
            if key in ("exposure", "exposure_time"):
                return self.exposure_time
            elif key == "gain":
                return self.gain
            elif key == "blacklevel":
                return self.blacklevel
            elif key == "binning":
                return self.binning
            elif key == "frame_rate":
                return self.frame_rate
            elif key in ("trigger_source", "trigger"):
                return self.trigger_source
            elif key in ("image_width", "width"):
                return int(self.SensorWidth)
            elif key in ("image_height", "height"):
                return int(self.SensorHeight)
            elif key == "model":
                return self.model
            elif key in ("isrgb",):
                return bool(self.isRGB)
            elif key in ("flat_fielding", "flatfielding"):
                return bool(self.isFlatfielding)
            else:
                self.__logger.warning(f"Unknown property '{property_name}'")
                return None
        except Exception as e:
            self.__logger.error(f"getPropertyValue('{property_name}') failed: {e}")
            return None

    # -------- Live control (callback-based) ---------------------------------------
    def start_live(self):
        """Start streaming using callback approach - no threading needed"""
        if self.is_streaming:
            return
            
        self.__logger.info("Starting Tucsen callback-based streaming...")
        self.flushBuffer()
        
        try:
            # Configure trigger mode for continuous capture
            m_tgr = TUCAM_TRIGGER_ATTR()
            try:
                TUCAM_Cap_GetTrigger(self.camera_handle, pointer(m_tgr))
                m_tgr.nTgrMode = TUCAM_CAPTURE_MODES.TUCCM_SEQUENCE.value
                m_tgr.nBufFrames = 10
                TUCAM_Cap_SetTrigger(self.camera_handle, m_tgr)
            except Exception as e:
                self.__logger.warning(f"Could not set trigger mode: {e}")

            # Allocate buffer
            m_frame = TUCAM_FRAME()
            m_frformat = TUFRM_FORMATS
            m_capmode = TUCAM_CAPTURE_MODES

            m_frame.pBuffer = 0
            m_frame.ucFormatGet = m_frformat.TUFRM_FMT_USUAl.value
            m_frame.uiRsdSize = 1

            ret = TUCAM_Buf_Alloc(self.camera_handle, pointer(m_frame))
            if ret != TUCAMRET.TUCAMRET_SUCCESS:
                raise Exception(f"Buffer allocation failed: {ret}")
                
            self.__logger.info(f"Buffer allocated successfully")

            # Start capture
            ret = TUCAM_Cap_Start(self.camera_handle, m_capmode.TUCCM_SEQUENCE.value)
            if ret != TUCAMRET.TUCAMRET_SUCCESS:
                raise Exception(f"Capture start failed: {ret}")
                
            self.__logger.info("Capture started successfully")
            self.is_streaming = True
            self.__logger.info("Tucsen callback-based streaming started")
            
        except Exception as e:
            self.__logger.error(f"Failed to start streaming: {e}")
            self.is_streaming = False
            raise

    def stop_live(self):
        """Stop streaming"""
        if not self.is_streaming:
            return
            
        self.__logger.info("Stopping Tucsen streaming...")
        
        try:
            # Stop capture and release buffer
            try:
                TUCAM_Buf_AbortWait(self.camera_handle)
            except Exception:
                pass
            try:
                TUCAM_Cap_Stop(self.camera_handle)
            except Exception:
                pass
            try:
                TUCAM_Buf_Release(self.camera_handle)
            except Exception:
                pass
                
            self.is_streaming = False
            self.__logger.info("Tucsen streaming stopped")
            
        except Exception as e:
            self.__logger.error(f"Error stopping streaming: {e}")

    suspend_live = stop_live

    def prepare_live(self):
        """Prepare for streaming (no-op for callback approach)"""
        pass



    # -------- Frame conversion ----------------------------------------------
    def _convert_frame_to_numpy(self, frame: "TUCAM_FRAME") -> Optional[np.ndarray]:
        try:
            if frame.uiImgSize == 0 or frame.pBuffer == 0:
                return None
            buf = create_string_buffer(frame.uiImgSize)
            pointer_data = c_void_p(frame.pBuffer + frame.usHeader)
            memmove(buf, pointer_data, frame.uiImgSize)
            data = bytes(buf)
            if frame.ucElemBytes == 1:
                dtype = np.uint8
            elif frame.ucElemBytes == 2:
                dtype = np.uint16
            else:
                self.__logger.warning(f"Unsupported elem size: {frame.ucElemBytes}")
                return None
            arr = np.frombuffer(data, dtype=dtype)
            if frame.ucChannels == 1:
                arr = arr.reshape((int(frame.usHeight), int(frame.usWidth)))
            elif frame.ucChannels == 3:
                arr = arr.reshape((int(frame.usHeight), int(frame.usWidth), 3))
            else:
                self.__logger.warning(f"Unsupported channels: {frame.ucChannels}")
                return None
            # Flatfield (optional)
            if self.isFlatfielding and self.flatfieldImage is not None:
                try:
                    arr = arr.astype(np.float32)
                    arr = arr - self.flatfieldImage
                except Exception:
                    pass
            return arr
        except Exception as e:
            self.__logger.error(f"Convert frame failed: {e}")
            return None

    # -------- Parameters & properties ---------------------------------------
    def get_camera_parameters(self):
        return {
            "model": self.model,
            "isRGB": self.isRGB,
            "width": self.SensorWidth,
            "height": self.SensorHeight,
            "exposure_time": self.exposure_time,
            "gain": self.gain,
            "blacklevel": self.blacklevel,
            "binning": self.binning,
        }

    def get_gain(self):
        try:
            return (self.gain, 0.0, 100.0)
        except Exception as e:
            self.__logger.error(f"Failed to get gain: {e}")
            return (None, None, None)

    def get_exposuretime(self):
        try:
            return (self.exposure_time, 0.1, 10000.0)
        except Exception as e:
            self.__logger.error(f"Failed to get exposure time: {e}")
            return (None, None, None)

    def set_exposure_time(self, exposure_time):
        try:
            self.exposure_time = exposure_time
            exposure_ms = float(exposure_time) 
            ret = TUCAM_Prop_SetValue(self.camera_handle, TUCAM_IDPROP.TUIDP_EXPOSURETM.value, c_double(exposure_ms), 0)
            if not self._ok(ret):
                self.__logger.warning(f"Set exposure returned {ret}")
        except Exception as e:
            self.__logger.error(f"Failed to set exposure time: {e}")

    def set_gain(self, gain):
        try:
            #
            self.gain = gain
            ret = TUCAM_Prop_SetValue(self.camera_handle, TUCAM_IDPROP.TUIDP_GLOBALGAIN.value, c_double(float(gain)), 0)
            if not self._ok(ret):
                self.__logger.warning(f"Set gain returned {ret}")
        except Exception as e:
            self.__logger.error(f"Failed to set gain: {e}")

    def set_blacklevel(self, blacklevel):
        try:
            self.blacklevel = blacklevel
            ret = TUCAM_Prop_SetValue(self.camera_handle, TUCAM_IDPROP.TUIDP_BLACKLEVEL.value, c_double(float(blacklevel)), 0)
            if not self._ok(ret):
                self.__logger.warning(f"Set blacklevel returned {ret}")
        except Exception as e:
            self.__logger.error(f"Failed to set blacklevel: {e}")

    def setBinning(self, binning=1):
        try:
            self.binning = binning
            # TODO: apply via Tucsen API if available for your model
        except Exception as e:
            self.__logger.error(f"Failed to set binning: {e}")

    # -------- Buffer & retrieval --------------------------------------------
    def getLast(self, returnFrameNumber: bool = False, timeout: float = 1.0, auto_trigger: bool = True):
        t0 = time.time()
        while not self.frame_buffer:
            if time.time() - t0 > timeout:
                return (None, None) if returnFrameNumber else None
            time.sleep(0.001)
        frame = self.frame_buffer[-1] if self.frame_buffer else None
        frame_id = self.frameid_buffer[-1] if self.frameid_buffer else -1
        return (frame, frame_id) if returnFrameNumber else frame

    def flushBuffer(self):
        self.frameid_buffer.clear()
        self.frame_buffer.clear()

    def getLastChunk(self):
        frames = list(self.frame_buffer)
        ids = list(self.frameid_buffer)
        self.flushBuffer()
        self.lastFrameFromBuffer = frames[-1] if frames else None
        return frames, ids

    # -------- Triggering -----------------------------------------------------
    def getTriggerTypes(self) -> List[str]:
        return ["Continuous", "Software Trigger", "External Trigger"]

    def getTriggerSource(self) -> str:
        return self.trigger_source

    def setTriggerSource(self, trigger_source):
        try:
            self.trigger_source = trigger_source
            ts = trigger_source.strip().lower()
            if ts in ("continuous", "continous", "free run"):
                val = 0
            elif ts in ("software", "software trigger"):
                val = 1
            elif ts in ("external", "external trigger"):
                val = 2
            else:
                val = 0
            try:
                TUCAM_Capa_SetValue(self.camera_handle, TUCAM_IDCAPA.TUIDC_TRIGGERMODES.value, val)
            except Exception:
                pass
        except Exception as e:
            self.__logger.error(f"Failed to set trigger source: {e}")

    def send_trigger(self):
        try:
            ret = TUCAM_Cap_DoSoftwareTrigger(self.camera_handle)
            return self._ok(ret)
        except Exception as e:
            self.__logger.error(f"Failed to send trigger: {e}")
            return False

    # -------- Flatfield stubs ------------------------------------------------
    def setFlatfieldImage(self, flatfieldImage, isFlatfielding):
        self.flatfieldImage = flatfieldImage
        self.isFlatfielding = bool(isFlatfielding)

    def recordFlatfieldImage(self, n=16, median=False):
        if not self._is_streaming.is_set():
            return
        frames = []
        t_end = time.time() + 2.0  # simple time cap
        while len(frames) < n and time.time() < t_end:
            f = self.getLast(timeout=0.2)
            if f is not None:
                frames.append(f.astype(np.float32))
        if frames:
            stack = np.stack(frames, axis=0)
            self.flatfieldImage = (np.median(stack, axis=0) if median else np.mean(stack, axis=0)).astype(np.float32)
            self.isFlatfielding = True

    # -------- Close ----------------------------------------------------------
    def close(self):
        try:
            if self._is_streaming.is_set():
                self.stop_live()
            if TUCSEN_PLATFORM == "linux":
                self._close_camera_linux()
            elif TUCSEN_PLATFORM == "windows":
                self._close_camera_windows()
        except Exception as e:
            self.__logger.error(f"Failed to close camera: {e}")

    def _close_camera_linux(self):
        if self.camera_handle:
            ret = TUCAM_Dev_Close(self.camera_handle)
            if not self._ok(ret):
                self.__logger.warning(f"Dev_Close returned {ret}")
        try:
            TUCAM_Api_Uninit()
        except Exception:
            pass
        self.is_connected = False
        self.__logger.info("Camera closed (Linux)")

    def _close_camera_windows(self):
        try:
            try:
                TUCAM_Buf_AbortWait(self.camera_handle)
            except Exception:
                pass
            try:
                TUCAM_Cap_Stop(self.camera_handle)
            except Exception:
                pass
            try:
                if self._m_frame is not None:
                    TUCAM_Buf_Release(self.camera_handle)
            except Exception:
                pass
            if self.camera_handle and self.camera_handle != 0:
                try:
                    ret = TUCAM_Dev_Close(self.camera_handle)
                    if not self._ok(ret):
                        self.__logger.warning(f"Dev_Close returned {ret}")
                except Exception as e:
                    self.__logger.warning(f"Error closing device: {e}")
            time.sleep(0.1)
            try:
                TUCAM_Api_Uninit()
            except Exception:
                pass
        finally:
            self.camera_handle = None
            self.is_connected = False
            self.__logger.info("Camera closed (Windows)")

    # Context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

