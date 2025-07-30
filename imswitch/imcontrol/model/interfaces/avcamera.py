import collections
import time
import numpy as np
import threading

from typing import Optional
from imswitch.imcommon.model import initLogger

# Try VmbPy first (official Allied Vision SDK), then fallback to legacy Vimba
isVmbPy = False
isVimba = False

try:
    # Try VmbPy first (official new SDK)
    from vmbpy import *
    isVmbPy = True
    print("VmbPy SDK loaded successfully")
except ImportError:
    try:
        # Fallback to legacy Vimba
        from vimba import (Vimba, FrameStatus, Frame, VimbaCameraError)
        isVimba = True
        print("Legacy Vimba SDK loaded successfully")
    except ImportError as e:
        print(e)
        print("Neither VmbPy nor legacy Vimba installed..")
        # Define dummy exception for when neither SDK is available
        class VimbaCameraError(Exception):
            pass

if not (isVmbPy or isVimba):
    print("No Allied Vision SDK available")


class CameraAV:
    def __init__(self, camera_id=None):
        """
        :param camera_id: Index (int) or ID (string) of the Allied Vision camera to open.
                          If None, the first available camera is used.
        """
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=True)

        if not (isVmbPy or isVimba):
            raise RuntimeError("Neither VmbPy nor legacy VimbaPython installed or found.")

        # Initialize based on available SDK
        if isVmbPy:
            # VmbPy - keep persistent context managers
            self._vmb_system = None
            self._camera_context = None
        else:
            # Legacy Vimba initialization
            self._vimba = Vimba.get_instance()
            self._vimba.__enter__()

        self._camera = None
        self._running = False
        self._streaming = False
        self.frame_buffer = collections.deque(maxlen=60)
        self.frame = None
        self.frame_id = 0
        self.frame_id_last = -1
        self._frame_lock = threading.Lock()

        self.model = "AlliedVisionCamera"
        self.sensor_width = 0
        self.sensor_height = 0
        
        # Add isRGB property for compatibility with MCTController
        self.isRGB = False

        # For storing any ROI parameters (OffsetX, OffsetY, Width, Height)
        self.hpos = 0
        self.vpos = 0
        self.hsize = 0
        self.vsize = 0

        self._open_camera(camera_id)
        self.__logger.debug("CameraAV initialized.")

    def _open_camera(self, camera_id):
        if isVmbPy:
            # VmbPy camera discovery and opening with persistent context managers
            self._vmb_system = VmbSystem.get_instance()
            self._vmb_system.__enter__()
            
            try:
                cams = self._vmb_system.get_all_cameras()
                if not cams:
                    raise RuntimeError("No Allied Vision cameras found.")

                if camera_id is None:
                    self._camera = cams[0]
                else:
                    # If camera_id is an integer, interpret as index
                    if isinstance(camera_id, int):
                        if camera_id < 0 or camera_id >= len(cams):
                            raise RuntimeError(f"Invalid camera index: {camera_id}")
                        self._camera = cams[camera_id]
                    else:
                        # Otherwise, treat it as a string-based camera ID
                        try:
                            self._camera = self._vmb_system.get_camera_by_id(camera_id)
                        except VmbCameraError as e:
                            raise RuntimeError(f"Failed to open camera with ID '{camera_id}'.") from e

                # Try to close camera if it was left open from previous session
                try:
                    self._camera.close()
                except:
                    pass  # Ignore if camera wasn't open
                    
                # Enter camera context and keep it persistent
                self._camera_context = self._camera.__enter__()
                
                # Try to adjust GeV packet size. This Feature is only available for GigE cameras.
                try:
                    streams = self._camera.get_streams()
                    if streams:
                        stream = streams[0]
                        if hasattr(stream, 'GVSPAdjustPacketSize'):
                            stream.GVSPAdjustPacketSize.run()
                            while not stream.GVSPAdjustPacketSize.is_done():
                                time.sleep(0.001)  # Small delay
                except (AttributeError, VmbFeatureError):
                    pass  # Not a GigE camera or feature not available
                
                # Set acquisition mode
                try:
                    self._camera.get_feature_by_name("AcquisitionMode").set("Continuous")
                except Exception as e:
                    self.__logger.warning(f"Could not set AcquisitionMode: {e}")
                
                # Get camera model name
                try:
                    self.model = self._camera.get_name()
                except:
                    self.model = "AlliedVisionCamera"
                    
                try:
                    # Read sensor dimensions
                    self.sensor_width = self._camera.get_feature_by_name("SensorWidth").get()
                    self.sensor_height = self._camera.get_feature_by_name("SensorHeight").get()
                except Exception:
                    self.sensor_width = 0
                    self.sensor_height = 0
                    
                # Auto-detect RGB capability based on pixel format
                self._detect_rgb()
                
            except Exception as e:
                self._cleanup_context_managers()
                raise e
        else:
            # Legacy Vimba camera discovery and opening
            cams = self._vimba.get_all_cameras()
            if not cams:
                raise RuntimeError("No Allied Vision cameras found.")

            if camera_id is None:
                self._camera = cams[0]
            else:
                # If camera_id is an integer, interpret as index
                if isinstance(camera_id, int):
                    if camera_id < 0 or camera_id >= len(cams):
                        raise RuntimeError(f"Invalid camera index: {camera_id}")
                    self._camera = cams[camera_id]
                else:
                    # Otherwise, treat it as a string-based camera ID
                    try:
                        self._camera = self._vimba.get_camera_by_id(camera_id)
                    except VimbaCameraError as e:
                        raise RuntimeError(f"Failed to open camera with ID '{camera_id}'.") from e

            # Try to close camera if it was left open from previous session
            try:
                self._camera.close()
            except:
                pass  # Ignore if camera wasn't open
                
            self._camera._open()
            self._camera.get_feature_by_name("AcquisitionMode").set("Continuous")
            self.model = self._camera.get_name()
            try:
                # Read sensor dimensions
                self.sensor_width = self._camera.get_feature_by_name("SensorWidth").get()
                self.sensor_height = self._camera.get_feature_by_name("SensorHeight").get()
            except Exception:
                self.sensor_width = 0
                self.sensor_height = 0

        # Default ROI equals full sensor
        self.hpos = 0
        self.vpos = 0
        self.hsize = self.sensor_width
        self.vsize = self.sensor_height

        self.__logger.debug(f"Opened camera '{self.model}' (ID/index: {camera_id}).")

    def _detect_rgb(self):
        """Auto-detect if camera supports RGB based on pixel format or model name"""
        try:
            if isVmbPy:
                # Check available pixel formats for color capability
                pixel_format = self._camera.get_feature_by_name("PixelFormat").get()
                # Common color formats: Bayer patterns, RGB, YUV
                color_formats = ['RGB', 'BGR', 'Bayer', 'YUV', 'Color']
                self.isRGB = any(fmt in str(pixel_format) for fmt in color_formats)
                
                # Also check model name for UC or color indicators
                model_name = self.model.upper()
                if 'UC' in model_name or 'COLOR' in model_name or 'C' in model_name:
                    self.isRGB = True
            else:
                # Legacy Vimba detection
                try:
                    pixel_format = self._camera.get_feature_by_name("PixelFormat").get()
                    color_formats = ['RGB', 'BGR', 'Bayer', 'YUV', 'Color']
                    self.isRGB = any(fmt in str(pixel_format) for fmt in color_formats)
                except:
                    self.isRGB = False
        except Exception as e:
            self.__logger.warning(f"Could not detect RGB capability: {e}")
            self.isRGB = False
            
        self.__logger.debug(f"Detected isRGB: {self.isRGB}")

    def _cleanup_context_managers(self):
        """Clean up VmbPy context managers"""
        if isVmbPy:
            try:
                if hasattr(self, '_camera_context') and self._camera_context:
                    self._camera.__exit__(None, None, None)
                    self._camera_context = None
            except Exception as e:
                self.__logger.warning(f"Error exiting camera context: {e}")
            
            try:
                if hasattr(self, '_vmb_system') and self._vmb_system:
                    self._vmb_system.__exit__(None, None, None)
                    self._vmb_system = None
            except Exception as e:
                self.__logger.warning(f"Error exiting VmbSystem context: {e}")

    @classmethod
    def cleanup_all_cameras(cls):
        """Class method to cleanup any lingering camera resources - useful for graceful restart"""
        if isVmbPy:
            try:
                # Try to shutdown any existing VmbSystem instances
                with VmbSystem.get_instance() as vmb:
                    cameras = vmb.get_all_cameras()
                    for cam in cameras:
                        try:
                            cam.close()
                        except:
                            pass  # Ignore errors during cleanup
            except Exception as e:
                print(f"Warning: Error during camera cleanup: {e}")
        else:
            try:
                # Try to cleanup legacy Vimba
                vimba = Vimba.get_instance()
                with vimba:
                    cameras = vimba.get_all_cameras()
                    for cam in cameras:
                        try:
                            cam.close()
                        except:
                            pass  # Ignore errors during cleanup
            except Exception as e:
                print(f"Warning: Error during legacy camera cleanup: {e}")


    
    def _frame_handler(self, cam: Camera, stream: Stream, frame: Frame):
        """Frame handler for asynchronous streaming"""
        if isVmbPy:
            # VmbPy frame handling - following the example pattern
            try:
                if frame.get_status() == FrameStatus.Complete:
                    with self._frame_lock:
                        data = frame.as_numpy_ndarray()
                        self.frame_id = frame.get_id()

                        # Apply ROI in software if the hardware ROI is not set
                        if self.vsize and self.hsize:
                            cropped = data[self.vpos:self.vpos + self.vsize,
                                          self.hpos:self.hpos + self.hsize]
                            if cropped.size == 0:
                                cropped = data
                            self.frame = cropped
                        else:
                            self.frame = data

                        self.frame_buffer.append(self.frame.copy())
                # Re-queue frame for VmbPy
                cam.queue_frame(frame)
            except Exception as e:
                self.__logger.error(f"Error in VmbPy frame handler: {e}")
        else:
            # Legacy Vimba frame handling
            if frame.get_status() == FrameStatus.Complete:
                with self._frame_lock:
                    data = frame.as_numpy_ndarray()
                    self.frame_id = frame.get_id()

                    # Apply ROI in software if the hardware ROI is not set
                    if self.vsize and self.hsize:
                        cropped = data[self.vpos:self.vpos + self.vsize,
                                      self.hpos:self.hpos + self.hsize]
                        if cropped.size == 0:
                            cropped = data
                        self.frame = cropped
                    else:
                        self.frame = data

                    self.frame_buffer.append(self.frame.copy())
            cam.queue_frame(frame)

    def start_live(self):
        if not self._running:
            self._running = True
        if not self._streaming:
            try:
                if isVmbPy:
                    # VmbPy asynchronous streaming following the example pattern
                    self._camera.start_streaming(
                        handler=self._frame_handler,
                        buffer_count=3,
                        allocation_mode=AllocationMode.AnnounceFrame
                    )
                    self._streaming = True
                    self.__logger.debug("VmbPy camera streaming started.")
                else:
                    # Legacy Vimba streaming
                    self._camera.start_streaming(
                        handler=self._frame_handler,
                        buffer_count=10
                    )
                    self._streaming = True
                    self.__logger.debug("Legacy Vimba camera streaming started.")
            except Exception as e:
                self.__logger.error(f"Failed to start streaming: {e}")
                self._streaming = False

    def stop_live(self):
        if self._streaming:
            try:
                self._camera.stop_streaming()
                self._streaming = False
                self.__logger.debug("Camera streaming stopped.")
            except Exception as e:
                self.__logger.warning(f"Error stopping camera streaming: {e}")
                self._streaming = False

    def suspend_live(self):
        # This method just stops acquisition without changing the _running state
        if self._streaming:
            try:
                self._camera.stop_streaming()
                self._streaming = False
                self.__logger.debug("Camera streaming suspended.")
            except Exception as e:
                self.__logger.warning(f"Error suspending camera streaming: {e}")
                self._streaming = False

    def close(self):
        # Stop streaming if active
        if self._streaming:
            try:
                self._camera.stop_streaming()
                self._streaming = False
            except Exception as e:
                self.__logger.warning(f"Error stopping streaming during close: {e}")
                
        # Close camera
        try:
            if isVmbPy:
                # VmbPy - no explicit close needed as we'll exit context managers
                pass
            else:
                # Legacy Vimba
                self._camera.close()
        except Exception as e:
            self.__logger.warning(f"Error closing camera: {e}")
            
        # Cleanup SDK context
        try:
            if isVmbPy:
                # Clean up VmbPy context managers
                self._cleanup_context_managers()
            else:
                # Exit legacy Vimba context if not already
                if hasattr(self, '_vimba') and self._vimba:
                    self._vimba.__exit__(None, None, None)
                    self._vimba = None
        except Exception as e:
            self.__logger.warning(f"Error during SDK cleanup: {e}")
            
        self.__logger.debug("Camera closed and SDK context cleaned up.")

    def setROI(self, hpos=None, vpos=None, hsize=None, vsize=None):
        # In principle, we can set the hardware ROI via features:
        # 'OffsetX', 'OffsetY', 'Width', 'Height'. If the camera supports it,
        # you can do so below. For safety, these lines are commented:
        #
        # if hpos is not None:
        #     self._camera.get_feature_by_name("OffsetX").set(hpos)
        # if vpos is not None:
        #     self._camera.get_feature_by_name("OffsetY").set(vpos)
        # if hsize is not None:
        #     self._camera.get_feature_by_name("Width").set(hsize)
        # if vsize is not None:
        #     self._camera.get_feature_by_name("Height").set(vsize)
        #
        # For now, we store them and do a "software" ROI in _frame_handler.
        if hpos is not None:
            self.hpos = int(hpos)
        if vpos is not None:
            self.vpos = int(vpos)
        if hsize is not None:
            self.hsize = int(hsize)
        if vsize is not None:
            self.vsize = int(vsize)
        self.frame_buffer.clear()
        self.__logger.debug(f"Set ROI to x={self.hpos}, y={self.vpos}, w={self.hsize}, h={self.vsize}")

    def getLast(self):
        # Return the most recent frame from buffer (streaming) or direct capture
        # The manager code uses is_resize, but we don't do anything with it here
        # (kept for compatibility).
        try:
            if self._streaming and len(self.frame_buffer) > 0:
                # Use the latest frame from the streaming buffer
                with self._frame_lock:
                    frame = self.frame_buffer[-1].copy()
                    return frame.copy() if frame is not None else np.zeros((100, 100))
            '''
            else:
                # Fallback to direct frame capture if not streaming
                if isVmbPy:
                    # VmbPy get frame method - use existing context managers
                    frame = self._camera.get_frame(timeout_ms=1000)
                    self.frame = frame.as_numpy_ndarray()
                else:
                    # Legacy Vimba get frame method  
                    frame = self._camera.get_frame(timeout_ms=1000)
                    if hasattr(frame, 'as_opencv_image'):
                        self.frame = frame.as_opencv_image()
                    else:
                        self.frame = frame.as_numpy_ndarray()
                return self.frame.copy() if self.frame is not None else np.zeros((100, 100))
            '''
        except Exception as e:
            self.__logger.warning(f"Error getting frame: {e}")
            # Return last known good frame or a placeholder
            if hasattr(self, 'frame') and self.frame is not None:
                return self.frame.copy()
            else:
                # Return a small placeholder frame to avoid crashes
                return np.zeros((100, 100), dtype=np.uint8)

    def getLastChunk(self):
        # Return all frames currently in buffer as a single 3D array if desired
        arr = np.array(self.frame_buffer, copy=True)
        self.frame_buffer.clear()
        return arr

    def flushBuffer(self):
        self.frame_buffer.clear()

    def openPropertiesGUI(self):
        # No-op for now
        pass

    def setPropertyValue(self, property_name, property_value):
        """
        Maps ImSwitch property names to AV camera features.
        property_name in ['exposure', 'gain', 'blacklevel', 'pixel_format', ...]
        """
        try:
            if property_name == "exposure":
                # Expect ms from Manager; convert to microseconds
                microseconds = max(1, int(property_value)) * 1000
                if isVmbPy:
                    # VmbPy - camera is already in context
                    self._camera.get_feature_by_name("ExposureTime").set(microseconds)
                else:
                    # Legacy Vimba
                    self._camera.get_feature_by_name("ExposureTime").set(microseconds)
            elif property_name == "gain":
                if isVmbPy:
                    self._camera.get_feature_by_name("Gain").set(float(property_value))
                else:
                    self._camera.get_feature_by_name("Gain").set(float(property_value))
            elif property_name == "blacklevel":
                if isVmbPy:
                    self._camera.get_feature_by_name("BlackLevel").set(float(property_value))
                else:
                    self._camera.get_feature_by_name("BlackLevel").set(float(property_value))
            elif property_name == "pixel_format":
                # Stopping streaming while changing pixel format is safer
                was_streaming = self._streaming
                if was_streaming:
                    self.stop_live()
                # Note: commented out as pixel format changes can be complex
                #if isVmbPy:
                #    self._camera.get_feature_by_name("PixelFormat").set(str(property_value))
                #else:
                #    self._camera.get_feature_by_name("PixelFormat").set(str(property_value))
                if was_streaming:
                    self.start_live()
            elif property_name == "isRGB":
                # Set isRGB property directly
                self.isRGB = bool(property_value)
            else:
                self.__logger.warning(f"Unsupported property: {property_name}")
                return False
            return property_value
        except Exception as e:
            self.__logger.error(f"Failed to set {property_name} to {property_value}: {e}")
            return False

    def getPropertyValue(self, property_name):
        try:
            if property_name == "exposure":
                if isVmbPy:
                    val = self._camera.get_feature_by_name("ExposureTime").get()
                else:
                    val = self._camera.get_feature_by_name("ExposureTime").get()
                return int(val // 1000)  # convert microseconds -> ms
            elif property_name == "gain":
                if isVmbPy:
                    return float(self._camera.get_feature_by_name("Gain").get())
                else:
                    return float(self._camera.get_feature_by_name("Gain").get())
            elif property_name == "blacklevel":
                if isVmbPy:
                    return float(self._camera.get_feature_by_name("BlackLevel").get())
                else:
                    return float(self._camera.get_feature_by_name("BlackLevel").get())
            elif property_name == "image_width":
                return int(self.sensor_width)
            elif property_name == "image_height":
                return int(self.sensor_height)
            elif property_name == "pixel_format":
                if isVmbPy:
                    return str(self._camera.get_feature_by_name("PixelFormat").get())
                else:
                    return str(self._camera.get_feature_by_name("PixelFormat").get())
            elif property_name == "isRGB":
                return self.isRGB
            else:
                self.__logger.warning(f"Unsupported property requested: {property_name}")
                return False
        except Exception as e:
            self.__logger.error(f"Failed to get {property_name}: {e}")
            return False
