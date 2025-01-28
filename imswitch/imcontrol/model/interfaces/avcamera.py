import collections
import time
import numpy as np

from typing import Optional
from imswitch.imcommon.model import initLogger

# If VimbaPython is not installed or not found, set isVimba to False
isVimba = True
try:
    from vimba import (Vimba, FrameStatus, Frame, VimbaCameraError)
except ImportError as e:
    print(e)
    print("No Vimba installed..")
    isVimba = False


class CameraAV:
    def __init__(self, camera_id=None):
        """
        :param camera_id: Index (int) or ID (string) of the Allied Vision camera to open.
                          If None, the first available camera is used.
        """
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=True)

        if not isVimba:
            raise RuntimeError("VimbaPython not installed or not found.")

        # Enter the Vimba context
        self._vimba = Vimba.get_instance()
        self._vimba.__enter__()

        self._camera = None
        self._running = False
        self._streaming = False
        self.frame_buffer = collections.deque(maxlen=60)
        self.frame = None
        self.frame_id = 0
        self.frame_id_last = -1

        self.model = "AlliedVisionCamera"
        self.sensor_width = 0
        self.sensor_height = 0

        # For storing any ROI parameters (OffsetX, OffsetY, Width, Height)
        self.hpos = 0
        self.vpos = 0
        self.hsize = 0
        self.vsize = 0

        self._open_camera(camera_id)
        self.__logger.debug("CameraAV initialized.")

    def _open_camera(self, camera_id):
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

    def _frame_handler(self, cam, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
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

            self.frame_buffer.append(self.frame)
        cam.queue_frame(frame)

    def start_live(self):
        if not self._running:
            self._running = True
        if not self._streaming:
            # TODO: THis is not working 
            #self._camera.start_streaming(
            #    handler=self._frame_handler,
            #    buffer_count=10
            #)
            self._streaming = True
            self.__logger.debug("Camera streaming started.")

    def stop_live(self):
        if self._streaming:
            self._camera.stop_streaming()
            self._streaming = False
            self.__logger.debug("Camera streaming stopped.")

    def suspend_live(self):
        # This method just stops acquisition without changing the _running state
        if self._streaming:
            self._camera.stop_streaming()
            self._streaming = False
            self.__logger.debug("Camera streaming suspended.")

    def close(self):
        # Stop streaming if active
        if self._streaming:
            self._camera.stop_streaming()
            self._streaming = False
        # Close camera
        try:
            self._camera.close()
        except Exception as e:
            self.__logger.warning(f"Error closing camera: {e}")
        # Exit Vimba context if not already
        if self._vimba:
            self._vimba.__exit__(None, None, None)
            self._vimba = None
        self.__logger.debug("Camera closed and Vimba context exited.")

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

    def getLast(self, is_resize=True):
        # Return the most recent frame from self.frame
        # The manager code uses is_resize, but we don't do anything with it here
        # (kept for compatibility).
        self.frame = self._camera.get_frame(timeout_ms=1000).as_opencv_image()
        return self.frame

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
                self._camera.get_feature_by_name("ExposureTime").set(microseconds)
            elif property_name == "gain":
                self._camera.get_feature_by_name("Gain").set(float(property_value))
            elif property_name == "blacklevel":
                self._camera.get_feature_by_name("BlackLevel").set(float(property_value))
            elif property_name == "pixel_format":
                # Stopping streaming while changing pixel format is safer
                was_streaming = self._streaming
                if was_streaming:
                    self.stop_live()
                #self._camera.get_feature_by_name("PixelFormat").set(str(property_value))
                if was_streaming:
                    self.start_live()
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
                val = self._camera.get_feature_by_name("ExposureTime").get()
                return int(val // 1000)  # convert microseconds -> ms
            elif property_name == "gain":
                return float(self._camera.get_feature_by_name("Gain").get())
            elif property_name == "blacklevel":
                return float(self._camera.get_feature_by_name("BlackLevel").get())
            elif property_name == "image_width":
                return int(self.sensor_width)
            elif property_name == "image_height":
                return int(self.sensor_height)
            elif property_name == "pixel_format":
                return str(self._camera.get_feature_by_name("PixelFormat").get())
            else:
                self.__logger.warning(f"Unsupported property requested: {property_name}")
                return False
        except Exception as e:
            self.__logger.error(f"Failed to get {property_name}: {e}")
            return False
