# image processing libraries
from dataclasses_json.api import _process_class
import numpy as np
import time
import cv2, queue, threading
from imswitch.imcommon.model import initLogger
from threading import Thread

import collections

class CameraOpenCV:
    def __init__(self, cameraindex=0, isRGB=False, isAutoParameters=True):
        super().__init__()
        # we are aiming to interface with webcams or arducams
        self.__logger = initLogger(self, tryInheritParent=False)

        # many to be purged
        self.model = "CameraOpenCV"

        # camera parameters
        self.blacklevel = 0
        self.exposure_time = 10
        self.analog_gain = 0
        self.pixel_format = "Mono8"

        self.frame_id_last = 0

        self.PreviewWidthRatio = 4
        self.PreviewHeightRatio = 4

        self.SensorWidth = 1000
        self.SensorHeight = 1000

        # reserve some space for the framebuffer
        self.NBuffer = 1
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)

        #%% starting the camera => self.camera  will be created
        self.cameraindex = cameraindex
        self.camera = None
        self.isRGB = isRGB
        self.isAutoParameters = isAutoParameters
        self.openCamera(self.cameraindex, self.SensorWidth, self.SensorHeight, self.isRGB)


    def start_live(self):
        # check if camera is open
        if not self.camera_is_open:
            self.camera_is_open = True
            self.openCamera(self.cameraindex, self.SensorWidth, self.SensorHeight, self.isRGB)

    def stop_live(self):
        pass
        #self.camera.release()
        #self.camera_is_open = False

    def suspend_live(self):
        pass
        #self.camera.release()
        #self.camera_is_open = False

    def prepare_live(self):
        pass

    def close(self):
        """Close camera and cleanup resources properly."""
        self.__logger.info("Closing camera...")
        self.camera_is_open = False
        
        # Wait for frame grabber thread to finish
        if hasattr(self, 'frameGrabberThread') and self.frameGrabberThread is not None:
            if self.frameGrabberThread.is_alive():
                self.__logger.debug("Waiting for frame grabber thread to exit...")
                self.frameGrabberThread.join(timeout=2.0)
                if self.frameGrabberThread.is_alive():
                    self.__logger.warning("Frame grabber thread did not exit cleanly")
        
        # Release camera
        if self.camera is not None:
            try:
                self.camera.release()
                self.__logger.debug("Camera released")
            except Exception as e:
                self.__logger.error(f"Error releasing camera: {e}")
            finally:
                self.camera = None
        
        self.__logger.info("Camera closed")

    def set_exposure_time(self,exposure_time):
        if self.isAutoParameters:
            return
        self.exposure_time = exposure_time
        try:
            self.camera.set(cv2.CAP_PROP_EXPOSURE, self.exposure_time)
        except Exception as e:
            self.__logger.error(e)
            self.__logger.debug("Error setting Exposure time in opencv camera")

    def set_analog_gain(self,analog_gain):
        if self.isAutoParameters:
            return
        self.analog_gain = analog_gain
        try:
            self.camera.set(cv2.CAP_PROP_EXPOSURE, self.analog_gain)
        except Exception as e:
            self.__logger.error(e)
            self.__logger.debug("Error setting Exposure time in opencv camera")

    def set_blacklevel(self,blacklevel):
        self.blacklevel = blacklevel
        self.__logger.debug("Error setting blacklevel time in opencv camera")

    def set_pixel_format(self,format):
        self.pixelformat = format
        self.__logger.debug("Error setting pixelformat time in opencv camera")

    def getLast(self, is_resize=True, returnFrameNumber=False):
        # get frame and save
        #TODO: Napari only displays 8Bit?
        if returnFrameNumber:
            return self.frame, self.frame_id_last
        return self.frame

    def getLastChunk(self):
        chunk = np.array(self.frame_buffer)
        #frameids = self.frame_buffer[1]
        self.__logger.debug("Buffer: "+str(len(self.frame_buffer))+"  "+str(chunk.shape))
        self.frame_buffer.clear()
        return chunk

    def setROI(self, hpos, vpos, hsize, vsize):
        pass

    def setPropertyValue(self, property_name, property_value):
        # Check if the property exists.
        if property_name == "gain":
            self.set_analog_gain(property_value)
        elif property_name == "exposure":
            self.set_exposure_time(property_value)
        elif property_name == "blacklevel":
            self.set_blacklevel(property_value)
        elif property_name == "pixel_format":
            self.stop_live()
            self.set_pixel_format(property_value)
            self.start_live()
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    def getPropertyValue(self, property_name):
        # Check if the property exists.
        if property_name == "gain":
            property_value = self.gain
        elif property_name == "exposure":
            property_value = self.exposure
        elif property_name == "blacklevel":
            property_value = self.blacklevel
        elif property_name == "image_width":
            property_value = self.SensorWidth
        elif property_name == "image_height":
            property_value = self.SensorHeight
        elif property_name == "pixel_format":
            property_value = self.PixelFormat
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    def openPropertiesGUI(self):
        pass
    
    def isCameraHealthy(self):
        """
        Check if camera is healthy and responding.
        
        Returns:
            bool: True if camera is open and frame grabber is running
        """
        if not self.camera_is_open:
            return False
        
        if self.camera is None or not self.camera.isOpened():
            return False
        
        if not hasattr(self, 'frameGrabberThread') or self.frameGrabberThread is None:
            return False
        
        if not self.frameGrabberThread.is_alive():
            self.__logger.warning("Frame grabber thread is not alive")
            return False
        
        return True

    def listAvailableUSBCameras(self):
        '''list available cameras that can be used through openCVs VideoCapture'''
        # list available cameras
        from sys import platform
        if platform == "linux" or platform == "linux2":
            import glob
            cameras = glob.glob('/dev/video*')
            cameras = [int(c.split('/')[-1].replace('video', '')) for c in cameras]
            self.__logger.debug("Available cameras: "+str(cameras))
        else:
            cameras = []
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append(i)
                    cap.release()

        return cameras
        # check if camera is open


    def openCamera(self, cameraindex, width, height, isRGB):
        """
        Automatically match the user-chosen 'cameraindex' to the discovered /dev/video entries on Linux.
        On other platforms, just use cameraindex as provided.
        Implements robust opening with retries and proper V4L2 buffer configuration.
        """
        from sys import platform
        if self.camera is not None:
            self.camera.release()
            time.sleep(0.5)  # Allow time for release

        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if platform == "linux" or platform == "linux2":
                    # Map the requested cameraindex to an actual /dev/video device
                    available_cameras = sorted(self.listAvailableUSBCameras())
                    if isinstance(cameraindex, int):
                        # If out of range, fallback to the last device or device 0
                        if 0 <= cameraindex < len(available_cameras):
                            dev_path = f"/dev/video{available_cameras[cameraindex]}"
                        else:
                            dev_path = f"/dev/video{available_cameras[-1]}" if available_cameras else "/dev/video0"
                    else:
                        # Assume the user gave a direct string path like "/dev/video2"
                        dev_path = str(cameraindex)

                    self.__logger.debug(f"Attempt {attempt + 1}/{max_retries}: Opening Linux device: {dev_path}")
                    
                    # Try multiple capture methods for UVC cameras
                    camera_opened = False
                    
                    # Method 1: V4L2 with MJPG (preferred for most cameras)
                    try:
                        self.camera = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
                        if self.camera.isOpened():
                            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                            # Test if we can actually read a frame
                            ret, test_frame = self.camera.read()
                            if ret and test_frame is not None:
                                self.__logger.debug(f"Successfully opened with V4L2+MJPG")
                                camera_opened = True
                            else:
                                self.__logger.warning("V4L2+MJPG opened but cannot read frames")
                                self.camera.release()
                    except Exception as e:
                        self.__logger.debug(f"V4L2+MJPG failed: {e}")
                        if self.camera is not None:
                            self.camera.release()
                    
                    # Method 2: V4L2 with YUYV (fallback for UVC cameras)
                    if not camera_opened:
                        try:
                            self.camera = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
                            if self.camera.isOpened():
                                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
                                ret, test_frame = self.camera.read()
                                if ret and test_frame is not None:
                                    self.__logger.debug(f"Successfully opened with V4L2+YUYV")
                                    camera_opened = True
                                else:
                                    self.__logger.warning("V4L2+YUYV opened but cannot read frames")
                                    self.camera.release()
                        except Exception as e:
                            self.__logger.debug(f"V4L2+YUYV failed: {e}")
                            if self.camera is not None:
                                self.camera.release()
                    
                    # Method 3: Default OpenCV (no specific backend)
                    if not camera_opened:
                        try:
                            self.camera = cv2.VideoCapture(cameraindex)
                            if self.camera.isOpened():
                                ret, test_frame = self.camera.read()
                                if ret and test_frame is not None:
                                    self.__logger.debug(f"Successfully opened with default backend")
                                    camera_opened = True
                                else:
                                    self.__logger.warning("Default backend opened but cannot read frames")
                                    self.camera.release()
                        except Exception as e:
                            self.__logger.debug(f"Default backend failed: {e}")
                            if self.camera is not None:
                                self.camera.release()
                    
                    if not camera_opened:
                        raise RuntimeError(f"Failed to open camera at {dev_path} - tried V4L2+MJPG, V4L2+YUYV, and default backend")
                    
                    # Configure V4L2 buffer to reduce timeout issues
                    # Set buffer size to minimum to reduce latency and timeout issues
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Set timeout for V4L2 (in milliseconds) - not all backends support this
                    try:
                        self.camera.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 second timeout
                    except:
                        pass  # Some OpenCV versions don't support this

                elif platform == "darwin":
                    # macOS: just use cameraindex directly
                    self.__logger.debug(f"Attempt {attempt + 1}/{max_retries}: Opening macOS camera: {cameraindex}")
                    self.camera = cv2.VideoCapture(cameraindex)
                else:
                    # Windows: use CAP_DSHOW when opening
                    self.__logger.debug(f"Attempt {attempt + 1}/{max_retries}: Opening Windows camera: {cameraindex}")
                    self.camera = cv2.VideoCapture(cameraindex, cv2.CAP_DSHOW)
                
                if not self.camera.isOpened():
                    raise RuntimeError(f"Camera not opened on attempt {attempt + 1}")

                self.__logger.debug("Camera opened successfully")
                
                # Set resolution
                #self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920.0)  # 4k/high_res
                #self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080.0) # 4k/high_res
                
                # Flush initial frames and verify camera is working
                successful_reads = 0
                max_warmup_attempts = 10
                
                for i in range(max_warmup_attempts):
                    ret, img = self.camera.read()
                    if ret and img is not None:
                        successful_reads += 1
                        if successful_reads >= 3:  # Need at least 3 successful reads
                            break
                    else:
                        self.__logger.warning(f"Warmup frame {i + 1} failed to read")
                    time.sleep(0.1)  # Small delay between reads
                
                if successful_reads < 3:
                    raise RuntimeError(f"Camera warmup failed - only {successful_reads} successful reads")
                
                # Get actual resolution
                actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.__logger.debug(f"Camera resolution: {actual_width}x{actual_height}")
                
                self.SensorHeight = img.shape[0]
                self.SensorWidth = img.shape[1]
                self.shape = (self.SensorWidth, self.SensorHeight)
                self.camera_is_open = True

                self.__logger.info(f"Camera opened successfully: {self.SensorWidth}x{self.SensorHeight}")
                
                # Start a thread to continuously grab frames
                self.frameGrabberThread = Thread(target=self.setFrameBuffer, args=(isRGB,))
                self.frameGrabberThread.daemon = True  # Make thread daemon so it exits with main program
                self.frameGrabberThread.start()
                
                return  # Success - exit retry loop
                
            except Exception as e:
                self.__logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                # Clean up failed attempt
                if self.camera is not None:
                    try:
                        self.camera.release()
                    except:
                        pass
                    self.camera = None
                
                # If this wasn't the last attempt, wait before retrying
                if attempt < max_retries - 1:
                    self.__logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # Last attempt failed
                    self.__logger.error(f"Failed to open camera after {max_retries} attempts")
                    self.camera_is_open = False
                    raise RuntimeError(f"Could not open camera after {max_retries} attempts: {e}")

    def setFrameBuffer(self, isRGB=True):
        """
        Continuously grab frames from camera in background thread.
        Implements robust error handling and recovery for V4L2 timeout issues.
        """
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        # Give camera time to initialize
        time.sleep(0.5)
        
        while self.camera_is_open:
            try:
                # For problematic UVC cameras, use read() directly instead of grab()+retrieve()
                # This can help with cameras that report True but return None frames
                ret, frame = self.camera.read()
                
                if not ret:
                    consecutive_failures += 1
                    self.__logger.warning(f"Failed to read frame - ret=False (attempt {consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.__logger.error("Too many consecutive frame read failures - stopping capture")
                        self.camera_is_open = False
                        break
                    
                    time.sleep(0.1)  # Brief pause before retry
                    continue
                
                if frame is None:
                    consecutive_failures += 1
                    self.__logger.warning(f"Failed to read frame - frame is None (attempt {consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.__logger.error("Too many consecutive None frames - stopping capture")
                        self.camera_is_open = False
                        break
                    
                    time.sleep(0.1)  # Brief pause before retry
                    continue
                
                # Validate frame has proper dimensions
                if frame.size == 0 or len(frame.shape) < 2:
                    consecutive_failures += 1
                    self.__logger.warning(f"Invalid frame dimensions: {frame.shape} (attempt {consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.__logger.error("Too many consecutive invalid frames - stopping capture")
                        self.camera_is_open = False
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Success - reset failure counter
                consecutive_failures = 0
                
                # Process frame
                self.frame_id_last += 1
                
                if not isRGB and len(frame.shape) > 2:
                    frame = np.uint8(np.mean(frame, -1))
                
                self.frame = np.flip(frame)
                self.frame_buffer.append(self.frame)
                
            except Exception as e:
                consecutive_failures += 1
                self.__logger.error(f"Frame capture exception (attempt {consecutive_failures}/{max_consecutive_failures}): {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    self.__logger.error("Too many consecutive exceptions - stopping capture")
                    self.camera_is_open = False
                    break
                
                time.sleep(0.1)  # Brief pause before retry
        
        self.__logger.info("Frame grabber thread exiting")
