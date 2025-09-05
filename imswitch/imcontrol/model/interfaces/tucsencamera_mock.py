import numpy as np
import time
import threading
import collections
from imswitch.imcommon.model import initLogger

class MockCameraTucsen:
    """Mock implementation of Tucsen camera for testing purposes."""

    def __init__(self, cameraNo=None, exposure_time=10000, gain=0, frame_rate=-1, blacklevel=100, isRGB=False, binning=1):
        self.__logger = initLogger(self, tryInheritParent=False)
        
        self.model = "MockTucsen"
        self.shape = (0, 0)
        self.is_connected = True
        self.is_streaming = False
        
        self.blacklevel = blacklevel
        self.exposure_time = exposure_time
        self.gain = gain
        self.frame_rate = frame_rate
        self.cameraNo = cameraNo if cameraNo is not None else 0
        self.binning = binning
        self.isRGB = isRGB
        
        # Mock sensor dimensions
        self.SensorHeight = 2048
        self.SensorWidth = 2048
        
        # Frame buffer
        self.NBuffer = 5
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameNumber = 0
        
        # Threading for mock frame generation
        self._thread = None
        self._stop_event = threading.Event()
        self.trigger_source = "Continuous"
        
        self.__logger.info(f"Mock Tucsen camera initialized: {self.SensorWidth}x{self.SensorHeight}")

    def _generate_mock_frame(self):
        """Generate a mock frame with some pattern."""
        # Create a simple pattern with noise
        frame = np.random.randint(100, 4000, (self.SensorHeight, self.SensorWidth), dtype=np.uint16)
        
        # Add some circular pattern
        y, x = np.ogrid[:self.SensorHeight, :self.SensorWidth]
        center_y, center_x = self.SensorHeight // 2, self.SensorWidth // 2
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 < (min(self.SensorHeight, self.SensorWidth) // 4) ** 2
        frame[mask] += 1000
        
        return frame

    def _frame_generation_thread(self):
        """Thread function to continuously generate frames."""
        while not self._stop_event.is_set():
            if self.is_streaming:
                frame = self._generate_mock_frame()
                self.frame_buffer.append(frame)
                self.frameid_buffer.append(self.frameNumber)
                self.frameNumber += 1
                
            # Simulate frame rate
            if self.frame_rate > 0:
                time.sleep(1.0 / self.frame_rate)
            else:
                time.sleep(0.1)  # Default 10 FPS

    def start_live(self):
        """Start live acquisition."""
        if self.is_streaming:
            self.__logger.warning("Mock camera is already streaming")
            return
            
        self.is_streaming = True
        self._stop_event.clear()
        
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._frame_generation_thread, daemon=True)
            self._thread.start()
            
        self.__logger.info("Mock Tucsen camera started streaming")

    def stop_live(self):
        """Stop live acquisition."""
        self.is_streaming = False
        self.__logger.info("Mock Tucsen camera stopped streaming")

    def suspend_live(self):
        """Suspend live acquisition."""
        self.stop_live()

    def prepare_live(self):
        """Prepare for live acquisition."""
        pass

    def close(self):
        """Close camera and cleanup."""
        self.is_streaming = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            
        self.__logger.info("Mock Tucsen camera closed")

    def getLast(self, returnFrameNumber: bool = False, timeout: float = 1.0, auto_trigger: bool = True):
        """Get the latest frame from the buffer."""
        # Handle software trigger
        if auto_trigger and self.trigger_source.lower() in ("software", "software trigger"):
            self.send_trigger()

        # Wait for frame
        t0 = time.time()
        while not self.frame_buffer:
            if time.time() - t0 > timeout:
                self.__logger.warning("Timeout waiting for mock frame")
                # Generate a frame immediately
                frame = self._generate_mock_frame()
                if returnFrameNumber:
                    return (frame, self.frameNumber)
                return frame
            time.sleep(0.001)

        frame = self.frame_buffer[-1] if self.frame_buffer else None
        frame_id = self.frameid_buffer[-1] if self.frameid_buffer else -1

        if returnFrameNumber:
            return (frame, frame_id)
        return frame

    def flushBuffer(self):
        """Clear the frame buffer."""
        self.frameid_buffer.clear()
        self.frame_buffer.clear()

    def getLastChunk(self):
        """Get all frames from buffer and clear it."""
        frames = list(self.frame_buffer)
        ids = list(self.frameid_buffer)
        self.flushBuffer()
        return frames, ids

    def setPropertyValue(self, property_name, property_value):
        """Set a camera property value."""
        if property_name == "exposure_time":
            self.exposure_time = property_value
        elif property_name == "gain":
            self.gain = property_value
        elif property_name == "blacklevel":
            self.blacklevel = property_value
        elif property_name == "binning":
            self.binning = property_value
        else:
            self.__logger.debug(f"Mock camera: set {property_name} = {property_value}")

    def getPropertyValue(self, property_name):
        """Get a camera property value."""
        if property_name == "exposure_time":
            return self.exposure_time
        elif property_name == "gain":
            return self.gain
        elif property_name == "blacklevel":
            return self.blacklevel
        elif property_name == "binning":
            return self.binning
        elif property_name == "image_width":
            return self.SensorWidth
        elif property_name == "image_height":
            return self.SensorHeight
        else:
            self.__logger.debug(f"Mock camera: get {property_name}")
            return 0

    def getTriggerTypes(self):
        """Return available trigger types."""
        return [
            "Continuous",
            "Software Trigger",
            "External Trigger"
        ]

    def getTriggerSource(self):
        """Get current trigger source."""
        return self.trigger_source

    def setTriggerSource(self, trigger_source):
        """Set trigger source."""
        self.trigger_source = trigger_source
        self.__logger.debug(f"Mock camera trigger source set to: {trigger_source}")

    def send_trigger(self):
        """Send software trigger."""
        if self.trigger_source.lower() in ("software", "software trigger"):
            # Generate a frame immediately
            frame = self._generate_mock_frame()
            self.frame_buffer.append(frame)
            self.frameid_buffer.append(self.frameNumber)
            self.frameNumber += 1
            return True
        return False

    def openPropertiesGUI(self):
        """Open camera properties GUI (placeholder)."""
        self.__logger.info("Mock Tucsen camera: Properties GUI would open here")

    def getFrameNumber(self):
        """Get current frame number."""
        return self.frameNumber

    def get_camera_parameters(self):
        """Get current camera parameters."""
        return {
            "model": self.model,
            "isRGB": self.isRGB,
            "width": self.SensorWidth,
            "height": self.SensorHeight,
            "exposure_time": self.exposure_time,
            "gain": self.gain,
            "blacklevel": self.blacklevel,
            "binning": self.binning
        }

    def get_gain(self):
        """Get current gain settings."""
        return (self.gain, 0.0, 100.0)  # current, min, max

    def get_exposuretime(self):
        """Get current exposure time settings."""
        return (self.exposure_time, 0.1, 10000.0)  # current, min, max

    def set_exposure_time(self, exposure_time):
        """Set camera exposure time."""
        self.exposure_time = exposure_time

    def set_gain(self, gain):
        """Set camera gain."""
        self.gain = gain

    def set_blacklevel(self, blacklevel):
        """Set camera black level."""
        self.blacklevel = blacklevel

    def setBinning(self, binning=1):
        """Set camera binning."""
        self.binning = binning

    def setFlatfieldImage(self, flatfieldImage, isFlatfielding):
        """Set flatfield image (mock implementation)."""
        self.__logger.debug("Mock camera: flatfield image set")

    def recordFlatfieldImage(self):
        """Record flatfield image (mock implementation)."""
        self.__logger.debug("Mock camera: flatfield image recorded")

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
