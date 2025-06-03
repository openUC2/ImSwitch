"""
Hardware Interface for Experiment Controller

Handles all direct hardware interactions including stage movement,
laser control, and camera operations.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from imswitch.imcommon.model import initLogger


class HardwareInterface:
    """Manages hardware interactions for experiment execution."""

    def __init__(self, master, comm_channel):
        """
        Initialize hardware interface.
        
        Args:
            master: Main controller with hardware managers
            comm_channel: Communication channel for signals
        """
        self._logger = initLogger(self)
        self._master = master
        self._commChannel = comm_channel
        
        # Initialize hardware references
        self._setup_hardware_references()
        
        # Set default speeds and acceleration
        self.SPEED_X_default = 20000
        self.SPEED_Y_default = 20000
        self.SPEED_Z_default = 20000
        self.ACCELERATION = 500000

    def _setup_hardware_references(self):
        """Setup references to hardware managers."""
        # Detector setup
        detector_names = self._master.detectorsManager.getAllDeviceNames()
        if detector_names:
            self.mDetector = self._master.detectorsManager[detector_names[0]]
            self.is_rgb = self.mDetector._camera.isRGB
            self.detector_pixel_size = self.mDetector.pixelSizeUm
        else:
            self.mDetector = None
            self.is_rgb = False
            self.detector_pixel_size = [1.0]

        # Illumination setup
        laser_names = self._master.lasersManager.getAllDeviceNames()
        led_names = self._master.LEDMatrixsManager.getAllDeviceNames()
        self.all_illumination_names = laser_names + led_names
        
        self.available_illuminations = []
        for device_name in self.all_illumination_names:
            try:
                # Try laser manager first
                device = self._master.lasersManager[device_name]
                self.available_illuminations.append(device)
            except KeyError:
                try:
                    # Try LED matrix manager
                    device = self._master.LEDMatrixsManager[device_name]
                    self.available_illuminations.append(device)
                except KeyError:
                    self._logger.warning(f"Could not find device: {device_name}")

        # Stage setup
        positioner_names = self._master.positionersManager.getAllDeviceNames()
        if positioner_names:
            stage_name = positioner_names[0]
            self.mStage = self._master.positionersManager[stage_name]
        else:
            self.mStage = None
            self._logger.warning("No stage/positioner found")

    def acquire_frame(self, channel: str = "Mono") -> np.ndarray:
        """
        Acquire a frame from the detector.
        
        Args:
            channel: Channel name for acquisition
            
        Returns:
            Acquired frame as numpy array
        """
        if not self.mDetector:
            self._logger.error("No detector available")
            return np.array([])
            
        self._logger.debug(f"Acquiring frame on channel {channel}")
        frame = self.mDetector.getLatestFrame()
        return frame

    def set_exposure_time_gain(self, exposure_time: float, gain: float):
        """
        Set camera exposure time and gain.
        
        Args:
            exposure_time: Exposure time in microseconds
            gain: Camera gain value
        """
        if gain is not None and gain >= 0:
            self._commChannel.sharedAttrs.sigAttributeSet(
                ['Detector', None, None, "gain"], gain
            )
            self._logger.debug(f"Setting gain to {gain}")
            
        if exposure_time is not None and exposure_time > 0:
            self._commChannel.sharedAttrs.sigAttributeSet(
                ['Detector', None, None, "exposureTime"], exposure_time
            )
            self._logger.debug(f"Setting exposure time to {exposure_time}")

    def set_laser_power(self, power: float, channel: str) -> Optional[float]:
        """
        Set laser power for specified channel.
        
        Args:
            power: Power value to set
            channel: Laser channel name
            
        Returns:
            Set power value or None if error
        """
        if channel not in self.all_illumination_names:
            self._logger.error(
                f"Channel {channel} not found in available lasers: "
                f"{self.all_illumination_names}"
            )
            return None
            
        try:
            laser = self._master.lasersManager[channel]
            laser.setValue(power)
            if laser.enabled == 0:
                laser.setEnabled(1)
            self._logger.debug(f"Setting laser power to {power} for channel {channel}")
            return power
        except KeyError:
            self._logger.error(f"Could not access laser {channel}")
            return None

    def move_stage_xy(self, posX: float = None, posY: float = None, 
                     relative: bool = False, speed: Tuple[float, float] = None) -> Tuple[float, float]:
        """
        Move stage to X,Y position.
        
        Args:
            posX: X position to move to
            posY: Y position to move to  
            relative: Whether movement is relative or absolute
            speed: Tuple of (speed_x, speed_y), uses defaults if None
            
        Returns:
            Tuple of final (posX, posY) position
        """
        if not self.mStage:
            self._logger.error("No stage available for movement")
            return (0.0, 0.0)
            
        if speed is None:
            speed = (self.SPEED_X_default, self.SPEED_Y_default)
            
        self._logger.debug(f"Moving stage to X={posX}, Y={posY}")
        
        self.mStage.move(
            value=(posX, posY), 
            speed=speed, 
            axis="XY",
            is_absolute=not relative, 
            is_blocking=True, 
            acceleration=self.ACCELERATION
        )
        
        self._commChannel.sigUpdateMotorPosition.emit([posX, posY])
        return (posX, posY)

    def move_stage_z(self, posZ: float, relative: bool = False, 
                    speed: float = None) -> float:
        """
        Move stage to Z position.
        
        Args:
            posZ: Z position to move to
            relative: Whether movement is relative or absolute
            speed: Z axis speed, uses default if None
            
        Returns:
            Final Z position
        """
        if not self.mStage:
            self._logger.error("No stage available for movement")
            return 0.0
            
        if speed is None:
            speed = self.SPEED_Z_default
            
        self._logger.debug(f"Moving stage to Z={posZ}")
        
        self.mStage.move(
            value=posZ, 
            speed=speed, 
            axis="Z",
            is_absolute=not relative, 
            is_blocking=True
        )
        
        new_position = self.mStage.getPosition()
        self._commChannel.sigUpdateMotorPosition.emit([new_position["Z"]])
        return new_position["Z"]

    def get_current_position(self) -> Dict[str, float]:
        """
        Get current stage position.
        
        Returns:
            Dictionary with current X, Y, Z positions
        """
        if not self.mStage:
            return {"X": 0.0, "Y": 0.0, "Z": 0.0}
        return self.mStage.getPosition()

    def start_detector_acquisition(self):
        """Start detector acquisition if not already running."""
        if self.mDetector and not self.mDetector._running:
            self.mDetector.startAcquisition()

    def stop_detector_acquisition(self):
        """Stop detector acquisition."""
        if self.mDetector:
            self.mDetector.stopAcquisition()

    def set_trigger_source(self, source: str):
        """
        Set detector trigger source.
        
        Args:
            source: Trigger source ("External trigger" or "Continuous")
        """
        if self.mDetector:
            self.mDetector.setTriggerSource(source)

    def flush_detector_buffers(self):
        """Flush detector buffers."""
        if self.mDetector:
            self.mDetector.flushBuffers()

    def get_detector_chunk(self):
        """Get chunk of frames from detector buffer."""
        if self.mDetector:
            return self.mDetector.getChunk()
        return np.array([]), np.array([])

    def perform_autofocus(self, min_z: float = 0, max_z: float = 0, 
                         step_size: float = 0):
        """
        Perform autofocus operation.
        
        Args:
            min_z: Minimum Z position for autofocus
            max_z: Maximum Z position for autofocus
            step_size: Step size for autofocus
        """
        self._logger.debug(
            f"Performing autofocus with parameters minZ={min_z}, "
            f"maxZ={max_z}, stepSize={step_size}"
        )
        # TODO: Connect this to actual autofocus implementation
        # This is a placeholder for now

    def turn_off_all_illumination(self, channel_list: list):
        """
        Turn off all illumination channels.
        
        Args:
            channel_list: List of illumination channels to turn off
        """
        for channel in channel_list:
            self.set_laser_power(0, channel)