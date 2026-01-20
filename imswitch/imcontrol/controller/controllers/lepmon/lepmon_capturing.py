"""
LepmonOS Image Capturing Module

This module handles the complete capture workflow for the Lepmon moth trap:
- Main capture loop with timing control
- Exposure/gain auto-adjustment
- Image quality verification
- Integration with LEDs, sensors, and camera

Mirrors capturing.py from LepmonOS_update.
"""

import os
import time
import math
from datetime import datetime, timedelta
from threading import Thread, Event
from typing import Optional, Callable, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class CaptureState(Enum):
    """Capture workflow state"""
    IDLE = "idle"
    STARTING = "starting"
    WAITING = "waiting"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class CaptureResult:
    """Result of a single capture operation"""
    success: bool = False
    image_path: str = ""
    code: str = ""
    exposure: float = 150.0
    gain: float = 7.0
    brightness: float = 0.0
    timestamp: str = ""
    sensor_data: Dict = field(default_factory=dict)
    error_message: str = ""


@dataclass
class CaptureSession:
    """Tracking for a capture session"""
    session_id: str = ""
    start_time: Optional[datetime] = None
    expected_images: int = 0
    captured_images: int = 0
    failed_captures: int = 0
    folder_path: str = ""
    uv_led_active: bool = False
    heater_active: bool = False
    current_exposure: float = 150.0
    current_gain: float = 7.0


class LepmonCapturing:
    """
    Capture controller for Lepmon moth trap.
    
    Implements the main capture loop from LepmonOS capturing.py:
    1. Wait for darkness (dusk threshold)
    2. Turn on UV LED (LepiLED)
    3. Capture images at configured interval
    4. Auto-adjust exposure/gain for optimal brightness
    5. Record sensor data with each capture
    6. Turn off UV at dawn
    
    This controller works with the modular components:
    - LepmonConfig for settings
    - LepmonLights for LED control
    - LepmonSensors for environmental data
    - LepmonTimes for timing calculations
    - LepmonOLED for display feedback
    """
    
    def __init__(self,
                 config: Any = None,
                 lights: Any = None,
                 sensors: Any = None,
                 times: Any = None,
                 oled: Any = None,
                 camera_capture_func: Optional[Callable] = None,
                 state_callback: Optional[Callable[[CaptureState, Dict], None]] = None,
                 image_callback: Optional[Callable[[CaptureResult], None]] = None):
        """
        Initialize capture controller.
        
        Args:
            config: LepmonConfig instance
            lights: LepmonLights instance
            sensors: LepmonSensors instance
            times: LepmonTimes instance
            oled: LepmonOLED instance
            camera_capture_func: Function to capture image (returns frame)
            state_callback: Callback for state changes
            image_callback: Callback for captured images
        """
        self.config = config
        self.lights = lights
        self.sensors = sensors
        self.times = times
        self.oled = oled
        self.camera_capture_func = camera_capture_func
        self.state_callback = state_callback
        self.image_callback = image_callback
        
        # State
        self.state = CaptureState.IDLE
        self.session: Optional[CaptureSession] = None
        self.stop_event = Event()
        self.capture_thread: Optional[Thread] = None
        
        # Capture settings (can be overridden from config)
        self.dusk_threshold = 90  # Lux
        self.capture_interval = 2  # Minutes
        self.initial_exposure = 150  # ms
        self.initial_gain = 7.0
        
        # Auto-exposure settings
        self.brightness_reference = 170
        self.brightness_tolerance = 8
        self.min_exposure = 100
        self.max_exposure = 170
        self.exposure_step = 5
        self.min_gain = 5.0
        self.max_gain = 15.0
        self.gain_step = 0.5
        
        # Gamma correction
        self.gamma_correction = True
        self.gamma_value = 1.5
        
        # Load settings from config if available
        if config:
            self._load_settings_from_config()
    
    def _load_settings_from_config(self):
        """Load settings from LepmonConfig"""
        if self.config:
            self.dusk_threshold = self.config.capture.dusk_threshold
            self.capture_interval = self.config.capture.interval
            self.initial_exposure = self.config.capture.initial_exposure
            self.initial_gain = self.config.capture.initial_gain
            
            self.brightness_reference = self.config.image_quality.brightness_reference
            self.brightness_tolerance = self.config.image_quality.brightness_tolerance
            self.min_exposure = self.config.image_quality.minimal_exposure
            self.max_exposure = self.config.image_quality.maximal_exposure
            self.exposure_step = self.config.image_quality.step_exposure
            self.min_gain = self.config.image_quality.minimal_gain
            self.max_gain = self.config.image_quality.maximal_gain
            self.gain_step = self.config.image_quality.step_gain
            self.gamma_correction = self.config.image_quality.gamma_correction
            self.gamma_value = self.config.image_quality.gamma_value
    
    def _set_state(self, new_state: CaptureState, info: Dict = None):
        """Update state and notify callback"""
        self.state = new_state
        if self.state_callback:
            self.state_callback(new_state, info or {})
    
    # ---------------------- Session Management ---------------------- #
    
    def start_capture_session(self,
                              folder_path: Optional[str] = None,
                              exposure: Optional[float] = None,
                              gain: Optional[float] = None,
                              interval: Optional[int] = None,
                              override_timecheck: bool = False) -> bool:
        """
        Start a new capture session.
        
        Args:
            folder_path: Path to save images (auto-created if None)
            exposure: Initial exposure in ms
            gain: Initial gain
            interval: Capture interval in minutes
            override_timecheck: Skip darkness check (for testing)
            
        Returns:
            True if session started successfully
        """
        if self.state != CaptureState.IDLE:
            print("Cannot start: capture already in progress")
            return False
        
        # Create session
        self.session = CaptureSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now(),
            current_exposure=exposure or self.initial_exposure,
            current_gain=gain or self.initial_gain,
            folder_path=folder_path or self._create_capture_folder(),
        )
        
        # Update interval if provided
        if interval:
            self.capture_interval = interval
        
        # Calculate expected images
        if self.times:
            exp_times = self.times.get_experiment_times()
            self.session.expected_images = self._calculate_expected_images(
                exp_times.start_time, exp_times.end_time, self.capture_interval
            )
        
        # Clear stop event and start thread
        self.stop_event.clear()
        self.capture_thread = Thread(
            target=self._capture_loop,
            args=(override_timecheck,),
            daemon=True
        )
        self.capture_thread.start()
        
        self._set_state(CaptureState.STARTING, {
            "session_id": self.session.session_id,
            "expected_images": self.session.expected_images,
        })
        
        return True
    
    def stop_capture_session(self):
        """Stop the current capture session"""
        if self.state == CaptureState.IDLE:
            return
        
        self._set_state(CaptureState.STOPPING)
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=10.0)
        
        # Turn off LEDs
        if self.lights:
            self.lights.all_off()
        
        self._set_state(CaptureState.IDLE, {
            "captured_images": self.session.captured_images if self.session else 0,
            "failed_captures": self.session.failed_captures if self.session else 0,
        })
        
        self.session = None
    
    def is_running(self) -> bool:
        """Check if capture is in progress"""
        return self.state not in [CaptureState.IDLE, CaptureState.ERROR]
    
    # ---------------------- Main Capture Loop ---------------------- #
    
    def _capture_loop(self, override_timecheck: bool = False):
        """
        Main capture loop - equivalent to LepmonOS capturing().
        
        This runs in a separate thread and implements the full capture workflow.
        """
        try:
            # Wait for darkness or skip if override
            if not override_timecheck:
                self._wait_for_darkness()
            
            if self.stop_event.is_set():
                return
            
            # Turn on UV LED
            self._start_uv_led()
            
            # Main capture loop
            while not self.stop_event.is_set():
                # Check if still in capture window
                if not override_timecheck and not self._is_in_capture_window():
                    print("Outside capture window - ending session")
                    break
                
                # Perform capture
                self._set_state(CaptureState.CAPTURING)
                result = self._capture_single_image()
                
                if result.success:
                    self.session.captured_images += 1
                    if self.image_callback:
                        self.image_callback(result)
                else:
                    self.session.failed_captures += 1
                
                # Wait for next capture interval
                self._set_state(CaptureState.WAITING, {
                    "next_capture_in": self.capture_interval * 60,
                    "captured_images": self.session.captured_images,
                })
                
                self._wait_for_interval()
            
            # End session - turn off UV
            self._stop_uv_led()
            
        except Exception as e:
            print(f"Capture loop error: {e}")
            self._set_state(CaptureState.ERROR, {"error": str(e)})
        finally:
            if self.lights:
                self.lights.all_off()
    
    def _wait_for_darkness(self):
        """Wait until ambient light falls below dusk threshold"""
        print(f"Waiting for darkness (threshold: {self.dusk_threshold} lux)")
        
        while not self.stop_event.is_set():
            if self.sensors:
                lux, _ = self.sensors.get_light()
                if lux <= self.dusk_threshold:
                    print(f"Darkness detected: {lux} lux")
                    break
                print(f"Current light: {lux} lux - waiting...")
            
            # Also check if in time window
            if self.times and self.times.is_in_capture_window():
                print("In capture time window")
                break
            
            # Update display
            if self.oled:
                self.oled.display_text("Warte auf", "Dunkelheit...", f"Aktuell: {lux:.0f} lux")
            
            time.sleep(60)  # Check every minute
    
    def _is_in_capture_window(self) -> bool:
        """Check if current time is within capture window"""
        if self.times:
            return self.times.is_in_capture_window()
        return True  # Default to always capture if no times module
    
    def _start_uv_led(self):
        """Turn on UV LED for moth attraction"""
        if self.lights:
            self.lights.lepiled_start()
        self.session.uv_led_active = True
        print("UV LED (LepiLED) activated")
        
        if self.oled:
            self.oled.show_message("uv_on")
    
    def _stop_uv_led(self):
        """Turn off UV LED"""
        if self.lights:
            self.lights.lepiled_ende()
        self.session.uv_led_active = False
        print("UV LED (LepiLED) deactivated")
        
        if self.oled:
            self.oled.show_message("uv_off")
    
    def _wait_for_interval(self):
        """Wait for the capture interval"""
        wait_seconds = self.capture_interval * 60
        
        # Wait in small increments to allow stopping
        elapsed = 0
        while elapsed < wait_seconds and not self.stop_event.is_set():
            time.sleep(min(10, wait_seconds - elapsed))
            elapsed += 10
    
    # ---------------------- Single Image Capture ---------------------- #
    
    def _capture_single_image(self) -> CaptureResult:
        """
        Capture a single image with all the workflow steps.
        
        Equivalent to snap_image() in LepmonOS Camera.py.
        """
        result = CaptureResult(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            exposure=self.session.current_exposure,
            gain=self.session.current_gain,
        )
        
        try:
            # Generate image code
            if self.config:
                result.code = self.config.create_image_code()
            else:
                result.code = f"IMG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Turn on visible LED (flash)
            if self.lights:
                self.lights.dim_up()
            
            # Capture frame
            if self.camera_capture_func:
                frame = self.camera_capture_func(
                    exposure=self.session.current_exposure,
                    gain=self.session.current_gain
                )
                
                if frame is not None:
                    # Calculate brightness and auto-adjust exposure
                    result.brightness = self._calculate_brightness(frame)
                    good_exposure = self._auto_adjust_exposure(result.brightness)
                    
                    # Apply gamma correction if enabled
                    if self.gamma_correction:
                        frame = self._apply_gamma_correction(frame)
                    
                    # Save image
                    result.image_path = self._save_image(frame, result.code)
                    result.success = True
                else:
                    result.error_message = "Camera returned no frame"
            else:
                # Simulation mode
                result.image_path = f"{self.session.folder_path}/{result.code}.jpg"
                result.success = True
                print(f"Simulated capture: {result.image_path}")
            
            # Turn off visible LED
            if self.lights:
                self.lights.dim_down()
            
            # Read sensor data
            if self.sensors:
                sensor_data, _ = self.sensors.read_all_sensors(
                    code=result.code,
                    local_time=datetime.now().strftime("%H:%M:%S")
                )
                result.sensor_data = self.sensors.to_dict(sensor_data)
            
            # Update session exposure/gain
            self.session.current_exposure = result.exposure
            self.session.current_gain = result.gain
            
            # Update display
            if self.oled:
                self.oled.show_message("capture_status",
                    count=self.session.captured_images + 1,
                    time=datetime.now().strftime("%H:%M"),
                    space=self._get_free_space_gb()
                )
            
        except Exception as e:
            result.error_message = str(e)
            print(f"Capture error: {e}")
        
        return result
    
    # ---------------------- Auto-Exposure ---------------------- #
    
    def _calculate_brightness(self, frame) -> float:
        """Calculate average brightness of frame"""
        try:
            import numpy as np
            if len(frame.shape) == 3:
                # Convert to grayscale
                gray = np.mean(frame, axis=2)
            else:
                gray = frame
            return float(np.mean(gray))
        except Exception:
            return 0.0
    
    def _auto_adjust_exposure(self, current_brightness: float) -> bool:
        """
        Auto-adjust exposure and gain based on brightness.
        
        Equivalent to calculate_Exposure_and_gain() in LepmonOS.
        
        Returns:
            True if exposure is good, False if adjustment needed
        """
        target = self.brightness_reference
        tolerance = self.brightness_tolerance
        
        # Check if within tolerance
        if abs(current_brightness - target) <= tolerance:
            return True
        
        # Adjust exposure first, then gain
        if current_brightness < target - tolerance:
            # Too dark - increase exposure
            if self.session.current_exposure < self.max_exposure:
                self.session.current_exposure = min(
                    self.session.current_exposure + self.exposure_step,
                    self.max_exposure
                )
            elif self.session.current_gain < self.max_gain:
                self.session.current_gain = min(
                    self.session.current_gain + self.gain_step,
                    self.max_gain
                )
        else:
            # Too bright - decrease exposure
            if self.session.current_exposure > self.min_exposure:
                self.session.current_exposure = max(
                    self.session.current_exposure - self.exposure_step,
                    self.min_exposure
                )
            elif self.session.current_gain > self.min_gain:
                self.session.current_gain = max(
                    self.session.current_gain - self.gain_step,
                    self.min_gain
                )
        
        return False
    
    def _apply_gamma_correction(self, frame):
        """Apply gamma correction for shadow brightening"""
        try:
            import numpy as np
            frame = frame / 255.0
            frame = np.power(frame, 1 / self.gamma_value)
            frame = (frame * 255).astype(np.uint8)
            return frame
        except Exception:
            return frame
    
    # ---------------------- File Management ---------------------- #
    
    def _create_capture_folder(self) -> str:
        """Create folder for capture session"""
        if self.config:
            folder_path = self.config.create_folder_path()
        else:
            folder_path = f"/tmp/lepmon_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def _save_image(self, frame, code: str) -> str:
        """Save image to disk"""
        try:
            import cv2
            filepath = os.path.join(self.session.folder_path, f"{code}.jpg")
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"Image save error: {e}")
            return ""
    
    def _get_free_space_gb(self) -> float:
        """Get free disk space in GB"""
        try:
            import shutil
            path = self.session.folder_path if self.session else "/tmp"
            total, used, free = shutil.disk_usage(path)
            return round(free / (1024**3), 1)
        except Exception:
            return 0.0
    
    def _calculate_expected_images(self, start_time: str, end_time: str, interval: int) -> int:
        """Calculate expected number of images based on time window"""
        try:
            start = datetime.strptime(start_time, "%H:%M:%S")
            end = datetime.strptime(end_time, "%H:%M:%S")
            
            # Handle midnight crossing
            if end <= start:
                end += timedelta(days=1)
            
            duration_seconds = (end - start).total_seconds()
            return math.floor(duration_seconds / (interval * 60)) + 1
        except Exception:
            return 0
    
    # ---------------------- Status ---------------------- #
    
    def get_status(self) -> Dict:
        """Get current capture status"""
        return {
            "state": self.state.value,
            "is_running": self.is_running(),
            "session": {
                "session_id": self.session.session_id if self.session else None,
                "start_time": self.session.start_time.isoformat() if self.session and self.session.start_time else None,
                "expected_images": self.session.expected_images if self.session else 0,
                "captured_images": self.session.captured_images if self.session else 0,
                "failed_captures": self.session.failed_captures if self.session else 0,
                "folder_path": self.session.folder_path if self.session else "",
                "uv_led_active": self.session.uv_led_active if self.session else False,
                "current_exposure": self.session.current_exposure if self.session else self.initial_exposure,
                "current_gain": self.session.current_gain if self.session else self.initial_gain,
            },
            "settings": {
                "dusk_threshold": self.dusk_threshold,
                "capture_interval": self.capture_interval,
                "brightness_reference": self.brightness_reference,
                "gamma_correction": self.gamma_correction,
            },
        }
