"""
LepmonController - Refactored

Moth trap controller for ImSwitch using modular components.
This controller integrates with the lepmon/ modules for GPIO, OLED, 
LEDs, sensors, capturing, and timing.

Provides FastAPI endpoints via @APIExport and WebSocket updates via Signal.
"""

import os
import time
import datetime
import numpy as np
import cv2
from datetime import timedelta
from threading import Thread, Event
from typing import Dict, Optional
from fastapi import Response
import io

from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcommon.framework import Signal
from ..basecontrollers import LiveUpdatedController

# Import modular Lepmon components
try:
    from .lepmon import (
        LepmonConfig, DEFAULT_CONFIG as LEPMON_DEFAULT_CONFIG,
        LepmonGPIO, LED_PINS, BUTTON_PINS, HAS_GPIO,
        LepmonOLED, HAS_OLED,
        LepmonLights,
        LepmonSensors, HAS_I2C,
        LepmonCapturing, CaptureState,
        LepmonTimes
    )
    HAS_LEPMON_MODULES = True
except ImportError as e:
    HAS_LEPMON_MODULES = False
    print(f"Lepmon modules not fully available: {e}")
    # Fallback definitions
    HAS_GPIO = False
    HAS_OLED = False
    HAS_I2C = False
    LED_PINS = {}
    BUTTON_PINS = {}

try:
    from PIL import Image
except ImportError:
    Image = None


# Default configuration for ImSwitch integration
DEFAULT_CONFIG = {
    "exposureTime": 100.0,
    "gain": 0.0,
    "timelapsePeriod": 60,
    "storagePath": "/mnt/usb_drive",
    "isRunning": False,
    "wasRunning": False,
    "numberOfFrames": 10,
    "experimentName": "LepMonTest",
    "fileFormat": "JPG",
    "frameRate": 1,
}


class LepmonController(LiveUpdatedController):
    """
    Lepmon Moth Trap Controller
    
    This controller orchestrates all Lepmon functionality:
    - Camera capture with auto-exposure
    - UV and visible LED control
    - OLED display updates
    - Button/menu handling
    - Sensor data collection
    - Astronomical time calculations
    
    Uses modular components from lepmon/ package for clean separation of concerns.
    All hardware operations support simulation mode when running on non-RPi systems.
    """

    # WebSocket Signals for frontend updates
    sigImagesTaken = Signal(int)
    sigIsRunning = Signal(bool)
    sigFocusSharpness = Signal(float)
    sigSensorUpdate = Signal(dict)
    sigDisplayUpdate = Signal(dict)
    sigButtonPressed = Signal(dict)
    sigLightStateChanged = Signal(dict)
    sigCaptureStateChanged = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)
        
        # Initialize experiment parameters
        self._master.LepmonManager.defaultConfig = DEFAULT_CONFIG
        self.mExperimentParameters = self._master.LepmonManager.defaultConfig.copy()
        
        self.is_measure = False
        self.imagesTaken = 0
        self.mFrame = None
        
        # Get camera detector
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detectorlepmonCam = self._master.detectorsManager[allDetectorNames[0]]
        
        # Initialize modular components
        self._init_lepmon_modules()
        
        # Initialize camera
        self.changeAutoExposureTime("auto")
        
        # HMI state
        self.menu_open = False
        self.current_menu_state = "main"
        self.hmi_stop_event = Event()
        
        # Start background threads
        self._start_background_threads()
        
        self._logger.info("LepmonController initialized with modular components")

    def _init_lepmon_modules(self):
        """Initialize all Lepmon modular components"""
        self._logger.info("Initializing Lepmon modules...")
        
        # Configuration
        self.lepmon_config = LepmonConfig() if HAS_LEPMON_MODULES else None
        
        # GPIO (buttons, status LEDs)
        self.lepmon_gpio = LepmonGPIO(
            button_callback=self._on_button_pressed,
            hardware_generation="Pro_Gen_2"
        ) if HAS_LEPMON_MODULES else None
        
        # OLED Display
        self.lepmon_oled = LepmonOLED(
            rotate=0,
            font_path=None,
            display_callback=self._on_display_updated,
            language="de"
        ) if HAS_LEPMON_MODULES else None
        
        # LED Control (UV + Visible)
        self.lepmon_lights = LepmonLights(
            hardware_generation="Pro_Gen_2",
            state_callback=self._on_light_state_changed
        ) if HAS_LEPMON_MODULES else None
        
        # Sensors
        self.lepmon_sensors = LepmonSensors() if HAS_LEPMON_MODULES else None
        
        # Time calculations
        self.lepmon_times = LepmonTimes(
            latitude=48.1351,  # Default: Munich
            longitude=11.5820
        ) if HAS_LEPMON_MODULES else None
        
        # Capturing workflow
        self.lepmon_capturing = LepmonCapturing(
            config=self.lepmon_config,
            lights=self.lepmon_lights,
            sensors=self.lepmon_sensors,
            times=self.lepmon_times,
            oled=self.lepmon_oled,
            camera_capture_func=self._capture_camera_frame,
            state_callback=self._on_capture_state_changed,
            image_callback=self._on_image_captured
        ) if HAS_LEPMON_MODULES else None
        
        # Update config with GPS if available
        if self.lepmon_config and self.lepmon_times:
            gps = self.lepmon_config.gps
            self.lepmon_times.update_location(gps.latitude, gps.longitude)
        
        self._logger.info(f"Modules initialized - GPIO: {HAS_GPIO}, OLED: {HAS_OLED}, I2C: {HAS_I2C}")

    def _start_background_threads(self):
        """Start background monitoring threads"""
        # Sensor polling thread
        self._sensor_thread_active = True
        self._sensor_thread = Thread(
            target=self._sensor_polling_loop,
            args=(10,),  # 10 second interval
            daemon=True
        )
        self._sensor_thread.start()
        
        # Button monitoring thread
        self._button_thread = Thread(
            target=self._button_monitoring_loop,
            daemon=True
        )
        self._button_thread.start()
        
        # Open HMI menu on startup
        self._open_hmi_menu()

    # ===================== Callbacks for Module Integration ===================== #

    def _on_button_pressed(self, button_name: str, state: bool):
        """Callback when a button is pressed"""
        self._logger.debug(f"Button pressed: {button_name} = {state}")
        self.sigButtonPressed.emit({"buttonName": button_name, "state": state})

    def _on_display_updated(self, content: dict):
        """Callback when OLED display is updated"""
        self.sigDisplayUpdate.emit(content)

    def _on_light_state_changed(self, state: dict):
        """Callback when LED state changes"""
        self.sigLightStateChanged.emit(state)

    def _on_capture_state_changed(self, state: CaptureState, info: dict):
        """Callback when capture state changes"""
        self._logger.info(f"Capture state: {state.value}")
        self.sigCaptureStateChanged.emit({
            "state": state.value,
            **info
        })
        
        # Update isRunning based on capture state
        if state == CaptureState.IDLE:
            self.is_measure = False
            self.sigIsRunning.emit(False)
        elif state == CaptureState.CAPTURING:
            self.is_measure = True
            self.sigIsRunning.emit(True)

    def _on_image_captured(self, result):
        """Callback when an image is captured"""
        self.imagesTaken += 1
        self.sigImagesTaken.emit(self.imagesTaken)
        self._logger.info(f"Image captured: {result.code} ({self.imagesTaken} total)")

    def _capture_camera_frame(self, exposure: float = None, gain: float = None):
        """Capture a frame from the camera - used by LepmonCapturing"""
        try:
            if exposure is not None:
                self.changeExposureTime(exposure)
            if gain is not None:
                self.changeGain(gain)
            
            frame = self.detectorlepmonCam.getLatestFrame()
            return frame
        except Exception as e:
            self._logger.error(f"Camera capture error: {e}")
            return None

    # ===================== Background Threads ===================== #

    def _sensor_polling_loop(self, interval: int):
        """Poll sensors at regular intervals"""
        while self._sensor_thread_active:
            try:
                if self.lepmon_sensors:
                    sensor_data, status = self.lepmon_sensors.read_all_sensors(
                        code=f"poll_{self.imagesTaken}",
                        local_time=datetime.datetime.now().strftime("%H:%M:%S")
                    )
                    self.sigSensorUpdate.emit(self.lepmon_sensors.to_dict(sensor_data))
            except Exception as e:
                self._logger.warning(f"Sensor polling error: {e}")
            
            time.sleep(interval)

    def _button_monitoring_loop(self):
        """Continuous button monitoring for HMI"""
        self._logger.info("Starting button monitoring loop")
        
        while not self.hmi_stop_event.is_set():
            try:
                if self.lepmon_gpio:
                    # Check all buttons
                    for button_name in BUTTON_PINS.keys():
                        if self.lepmon_gpio.button_pressed(button_name):
                            self._handle_button_press(button_name)
            except Exception as e:
                self._logger.warning(f"Button monitoring error: {e}")
            
            time.sleep(0.05)  # 50ms polling

    def _handle_button_press(self, button_name: str):
        """Handle button press based on current menu state"""
        self._logger.debug(f"Handling button: {button_name} in state: {self.current_menu_state}")
        
        if not self.menu_open:
            if button_name == "enter":
                self.menu_open = True
                self._logger.info("Menu opened")
                if self.lepmon_oled:
                    self.lepmon_oled.display_text("Eingabe Menü", "geöffnet", "")
                if self.lepmon_gpio:
                    self.lepmon_gpio.turn_on_led("blau")
            return
        
        # Menu is open - handle navigation
        if self.current_menu_state == "main":
            if button_name == "rechts":
                self._run_focus_mode()
            elif button_name == "unten":
                self._handle_location_menu()
            elif button_name == "oben":
                self._handle_update_menu()

    # ===================== HMI Menu Functions ===================== #

    def _open_hmi_menu(self):
        """Open the HMI menu system"""
        try:
            self.menu_open = False
            if self.lepmon_gpio:
                self.lepmon_gpio.turn_on_led("blau")
            if self.lepmon_oled:
                self.lepmon_oled.display_text(
                    "Menü öffnen:",
                    "bitte Enter drücken",
                    "(rechts unten)"
                )
            self._logger.info("HMI menu ready - press Enter to open")
        except Exception as e:
            self._logger.error(f"Failed to open HMI menu: {e}")

    def _run_focus_mode(self):
        """Run focus assistance mode for 15 seconds"""
        self._logger.info("Focus mode activated")
        if self.lepmon_oled:
            self.lepmon_oled.display_text("Fokussierhilfe", "aktiviert", "15 Sekunden")
        
        start_time = time.time()
        while time.time() - start_time < 15.0:
            try:
                frame = self.detectorlepmonCam.getLatestFrame()
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    self.sigFocusSharpness.emit(float(sharpness))
            except Exception as e:
                self._logger.warning(f"Focus mode error: {e}")
            time.sleep(0.5)

    def _handle_location_menu(self):
        """Handle location code menu"""
        self._logger.info("Location menu opened")
        if self.lepmon_oled:
            self.lepmon_oled.display_text("Bitte Land,", "Provinz und", "Stadtcode wählen")
        time.sleep(2)

    def _handle_update_menu(self):
        """Handle update menu"""
        self._logger.info("Update menu opened")
        if self.lepmon_oled:
            self.lepmon_oled.display_text("Update Menü", "geöffnet", "")
        time.sleep(2)

    # ===================== API Endpoints - Status ===================== #

    @APIExport()
    def getStatus(self) -> dict:
        """Get overall Lepmon status"""
        return {
            "isRunning": self.is_measure,
            "currentImageCount": self.imagesTaken,
            "serverTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "freeSpace": self._compute_free_space(),
            "captureState": self.lepmon_capturing.state.value if self.lepmon_capturing else "unknown"
        }

    @APIExport()
    def getInitialParams(self) -> dict:
        """Get camera/timelapse settings"""
        return {
            "exposureTime": self.mExperimentParameters["exposureTime"],
            "gain": self.mExperimentParameters["gain"],
            "timelapsePeriod": self.mExperimentParameters["timelapsePeriod"],
            "storagePath": self.mExperimentParameters["storagePath"],
        }

    @APIExport()
    def getHardwareStatus(self) -> dict:
        """Get hardware component status"""
        light_states = self.lepmon_lights.get_state() if self.lepmon_lights else {}
        display_content = self.lepmon_oled.get_current_content() if self.lepmon_oled else {}
        
        return {
            "lightStates": light_states,
            "lcdDisplay": display_content,
            "buttonStates": self.lepmon_gpio.get_button_states() if self.lepmon_gpio else {},
            "hardwareStatus": {
                "gpio_available": HAS_GPIO,
                "oled_available": HAS_OLED,
                "i2c_available": HAS_I2C,
                "simulation_mode": not HAS_GPIO
            },
            "availableLEDs": list(LED_PINS.keys()) if LED_PINS else [],
            "availableButtons": list(BUTTON_PINS.keys()) if BUTTON_PINS else []
        }

    @APIExport()
    def getSensorData(self) -> dict:
        """Get current sensor readings"""
        if self.lepmon_sensors:
            data, _ = self.lepmon_sensors.read_all_sensors()
            return {
                "success": True,
                "data": self.lepmon_sensors.to_dict(data),
                "timestamp": datetime.datetime.now().isoformat()
            }
        return {"success": False, "message": "Sensors not available"}

    @APIExport()
    def getLepmonConfig(self) -> dict:
        """Get complete Lepmon configuration"""
        if self.lepmon_config:
            return {
                "success": True,
                "config": self.lepmon_config.to_frontend_config()
            }
        return {"success": False, "message": "Config not available"}

    @APIExport()
    def getExperimentTimes(self) -> dict:
        """Get calculated experiment times (sunrise/sunset)"""
        if self.lepmon_times:
            sun_times = self.lepmon_times.get_sun_times()
            exp_times = self.lepmon_times.get_experiment_times()
            return {
                "success": True,
                "sun": self.lepmon_times.sun_times_to_dict(sun_times),
                "experiment": {
                    "start": exp_times.start_time,
                    "end": exp_times.end_time,
                    "lepiled_end": exp_times.lepiled_end_time
                }
            }
        return {"success": False, "message": "Times module not available"}

    @APIExport()
    def getCaptureStatus(self) -> dict:
        """Get detailed capture session status"""
        if self.lepmon_capturing:
            return {
                "success": True,
                **self.lepmon_capturing.get_status()
            }
        return {"success": False, "message": "Capture module not available"}

    @APIExport()
    def getHMIStatus(self) -> dict:
        """Get HMI menu status"""
        return {
            "success": True,
            "hmi_open": self.menu_open,
            "current_menu_state": self.current_menu_state,
            "monitoring_active": not self.hmi_stop_event.is_set()
        }

    # ===================== API Endpoints - Actions ===================== #

    @APIExport(requestType="POST")
    def startExperiment(self,
                        exposureTime: float = 100.0,
                        gain: float = 0.0,
                        timelapsePeriod: int = 60,
                        overrideTimecheck: bool = False) -> dict:
        """Start a capture experiment"""
        self._logger.info(f"Starting experiment: exposure={exposureTime}, interval={timelapsePeriod}")
        
        # Update camera settings
        self.changeAutoExposureTime("manual")
        self.changeExposureTime(exposureTime)
        self.changeGain(gain)
        
        # Store parameters
        self.mExperimentParameters.update({
            "exposureTime": exposureTime,
            "gain": gain,
            "timelapsePeriod": timelapsePeriod
        })
        
        # Start capture using modular capturing component
        if self.lepmon_capturing:
            success = self.lepmon_capturing.start_capture_session(
                exposure=exposureTime,
                gain=gain,
                interval=timelapsePeriod // 60,  # Convert seconds to minutes
                override_timecheck=overrideTimecheck
            )
            if success:
                self.is_measure = True
                self.sigIsRunning.emit(True)
                return {"success": True, "message": "Experiment started"}
            return {"success": False, "message": "Failed to start capture session"}
        
        # Fallback to legacy thread if modules not available
        return self._start_legacy_experiment()

    @APIExport(requestType="POST")
    def stopExperimentLepmon(self) -> dict:
        """Stop the current experiment"""
        self._logger.info("Stopping experiment")
        
        if self.lepmon_capturing:
            self.lepmon_capturing.stop_capture_session()
        
        self.is_measure = False
        self.sigIsRunning.emit(False)
        
        return {"success": True, "message": "Experiment stopped"}

    @APIExport(requestType="POST")
    def focusMode(self) -> dict:
        """Activate 15-second focus mode"""
        Thread(target=self._run_focus_mode, daemon=True).start()
        return {"success": True, "message": "Focus mode started for 15s"}

    @APIExport(requestType="POST")
    def setLightState(self, lightName: str, state: bool) -> dict:
        """Control individual LED"""
        try:
            if self.lepmon_lights:
                if state:
                    if lightName == "UV_LED" or lightName == "lepiled":
                        self.lepmon_lights.lepiled_start()
                    elif lightName == "Visible_LED" or lightName == "dimmer":
                        self.lepmon_lights.dim_up()
                    else:
                        # Try GPIO for status LEDs
                        if self.lepmon_gpio:
                            self.lepmon_gpio.turn_on_led(lightName)
                else:
                    if lightName == "UV_LED" or lightName == "lepiled":
                        self.lepmon_lights.lepiled_ende()
                    elif lightName == "Visible_LED" or lightName == "dimmer":
                        self.lepmon_lights.dim_down()
                    else:
                        if self.lepmon_gpio:
                            self.lepmon_gpio.turn_off_led(lightName)
                
                return {"success": True, "message": f"LED {lightName} {'on' if state else 'off'}"}
            return {"success": False, "message": "Lights module not available"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    @APIExport(requestType="POST")
    def setAllLightsOff(self) -> dict:
        """Turn off all LEDs"""
        if self.lepmon_lights:
            self.lepmon_lights.all_off()
        if self.lepmon_gpio:
            for led in LED_PINS.keys():
                self.lepmon_gpio.turn_off_led(led)
        return {"success": True, "message": "All lights off"}

    @APIExport(requestType="POST")
    def updateDisplay(self, line1: str = "", line2: str = "", line3: str = "") -> dict:
        """Update OLED display"""
        if self.lepmon_oled:
            self.lepmon_oled.display_text(line1, line2, line3)
            return {"success": True, "message": "Display updated"}
        return {"success": False, "message": "OLED not available"}

    @APIExport(requestType="POST")
    def showMessage(self, messageKey: str, **kwargs) -> dict:
        """Show a predefined message on display"""
        if self.lepmon_oled:
            self.lepmon_oled.show_message(messageKey, **kwargs)
            return {"success": True, "message": f"Showing: {messageKey}"}
        return {"success": False, "message": "OLED not available"}

    @APIExport(requestType="POST")
    def simulateButtonPress(self, buttonName: str) -> dict:
        """Simulate a button press for testing"""
        if self.lepmon_gpio:
            self.lepmon_gpio.simulate_button_press(buttonName)
            return {"success": True, "message": f"Button {buttonName} simulated"}
        return {"success": False, "message": "GPIO not available"}

    @APIExport(requestType="POST")
    def setSensorData(self, innerTemp: float = None, outerTemp: float = None, humidity: float = None) -> dict:
        """Override sensor data (for testing)"""
        # This would override simulated sensor values
        self.sigSensorUpdate.emit({
            "innerTemp": innerTemp,
            "outerTemp": outerTemp,
            "humidity": humidity
        })
        return {"success": True, "message": "Sensor data updated"}

    @APIExport(requestType="POST")
    def setGPSCoordinates(self, latitude: float, longitude: float) -> dict:
        """Update GPS coordinates for time calculations"""
        if self.lepmon_times:
            self.lepmon_times.update_location(latitude, longitude)
            return {"success": True, "message": "GPS coordinates updated"}
        return {"success": False, "message": "Times module not available"}

    @APIExport(requestType="POST")
    def openHMI(self) -> dict:
        """Open HMI menu"""
        self._open_hmi_menu()
        return {"success": True, "message": "HMI menu opened"}

    @APIExport(requestType="POST")
    def closeHMI(self) -> dict:
        """Close HMI menu"""
        self.menu_open = False
        if self.lepmon_gpio:
            self.lepmon_gpio.turn_off_led("blau")
        if self.lepmon_oled:
            self.lepmon_oled.display_text("HMI geschlossen", "", "")
        return {"success": True, "message": "HMI closed"}

    @APIExport(requestType="POST")
    def reboot(self) -> dict:
        """Request system reboot"""
        self._logger.warning("Reboot requested")
        # In production: os.system("sudo reboot")
        return {"success": True, "message": "Reboot initiated (mock)"}

    # ===================== Image Capture ===================== #

    '''
    @APIExport(requestType="POST")
    def snapImage(self, format: str = "jpg", exposure: float = None) -> dict:
        """Capture a single image"""
        try:
            if exposure:
                self.changeExposureTime(exposure)
            
            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            frame = self.snapImagelepmonCam(filename, fileFormat=format.upper())
            
            self.imagesTaken += 1
            self.sigImagesTaken.emit(self.imagesTaken)
            
            return {
                "success": True,
                "filename": filename,
                "format": format,
                "imageCount": self.imagesTaken
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    '''
    
    @APIExport(runOnUIThread=True)
    def returnLastSnappedImage(self) -> Response:
        """Return the last captured image as PNG"""
        try:
            arr = self.mFrame if self.mFrame is not None else self.detectorlepmonCam.getLatestFrame()
            if arr is None:
                raise RuntimeError("No image available")
            
            im = Image.fromarray(arr.astype(np.uint8))
            with io.BytesIO() as buf:
                im = im.convert("L")
                im.save(buf, format="PNG")
                im_bytes = buf.getvalue()
            
            headers = {"Content-Disposition": 'inline; filename="image.png"'}
            return Response(im_bytes, headers=headers, media_type="image/png")
        except Exception as e:
            raise RuntimeError(f"No image available: {e}") from e

    @APIExport(runOnUIThread=True)
    def snapImagelepmonCam(self, fileName: str = None, fileFormat: str = "JPG"):
        """Capture and save image to disk"""
        if not fileName:
            fileName = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        frame = self.detectorlepmonCam.getLatestFrame()
        if frame is None:
            self._logger.warning("No frame from camera")
            return None
        
        self.mFrame = frame
        
        if fileFormat.upper() == "TIF":
            import tifffile as tif
            tif.imwrite(fileName + ".tif", frame, append=False)
        elif fileFormat.upper() == "JPG":
            cv2.imwrite(fileName + ".jpg", frame)
        elif fileFormat.upper() == "PNG":
            cv2.imwrite(fileName + ".png", frame)
        
        return frame

    # ===================== Camera Controls ===================== #

    @APIExport(runOnUIThread=True)
    def changeExposureTime(self, value: float):
        """Set camera exposure time"""
        try:
            self.mExperimentParameters["exposureTime"] = value
            self.detectorlepmonCam.setParameter(name="exposure", value=value)
        except Exception as e:
            self._logger.error(f"Could not set exposure: {e}")

    @APIExport(runOnUIThread=True)
    def changeAutoExposureTime(self, value: str):
        """Set auto-exposure mode ('auto' or 'manual')"""
        try:
            self.detectorlepmonCam.setParameter(name="exposure_mode", value=value)
        except Exception as e:
            self._logger.error(f"Could not set exposure mode: {e}")

    @APIExport(runOnUIThread=True)
    def changeGain(self, value: float):
        """Set camera gain"""
        try:
            self.mExperimentParameters["gain"] = value
            self.detectorlepmonCam.setGain(value)
        except Exception as e:
            self._logger.error(f"Could not set gain: {e}")

    # ===================== Helper Functions ===================== #

    def _compute_free_space(self) -> str:
        """Compute free disk space"""
        try:
            usage = dirtools.getDiskusage()
            free_prc = 100.0 - (usage * 100)
            return f"{free_prc:.1f}% free"
        except:
            return "unknown"

    def _start_legacy_experiment(self) -> dict:
        """Fallback experiment loop without modular components"""
        def experiment_thread():
            self.is_measure = True
            self.imagesTaken = 0
            self.sigIsRunning.emit(True)
            
            while self.is_measure:
                self.imagesTaken += 1
                self.sigImagesTaken.emit(self.imagesTaken)
                
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    self.snapImagelepmonCam(filename)
                except Exception as e:
                    self._logger.error(f"Capture error: {e}")
                
                # Wait for interval
                for _ in range(self.mExperimentParameters["timelapsePeriod"]):
                    if not self.is_measure:
                        break
                    time.sleep(1)
            
            self.sigIsRunning.emit(False)
        
        Thread(target=experiment_thread, daemon=True).start()
        return {"success": True, "message": "Legacy experiment started"}

    # ===================== Cleanup ===================== #

    def closeEvent(self):
        """Cleanup on controller close"""
        self._sensor_thread_active = False
        self.hmi_stop_event.set()
        
        if self.lepmon_capturing:
            self.lepmon_capturing.stop_capture_session()
        
        if self.lepmon_lights:
            self.lepmon_lights.all_off()
        
        if self.lepmon_gpio:
            self.lepmon_gpio.cleanup()
        
        self._logger.info("LepmonController cleanup complete")

    def __del__(self):
        """Destructor"""
        self.closeEvent()
