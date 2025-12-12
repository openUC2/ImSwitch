import io
import time
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageFile
from fastapi import Response
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import threading
from imswitch.imcommon.framework import Thread, Signal
from imswitch.imcommon.model import initLogger, APIExport, dirtools
from ..basecontrollers import ImConWidgetController
from imswitch import IS_HEADLESS

# Import extracted modules
from imswitch.imcontrol.controller.pidcontroller import PIDController
from imswitch.imcontrol.controller.loggingutils import FocusLockCSVLogger
from imswitch.imcontrol.controller.focusmetrics import FocusMetricFactory, FocusConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Dataclasses (API-stable)
# =========================
@dataclass
class FocusLockParams:
    focus_metric: str = "peak" # "astigmatism", "gaussian", "gradient", "peak"
    crop_center: Optional[List[int]] = None
    crop_size: Optional[int] = None
    gaussian_sigma: float = 11.0
    background_threshold: float = 1.0
    update_freq: float = 1.0
    two_foci_enabled: bool = False
    z_stack_enabled: bool = False
    z_step_limit_nm: float = 40.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focus_metric": self.focus_metric,
            "crop_center": self.crop_center,
            "crop_size": self.crop_size,
            "gaussian_sigma": self.gaussian_sigma,
            "background_threshold": self.background_threshold,
            "update_freq": self.update_freq,
            "two_foci_enabled": self.two_foci_enabled,
            "z_stack_enabled": self.z_stack_enabled,
            "z_step_limit_nm": self.z_step_limit_nm,
        }


@dataclass
class PIControllerParams:
    # API-compatible: keep name/fields; extend silently
    kp: float = 0.0
    ki: float = 0.0
    set_point: float = 0.0
    safety_distance_limit: float = 500.0   # treated as travel budget (µm)
    safety_move_limit: float = 50         # per-update clamp (µm)
    min_step_threshold: float = 2      # deadband (µm)
    safety_motion_active: bool = False
    # New (does not break API)
    kd: float = 0.0
    scale_um_per_unit: float = 1.0         # focus-units -> µm
    sample_time: float = 0.1               # s, updated from update_freq
    output_lowpass_alpha: float = 0.0      # 0..1 smoothing of controller output
    integral_limit: float = 100.0          # anti-windup (controller units)
    meas_lowpass_alpha: float = 0.0        # pre-filter focus value (0..1 EMA)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kp": self.kp,
            "ki": self.ki,
            "set_point": self.set_point,
            "safety_distance_limit": self.safety_distance_limit,
            "safety_move_limit": self.safety_move_limit,
            "min_step_threshold": self.min_step_threshold,
            "safety_motion_active": self.safety_motion_active,
            # expose new params too (non-breaking)
            "kd": self.kd,
            "scale_um_per_unit": self.scale_um_per_unit,
            "sample_time": self.sample_time,
            "output_lowpass_alpha": self.output_lowpass_alpha,
            "integral_limit": self.integral_limit,
            "meas_lowpass_alpha": self.meas_lowpass_alpha,
        }


@dataclass
class CalibrationParams:
    from_position: float = 49.0
    to_position: float = 51.0
    num_steps: int = 20
    settle_time: float = 0.5
    scan_range_um: float = 2.0  # Range to scan around current position (±scan_range_um/2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_position": self.from_position,
            "to_position": self.to_position,
            "num_steps": self.num_steps,
            "settle_time": self.settle_time,
            "scan_range_um": self.scan_range_um,
        }


@dataclass
class CalibrationData:
    """Enhanced calibration data structure with lookup table and metadata."""
    position_data: List[float]
    focus_data: List[float]
    polynomial_coeffs: Optional[List[float]]
    sensitivity_nm_per_unit: float
    r_squared: float
    linear_range: Tuple[float, float]  # valid focus range for linear approximation
    timestamp: float
    lookup_table: Optional[Dict[float, float]] = None  # focus_value -> z_position

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_data": self.position_data,
            "focus_data": self.focus_data,
            "polynomial_coeffs": self.polynomial_coeffs,
            "sensitivity_nm_per_unit": self.sensitivity_nm_per_unit,
            "r_squared": self.r_squared,
            "linear_range": list(self.linear_range),
            "timestamp": self.timestamp,
            "lookup_table": self.lookup_table,
        }


@dataclass
class FocusLockState:
    is_measuring: bool = False
    is_locked: bool = False
    about_to_lock: bool = False
    current_focus_value: float = 0.0
    current_position: float = 0.0
    measurement_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_measuring": self.is_measuring,
            "is_locked": self.is_locked,
            "about_to_lock": self.about_to_lock,
            "current_focus_value": self.current_focus_value,
            "current_position": self.current_position,
            "measurement_active": self.measurement_active,
        }


# =========================
# Controller
# =========================
MAX_SPEED = 2000  # µm/s
class FocusLockController(ImConWidgetController):
    """Linked to FocusLockWidget. Public API (APIExport) kept stable."""

    sigFocusValueUpdate = Signal(object)       # Renamed from sigUpdateFocusValue for consistency
    sigFocusLockStateChanged = Signal(object)  # (state_dict)
    sigCalibrationProgress = Signal(object)    # (progress_dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        if self._setupInfo.focusLock is None:
            return

        self.camera = self._setupInfo.focusLock.camera
        self.positioner = self._setupInfo.focusLock.positioner
        try:
            self.stage = self._master.positionersManager[self.positioner]
        except KeyError:
            self._logger.error(f"Positioner '{self.positioner}' not found using first in list.")
            self.positioner = self._master.positionersManager.getAllDeviceNames()[0]
            self.stage = self._master.positionersManager[self.positioner]


        # Internal Z position tracking
        self.currentZPosition = self.stage.getPosition()["Z"]
        self._commChannel.sigUpdateMotorPosition.connect(self._onMotorPositionUpdate)

        # Params - Consolidated focus parameters
        self._focus_params = FocusLockParams(
            focus_metric=getattr(self._setupInfo.focusLock, "focusLockMetric", "peak"),
            crop_center=getattr(self._setupInfo.focusLock, "cropCenter", None),
            crop_size=getattr(self._setupInfo.focusLock, "cropSize", None),
            update_freq=self._setupInfo.focusLock.updateFreq or 10,
        )
        # Initialize focus metric computer using extracted module 
        focus_config = FocusConfig(
            gaussian_sigma=self._focus_params.gaussian_sigma,
            background_threshold=self._focus_params.background_threshold,
            crop_radius=self._focus_params.crop_size or 300,
            enable_gaussian_blur=True,
            peak_distance=200
        )
        self._focus_metric = FocusMetricFactory.create(self._focus_params.focus_metric, focus_config)

        # Camera ROI settings
        fovWidth = getattr(self._setupInfo.focusLock, "fovWidth", 1024)
        fovHeight = getattr(self._setupInfo.focusLock, "fovHeight", fovWidth) # if not set, assume square FOV
        fovCenter = getattr(self._setupInfo.focusLock, "fovCenter", [None, None])
        if fovCenter[0] is None or fovCenter[1] is None:
            # Default to image center
            cam = self._master.detectorsManager[self.camera]
            imageWidth, imageHeight = cam.fullShape[0], cam.fullShape[1]
            fovCenter = [imageWidth // 2, imageHeight // 2]
        # set up the FOV on the camera side
        try:
            cam = self._master.detectorsManager[self.camera]
            # def setROI(self,hpos=None,vpos=None,hsize=None,vsize=None):
            # computing coordinates for cropping the image
            imageWidth, imageHeight = cam.fullShape[0], cam.fullShape[1]
            hpos = imageWidth//2 - fovWidth//2
            vpos = imageHeight//2 - fovHeight//2
            cam.crop(hpos = hpos, vpos = vpos, hsize = fovWidth, vsize = fovHeight)
        except Exception as e:
            self._logger.error(f"Failed to set camera ROI: {e}")

        # Laser (optional)
        laserName = getattr(self._setupInfo.focusLock, "laserName", None)
        laserValue = getattr(self._setupInfo.focusLock, "laserValue", None)
        if laserName and laserValue is not None:
            try:
                self._master.lasersManager[laserName].setEnabled(True)
                self._master.lasersManager[laserName].setValue(laserValue)
            except KeyError:
                self._logger.error(f"Laser '{laserName}' not found. Cannot set power to {laserValue}.")

        # PI parameters (API names preserved), add extras
        piKp = getattr(self._setupInfo.focusLock, "piKp", 0.0)
        piKi = getattr(self._setupInfo.focusLock, "piKi", 0.0)
        piKd = getattr(self._setupInfo.focusLock, "piKd", 0.0)
        setPoint = getattr(self._setupInfo.focusLock, "setPoint", 0.0)
        safety_distance_limit = getattr(self._setupInfo.focusLock, "safetyDistanceLimit", 500.0)
        safetyMoveLimit = getattr(self._setupInfo.focusLock, "safetyMoveLimit", 50.0)
        minStepThreshold = getattr(self._setupInfo.focusLock, "minStepThreshold", 2)
        safety_motion_active = getattr(self._setupInfo.focusLock, "safetyMotionActive", False)
        scale_um_per_unit = getattr(self._setupInfo.focusLock, "scaleUmPerUnit", 100.0) # scale that translates focus units to microns
        output_lowpass_alpha = getattr(self._setupInfo.focusLock, "outputLowpassAlpha", 0.0)
        integral_limit = getattr(self._setupInfo.focusLock, "integralLimit", 100.0)
        meas_lowpass_alpha = getattr(self._setupInfo.focusLock, "measLowpassAlpha", 0.0)

        self._pi_params = PIControllerParams(
            kp=piKp, ki=piKi, kd=piKd, set_point=setPoint,
            safety_distance_limit=safety_distance_limit,
            safety_move_limit=safetyMoveLimit,
            min_step_threshold=minStepThreshold,
            safety_motion_active=safety_motion_active,
            scale_um_per_unit=scale_um_per_unit,
            sample_time=1.0 / (self._focus_params.update_freq or 10.0),
            output_lowpass_alpha=output_lowpass_alpha,
            integral_limit=integral_limit,
            meas_lowpass_alpha=meas_lowpass_alpha,
        )

        self._calib_params = CalibrationParams()
        self._state = FocusLockState()

        # Calibration data storage - this is the key integration point for PID controller
        self._current_calibration: Optional[CalibrationData] = None

        # Current focus value (renamed from setPointSignal for clarity)
        self.current_focus_value = 0.0

        # Lock state variables
        self.locked = False
        self.aboutToLock = False

        # Thread control
        self.__isPollingFramesActive = False
        self.pollingFrameUpdatePeriode = 1.0 / self._focus_params.update_freq

        # About-to-lock logic
        self.aboutToLockDiffMax = 0.4

        # Data buffers for plotting
        self.buffer = 40
        self.currPoint = 0
        self.setPointData = np.zeros(self.buffer, dtype=float)
        self.timeData = np.zeros(self.buffer, dtype=float)
        self.reduceImageScaleFactor = 4

        # Travel budget tracking
        self._travel_used_um = 0.0

        # Measurement smoothing
        self._meas_filt = None

        # deubgging?
        self.is_debug = False

        # Threads
        self._focusCalibThread = FocusCalibThread(self)

        # CSV logging setup using extracted module
        try:
            csv_log_dir = os.path.join(dirtools.UserFileDirs.Root, "FocusLockController")
            self._csv_logger = FocusLockCSVLogger(csv_log_dir)
            self._logger.info(f"CSV logging initialized at: {csv_log_dir}")
        except Exception as e:
            self._logger.error(f"Failed to setup CSV logging: {e}")
            self._csv_logger = None

        # PID instance using extracted module (kept as self.pi for API stability)
        self.pi: Optional[PIDController] = None


    def __del__(self):
        try:
            self.__isPollingFramesActive = False
        except Exception:
            pass
        try:
            if hasattr(self, "_master") and hasattr(self, "camera"):
                self._master.detectorsManager[self.camera].stopAcquisition()
        except Exception:
            pass
        try:
            if hasattr(self, "ESP32Camera"):
                self.ESP32Camera.stopStreaming()
        except Exception:
            pass
        if hasattr(super(), "__del__"):
            try:
                super().__del__()
            except Exception:
                pass

    def updateThread(self):
        if not self.__isPollingFramesActive:
            self.__isPollingFramesActive = True
            self._pollFramesThread = threading.Thread(target=self._pollFrames, name="FocusLockPollFramesThread")
            self._pollFramesThread.daemon = True
            self._pollFramesThread.start()

    def _onMotorPositionUpdate(self, pos: Dict[str, float]):
        # Handle different signal formats from positioner
        if isinstance(pos, dict) and self.positioner in pos and "Z" in pos[self.positioner]:
            self.currentZPosition = pos[self.positioner]["Z"]

    # =========================
    # API: Params/state
    # =========================
    @APIExport(runOnUIThread=True)
    def getFocusLockParams(self) -> Dict[str, Any]:
        return self._focus_params.to_dict()

    @APIExport(runOnUIThread=True)
    def setFocusLockParams(self, **kwargs) -> Dict[str, Any]:
        for key, value in kwargs.items():
            if hasattr(self._focus_params, key):
                setattr(self._focus_params, key, value)
                if key == "focus_metric":
                    # Update focus metric computer
                    focus_config = FocusConfig(
                        gaussian_sigma=self._focus_params.gaussian_sigma,
                        background_threshold=self._focus_params.background_threshold,
                        crop_radius=self._focus_params.crop_size or 300,
                    )
                    self._focus_metric = FocusMetricFactory.create(value, focus_config)
                elif key == "update_freq":
                    self.pollingFrameUpdatePeriode = 1.0 / max(1e-3, float(value))
                    # keep PID dt in sync
                    self._pi_params.sample_time = self.pollingFrameUpdatePeriode
                    if self.pi:
                        self.pi.update_parameters(sample_time=self._pi_params.sample_time)
                elif key in ["gaussian_sigma", "background_threshold", "crop_size"]:
                    # Update focus metric config
                    self._focus_metric.update_config(**{key: value})
        return self._focus_params.to_dict()

    @APIExport(runOnUIThread=True)
    def getPIControllerParams(self) -> Dict[str, Any]:
        return self._pi_params.to_dict()

    @APIExport(runOnUIThread=True)
    def setPIControllerParams(self, **kwargs) -> Dict[str, Any]:
        for key, value in kwargs.items():
            if hasattr(self._pi_params, key):
                setattr(self._pi_params, key, value)
        if hasattr(self, "pi") and self.pi:
            self.pi.set_parameters(self._pi_params.kp, self._pi_params.ki)
            self.pi.update_parameters(
                kd=self._pi_params.kd,
                set_point=self._pi_params.set_point,
                sample_time=self._pi_params.sample_time,
                integral_limit=self._pi_params.integral_limit,
                output_lowpass_alpha=self._pi_params.output_lowpass_alpha,
            )
        if not IS_HEADLESS:
            self._widget.setKp(self._pi_params.kp)
            self._widget.setKi(self._pi_params.ki)
        return self._pi_params.to_dict()

    @APIExport(runOnUIThread=True)
    def getCalibrationParams(self) -> Dict[str, Any]:
        return self._calib_params.to_dict()

    @APIExport(runOnUIThread=True)
    def setCalibrationParams(self, **kwargs) -> Dict[str, Any]:
        for key, value in kwargs.items():
            if hasattr(self._calib_params, key):
                setattr(self._calib_params, key, value)
        return self._calib_params.to_dict()

    @APIExport(runOnUIThread=True)
    def getFocusLockState(self) -> Dict[str, Any]:
        self._state.is_locked = self.locked
        self._state.about_to_lock = self.aboutToLock
        self._state.current_focus_value = self.current_focus_value
        self._state.current_position = self.currentZPosition
        self._state.measurement_active = self._state.is_measuring or self.locked or self.aboutToLock
        return self._state.to_dict()

    # =========================
    # API: Measurement control
    # =========================
    @APIExport(runOnUIThread=True)
    def startFocusMeasurement(self) -> bool:
        # Start polling and create a new focus metric instance to reset any internal state
        focus_config = FocusConfig(
            gaussian_sigma=self._focus_params.gaussian_sigma,
            background_threshold=self._focus_params.background_threshold,
            crop_radius=self._focus_params.crop_size or 300,
            enable_gaussian_blur=True,
        )
        self._focus_metric = FocusMetricFactory.create(self._focus_params.focus_metric, focus_config)

        self.updateThread()
        # Camera acquisition
        try:
            self._master.detectorsManager[self.camera].startAcquisition()
        except Exception as e:
            self._logger.error(f"Failed to start acquisition on camera '{self.camera}': {e}")

        try:
            if not self._state.is_measuring:
                self._state.is_measuring = True
                self._emitStateChangedSignal()
                self._logger.info("Focus measurement started")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to start focus measurement: {e}")
            return False

    @APIExport(runOnUIThread=True)
    def stopFocusMeasurement(self) -> bool:
        # Camera acquisition
        try:
            self._master.detectorsManager[self.camera].stopAcquisition()
        except Exception as e:
            self._logger.error(f"Failed to start acquisition on camera '{self.camera}': {e}")

        try:
            if self._state.is_measuring:
                self._state.is_measuring = False
                self.unlockFocus()
                self._emitStateChangedSignal()
                self._logger.info("Focus measurement stopped")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to stop focus measurement: {e}")
            return False

    # =========================
    # API: Lock control
    # =========================
    @APIExport(runOnUIThread=True)
    def enableFocusLock(self, enable: bool = True) -> bool:
        try:
            if enable and not self.locked:
                if not self._state.is_measuring:
                    self.startFocusMeasurement()
                # Use internal Z position or fallback to hardware query
                self.lockFocus(self.currentZPosition)
                return True
            elif not enable and self.locked:
                self.unlockFocus()
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to enable/disable focus lock: {e}")
            return False

    @APIExport(runOnUIThread=True)
    def isFocusLocked(self) -> bool:
        return self.locked

    def _emitStateChangedSignal(self):
        """Emit signal for UI and external listeners to update their state."""
        self.sigFocusLockStateChanged.emit(self.getFocusLockState())

    # =========================
    # Legacy-compatible methods
    # =========================
    @APIExport(runOnUIThread=True)
    def unlockFocus(self):
        if self.locked:
            self.locked = False
            if self.pi:
                self.pi.reset()
            self._travel_used_um = 0.0
            # emit signal to update button state
            self._emitStateChangedSignal()
            self._logger.info("Focus unlocked")

    @APIExport(runOnUIThread=True)
    def toggleFocus(self, toLock: bool = None):
        self.aboutToLock = False
        if (not IS_HEADLESS and self._widget.lockButton.isChecked()) or (toLock is not None and toLock and not self.locked):
            self.lockFocus(self.stage.getPosition()["Z"])
            if not IS_HEADLESS:
                self._widget.lockButton.setText("Unlock")
        else:
            self.unlockFocus()
            if not IS_HEADLESS:
                self._widget.lockButton.setText("Lock")

    def cameraDialog(self):
        try:
            self._master.detectorsManager[self.camera].openPropertiesDialog()
        except Exception as e:
            self._logger.error(f"Failed to open camera dialog: {e}")

    @APIExport(runOnUIThread=True)
    def getCalibrationResults(self) -> Dict[str, Any]:
        """Get the results from the last calibration run."""
        return self._focusCalibThread.getData()

    @APIExport(runOnUIThread=True)
    def isCalibrationRunning(self) -> bool:
        """Check if a calibration is currently in progress."""
        return self._focusCalibThread.isRunning()

    def showCalibrationCurve(self):
        """Display the calibration curve (GUI only)."""
        if hasattr(self._focusCalibThread, 'show'):
            self._focusCalibThread.show()

    def twoFociVarChange(self):
        """Toggle two foci mode."""
        self._focus_params.two_foci_enabled = not self._focus_params.two_foci_enabled

    def zStackVarChange(self):
        """Toggle Z stack mode."""
        self._focus_params.z_stack_enabled = not self._focus_params.z_stack_enabled

    @APIExport(runOnUIThread=True)
    def setExposureTime(self, exposure_time: float):
        try:
            self._master.detectorsManager[self.camera].setParameter('exposure', exposure_time)
            self._logger.debug(f"Set exposure time to {exposure_time}")
        except Exception as e:
            self._logger.error(f"Failed to set exposure time: {e}")

    @APIExport(runOnUIThread=True)
    def setGain(self, gain: float):
        try:
            self._master.detectorsManager[self.camera].setParameter('gain', gain)
            self._logger.debug(f"Set gain to {gain}")
        except Exception as e:
            self._logger.error(f"Failed to set gain: {e}")

    # =========================
    # Single Source of Truth: Frame Capture and Focus Computation
    # =========================
    def _captureAndComputeFocus(self, num_frames: int = 2, is_debug: bool = False) -> Dict[str, Any]:
        """Capture frame(s) and compute focus value - single source of truth.
        
        This method is the ONLY place where frames are captured and processed
        for focus computation. All other methods (polling, calibration, autofocus)
        should use this method to ensure consistent processing.
        
        Args:
            num_frames: Number of frames to average (default 2 for buffer clearing)
            
        Returns:
            Dict containing:
            - focus_value: float - computed focus metric value
            - cropped_image: np.ndarray - the cropped image used for computation
            - timestamp: float - capture timestamp
            - raw_result: dict - full result from focus metric compute
            - valid: bool - whether the computation succeeded
        """
        timestamp = time.time()
        result = {
            "focus_value": 0.0,
            "cropped_image": None,
            "timestamp": timestamp,
            "raw_result": {},
            "valid": False,
        }
        
        try:
            # Step 1: Capture frame(s) and average if multiple
            frames = []
            for _ in range(num_frames):
                frame = self._master.detectorsManager[self.camera].getLatestFrame()
                if frame is not None:
                    frames.append(frame)
            
            if not frames:
                self._logger.warning("No frames captured from camera")
                return result
            
            # Average frames if multiple
            im = np.mean(np.array(frames), axis=0) if len(frames) > 1 else frames[0]
            
            # Step 2: Crop image using consistent parameters from _focus_params
            crop_size = self._focus_params.crop_size
            crop_center = self._focus_params.crop_center
            
            try:
                import NanoImagingPack as nip
                cropped_im = nip.extract(
                    img=im,
                    ROIsize=(crop_size, crop_size),
                    centerpos=crop_center,
                    PadValue=0.0,
                    checkComplex=True,
                )
            except Exception:
                cropped_im = self.extract(im, crop_size=crop_size, crop_center=crop_center)
            
            result["cropped_image"] = cropped_im
            
            # Step 3: Compute focus metric using the configured focus metric
            focus_result = self._focus_metric.compute(cropped_im)
            focus_value = focus_result.get("focus", None)
            
            '''
            is_debug => plot and visualize the focusmetric and current image by using the following parameters:
            {
            "t": ts,
            "focus": focus_value,
            "left_peak_x": left_peak,
            "right_peak_x": right_peak,
            "x_peak_distance": x_peak_distance,
            "proj_x": projx_s.astype(float),
            "avg_peak_distance": self._get_average_distance(),
            "peak_history_length": len(self.peak_distances),
            "compute_ms": (time.time() - t0) * 1000.0,
            }
            '''
            if is_debug:
                try:
                    # plot the current image and plot the focus metric result
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use("Agg")

                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].imshow(cropped_im, cmap='gray')
                    axs[0].set_title('Cropped Image for Focus Computation')
                    # aslo plot the projection and peaks from focus_result if available
                    if "proj_x" in focus_result:
                        axs[1].plot(focus_result["proj_x"], label='Projection X')
                    if "left_peak_x" in focus_result and "right_peak_x" in focus_result:
                        axs[1].axvline(focus_result["left_peak_x"], color='r', linestyle='--', label='Left Peak')
                        if focus_result["right_peak_x"] is not None:
                            axs[1].axvline(focus_result["right_peak_x"], color='g', linestyle='--', label='Right Peak')
                        # add current z-position and focus value to the plot title
                        axs[1].set_title(f'Focus Value: {focus_value:.2f} at Z: {self.currentZPosition:.2f} µm')
                    else:
                        axs[1].set_title(f'Focus Value: {focus_value:.2f} at Z: {self.currentZPosition:.2f} µm')
                        axs[1].legend() 
                    plt.savefig(f"focus_metric_debug_{int(timestamp)}.png")
                    plt.close(fig)
                except Exception as e:
                    self._logger.error(f"Failed to generate debug plot: {e}")
            if focus_value is None or np.isnan(focus_value):
                self._logger.debug("Invalid focus value computed (None or NaN)")
                return result
            
            result["focus_value"] = float(focus_value)
            result["raw_result"] = focus_result
            result["timestamp"] = focus_result.get("t", timestamp)
            result["valid"] = True
            
            # Update internal state
            self.current_focus_value = result["focus_value"]
            self.cropped_im = cropped_im
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error in _captureAndComputeFocus: {e}")
            return result

    def _pollFrames(self):
        tLast = 0
        # Store a history of the last values and filter out outliers
        self._focus_metric.reset_history()
        try:
            while self.__isPollingFramesActive:
                if (time.time() - tLast) < self.pollingFrameUpdatePeriode:
                    time.sleep(0.001)
                    continue
                tLast = time.time()
                if not self._state.is_measuring and not self.locked and not self.aboutToLock:
                    continue
                
                # Use single source of truth for frame capture and focus computation
                capture_result = self._captureAndComputeFocus(num_frames=2, is_debug= self.is_debug)
                
                if not capture_result["valid"]:
                    continue
                
                # Get current timestamp for logging
                current_timestamp = capture_result["timestamp"]

                # Emit enhanced focus value signal
                focus_data = {
                    "focus_value": self.current_focus_value,
                    "timestamp": current_timestamp,
                    "is_locked": self.locked,
                    "current_position": self.currentZPosition,
                    "focus_metric": self._focus_params.focus_metric,
                    "focus_setpoint": self._pi_params.set_point if self.pi else 0,
                }
                self.sigFocusValueUpdate.emit(focus_data)

                # Initialize variables for CSV logging
                pi_output = None
                step_um = 0.0

                # === Control action (relative moves only) ===
                if self.locked and self.pi is not None:
                    meas = float(self.current_focus_value)
                    if self._pi_params.meas_lowpass_alpha > 0.0:
                        a = self._pi_params.meas_lowpass_alpha
                        self._meas_filt = meas if self._meas_filt is None else a * self._meas_filt + (1 - a) * meas
                        meas_for_pid = self._meas_filt
                    else:
                        meas_for_pid = meas

                    u = self.pi.update(meas_for_pid)  # controller units
                    pi_output = u  # Store for logging
                    # Use calibration-based scale factor if available, otherwise fall back to configured scale
                    scale_factor = self._getCalibrationBasedScale()
                    step_um = u * scale_factor  # convert to µm

                    # Deadband: skip small movements
                    if abs(step_um) < self._pi_params.min_step_threshold:
                        step_um = 0.0

                    # Per-update clamp
                    limit = abs(self._pi_params.safety_move_limit)
                    step_um = max(min(step_um, limit), -limit)

                    if step_um != 0.0:
                        # Use relative movement
                        self.stage.move(value=step_um, axis="Z", speed=MAX_SPEED, is_blocking=False, is_absolute=False)
                        self._travel_used_um += abs(step_um)
                        # Travel budget acts like safety_distance_limit
                        if self._pi_params.safety_motion_active and self._travel_used_um > self._pi_params.safety_distance_limit:
                            self._logger.warning("Travel budget exceeded; unlocking focus.")
                            self.unlockFocus()

                elif self.aboutToLock:
                    if not hasattr(self, "aboutToLockDataPoints"):
                        self.aboutToLockDataPoints = np.zeros(5, dtype=float)
                    self.aboutToLockUpdate()

                # Log focus measurement to CSV using extracted logging module
                if (self._state.is_measuring or self.locked or self.aboutToLock) and self._csv_logger:
                    try:
                        self._csv_logger.log_focus_lock_data(
                            focus_value=float(self.current_focus_value),
                            is_locked=self.locked,
                            current_position=self.currentZPosition,
                            timestamp=current_timestamp,
                            pi_output=pi_output if self.locked else None,
                            focus_metric=self._focus_params.focus_metric,
                            crop_size=self._focus_params.crop_size,
                            crop_center=str(self._focus_params.crop_center) if self._focus_params.crop_center else None,
                            step_size_um=step_um if self.locked else None,
                            travel_used_um=self._travel_used_um,
                        )
                    except Exception as e:
                        self._logger.error(f"Failed to log focus measurement: {e}")

                    # Update plotting buffers
                    self.updateSetPointData()
        except Exception as e:
            self._logger.error(f"Error in frame polling thread: {e}")
            self.__isPollingFramesActive = False

    @APIExport(runOnUIThread=True)
    def setParamsAstigmatism(self, gaussian_sigma: float, background_threshold: float,
                        crop_size: int, crop_center: Optional[List[int]] = None):
        self._focus_params.gaussian_sigma = float(gaussian_sigma)
        self._focus_params.background_threshold = float(background_threshold)
        self._focus_params.crop_size = int(crop_size)
        if crop_center is not None:
            self._focus_params.crop_center = crop_center

        # Update focus metric config
        self.setFocusLockParams(**self.getParamsAstigmatism())


    @APIExport(runOnUIThread=True)
    def getParamsAstigmatism(self):
        return {
            "gaussian_sigma": self._focus_params.gaussian_sigma,
            "background_threshold": self._focus_params.background_threshold,
            "crop_size": self._focus_params.crop_size,
            "crop_center": self._focus_params.crop_center,
        }

    def aboutToLockUpdate(self):
        self.aboutToLockDataPoints = np.roll(self.aboutToLockDataPoints, 1)
        self.aboutToLockDataPoints[0] = float(self.current_focus_value)
        averageDiff = float(np.std(self.aboutToLockDataPoints))
        if averageDiff < self.aboutToLockDiffMax:
            # Use internal Z position or fallback to hardware query
            self.lockFocus(self.currentZPosition)
            self.aboutToLock = False

    def updateSetPointData(self):
        '''
        Update the data buffer for plotting the focus value over time.
        '''
        if self.currPoint < self.buffer:
            self.setPointData[self.currPoint] = self.current_focus_value
            self.timeData[self.currPoint] = 0.0
        else:
            self.setPointData = np.roll(self.setPointData, -1)
            self.setPointData[-1] = self.current_focus_value
            self.timeData = np.roll(self.timeData, -1)
            self.timeData[-1] = 0.0
        self.currPoint += 1

    @APIExport(runOnUIThread=True)
    def setPIParameters(self, kp: float, ki: float):
        self._pi_params.kp = float(kp)
        self._pi_params.ki = float(ki)
        if not self.pi:
            self.pi = PIDController(
                set_point=self._pi_params.set_point,
                kp=self._pi_params.kp, ki=self._pi_params.ki, kd=self._pi_params.kd,
                sample_time=self._pi_params.sample_time,
                integral_limit=self._pi_params.integral_limit,
                output_lowpass_alpha=self._pi_params.output_lowpass_alpha,
            )
        else:
            self.pi.set_parameters(kp, ki)
        if not IS_HEADLESS:
            self._widget.setKp(kp)
            self._widget.setKi(ki)

    @APIExport(runOnUIThread=True)
    def getPIParameters(self) -> Tuple[float, float]:
        return self._pi_params.kp, self._pi_params.ki

    def updatePI(self) -> float:
        """Kept for compatibility; returns last computed move in µm (no position reads)."""
        if not self.locked or not self.pi:
            return 0.0
        meas = float(self.current_focus_value)
        if self._pi_params.meas_lowpass_alpha > 0.0:
            a = self._pi_params.meas_lowpass_alpha
            self._meas_filt = meas if self._meas_filt is None else a * self._meas_filt + (1 - a) * meas
            meas_for_pid = self._meas_filt
        else:
            meas_for_pid = meas
        u = self.pi.update(meas_for_pid)
        # Use calibration-based scale factor if available, otherwise fall back to configured scale
        scale_factor = self._getCalibrationBasedScale()
        step_um = u * scale_factor
        # apply deadband + clamp, mirror of _pollFrames logic
        if abs(step_um) < self._pi_params.min_step_threshold:
            step_um = 0.0
        limit = abs(self._pi_params.safety_move_limit)
        step_um = max(min(step_um, limit), -limit)
        return step_um

    def _getCalibrationBasedScale(self) -> float:
        """Get the scale factor from calibration data if available, otherwise use default.
        
        Note: The scale factor can be negative depending on the optical setup,
        which is valid and handled correctly.
        """
        if self._current_calibration:
            # Convert nm per unit to µm per unit for consistency with existing code
            return self._current_calibration.sensitivity_nm_per_unit / 1000.0
        else:
            # Fall back to configured scale factor if no calibration available
            return self._pi_params.scale_um_per_unit

    @APIExport(runOnUIThread=True)
    def getCurrentFocusValue(self) -> Dict[str, Any]:
        """Get the current focus value and metadata.

        Returns:
            Dict containing:
            - focus_value: float - current focus metric value
            - timestamp: float - measurement timestamp
            - is_measuring: bool - whether continuous measurement is active
            - is_locked: bool - whether focus lock is engaged
        """
        return {
            "focus_value": float(self.current_focus_value),
            "timestamp": time.time(),
            "is_measuring": self._state.is_measuring,
            "is_locked": self.locked,
        }

    def lockFocus(self, zpos):
        if self.locked:
            return

        # Setpoint is current measured focus
        self._pi_params.set_point = float(self.current_focus_value)
        self.pi = PIDController(
            set_point=self._pi_params.set_point,
            kp=self._pi_params.kp,
            ki=self._pi_params.ki,
            kd=self._pi_params.kd,
            sample_time=self._pi_params.sample_time,
            integral_limit=self._pi_params.integral_limit,
            output_lowpass_alpha=self._pi_params.output_lowpass_alpha,
        )
        self.locked = True
        self._travel_used_um = 0.0

        self.updateZStepLimits()
        self._emitStateChangedSignal()
        self._logger.info(f"Focus locked at position {zpos} with set point {self.current_focus_value}")

    def updateZStepLimits(self):
        """Update Z step limits from configuration."""
        try:
            if not IS_HEADLESS and hasattr(self, '_widget'):
                self._focus_params.z_step_limit_nm = float(self._widget.zStepFromEdit.text())
        except Exception:
            pass  # Use default from focus params

    @staticmethod
    def extract(marray: np.ndarray, crop_size: Optional[int] = None, crop_center: Optional[List[int]] = None) -> np.ndarray:
        h, w = marray.shape[:2]
        if crop_center is None:
            center_x, center_y = w // 2, h // 2
        else:
            center_x, center_y = int(crop_center[0]), int(crop_center[1])

        if crop_size is None:
            crop_size = min(h, w) // 2
        crop_size = int(crop_size)

        half = crop_size // 2
        x_start = max(0, center_x - half)
        y_start = max(0, center_y - half)
        x_end = min(w, x_start + crop_size)
        y_end = min(h, y_start + crop_size)
        x_start = max(0, x_end - crop_size)
        y_start = max(0, y_end - crop_size)
        return marray[y_start:y_end, x_start:x_end]

    @APIExport(runOnUIThread=True)
    def setZStepLimit(self, limit_nm: float):
        self._focus_params.z_step_limit_nm = float(limit_nm)
        self.updateZStepLimits()
        return self._focus_params.z_step_limit_nm

    @APIExport(runOnUIThread=True)
    def getZStepLimit(self) -> float:
        return self._focus_params.z_step_limit_nm

    @APIExport(runOnUIThread=True)
    def returnLastCroppedImage(self) -> Response:
        if self._state.is_measuring: # or self.locked or self.aboutToLock:
            pass
        try:
            arr = self.cropped_im
            im = Image.fromarray(arr.astype(np.uint8))
            with io.BytesIO() as buf:
                im = im.convert("L")
                im.save(buf, format="PNG")
                im_bytes = buf.getvalue()
            headers = {"Content-Disposition": 'inline; filename="crop.png"'}
            return Response(im_bytes, headers=headers, media_type="image/png")
        except Exception as e:
            raise RuntimeError("No cropped image available. Please run update() first.") from e

    @APIExport(runOnUIThread=True)
    def returnLastImage(self) -> Response:
        try:
            if not self._master.detectorsManager[self.camera]._running:
                self._master.detectorsManager[self.camera].startAcquisition()
        except Exception as e:
            self._logger.error(f"Failed to start acquisition on camera '{self.camera}': {e}")

        lastFrame = self._master.detectorsManager[self.camera].getLatestFrame()
        if lastFrame is None:
            self._logger.error("No image available from camera.")
            return Response(status_code=404)
        lastFrame = lastFrame/np.max(lastFrame)*512.0
        lastFrame = lastFrame[::self.reduceImageScaleFactor, ::self.reduceImageScaleFactor]
        if lastFrame is None:
            raise RuntimeError("No image available. Please run update() first.")
        try:
            im = Image.fromarray(lastFrame.astype(np.uint8))
            with io.BytesIO() as buf:
                im.save(buf, format="PNG")
                im_bytes = buf.getvalue()
            headers = {"Content-Disposition": 'inline; filename="last_image.png"'}
            return Response(im_bytes, headers=headers, media_type="image/png")
        except Exception as e:
            raise RuntimeError("Failed to convert last image to PNG.") from e

    @APIExport(runOnUIThread=True, requestType="POST")
    def setCropFrameParameters(self, crop_size: int, crop_center: List[int] = None, frameSize: List[int] = None):
        detectorSize = self._master.detectorsManager[self.camera].shape

        self._focus_params.crop_size = int(crop_size * self.reduceImageScaleFactor)
        if crop_center is None:
            _crop_center = [detectorSize[1] // 2, detectorSize[0] // 2]
        else:
            crop_center = [int(crop_center[1] * self.reduceImageScaleFactor), int(crop_center[0] * self.reduceImageScaleFactor)]
        if self._focus_params.crop_size < 100:
            self._focus_params.crop_size = 100
        detectorSize = self._master.detectorsManager[self.camera].shape
        if self._focus_params.crop_size > detectorSize[0] or self._focus_params.crop_size > detectorSize[1]:
            raise ValueError(f"Crop size {self._focus_params.crop_size} exceeds detector size {detectorSize}.")
        if crop_center is None:
            crop_center = [self._focus_params.crop_size // 2, self._focus_params.crop_size // 2]
        self._focus_params.crop_center = crop_center
        self._logger.info(f"Set crop parameters: size={self._focus_params.crop_size}, center={self._focus_params.crop_center}")

        # Save the crop parameters to config file
        self.saveCropParameters()

    def saveCropParameters(self):
        """Save the current crop parameters to the config file."""
        try:
            # Save crop size and center to setup info
            if hasattr(self, '_setupInfo') and hasattr(self._setupInfo, 'focusLock'):
                # Set the crop parameters in the setup info
                self._setupInfo.focusLock.crop_size = self._focus_params.crop_size
                self._setupInfo.focusLock.crop_center = self._focus_params.crop_center

                # Save the updated setup info to config file
                from imswitch.imcontrol.model import configfiletools
                configfiletools.saveSetupInfo(configfiletools.loadOptions()[0], self._setupInfo)

                self._logger.info(f"Saved crop parameters to config: size={self._focus_params.crop_size}, center={self._focus_params.crop_center}")
        except Exception as e:
            self._logger.error(f"Could not save crop parameters: {e}")
            return


    @APIExport(runOnUIThread=True)
    def getCurrentFocusValue(self) -> Dict[str, Any]:
        """Get the current focus value and metadata.

        Returns:
            Dict containing:
            - focus_value: float - current focus metric value
            - timestamp: float - measurement timestamp
            - is_measuring: bool - whether continuous measurement is active
            - is_locked: bool - whether focus lock is engaged
        """
        return {
            "focus_value": float(self.current_focus_value),
            "timestamp": time.time(),
            "is_measuring": self._state.is_measuring,
            "is_locked": self.locked,
        }

    @APIExport(runOnUIThread=True)
    def performOneStepAutofocus(self, target_focus_setpoint: Optional[float] = None,
                                move_to_focus: bool = True,
                                max_attempts: int = 3,
                                threshold_um: float = 0.5) -> Dict[str, Any]:
        """Perform one-shot hardware-based autofocus using calibration data.

        This is designed for wellplate scanning where we need fast, accurate focusing
        at each XY position without a full Z-sweep. Instead of doing a full Z-sweep,
        we:
        1. Measure the current focus value (e.g., laser spot position)
        2. Calculate the offset from target setpoint (from previous focused position)
        3. Convert the offset to Z movement using calibration scale factor
        4. Move to correct Z position

        Workflow:
        1. Capture single frame from focus camera
        2. Calculate focus metric (e.g., laser peak position in pixels)
        3. Calculate offset from target setpoint: delta_focus = current - target
        4. Convert to Z offset: delta_z = delta_focus * scale_factor (µm/unit)
        5. Move Z stage by delta_z (if move_to_focus=True)
        6. Optionally iterate if offset too large

        Args:
            target_focus_setpoint: Target focus value to reach (e.g., from previous
                                  focused position). If None, uses self._pi_params.set_point
            move_to_focus: If True, move Z stage to calculated focus position
            max_attempts: Maximum number of correction iterations (default 3)
            threshold_um: Success threshold in µm (default 0.5µm)

        Returns:
            Dict containing:
            - success: bool - whether autofocus succeeded
            - current_focus_value: float - measured focus metric
            - target_focus_setpoint: float - target focus value used
            - focus_offset: float - offset from setpoint (focus units)
            - z_offset: float - calculated Z offset (µm)
            - moved: bool - whether stage was moved
            - num_attempts: int - number of iterations performed
            - final_error_um: float - final positioning error
        """
        if not self._current_calibration:
            self._logger.error("One-step autofocus requires calibration. Run calibration first.")
            return {
                "success": False,
                "error": "No calibration data available",
                "current_focus_value": 0.0,
                "target_focus_setpoint": None,
                "focus_offset": None,
                "z_offset": None,
                "moved": False,
                "num_attempts": 0,
                "final_error_um": None
            }

        initial_z_position = self.currentZPosition
        # Use provided setpoint or fall back to stored setpoint
        if target_focus_setpoint is None:
            target_focus_setpoint = self._pi_params.set_point
            if target_focus_setpoint == 0.0:
                self._logger.warning(
                    "No target setpoint provided and no stored setpoint. "
                    "Please provide target_focus_setpoint or run focus lock first."
                )

        # Ensure laser is on if configured
        laserName = getattr(self._setupInfo.focusLock, "laserName", None)
        laserValue = getattr(self._setupInfo.focusLock, "laserValue", None)
        if laserName and laserValue is not None and False : # TODO: We assume the laser is on always for now
            try:
                self._master.lasersManager[laserName].setValue(laserValue)
                self._master.lasersManager[laserName].setEnabled(True)
            except Exception as e:
                self._logger.warning(f"Could not enable focus laser: {e}")

        success = False
        num_attempts = 0
        final_error_um = None
        current_focus_value = 0.0
        focus_offset = 0.0
        z_offset = 0.0

        for attempt in range(max_attempts):
            num_attempts += 1

            # Use single source of truth for frame capture and focus computation
            capture_result = self._captureAndComputeFocus(num_frames=1, is_debug=self.is_debug)
            
            if not capture_result["valid"]:
                self._logger.error("Failed to capture and compute focus")
                # move back to original position if we moved
                self.stage
                return {
                    "success": False,
                    "error": "Frame capture or focus computation failed",
                    "current_focus_value": 0.0,
                    "target_focus_setpoint": float(target_focus_setpoint),
                    "focus_offset": None,
                    "z_offset": None,
                    "moved": False,
                    "num_attempts": num_attempts,
                    "final_error_um": None
                }

            current_focus_value = capture_result["focus_value"]

            # Calculate focus offset from target setpoint
            focus_offset =  target_focus_setpoint - current_focus_value

            # Convert focus offset to Z offset using calibration scale factor
            # Scale factor is in µm per focus unit (e.g., µm per pixel)
            scale_factor = self._getCalibrationBasedScale()
            z_offset = focus_offset * scale_factor
            final_error_um = abs(z_offset)

            self._logger.debug(
                f"One-step AF attempt {attempt + 1}/{max_attempts}: "
                    f"current_focus={current_focus_value:.2f}, target={target_focus_setpoint:.2f}, "
                f"focus_offset={focus_offset:.2f}, scale={scale_factor:.3f}µm/unit, "
                f"z_offset={z_offset:.2f}µm"
            )

            # Check if we're within threshold
            if final_error_um < threshold_um:
                success = True
                self._logger.info(
                    f"One-step autofocus successful after {num_attempts} attempts. "
                    f"Error: {final_error_um:.3f}µm"
                )
                break

            # Move to target position (if requested)
            if move_to_focus:
                try:
                    # Safety check: limit single move based on calibration scan range
                    max_single_move_um = self._calib_params.scan_range_um * 10  # 10x scan range as safety
                    if abs(z_offset) > max_single_move_um:
                        self._logger.warning(
                            f"Calculated Z offset ({z_offset:.2f}µm) exceeds safety limit "
                            f"({max_single_move_um}µm). Clamping move."
                        )
                        z_offset = np.sign(z_offset) * max_single_move_um

                    # Move stage (relative movement)
                    self.stage.move(value=z_offset, axis="Z", is_absolute=False, speed=10000, is_blocking=True)
                    time.sleep(0.3)  # Small settle time

                except Exception as e:
                    self._logger.error(f"Failed to move Z stage: {e}")
                    return {
                        "success": False,
                        "error": f"Stage movement failed: {e}",
                        "current_focus_value": float(current_focus_value),
                        "target_focus_setpoint": float(target_focus_setpoint),
                        "focus_offset": float(focus_offset),
                        "z_offset": float(z_offset),
                        "moved": False,
                        "num_attempts": num_attempts,
                        "final_error_um": float(final_error_um)
                    }
            else:
                # Not moving, so we're done after first measurement
                break

        return {
            "success": success,
            "current_focus_value": float(current_focus_value),
            "target_focus_setpoint": float(target_focus_setpoint),
            "focus_offset": float(focus_offset),
            "z_offset": float(z_offset),
            "moved": move_to_focus,
            "num_attempts": num_attempts,
            "final_error_um": float(final_error_um)
        }



    @APIExport(runOnUIThread=True, requestType="POST")
    def setCalibrationParams(self, **kwargs) -> Dict[str, Any]:
        """Update calibration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._calib_params, key):
                setattr(self._calib_params, key, value)
        return self._calib_params.to_dict()

    @APIExport(runOnUIThread=True)
    def getCalibrationParams(self) -> Dict[str, Any]:
        """Get current calibration parameters."""
        return self._calib_params.to_dict()

    @APIExport(runOnUIThread=True)
    def getCalibrationStatus(self) -> Dict[str, Any]:
        """Get calibration status and data."""
        if self._current_calibration:
            return {
                "calibrated": True,
                "calibration_active": self._focusCalibThread.isRunning(),
                "sensitivity_nm_per_unit": self._current_calibration.sensitivity_nm_per_unit,
                "r_squared": self._current_calibration.r_squared,
                "timestamp": self._current_calibration.timestamp,
                "pid_integration": True,
            }
        else:
            return {
                "calibrated": False,
                "calibration_active": self._focusCalibThread.isRunning(),
                "sensitivity_nm_per_unit": 0.0,
                "r_squared": 0.0,
                "timestamp": 0.0,
                "pid_integration": False,
            }

    @APIExport(runOnUIThread=True, requestType="POST")
    def runFocusCalibrationDynamic(self, scan_range_um: float = 2.0, num_steps: int = 20, settle_time: float = 0.5, initial_z_position: float = None) -> Dict[str, Any]:
        """Run focus calibration with dynamic range around current position."""
        # Update calibration parameters for dynamic range
        self._calib_params.scan_range_um = scan_range_um
        self._calib_params.num_steps = num_steps
        self._calib_params.settle_time = settle_time

        # Start calibration (uses dynamic range automatically now)
        if hasattr(self, '_focusCalibThread') and self._focusCalibThread.isRunning():
            return {"error": "Calibration already running"}

        self._focusCalibThread = FocusCalibThread(self, initial_z_position=initial_z_position)  # Pass initial_z_position to thread
        self._focusCalibThread.runThread()

        return {
            "message": "Dynamic calibration started",
            "scan_range_um": scan_range_um,
            "num_steps": num_steps,
            "settle_time": settle_time,
        }


# =========================
# Calibration thread
# =========================
class FocusCalibThread(object):
    """Thread for running focus calibration scans.
    
    Uses the controller's _captureAndComputeFocus method as the single source
    of truth for focus value computation.
    """
    
    def __init__(self, controller, initial_z_position=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller = controller
        self.signalData: List[float] = []
        self.positionData: List[float] = []
        self.poly = None
        self.calibrationResult = None
        self._isRunning = False
        self.initial_z_position = initial_z_position

    def isRunning(self) -> bool:
        return self._isRunning

    def stopThread(self):
        self._isRunning = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def runThread(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        self._thread = thread

    def run(self):
        try:
            self.signalData = []
            self.positionData = []
            self._isRunning = True

            if self.initial_z_position is not None:
                try:
                    self._controller._master.positionersManager[self._controller.positioner].move(
                        value=self.initial_z_position, axis="Z", speed=MAX_SPEED, 
                        is_blocking=True, is_absolute=True
                    )
                    time.sleep(1.0)  # Wait for move to complete and settle
                    self._controller._logger.info(f"Moved to initial Z position: {self.initial_z_position}µm")
                except Exception as e:
                    self._controller._logger.error(f"Failed to move to initial Z position {self.initial_z_position}µm: {e}")

            calib_params = self._controller._calib_params

            # Scan around the current position
            initialZPosition = self._controller.currentZPosition
            try:
                # Use scan_range_um parameter to define range around current position
                half_range = calib_params.scan_range_um / 2.0
                from_val = initialZPosition - half_range
                to_val = initialZPosition + half_range
                self._controller._logger.info(
                    f"Dynamic calibration: scanning {calib_params.scan_range_um}µm "
                    f"around current Z={initialZPosition:.3f}µm"
                )
            except Exception as e:
                # Fall back to fixed positions if current position can't be determined
                self._controller._logger.warning(f"Could not get current Z position, using fixed range: {e}")
                from_val = calib_params.from_position
                to_val = calib_params.to_position

            scan_list = np.round(np.linspace(from_val, to_val, calib_params.num_steps), 2)

            self._controller.sigCalibrationProgress.emit({
                "event": "calibration_started",
                "total_steps": len(scan_list),
                "from_position": from_val,
                "to_position": to_val,
                "scan_range_um": calib_params.scan_range_um,
            })

            for i, zpos in enumerate(scan_list):
                # Stop thread if requested
                if not self._isRunning:
                    self._controller._logger.info("Calibration thread stopped.")
                    break

                # Move to position
                self._controller._master.positionersManager[self._controller.positioner].move(
                    value=zpos, axis="Z", speed=MAX_SPEED, is_blocking=True, is_absolute=True
                )
                time.sleep(calib_params.settle_time)
                
                # Use single source of truth for focus computation
                capture_result = self._controller._captureAndComputeFocus(num_frames=2, is_debug=self._controller.is_debug)
                
                if capture_result["valid"]:
                    focus_signal = capture_result["focus_value"]
                else:
                    # Fall back to cached value if capture failed
                    focus_signal = float(self._controller.current_focus_value)
                    self._controller._logger.warning(
                        f"Frame capture failed at Z={zpos:.3f}, using cached signal"
                    )
                
                self.signalData.append(focus_signal)
                self.positionData.append(zpos)

                self._controller.sigCalibrationProgress.emit({
                    "event": "calibration_progress",
                    "step": i + 1,
                    "total_steps": len(scan_list),
                    "position": zpos,
                    "focus_value": focus_signal,
                    "progress_percent": ((i + 1) / len(scan_list)) * 100,
                })
                self._controller._logger.info(
                    f"Calibration step {i+1}/{len(scan_list)}: Z={zpos:.3f}, Focus value={focus_signal:.4f}"
                )
            
            # Enhanced calibration data structure and PID integration
            self.poly = np.polyfit(self.positionData, self.signalData, 1)
            self.calibrationResult = np.around(self.poly, 4)

            # Calculate enhanced calibration metrics
            r_squared = self._calculate_r_squared()
            sensitivity_nm_per_unit = self._get_sensitivity_nm_per_px()

            # Determine linear range (where polynomial fits well)
            focus_min, focus_max = min(self.signalData), max(self.signalData)
            linear_range = (focus_min, focus_max)

            # Create lookup table for focus value -> z position conversion
            lookup_table = {}
            if len(self.positionData) > 1:
                for focus_val, z_pos in zip(self.signalData, self.positionData):
                    lookup_table[float(focus_val)] = float(z_pos)

            # Create and store calibration data for PID controller integration
            calibration_data = CalibrationData(
                position_data=list(self.positionData),
                focus_data=list(self.signalData),
                polynomial_coeffs=self.poly.tolist() if self.poly is not None else None,
                sensitivity_nm_per_unit=sensitivity_nm_per_unit,
                r_squared=r_squared,
                linear_range=linear_range,
                timestamp=time.time(),
                lookup_table=lookup_table,
            )

            # Store calibration data in controller for PID integration
            self._controller._current_calibration = calibration_data
            self._controller._logger.info(
                f"Calibration completed: sensitivity={sensitivity_nm_per_unit:.1f} nm/unit, R²={r_squared:.4f}"
            )

            # Save calibration plot if matplotlib available
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                y_fit = np.polyval(self.poly, self.positionData)
                plt.figure()
                plt.plot(self.positionData, self.signalData, "o", label="Data")
                plt.plot(self.positionData, y_fit, "-", label="Fit")
                plt.xlabel("Z Position (µm)")
                plt.ylabel("Focus Value")
                plt.title("Focus Calibration with factor: {:.1f} nm/unit".format(sensitivity_nm_per_unit))
                plt.legend()
                plt.savefig("calibration_plot.png")
                plt.close()
            except Exception:
                pass

            self._controller.sigCalibrationProgress.emit({
                "event": "calibration_completed",
                "coefficients": self.poly.tolist(),
                "r_squared": r_squared,
                "sensitivity_nm_per_px": sensitivity_nm_per_unit,
                "calibration_data": calibration_data.to_dict(),
            })

            self.show()
            # Move back to initial position
            self._controller._master.positionersManager[self._controller.positioner].move(
                value=initialZPosition, axis="Z", speed=MAX_SPEED, is_blocking=True, is_absolute=True
            )
            self._isRunning = False
        except Exception as e:
            self._controller._logger.error(f"Error in calibration thread: {e}")
            self._isRunning = False

    def _calculate_r_squared(self) -> float:
        if self.poly is None or len(self.signalData) == 0:
            return 0.0
        y_pred = np.polyval(self.poly, self.positionData)
        ss_res = np.sum((self.signalData - y_pred) ** 2)
        ss_tot = np.sum((self.signalData - np.mean(self.signalData)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def _get_sensitivity_nm_per_px(self) -> float:
        if self.poly is None or self.poly[0] == 0:
            return 0.0
        return float(1000 / self.poly[0])

    def show(self):
        """Display calibration results (GUI or headless mode)."""
        if IS_HEADLESS or not hasattr(self._controller, '_widget'):
            # Enhanced headless mode signaling
            if self._controller._current_calibration:
                # Send comprehensive calibration signal for headless mode
                headless_signal_data = {
                    "event": "calibration_display_update",
                    "calibration_text": f"1 unit → {self._controller._current_calibration.sensitivity_nm_per_unit:.1f} nm",
                    "calibration_data": self._controller._current_calibration.to_dict(),
                    "pid_integration_active": True,
                    "timestamp": time.time(),
                }
                self._controller.sigCalibrationProgress.emit(headless_signal_data)
                self._controller._logger.info(f"Headless calibration display: {headless_signal_data['calibration_text']}")
            else:
                # Send invalid calibration signal
                headless_signal_data = {
                    "event": "calibration_display_update",
                    "calibration_text": "Calibration invalid",
                    "calibration_data": None,
                    "pid_integration_active": False,
                    "timestamp": time.time(),
                }
                self._controller.sigCalibrationProgress.emit(headless_signal_data)
            return

        # GUI mode - update widget display
        if self.poly is None or self.poly[0] == 0:
            cal_text = "Calibration invalid"
        else:
            cal_nm = self._get_sensitivity_nm_per_px()
            cal_text = f"1 px --> {cal_nm:.1f} nm"
        try:
            self._controller._widget.calibrationDisplay.setText(cal_text)
        except AttributeError:
            pass

    def getData(self) -> Dict[str, Any]:
        """Return enhanced calibration data - now integrated with PID controller."""
        enhanced_data = {
            "signalData": self.signalData,
            "positionData": self.positionData,
            "poly": self.poly.tolist() if self.poly is not None else None,
            "calibrationResult": self.calibrationResult.tolist() if self.calibrationResult is not None else None,
            "r_squared": self._calculate_r_squared(),
            "sensitivity_nm_per_px": self._get_sensitivity_nm_per_px(),
        }

        # Add enhanced calibration data if available
        if hasattr(self._controller, '_current_calibration') and self._controller._current_calibration:
            enhanced_data["calibration_data"] = self._controller._current_calibration.to_dict()
            enhanced_data["pid_integration_active"] = True
        else:
            enhanced_data["pid_integration_active"] = False

        return enhanced_data
