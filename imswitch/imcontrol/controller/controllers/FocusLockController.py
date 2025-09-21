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
    focus_metric: str = "astigmatism" # "astigmatism", "gaussian", "gradient"
    crop_center: Optional[List[int]] = None
    crop_size: Optional[int] = None
    gaussian_sigma: float = 11.0
    background_threshold: float = 40.0
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
            focus_metric=getattr(self._setupInfo.focusLock, "focusLockMetric", "astigmatism"),
            crop_center=getattr(self._setupInfo.focusLock, "cropCenter", None),
            crop_size=getattr(self._setupInfo.focusLock, "cropSize", None),
            update_freq=self._setupInfo.focusLock.updateFreq or 10,
        )
        # TODO: if there are no crop settings, we should find them automatically by detecing the maximum intensity spot in the image and crop 300 x/y around it, but only if the laser is on
        # Initialize focus metric computer using extracted module
        focus_config = FocusConfig(
            gaussian_sigma=self._focus_params.gaussian_sigma,
            background_threshold=self._focus_params.background_threshold,
            crop_radius=self._focus_params.crop_size or 300,
            enable_gaussian_blur=True,
        )
        self._focus_metric = FocusMetricFactory.create(self._focus_params.focus_metric, focus_config)

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
        safetyMoveLimit = getattr(self._setupInfo.focusLock, "safetyMoveLimit", 20.0)
        minStepThreshold = getattr(self._setupInfo.focusLock, "minStepThreshold", 0.002)
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
        
        # Lock state variables (cleaned up)
        self.locked = False
        self.aboutToLock = False
        
        # Legacy compatibility variables (TODO: remove these gradually)
        self.setPointSignal = 0.0  # Mirrors current_focus_value for compatibility
        self.twoFociVar = self._focus_params.two_foci_enabled
        self.zStackVar = self._focus_params.z_stack_enabled
        
        # Thread control
        self.__isPollingFramesActive = True
        self.pollingFrameUpdatePeriode = 1.0 / self._focus_params.update_freq
        
        # About-to-lock logic
        self.aboutToLockDiffMax = 0.4
        
        # Data buffers for plotting
        self.buffer = 40
        self.currPoint = 0
        self.setPointData = np.zeros(self.buffer, dtype=float)
        self.timeData = np.zeros(self.buffer, dtype=float)
        self.reduceImageScaleFactor = 1

        # Travel budget tracking
        self._travel_used_um = 0.0

        # Measurement smoothing
        self._meas_filt = None

        # Camera acquisition
        try:
            self._master.detectorsManager[self.camera].startAcquisition()
        except Exception as e:
            self._logger.error(f"Failed to start acquisition on camera '{self.camera}': {e}")

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

        # Start polling
        self.updateThread() # TODO: Shall we do that from the beginning? 

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
        self._pollFramesThread = threading.Thread(target=self._pollFrames, name="FocusLockPollFramesThread")
        self._pollFramesThread.daemon = True
        self._pollFramesThread.start()

    def _onMotorPositionUpdate(self, pos: Dict[str, float]):
        if type(pos) != list and "Z" in pos[self.positioner]: # TODO: Seems like the signal is not unified
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
                elif key == "two_foci_enabled":
                    self.twoFociVar = value
                elif key == "z_stack_enabled":
                    self.zStackVar = value
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
        try:
            if not self._state.is_measuring:
                self._state.is_measuring = True
                self._emitStateChangedSignal() # TODO: What is this good for? Actually needed? 
                self._logger.info("Focus measurement started")
                return True
            return False
        except Exception as e:
            self._logger.error(f"Failed to start focus measurement: {e}")
            return False

    @APIExport(runOnUIThread=True)
    def stopFocusMeasurement(self) -> bool:
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

    def _emitStateChangedSignal(self): # TODO: What is this good for? Actually needed?
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
            if not IS_HEADLESS:
                self._widget.lockButton.setChecked(False)
                try:
                    self._widget.focusPlot.removeItem(self._widget.focusLockGraph.lineLock)
                except Exception:
                    pass

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
        # TODO: This interface has changed ? 
        self._focusCalibThread.getData()

    @APIExport(runOnUIThread=True)
    def isCalibrationRunning(self) -> bool:
        # TODO: This interface has changed ?
        self._focusCalibThread.isRunning()

    def showCalibrationCurve(self):
        # TODO: This interface has changed ?
        self._focusCalibThread.showCalibrationCurve()

    def twoFociVarChange(self):
        self.twoFociVar = not self.twoFociVar
        self._focus_params.two_foci_enabled = self.twoFociVar

    def zStackVarChange(self):
        self.zStackVar = not self.zStackVar
        self._focus_params.z_stack_enabled = self.zStackVar

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

    def _pollFrames(self):
        tLast = 0
        # store a history of the last 5 values and filter out outliers
        last_values = []
        nFreeBufferFrames = 3
        while self.__isPollingFramesActive:
            
            if (time.time() - tLast) < self.pollingFrameUpdatePeriode:
                time.sleep(0.001)
                continue
            self._logger.debug("Current frame polling frequency: %.2f Hz", 1.0 / (time.time() - tLast))
            tLast = time.time()
            if not self._state.is_measuring and not self.locked and not self.aboutToLock:
                continue
            
            for i in range(nFreeBufferFrames): # Kinda clear buffer and wait a bit? 
                im = self._master.detectorsManager[self.camera].getLatestFrame()

            # Crop (prefer NiP if present)
            try:
                import NanoImagingPack as nip
                self.cropped_im = nip.extract(
                    img=im,
                    ROIsize=(self._focus_params.crop_size, self._focus_params.crop_size),
                    centerpos=self._focus_params.crop_center,
                    PadValue=0.0,
                    checkComplex=True,
                )
            except Exception:
                self.cropped_im = self.extract(im, crop_size=self._focus_params.crop_size,
                                               crop_center=self._focus_params.crop_center)


            # Compute focus value using extracted focus metrics module
            focus_result = self._focus_metric.compute(self.cropped_im)
            self.current_focus_value = focus_result.get("focus", 0.0)
            
            # TODO: Remove outliers in PID loop
            '''
            if len(last_values) >= 5:
                last_values.pop(0)
            last_values.append(self.current_focus_value)
            if len(last_values) == 5:
                median = np.median(last_values)
                diffs = [abs(v - median) for v in last_values]
                max_diff = max(diffs)
                if max_diff > self.aboutToLockDiffMax:
                    last_values.pop(diffs.index(max_diff))
                    self.current_focus_value = np.mean(last_values)
            '''
            # Legacy compatibility # TODO: remove all legacy vars
            self.setPointSignal = self._pi_params.set_point if self.pi else 0 # TODO: This should be the user-set set point, not current focus value 

            # Get current timestamp for logging
            current_timestamp = focus_result.get("t", time.time())

            # Emit enhanced focus value signal
            focus_data = {
                "focus_value": self.current_focus_value,
                "timestamp": current_timestamp,
                "is_locked": self.locked,
                "current_position": self.currentZPosition,
                "focus_metric": self._focus_params.focus_metric,
                "focus_result": focus_result,  # Include full result for debugging
                "focus_setpoint": self._pi_params.set_point if self.pi else 0,
            }
            self.sigFocusValueUpdate.emit(focus_data)

            # Initialize variables for CSV logging
            pi_output = None


            # === Control action (relative moves only) ===
            if self.locked and self.pi is not None:
                meas = float(self.current_focus_value)
                if self._pi_params.meas_lowpass_alpha > 0.0:
                    a = self._pi_params.meas_lowpass_alpha
                    self._meas_filt = meas if self._meas_filt is None else a * self._meas_filt + (1 - a) * meas
                    meas_for_pid = self._meas_filt
                else:
                    meas_for_pid = meas

                u = self.pi.update(meas_for_pid)                       # controller units
                pi_output = u  # Store for logging
                # Use calibration-based scale factor if available, otherwise fall back to configured scale
                scale_factor = self._getCalibrationBasedScale()
                step_um = u * scale_factor        # convert to µm

                # clamping if below threshold
                # deadband
                if abs(step_um) < self._pi_params.min_step_threshold:
                    step_um = 0.0

                # per-update clamp & optional safety gating
                limit = abs(self._pi_params.safety_move_limit) if self._pi_params.safety_motion_active else abs(self._pi_params.safety_move_limit)
                step_um = max(min(step_um, limit), -limit)

                if step_um != 0.0:
                    # Use absolute movement instead of relative
                    new_z_position = self.currentZPosition + step_um
                    self.stage.move(value=new_z_position, axis="Z", speed=MAX_SPEED, is_blocking=True, is_absolute=True)
                    self._travel_used_um += abs(step_um) # TODO: Still not sure if we need to use this! 
                    # travel budget acts like safety_distance_limit
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
                        step_size_um=step_um if self.locked and 'step_um' in locals() else None,
                        travel_used_um=self._travel_used_um,
                    )
                except Exception as e:
                    self._logger.error(f"Failed to log focus measurement: {e}")

            # Update plotting buffers
            self.updateSetPointData()

    @APIExport(runOnUIThread=True)
    def setParamsAstigmatism(self, gaussianSigma: float, backgroundThreshold: float,
                        cropSize: int, cropCenter: Optional[List[int]] = None):
        self._focus_params.gaussian_sigma = float(gaussianSigma)
        self._focus_params.background_threshold = float(backgroundThreshold)
        self._focus_params.crop_size = int(cropSize)
        if cropCenter is None:
            cropCenter = [cropSize // 2, cropSize // 2]
        self._focus_params.crop_center = cropCenter

        self.gaussianSigma = float(gaussianSigma)
        self.backgroundThreshold = float(backgroundThreshold)
        self.cropSize = int(cropSize)
        if cropCenter is None:
            cropCenter = [self.cropSize // 2, self.cropSize // 2]
        self.cropCenter = np.asarray(cropCenter, dtype=int)
        # TODO: self.cropCenter / self.cropSize are not used 

    @APIExport(runOnUIThread=True)
    def getParamsAstigmatism(self):
        return {
            "gaussianSigma": self._focus_params.gaussian_sigma,
            "backgroundThreshold": self._focus_params.background_threshold,
            "cropSize": self._focus_params.crop_size,
            "cropCenter": self._focus_params.crop_center,
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
        """Get the scale factor from calibration data if available, otherwise use default."""
        if self._current_calibration: # and self._current_calibration.sensitivity_nm_per_unit > 0: # TODO: I think it's fine if the factor is negative!
            # Convert nm per unit to µm per unit for consistency with existing code
            return self._current_calibration.sensitivity_nm_per_unit / 1000.0
        else:
            # Fall back to configured scale factor if no calibration available
            return self._pi_params.scale_um_per_unit

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
        lastFrame = self._master.detectorsManager[self.camera].getLatestFrame()
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
    def setCropFrameParameters(self, cropSize: int, cropCenter: List[int] = None, frameSize: List[int] = None):
        detectorSize = self._master.detectorsManager[self.camera].shape
        if frameSize is None:
            mRatio = 1 / self.reduceImageScaleFactor
        else:
            mRatio = detectorSize[0] / frameSize[0]
        self._focus_params.crop_size = int(cropSize * mRatio)
        if cropCenter is None:
            cropCenter = [detectorSize[1] // 2, detectorSize[0] // 2]
        else:
            cropCenter = [int(cropCenter[1] * mRatio), int(cropCenter[0] * mRatio)]
        if cropSize < 100:
            cropSize = 100
        detectorSize = self._master.detectorsManager[self.camera].shape
        if cropSize > detectorSize[0] or cropSize > detectorSize[1]:
            raise ValueError(f"Crop size {cropSize} exceeds detector size {detectorSize}.")
        if cropCenter is None:
            cropCenter = [cropSize // 2, cropSize // 2]
        self._focus_params.crop_center = cropCenter
        self._logger.info(f"Set crop parameters: size={self._focus_params.crop_size}, center={self._focus_params.crop_center}")
        
        # Save the crop parameters to config file
        self.saveCropParameters()

    def saveCropParameters(self):
        """Save the current crop parameters to the config file."""
        try:
            # Save crop size and center to setup info
            if hasattr(self, '_setupInfo') and hasattr(self._setupInfo, 'focusLock'):
                # Set the crop parameters in the setup info
                self._setupInfo.focusLock.cropSize = self._focus_params.crop_size
                self._setupInfo.focusLock.cropCenter = self._focus_params.crop_center
                
                # Save the updated setup info to config file
                from imswitch.imcontrol.model import configfiletools
                configfiletools.saveSetupInfo(configfiletools.loadOptions()[0], self._setupInfo)
                
                self._logger.info(f"Saved crop parameters to config: size={self._focus_params.crop_size}, center={self._focus_params.crop_center}")
        except Exception as e:
            self._logger.error(f"Could not save crop parameters: {e}")
            return


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
    def __init__(self, controller, initial_z_position=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._focusValueComputeController = controller # TODO: I think this is not ideal - rather work with the signal that pushes the new focus value - we should also rename this as setPointSignal is misleading. rather focusValueSignal or similar
        # hence instead of suppliyng hte full controller, we should only supply the positioner function and the signal for the focus value (via. connect())
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
        self.signalData = []
        self.positionData = []
        self._isRunning = True

        if self.initial_z_position is not None:
            try:
                self._focusValueComputeController._master.positionersManager[self._focusValueComputeController.positioner].move(value=self.initial_z_position, axis="Z", speed=MAX_SPEED, is_blocking=True, is_absolute=True)
                time.sleep(1.0)  # wait for move to complete and settle
                self._focusValueComputeController._logger.info(f"Moved to initial Z position: {self.initial_z_position}µm")
            except Exception as e:
                self._focusValueComputeController._logger.error(f"Failed to move to initial Z position {self.initial_z_position}µm: {e}")
        
        calib_params = self._focusValueComputeController._calib_params

        # IMPROVEMENT #1: Scan around the current position instead of always from->to
        initialZPosition = self._focusValueComputeController.currentZPosition
        try:
            # Use scan_range_um parameter to define range around current position
            half_range = calib_params.scan_range_um / 2.0
            from_val = initialZPosition - half_range
            to_val = initialZPosition + half_range
            self._focusValueComputeController._logger.info(f"Dynamic calibration: scanning {calib_params.scan_range_um}µm around current Z={initialZPosition:.3f}µm")
        except Exception as e:
            # Fall back to fixed positions if current position can't be determined
            self._focusValueComputeController._logger.warning(f"Could not get current Z position, using fixed range: {e}")
            from_val = calib_params.from_position
            to_val = calib_params.to_position

        scan_list = np.round(np.linspace(from_val, to_val, calib_params.num_steps), 2)

        self._focusValueComputeController.sigCalibrationProgress.emit({
            "event": "calibration_started",
            "total_steps": len(scan_list),
            "from_position": from_val,
            "to_position": to_val,
            "scan_range_um": calib_params.scan_range_um,
        })


        for i, zpos in enumerate(scan_list):
            # stop thread if requested
            if not self._isRunning:
                self._focusValueComputeController._logger.info("Calibration thread stopped.")
                break
            
            # Move to position (fix bug: was using 'z' instead of 'zpos')
            self._focusValueComputeController._master.positionersManager[self._focusValueComputeController.positioner].move(value=zpos, axis="Z", speed=MAX_SPEED, is_blocking=True, is_absolute=True)
            time.sleep(calib_params.settle_time)
            # TODO: If the computing thread is not running, we have to start it 
            # TODO: We should implement a callback on the signal update instead of polling twice here - connect() to the focus value update signal # 
            focus_signal = float(self._focusValueComputeController.current_focus_value) # TODO: This interface seems odd, we have to improve this I guess since we cannot guarantee that setPointSignal is updated after we move the step - but when we explicitly grab a frame, we can guarantee that but compute twice - so rather wait for the next updated value or poll actively from the _controller ?  - we could have a counter in the focus computation thread that returns the id - we would need to wait for the next frame id (i.e. we can actually use the frame id as a return from the camera )
            self._focusValueComputeController._logger.warning(f"Frame acquisition failed at Z={zpos:.3f}, using cached signal")
            self.signalData.append(focus_signal)
            self.positionData.append(zpos)

            self._focusValueComputeController.sigCalibrationProgress.emit({
                "event": "calibration_progress",
                "step": i + 1,
                "total_steps": len(scan_list),
                "position": zpos,
                "focus_value": focus_signal,
                "progress_percent": ((i + 1) / len(scan_list)) * 100,
            })
            self._focusValueComputeController._logger.info(f"Calibration step {i+1}/{len(scan_list)}: Z={zpos:.3f}, Focus value={focus_signal:.4f}")
        # IMPROVEMENT #3: Enhanced calibration data structure and PID integration
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
        
        # CRITICAL: Create and store calibration data for PID controller integration
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
        self._focusValueComputeController._current_calibration = calibration_data
        self._focusValueComputeController._logger.info(f"Calibration completed: sensitivity={sensitivity_nm_per_unit:.1f} nm/unit, R²={r_squared:.4f}")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.plot(self.positionData, self.signalData,"o")
            plt.plot(self.positionData, self.signalFit,"x")
            plt.savefig("calibration_plot.png")
            plt.close()
        except Exception as e:
            pass
        
        self._focusValueComputeController.sigCalibrationProgress.emit({
            "event": "calibration_completed",
            "coefficients": self.poly.tolist(),
            "r_squared": r_squared,
            "sensitivity_nm_per_px": sensitivity_nm_per_unit,
            "calibration_data": calibration_data.to_dict(),
        })

        self.show()
        # move back to initial position 
        self._focusValueComputeController._master.positionersManager[self._focusValueComputeController.positioner].move(value=initialZPosition, axis="Z", speed=MAX_SPEED, is_blocking=True, is_absolute=True)
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
        if IS_HEADLESS or not hasattr(self._focusValueComputeController, '_widget'):
            # IMPROVEMENT #4: Enhanced headless mode signaling
            if self._focusValueComputeController._current_calibration:
                # Send comprehensive calibration signal for headless mode
                headless_signal_data = {
                    "event": "calibration_display_update",
                    "calibration_text": f"1 unit → {self._focusValueComputeController._current_calibration.sensitivity_nm_per_unit:.1f} nm",
                    "calibration_data": self._focusValueComputeController._current_calibration.to_dict(),
                    "pid_integration_active": True,
                    "timestamp": time.time(),
                }
                self._focusValueComputeController.sigCalibrationProgress.emit(headless_signal_data)
                self._focusValueComputeController._logger.info(f"Headless calibration display: {headless_signal_data['calibration_text']}")
            else:
                # Send invalid calibration signal
                headless_signal_data = {
                    "event": "calibration_display_update", 
                    "calibration_text": "Calibration invalid",
                    "calibration_data": None,
                    "pid_integration_active": False,
                    "timestamp": time.time(),
                }
                self._focusValueComputeController.sigCalibrationProgress.emit(headless_signal_data)
            return
            
        # GUI mode - update widget display
        if self.poly is None or self.poly[0] == 0:
            cal_text = "Calibration invalid"
        else:
            cal_nm = self._get_sensitivity_nm_per_px()
            cal_text = f"1 px --> {cal_nm:.1f} nm"
        try:
            self._focusValueComputeController._widget.calibrationDisplay.setText(cal_text)
        except AttributeError:
            pass

    def getData(self) -> Dict[str, Any]:
        """Return enhanced calibration data - now integrated with PID controller."""
        # Return both legacy format and enhanced CalibrationData
        enhanced_data = {
            "signalData": self.signalData,
            "positionData": self.positionData,
            "poly": self.poly.tolist() if self.poly is not None else None,
            "calibrationResult": self.calibrationResult.tolist() if self.calibrationResult is not None else None,
            "r_squared": self._calculate_r_squared(),
            "sensitivity_nm_per_px": self._get_sensitivity_nm_per_px(),
        }
        
        # Add enhanced calibration data if available  
        if hasattr(self._focusValueComputeController, '_current_calibration') and self._focusValueComputeController._current_calibration:
            enhanced_data["calibration_data"] = self._focusValueComputeController._current_calibration.to_dict()
            enhanced_data["pid_integration_active"] = True
        else:
            enhanced_data["pid_integration_active"] = False
            
        return enhanced_data
