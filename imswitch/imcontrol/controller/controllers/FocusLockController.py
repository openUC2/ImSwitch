import io
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageFile
from fastapi import Response
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import threading
from imswitch.imcommon.framework import Thread, Signal
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from imswitch import IS_HEADLESS

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Dataclasses (API-stable)
# =========================
@dataclass
class FocusLockParams:
    focus_metric: str = "JPG" # "JPG", "astigmatism", "gaussian", "gradient"
    crop_center: Optional[List[int]] = None
    crop_size: Optional[int] = None
    gaussian_sigma: float = 11.0
    background_threshold: float = 40.0
    update_freq: float = 10.0
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
    safety_move_limit: float = 3.0         # per-update clamp (µm)
    min_step_threshold: float = 0.002      # deadband (µm)
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_position": self.from_position,
            "to_position": self.to_position,
            "num_steps": self.num_steps,
            "settle_time": self.settle_time,
        }


@dataclass
class FocusLockState:
    is_measuring: bool = False
    is_locked: bool = False
    about_to_lock: bool = False
    current_focus_value: float = 0.0
    lock_position: float = 0.0
    current_position: float = 0.0
    measurement_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_measuring": self.is_measuring,
            "is_locked": self.is_locked,
            "about_to_lock": self.about_to_lock,
            "current_focus_value": self.current_focus_value,
            "lock_position": self.lock_position,
            "current_position": self.current_position,
            "measurement_active": self.measurement_active,
        }


class _PID:
    """Discrete 2-DOF style (β on SP via caller if needed), derivative on measurement, anti-windup."""

    def __init__(self, set_point: float, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0,
                 sample_time: float = 0.1, integral_limit: float = 100.0, output_lowpass_alpha: float = 0.0):
        self.set_point = float(set_point)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.dt = max(float(sample_time), 1e-6)
        self.integral = 0.0
        self.integral_limit = float(integral_limit)
        self.last_meas = None
        self.out = 0.0
        self.alpha = float(output_lowpass_alpha)  # output EMA 0..1
        # derivative filter
        self.dpv_f = 0.0
        self.dpv_alpha = 0.85  # fixed light smoothing; can be exposed if needed

    def setParameters(self, kp: float, ki: float):
        self.kp, self.ki = float(kp), float(ki)

    def updateParameters(self, **kwargs):
        for k, v in kwargs.items():
            if k == "kp": self.kp = float(v)
            elif k == "ki": self.ki = float(v)
            elif k == "kd": self.kd = float(v)
            elif k == "sample_time": self.dt = max(float(v), 1e-6)
            elif k == "integral_limit": self.integral_limit = float(v)
            elif k == "output_lowpass_alpha": self.alpha = float(v)
            elif k == "set_point": self.set_point = float(v)

    def update(self, meas: float) -> float:
        e = self.set_point - float(meas)

        # integral w/ clamp (anti-windup)
        self.integral += e * self.dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

        # derivative on measurement w/ light low-pass
        if self.last_meas is None:
            d_meas = 0.0
        else:
            d_meas = (float(meas) - self.last_meas) / self.dt
        self.last_meas = float(meas)
        self.dpv_f = self.dpv_alpha * self.dpv_f + (1.0 - self.dpv_alpha) * d_meas

        u_raw = self.kp * e + self.ki * self.integral - self.kd * self.dpv_f
        self.out = self.alpha * self.out + (1.0 - self.alpha) * u_raw if self.alpha > 0.0 else u_raw
        return self.out

    def restart(self):
        self.integral = 0.0
        self.last_meas = None
        self.out = 0.0


# =========================
# Controller
# =========================
class FocusLockController(ImConWidgetController):
    """Linked to FocusLockWidget. Public API (APIExport) kept stable."""

    sigUpdateFocusValue = Signal(object)       # (focus_data_dict)
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

        # Params
        self._focus_params = FocusLockParams(
            focus_metric=getattr(self._setupInfo.focusLock, "focusLockMetric", "JPG"),
            crop_center=getattr(self._setupInfo.focusLock, "cropCenter", None),
            crop_size=getattr(self._setupInfo.focusLock, "cropSize", None),
            update_freq=self._setupInfo.focusLock.updateFreq or 10,
        )

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
        safetyMoveLimit = getattr(self._setupInfo.focusLock, "safetyMoveLimit", 3.0)
        minStepThreshold = getattr(self._setupInfo.focusLock, "minStepThreshold", 0.002)
        safety_motion_active = getattr(self._setupInfo.focusLock, "safetyMotionActive", False)
        scale_um_per_unit = getattr(self._setupInfo.focusLock, "scaleUmPerUnit", 1.0)
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

        # Legacy/GUI mirrors
        self.setPointSignal = 0.0
        self.locked = False
        self.aboutToLock = False
        self.zStackVar = self._focus_params.z_stack_enabled
        self.twoFociVar = self._focus_params.two_foci_enabled
        self.noStepVar = True
        self.__isPollingFramesActive = True
        self.pollingFrameUpdateRate = 1.0 / self._focus_params.update_freq
        self.zStepLimLo = 0.0
        self.aboutToLockDiffMax = 0.4
        self.lockPosition = 0.0
        self.currentPosition = 0.0  # kept for API/state, not used in loop
        self.lastPosition = 0.0
        self.buffer = 40
        self.currPoint = 0
        self.setPointData = np.zeros(self.buffer, dtype=float)
        self.timeData = np.zeros(self.buffer, dtype=float)
        self.reduceImageScaleFactor = 1

        self.gaussianSigma = self._focus_params.gaussian_sigma
        self.backgroundThreshold = self._focus_params.background_threshold
        self.cropCenter = self._focus_params.crop_center
        self.cropSize = self._focus_params.crop_size
        self.kp = self._pi_params.kp
        self.ki = self._pi_params.ki

        # Travel budget (use safety_distance_limit semantics)
        self._travel_used_um = 0.0

        # Measurement smoothing
        self._meas_filt = None

        # Camera acquisition
        try:
            self._master.detectorsManager[self.camera].startAcquisition()
        except Exception as e:
            self._logger.error(f"Failed to start acquisition on camera '{self.camera}': {e}")

        # Threads
        self.__processDataThread = ProcessDataThread(self)
        self.__focusCalibThread = FocusCalibThread(self)
        self.__processDataThread.setFocusLockMetric(self._focus_params.focus_metric)

        # PID instance (kept as self.pi for API stability)
        self.pi: Optional[_PID] = None

        # Start polling
        self.updateThread()

        # GUI bindings
        if not IS_HEADLESS:
            self._widget.setKp(self._pi_params.kp)
            self._widget.setKi(self._pi_params.ki)
            self._widget.kpEdit.textChanged.connect(self.unlockFocus)
            self._widget.kiEdit.textChanged.connect(self.unlockFocus)
            self._widget.lockButton.clicked.connect(self.toggleFocus)
            self._widget.camDialogButton.clicked.connect(self.cameraDialog)
            self._widget.focusCalibButton.clicked.connect(self.focusCalibrationStart)
            self._widget.calibCurveButton.clicked.connect(self.showCalibrationCurve)
            self._widget.zStackBox.stateChanged.connect(self.zStackVarChange)
            self._widget.twoFociBox.stateChanged.connect(self.twoFociVarChange)
            self._widget.sigSliderExpTValueChanged.connect(self.setExposureTime)
            self._widget.sigSliderGainValueChanged.connect(self.setGain)

    def __del__(self):
        try:
            self.__isPollingFramesActive = False
            self.__processDataThread.quit()
            self.__processDataThread.wait()
        except Exception:
            pass
        try:
            self.__focusCalibThread.quit()
            self.__focusCalibThread.wait()
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
                    self.__processDataThread.setFocusLockMetric(value)
                elif key == "two_foci_enabled":
                    self.twoFociVar = value
                elif key == "z_stack_enabled":
                    self.zStackVar = value
                elif key == "update_freq":
                    self.pollingFrameUpdateRate = 1.0 / max(1e-3, float(value))
                    # keep PID dt in sync
                    self._pi_params.sample_time = self.pollingFrameUpdateRate
                    if self.pi:
                        self.pi.updateParameters(sample_time=self._pi_params.sample_time)
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
            self.pi.setParameters(self._pi_params.kp, self._pi_params.ki)
            self.pi.updateParameters(
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
        self._state.current_focus_value = self.setPointSignal
        self._state.lock_position = self.lockPosition
        self._state.current_position = self.currentPosition
        self._state.measurement_active = hasattr(self, '__processDataThread') and self.__processDataThread.isRunning()
        return self._state.to_dict()

    # =========================
    # API: Measurement control
    # =========================
    @APIExport(runOnUIThread=True)
    def startFocusMeasurement(self) -> bool:
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
                zpos = self.stage.getPosition()["Z"]
                self.lockFocus(zpos)
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
        self.sigFocusLockStateChanged.emit(self.getFocusLockState())

    # =========================
    # Legacy-compatible methods
    # =========================
    @APIExport(runOnUIThread=True)
    def unlockFocus(self):
        if self.locked:
            self.locked = False
            if self.pi:
                self.pi.restart()
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
            zpos = self.stage.getPosition()["Z"]
            self.lockFocus(zpos)
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
    def focusCalibrationStart(self):
        self.__focusCalibThread.start()

    @APIExport(runOnUIThread=True)
    def runFocusCalibration(self, from_position: Optional[float] = None,
                            to_position: Optional[float] = None,
                            num_steps: Optional[int] = None,
                            settle_time: Optional[float] = None) -> Dict[str, Any]:
        if from_position is not None:
            self._calib_params.from_position = from_position
        if to_position is not None:
            self._calib_params.to_position = to_position
        if num_steps is not None:
            self._calib_params.num_steps = num_steps
        if settle_time is not None:
            self._calib_params.settle_time = settle_time
        self.__focusCalibThread.start()
        self.__focusCalibThread.wait()
        return self.__focusCalibThread.getData()

    @APIExport(runOnUIThread=True)
    def getCalibrationResults(self) -> Dict[str, Any]:
        return self.__focusCalibThread.getData()

    @APIExport(runOnUIThread=True)
    def isCalibrationRunning(self) -> bool:
        return self.__focusCalibThread.isRunning()

    def showCalibrationCurve(self):
        if not IS_HEADLESS:
            self._widget.showCalibrationCurve(self.__focusCalibThread.getData())

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
        while self.__isPollingFramesActive:
            time.sleep(self.pollingFrameUpdateRate)

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

            if not self._state.is_measuring and not self.locked and not self.aboutToLock:
                continue

            self.setPointSignal = self.__processDataThread.update(self.cropped_im, self.twoFociVar)

            # Emit enhanced focus value
            focus_data = {
                "focus_value": self.setPointSignal,
                "timestamp": time.time(),
                "is_locked": self.locked,
                "lock_position": self.lockPosition if self.locked else None,
                "current_position": 0,
                "focus_metric": self._focus_params.focus_metric,
            }
            self.sigUpdateFocusValue.emit(focus_data)

            # === Control action (relative moves only) ===
            if self.locked and self.pi is not None:
                meas = float(self.setPointSignal)
                if self._pi_params.meas_lowpass_alpha > 0.0:
                    a = self._pi_params.meas_lowpass_alpha
                    self._meas_filt = meas if self._meas_filt is None else a * self._meas_filt + (1 - a) * meas
                    meas_for_pid = self._meas_filt
                else:
                    meas_for_pid = meas

                u = self.pi.update(meas_for_pid)                       # controller units
                step_um = u * self._pi_params.scale_um_per_unit        # convert to µm

                # deadband
                if abs(step_um) < self._pi_params.min_step_threshold:
                    step_um = 0.0

                # per-update clamp & optional safety gating
                limit = abs(self._pi_params.safety_move_limit) if self._pi_params.safety_motion_active else abs(self._pi_params.safety_move_limit)
                step_um = max(min(step_um, limit), -limit)

                if step_um != 0.0:
                    self.stage.move(value=step_um, axis="Z", speed=1000, is_blocking=False, is_absolute=False)
                    self._travel_used_um += abs(step_um)
                    # travel budget acts like safety_distance_limit
                    if self._pi_params.safety_motion_active and self._travel_used_um > self._pi_params.safety_distance_limit:
                        self._logger.warning("Travel budget exceeded; unlocking focus.")
                        self.unlockFocus()

            elif self.aboutToLock:
                if not hasattr(self, "aboutToLockDataPoints"):
                    self.aboutToLockDataPoints = np.zeros(5, dtype=float)
                self.aboutToLockUpdate()

            # Update plotting buffers
            self.updateSetPointData()
            if not IS_HEADLESS:
                try:
                    self._widget.camImg.setImage(im)
                    if self.currPoint < self.buffer:
                        self._widget.focusPlotCurve.setData(self.timeData[1:self.currPoint],
                                                            self.setPointData[1:self.currPoint])
                    else:
                        self._widget.focusPlotCurve.setData(self.timeData, self.setPointData)
                except Exception:
                    pass

        # (kept nested as in original)
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
        self.aboutToLockDataPoints[0] = float(self.setPointSignal)
        averageDiff = float(np.std(self.aboutToLockDataPoints))
        if averageDiff < self.aboutToLockDiffMax:
            zpos = self.stage.getPosition()["Z"]
            self.lockFocus(zpos)
            self.aboutToLock = False

    def updateSetPointData(self):
        if self.currPoint < self.buffer:
            self.setPointData[self.currPoint] = self.setPointSignal
            self.timeData[self.currPoint] = 0.0
        else:
            self.setPointData = np.roll(self.setPointData, -1)
            self.setPointData[-1] = self.setPointSignal
            self.timeData = np.roll(self.timeData, -1)
            self.timeData[-1] = 0.0
        self.currPoint += 1

    @APIExport(runOnUIThread=True)
    def setPIParameters(self, kp: float, ki: float):
        self._pi_params.kp = float(kp)
        self._pi_params.ki = float(ki)
        if not self.pi:
            self.pi = _PID(
                set_point=self._pi_params.set_point,
                kp=self._pi_params.kp, ki=self._pi_params.ki, kd=self._pi_params.kd,
                sample_time=self._pi_params.sample_time,
                integral_limit=self._pi_params.integral_limit,
                output_lowpass_alpha=self._pi_params.output_lowpass_alpha,
            )
        else:
            self.pi.setParameters(kp, ki)
        self.ki = ki
        self.kp = kp
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
        meas = float(self.setPointSignal)
        if self._pi_params.meas_lowpass_alpha > 0.0:
            a = self._pi_params.meas_lowpass_alpha
            self._meas_filt = meas if self._meas_filt is None else a * self._meas_filt + (1 - a) * meas
            meas_for_pid = self._meas_filt
        else:
            meas_for_pid = meas
        u = self.pi.update(meas_for_pid)
        step_um = u * self._pi_params.scale_um_per_unit
        # apply deadband + clamp, mirror of _pollFrames logic
        if abs(step_um) < self._pi_params.min_step_threshold:
            step_um = 0.0
        limit = abs(self._pi_params.safety_move_limit)
        step_um = max(min(step_um, limit), -limit)
        return step_um

    def lockFocus(self, zpos):
        if self.locked:
            return
        if IS_HEADLESS:
            kp, ki = self._pi_params.kp, self._pi_params.ki
        else:
            kp = float(self._widget.kpEdit.text())
            ki = float(self._widget.kiEdit.text())
            self._pi_params.kp = kp
            self._pi_params.ki = ki

        # Setpoint is current measured focus
        self._pi_params.set_point = float(self.setPointSignal)
        self.pi = _PID(
            set_point=self._pi_params.set_point,
            kp=self._pi_params.kp,
            ki=self._pi_params.ki,
            kd=self._pi_params.kd,
            sample_time=self._pi_params.sample_time,
            integral_limit=self._pi_params.integral_limit,
            output_lowpass_alpha=self._pi_params.output_lowpass_alpha,
        )
        self.lockPosition = float(zpos)  # kept for legacy visualization only
        self.locked = True
        self._travel_used_um = 0.0

        if not IS_HEADLESS:
            try:
                self._widget.focusLockGraph.lineLock = self._widget.focusPlot.addLine(y=self.setPointSignal, pen="r")
                self._widget.lockButton.setChecked(True)
            except Exception:
                pass

        self.updateZStepLimits()
        self._emitStateChangedSignal()
        self._logger.info(f"Focus locked at position {zpos} with set point {self.setPointSignal}")

    def updateZStepLimits(self):
        try:
            if not IS_HEADLESS and hasattr(self, '_widget'):
                self.zStepLimLo = 0.001 * float(self._widget.zStepFromEdit.text())
                self._focus_params.z_step_limit_nm = float(self._widget.zStepFromEdit.text())
            else:
                self.zStepLimLo = 0.001 * self._focus_params.z_step_limit_nm
        except Exception:
            self.zStepLimLo = 0.001 * self._focus_params.z_step_limit_nm

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


# =========================
# Processing thread
# =========================
class ProcessDataThread(Thread):
    def __init__(self, controller, *args, **kwargs):
        self._controller = controller
        super().__init__(*args, **kwargs)
        self.focusLockMetric: Optional[str] = None

    def setFocusLockMetric(self, focuslockMetric: str):
        self.focusLockMetric = focuslockMetric

    def getCroppedImage(self) -> np.ndarray:
        if hasattr(self, "imagearraygf"):
            return self.imagearraygf
        raise RuntimeError("No image processed yet. Please run update() first.")

    def _jpeg_size_metric(self, img: np.ndarray) -> int:
        if img.dtype != np.uint8:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_u8 = img
        success, buffer = cv2.imencode(".jpg", img_u8, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            self._controller._logger.warning("Failed to encode image to JPEG.")
            return 0
        return int(len(buffer))

    def update(self, cropped_img: np.ndarray, twoFociVar: bool) -> float:
        self.imagearraygf = cropped_img
        if self.focusLockMetric == "JPG":
            focusMetricGlobal = float(self._jpeg_size_metric(self.imagearraygf))
        elif self.focusLockMetric == "astigmatism":
            config = FocusConfig(
                gaussian_sigma=float(self._controller._focus_params.gaussian_sigma),
                background_threshold=int(self._controller._focus_params.background_threshold),
                crop_radius=int(self._controller._focus_params.crop_size or 300),
                enable_gaussian_blur=True,
            )
            focus_metric = FocusMetric(config)
            result = focus_metric.compute(self.imagearraygf)
            focusMetricGlobal = float(result["focus"])
            self._controller._logger.debug(
                f"Focus computation result: {result}, Focus value: {result['focus']:.4f}, Timestamp: {result['t']}"
            )
        else:
            self.imagearraygf = gaussian_filter(self.imagearraygf.astype(float), 7)
            if twoFociVar:
                allmaxcoords = peak_local_max(self.imagearraygf, min_distance=60)
                size = allmaxcoords.shape[0]
                if size >= 2:
                    maxvals = np.full(2, -np.inf)
                    maxvalpos = np.zeros(2, dtype=int)
                    for n in range(size):
                        val = self.imagearraygf[allmaxcoords[n][0], allmaxcoords[n][1]]
                        if val > maxvals[0]:
                            if val > maxvals[1]:
                                maxvals[0] = maxvals[1]
                                maxvals[1] = val
                                maxvalpos[0] = maxvalpos[1]
                                maxvalpos[1] = n
                            else:
                                maxvals[0] = val
                                maxvalpos[0] = n
                    xcenter = allmaxcoords[maxvalpos[0]][0]
                    ycenter = allmaxcoords[maxvalpos[0]][1]
                    if allmaxcoords[maxvalpos[1]][1] < ycenter:
                        xcenter = allmaxcoords[maxvalpos[1]][0]
                        ycenter = allmaxcoords[maxvalpos[1]][1]
                    centercoords2 = np.array([xcenter, ycenter])
                else:
                    centercoords = np.where(self.imagearraygf == np.max(self.imagearraygf))
                    centercoords2 = np.array([centercoords[0][0], centercoords[1][0]])
            else:
                centercoords = np.where(self.imagearraygf == np.max(self.imagearraygf))
                centercoords2 = np.array([centercoords[0][0], centercoords[1][0]])

            subsizey = 50
            subsizex = 50
            h, w = self.imagearraygf.shape[:2]
            xlow = max(0, int(centercoords2[0] - subsizex))
            xhigh = min(h, int(centercoords2[0] + subsizex))
            ylow = max(0, int(centercoords2[1] - subsizey))
            yhigh = min(w, int(centercoords2[1] + subsizex))

            self.imagearraygfsub = self.imagearraygf[xlow:xhigh, ylow:yhigh]
            massCenter = np.array(ndi.center_of_mass(self.imagearraygfsub))
            focusMetricGlobal = float(massCenter[1] + centercoords2[1])

        return focusMetricGlobal


# =========================
# Calibration thread
# =========================
class FocusCalibThread(Thread):
    def __init__(self, controller, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller = controller
        self.signalData: List[float] = []
        self.positionData: List[float] = []
        self.poly = None
        self.calibrationResult = None

    def run(self):
        self.signalData = []
        self.positionData = []

        calib_params = self._controller._calib_params

        if not IS_HEADLESS and hasattr(self._controller, '_widget'):
            try:
                from_val = float(self._controller._widget.calibFromEdit.text())
                to_val = float(self._controller._widget.calibToEdit.text())
            except (ValueError, AttributeError):
                from_val = calib_params.from_position
                to_val = calib_params.to_position
        else:
            from_val = calib_params.from_position
            to_val = calib_params.to_position

        scan_list = np.round(np.linspace(from_val, to_val, calib_params.num_steps), 2)

        self._controller.sigCalibrationProgress.emit({
            "event": "calibration_started",
            "total_steps": len(scan_list),
            "from_position": from_val,
            "to_position": to_val,
        })

        for i, z in enumerate(scan_list):
            self._controller._master.positionersManager[self._controller.positioner].setPosition(z, 0)
            time.sleep(calib_params.settle_time)
            focus_signal = float(self._controller.setPointSignal)
            actual_position = float(self._controller._master.positionersManager[self._controller.positioner].get_abs())

            self.signalData.append(focus_signal)
            self.positionData.append(actual_position)

            self._controller.sigCalibrationProgress.emit({
                "event": "calibration_progress",
                "step": i + 1,
                "total_steps": len(scan_list),
                "position": actual_position,
                "focus_value": focus_signal,
                "progress_percent": ((i + 1) / len(scan_list)) * 100,
            })

        self.poly = np.polyfit(self.positionData, self.signalData, 1)
        self.calibrationResult = np.around(self.poly, 4)

        self._controller.sigCalibrationProgress.emit({
            "event": "calibration_completed",
            "coefficients": self.poly.tolist(),
            "r_squared": self._calculate_r_squared(),
            "sensitivity_nm_per_px": self._get_sensitivity_nm_per_px(),
        })

        self.show()

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
        if IS_HEADLESS or not hasattr(self._controller, '_widget'):
            return
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
        return {
            "signalData": self.signalData,
            "positionData": self.positionData,
            "poly": self.poly.tolist() if self.poly is not None else None,
            "calibrationResult": self.calibrationResult.tolist() if self.calibrationResult is not None else None,
            "r_squared": self._calculate_r_squared(),
            "sensitivity_nm_per_px": self._get_sensitivity_nm_per_px(),
        }


# =========================
# Focus metric
# =========================
@dataclass
class FocusConfig:
    gaussian_sigma: float = 11.0
    background_threshold: int = 40
    crop_radius: int = 300
    enable_gaussian_blur: bool = True


class FocusMetric:
    def __init__(self, config: Optional[FocusConfig] = None):
        self.config = config or FocusConfig()

    @staticmethod
    def gaussian_1d(xdata: np.ndarray, i0: float, x0: float, sigma: float, amp: float) -> np.ndarray:
        x = xdata
        x0 = float(x0)
        return i0 + amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def double_gaussian_1d(xdata: np.ndarray, i0: float, x0: float, sigma: float, amp: float, dist: float) -> np.ndarray:
        x = xdata
        x0 = float(x0)
        return (
            i0
            + amp * np.exp(-((x - (x0 - dist / 2)) ** 2) / (2 * sigma ** 2))
            + amp * np.exp(-((x - (x0 + dist / 2)) ** 2) / (2 * sigma ** 2))
        )

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            im = np.mean(frame, axis=-1).astype(np.uint8)
        else:
            im = frame.astype(np.uint8)

        im = im.astype(float)

        if self.config.crop_radius > 0:
            im_gauss = gaussian_filter(im, sigma=111)
            max_coord = np.unravel_index(np.argmax(im_gauss), im_gauss.shape)
            h, w = im.shape
            y_min = max(0, max_coord[0] - self.config.crop_radius)
            y_max = min(h, max_coord[0] + self.config.crop_radius)
            x_min = max(0, max_coord[1] - self.config.crop_radius)
            x_max = min(w, max_coord[1] + self.config.crop_radius)
            im = im[y_min:y_max, x_min:x_max]

        if self.config.enable_gaussian_blur:
            im = gaussian_filter(im, sigma=self.config.gaussian_sigma)

        im = im - np.mean(im) / 2.0
        im[im < self.config.background_threshold] = 0
        return im

    def preprocess_frame_rainer(self, frame: np.ndarray) -> np.ndarray:
        return self.preprocess_frame(frame)

    def compute_projections(self, im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        projX = np.mean(im, axis=0)
        projY = np.mean(im, axis=1)
        return projX, projY

    def fit_projections(self, projX: np.ndarray, projY: np.ndarray, isDoubleGaussX: bool = False) -> Tuple[float, float]:
        h1, w1 = len(projY), len(projX)
        x = np.arange(w1)
        y = np.arange(h1)

        i0_x = float(np.mean(projX))
        amp_x = float(np.max(projX) - i0_x)
        sigma_x_init = float(np.std(projX))
        i0_y = float(np.mean(projY))
        amp_y = float(np.max(projY) - i0_y)
        sigma_y_init = float(np.std(projY))

        if isDoubleGaussX:
            init_guess_x = [i0_x, w1 / 2, sigma_x_init, amp_x, 100.0]
        else:
            init_guess_x = [i0_x, w1 / 2, sigma_x_init, amp_x]
        init_guess_y = [i0_y, h1 / 2, sigma_y_init, amp_y]

        try:
            if isDoubleGaussX:
                popt_x, _ = curve_fit(self.double_gaussian_1d, x, projX, p0=init_guess_x, maxfev=50000)
                sigma_x = abs(float(popt_x[2]))
            else:
                popt_x, _ = curve_fit(self.gaussian_1d, x, projX, p0=init_guess_x, maxfev=50000)
                sigma_x = abs(float(popt_x[2]))

            popt_y, _ = curve_fit(self.gaussian_1d, y, projY, p0=init_guess_y, maxfev=50000)
            sigma_y = abs(float(popt_y[2]))
        except Exception:
            sigma_x = float(np.std(projX))
            sigma_y = float(np.std(projY))

        return sigma_x, sigma_y

    def compute(self, frame: np.ndarray) -> Dict[str, Any]:
        timestamp = time.time()
        try:
            im = self.preprocess_frame(frame)
            projX, projY = self.compute_projections(im)
            sigma_x, sigma_y = self.fit_projections(projX, projY)
            focus_value = 12334567 if sigma_y == 0 else float(sigma_x / sigma_y)
        except Exception:
            focus_value = 12334567
        return {"t": timestamp, "focus": focus_value}

    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
