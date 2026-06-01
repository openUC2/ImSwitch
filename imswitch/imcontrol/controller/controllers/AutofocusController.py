import time
import numpy as np
import threading
from enum import Enum
from scipy.optimize import curve_fit
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from skimage.filters import gaussian
from imswitch.imcommon.framework import Signal
import cv2
import queue

# Global axis for Z-positioning
gAxis = "Z"


class AutofocusState(Enum):
    """State machine for autofocus lifecycle."""
    IDLE = "idle"
    STARTING = "starting"
    SCANNING = "scanning"
    FITTING = "fitting"
    MOVING_TO_FOCUS = "moving_to_focus"
    ABORTED = "aborted"
    FINISHED = "finished"
    ERROR = "error"


def _gaussian(x, a, x0, sigma, c):
    return a * np.exp(-0.5 * ((x - x0) / (sigma + 1e-12)) ** 2) + c


def _robust_gaussian_fit(x, y):
    """
    Fit a 1D Gaussian to (x, y). Returns (x0, fit_y) where:
      - x0 is the fitted center
      - fit_y is the Gaussian evaluated on x (or None on failure)
    Falls back to argmax if fit fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    a0 = float(np.max(y) - np.min(y) + 1e-9)
    x0_0 = float(x[np.argmax(y)])
    sigma0 = max((np.max(x) - np.min(x)) / 5.0, 1e-6)
    c0 = float(np.min(y))

    p0 = [a0, x0_0, sigma0, c0]
    bounds = ([0.0, np.min(x) - abs(3 * sigma0), 1e-6, -np.inf],
              [np.inf, np.max(x) + abs(3 * sigma0), np.inf, np.inf])

    try:
        popt, _ = curve_fit(_gaussian, x, y, p0=p0, bounds=bounds, maxfev=20000)
        a, x0, sigma, c = popt
        fit_y = _gaussian(x, a, x0, sigma, c)
        return float(x0), fit_y
    except Exception:
        return float(x0_0), None


class MovementController:
    """
    Asynchronous mover with abort capability:
      .move_to_position(value, axis, speed, is_absolute)
      .is_target_reached()
      .abort() - stops current motion immediately
    """
    def __init__(self, stages):
        self.stages = stages
        self._lock = threading.Lock()
        self.target_reached = True
        self.target_position = None
        self.axis = None
        self.speed = None
        self.is_absolute = True
        self._thread = None
        self._abort_flag = False

    def move_to_position(self, value, axis=gAxis, speed=None, is_absolute=True):
        with self._lock:
            self._abort_flag = False
            self.target_reached = False
            self.target_position = value
            self.axis = axis
            self.speed = speed
            self.is_absolute = is_absolute
        self._thread = threading.Thread(target=self._move, daemon=True)
        self._thread.start()

    def _move(self):
        with self._lock:
            value = self.target_position
            axis = self.axis
            speed = self.speed
            is_absolute = self.is_absolute
        try:
            # Check abort flag before starting move
            if self._abort_flag:
                return
            self.stages.move(value=value, axis=axis, speed=speed, is_absolute=is_absolute, is_blocking=True)
        finally:
            with self._lock:
                self.target_reached = True

    def is_target_reached(self):
        with self._lock:
            return bool(self.target_reached)

    def abort(self):
        """Abort current motion immediately by calling forceStop on the stage."""
        with self._lock:
            self._abort_flag = True
        # Call forceStop on stage if available
        if hasattr(self.stages, 'forceStop'):
            try:
                if self.axis:
                    self.stages.forceStop(self.axis)
                else:
                    self.stages.forceStop(gAxis)
            except Exception:
                pass
        with self._lock:
            self.target_reached = True

    def is_aborted(self):
        """Check if abort was requested."""
        with self._lock:
            return bool(self._abort_flag)


class AutofocusController(ImConWidgetController):
    """Linked to AutofocusWidget."""
    sigUpdateFocusPlot = Signal(object, object)   # x, y
    sigUpdateFocusValue = Signal(object)          # {"bestzpos": float}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        # Thread-safe state management
        self._stateLock = threading.Lock()
        self._autofocusState = AutofocusState.IDLE
        self.isAutofusRunning = False  # keep attribute name for compatibility
        self.isLiveMonitoring = False  # for live focus value monitoring
        self._liveMonitoringThread = None
        self._liveMonitoringPeriod = 0.5  # default period in seconds
        self._focusMethod = "LAPE"  # default focus measurement method
        self._liveMonitoringCropsize = 2048  # default crop size for live monitoring
        self._liveMonitoringNGauss = 0  # Gaussian blur sigma (0 = no blur, set to match autofocus if needed)
        self._liveMonitoringBinning = 1  # Binning factor (1 = no binning, set to match autofocus if needed)

        # Safety configuration - default software limits
        self._minZ = -np.inf  # Will be updated from stage manager if available
        self._maxZ = np.inf   # Will be updated from stage manager if available
        self._positionValidated = False  # Track if position has been validated since startup

        if self._setupInfo.autofocus is not None:
            self.cameraName = self._setupInfo.autofocus.camera
            self.stageName = self._setupInfo.autofocus.positioner
        else:
            self.cameraName = self._master.detectorsManager.getAllDeviceNames()[0]
            self.stageName = self._master.positionersManager.getAllDeviceNames()[0]

        try:
            self.camera = self._master.detectorsManager[self.cameraName]
        except Exception:
            self.cameraName = self._master.detectorsManager.getAllDeviceNames()[0]
            self.camera = self._master.detectorsManager[self.cameraName]
        try:
            self.stages = self._master.positionersManager[self.stageName]
        except Exception:
            self.stageName = self._master.positionersManager.getAllDeviceNames()[0]
            self.stages = self._master.positionersManager[self.stageName]

        # Initialize safety limits from stage manager if available
        self._initializeSafetyLimits()

        self._commChannel.sigAutoFocus.connect(self.autoFocus)

        self._moveController = MovementController(self.stages)

    def __del__(self):
        try:
            self._setAutofocusState(AutofocusState.IDLE)
            self.isAutofusRunning = False
            self.isLiveMonitoring = False
            if hasattr(self, '_AutofocusThead') and self._AutofocusThead and self._AutofocusThead.is_alive():
                self._AutofocusThead.join(timeout=1.0)
            if hasattr(self, '_liveMonitoringThread') and self._liveMonitoringThread and self._liveMonitoringThread.is_alive():
                self._liveMonitoringThread.join(timeout=1.0)
        except Exception:
            pass
        if hasattr(super(), '__del__'):
            super().__del__()

    def _initializeSafetyLimits(self):
        """Initialize safety limits from stage manager if available."""
        try:
            # Try to get limits from stage manager (ESP32StageManager has these)
            if hasattr(self.stages, 'minZ'):
                self._minZ = self.stages.minZ
            if hasattr(self.stages, 'maxZ'):
                self._maxZ = self.stages.maxZ
            self.__logger.info(f"Autofocus safety limits initialized: minZ={self._minZ}, maxZ={self._maxZ}")
        except Exception as e:
            self.__logger.warning(f"Could not initialize safety limits from stage: {e}")

    def _setAutofocusState(self, state: AutofocusState):
        """Thread-safe state setter."""
        with self._stateLock:
            self._autofocusState = state
            self.isAutofusRunning = state not in [AutofocusState.IDLE, AutofocusState.FINISHED, AutofocusState.ERROR, AutofocusState.ABORTED]

    def _getAutofocusState(self) -> AutofocusState:
        """Thread-safe state getter."""
        with self._stateLock:
            return self._autofocusState

    def _getSafeCurrentZ(self, axis: str = gAxis) -> tuple:
        """
        Get current Z position with validation against hardware limits.

        Returns:
            tuple: (position, is_valid, error_message)
                - position: float or None if invalid
                - is_valid: bool
                - error_message: str or None
        """
        try:
            # Get fresh position from hardware
            pos_dict = self.stages.getPosition()
            if pos_dict is None:
                return None, False, "Failed to read position from stage"

            if axis not in pos_dict:
                return None, False, f"Axis '{axis}' not found in position data"

            current_z = float(pos_dict[axis])

            # Validate against NaN
            if np.isnan(current_z):
                return None, False, "Position is NaN - stage position not initialized"

            # Validate against limits
            if current_z < self._minZ:
                return None, False, f"Current position {current_z} is below minimum limit {self._minZ}"
            if current_z > self._maxZ:
                return None, False, f"Current position {current_z} is above maximum limit {self._maxZ}"

            self._positionValidated = True
            return current_z, True, None

        except Exception as e:
            return None, False, f"Error reading position: {e}"

    def _clampPosition(self, position: float, axis: str = gAxis) -> float:
        """
        Clamp a position to hardware limits.

        Args:
            position: Target position
            axis: Axis name

        Returns:
            Clamped position within safe bounds
        """
        clamped = np.clip(position, self._minZ, self._maxZ)
        if clamped != position:
            self.__logger.warning(f"Position {position} clamped to {clamped} (limits: {self._minZ} to {self._maxZ})")
        return float(clamped)

    def _validateScanRange(self, center: float, rangez: float, axis: str = gAxis) -> tuple:
        """
        Validate and potentially adjust scan range to stay within limits.

        Args:
            center: Center position for scan
            rangez: Scan range (±rangez from center)
            axis: Axis name

        Returns:
            tuple: (adjusted_rangez, is_valid, warning_message)
        """
        scan_min = center - abs(rangez) / 2
        scan_max = center + abs(rangez) / 2

        # Check if scan would exceed limits
        if scan_min < self._minZ or scan_max > self._maxZ:
            # Try to adjust range to fit within limits
            available_down = center - self._minZ
            available_up = self._maxZ - center
            adjusted_rangez = 2 * min(available_down, available_up, abs(rangez) / 2)

            if adjusted_rangez < abs(rangez) * 0.1:  # Less than 10% of requested range
                return adjusted_rangez, False, f"Scan range too restricted: only {adjusted_rangez:.1f} available vs {rangez} requested"

            return adjusted_rangez, True, f"Scan range adjusted from {rangez} to {adjusted_rangez:.1f} to fit within limits"

        return rangez, True, None

    @APIExport(runOnUIThread=True)
    def getAutofocusStatus(self):
        """
        Get current autofocus status for frontend synchronization.

        Returns:
            dict with state information
        """
        current_z, is_valid, error = self._getSafeCurrentZ()
        return {
            "state": self._getAutofocusState().value,
            "isRunning": self.isAutofusRunning,
            "isLiveMonitoring": self.isLiveMonitoring,
            "currentZ": current_z,
            "positionValid": is_valid,
            "positionError": error,
            "minZ": self._minZ if self._minZ != -np.inf else None,
            "maxZ": self._maxZ if self._maxZ != np.inf else None,
        }



    @APIExport(runOnUIThread=True)
    def autoFocus(self, rangez: int = 100, resolutionz: int = 10, defocusz: int = 0, tSettle:float=0.1, isDebug:bool=False,
                               nGauss:int=0, nCropsize:int=2048, focusAlgorithm:str="LAPE", static_offset:float=0.0, twoStage:bool=False, twoStageDivisor:int=10):
        """
        Step-scan autofocus with Gaussian peak fit.

        Args:
            rangez: Z-range to scan (±rangez from current position)
            resolutionz: Step size in Z
            defocusz: Defocus offset (currently unused)
            tSettle: Settling time between steps (seconds)
            isDebug: Save debug images if True
            nGauss: Gaussian blur sigma (0 to disable)
            nCropsize: Crop size for focus calculation
            focusAlgorithm: Focus measurement method ("LAPE", "GLVA", or "JPEG")
            static_offset: Static offset to add to final focus position
            twoStage: If True, perform coarse scan followed by fine scan
            twoStageDivisor: Divisor for fine scan range and resolution (default 10)

        Returns:
            dict with status information or None if started successfully
        """
        # Thread-safe state check
        with self._stateLock:
            if self._autofocusState not in [AutofocusState.IDLE, AutofocusState.FINISHED, AutofocusState.ERROR, AutofocusState.ABORTED]:
                self.__logger.warning(f"Autofocus already running (state: {self._autofocusState.value})")
                return {"status": "error", "message": f"Autofocus already running (state: {self._autofocusState.value})"}
            self._autofocusState = AutofocusState.STARTING
            self.isAutofusRunning = True

        # Validate current position before starting
        current_z, is_valid, error_msg = self._getSafeCurrentZ()
        if not is_valid:
            self.__logger.error(f"Cannot start autofocus: {error_msg}")
            self._setAutofocusState(AutofocusState.ERROR)
            return {"status": "error", "message": error_msg}

        # Validate scan range
        adjusted_rangez, range_valid, range_warning = self._validateScanRange(current_z, rangez)
        if range_warning:
            self.__logger.warning(range_warning)
        if not range_valid:
            self.__logger.error(f"Cannot start autofocus: {range_warning}")
            self._setAutofocusState(AutofocusState.ERROR)
            return {"status": "error", "message": range_warning}

        self._AutofocusThead = threading.Thread(
            target=self.doAutofocusBackground,
            args=(adjusted_rangez, resolutionz, defocusz, gAxis, tSettle, isDebug, nGauss, nCropsize, focusAlgorithm, static_offset, twoStage, twoStageDivisor),
            daemon=True
        )
        self._AutofocusThead.start()
        return {"status": "started", "rangez": adjusted_rangez, "centerZ": current_z}

    @APIExport(runOnUIThread=True)
    def autoFocusFast(self, sweep_range: float = 150.0, speed: float = None, defocusz: int = 0, axis: str = gAxis,
                      nCropsize: int = 2048, focusAlgorithm: str = "LAPE", static_offset: float = 0.0):
        """
        Continuous fast-sweep autofocus WITHOUT continuous Z-readback:
          - Move to z0 + sweep_range, then sweep down to z0 - sweep_range.
          - Record a timestamp for every image.
          - After the sweep, build Z positions by linear time mapping using
            z(t) = z_start + v_eff * (t - t_start), where
            v_eff = (z_end - z_start) / (t_end - t_start).
          - Fit Gaussian to (z, focus) and move to center.

        Returns:
            dict with status information
        """
        # Thread-safe state check
        with self._stateLock:
            if self._autofocusState not in [AutofocusState.IDLE, AutofocusState.FINISHED, AutofocusState.ERROR, AutofocusState.ABORTED]:
                self.__logger.warning(f"Autofocus already running (state: {self._autofocusState.value})")
                return {"status": "error", "message": f"Autofocus already running (state: {self._autofocusState.value})"}
            self._autofocusState = AutofocusState.STARTING
            self.isAutofusRunning = True

        # Validate current position before starting
        current_z, is_valid, error_msg = self._getSafeCurrentZ(axis)
        if not is_valid:
            self.__logger.error(f"Cannot start fast autofocus: {error_msg}")
            self._setAutofocusState(AutofocusState.ERROR)
            return {"status": "error", "message": error_msg}

        # Validate sweep range
        adjusted_range, range_valid, range_warning = self._validateScanRange(current_z, sweep_range * 2)  # *2 because sweep is ±sweep_range
        if range_warning:
            self.__logger.warning(range_warning)
        adjusted_sweep = adjusted_range / 2

        self._AutofocusThead = threading.Thread(
            target=self._doAutofocusFastBackground_timeMapped,
            args=(adjusted_sweep, speed, defocusz, axis, nCropsize, focusAlgorithm, static_offset),
            daemon=True
        )
        self._AutofocusThead.start()
        return {"status": "started", "sweep_range": adjusted_sweep, "centerZ": current_z}

    @APIExport(runOnUIThread=True)
    def stopAutofocus(self):
        """
        Stop autofocus immediately.

        This will:
        1. Set abort flag to stop scanning loop
        2. Call forceStop on stage to halt motion immediately
        3. Abort MovementController if active
        4. Update state machine

        Returns:
            dict with status information
        """
        self.__logger.info("Stop autofocus requested")

        # Set abort state
        self._setAutofocusState(AutofocusState.ABORTED)
        self.isAutofusRunning = False

        # Abort MovementController (this will call forceStop on stage)
        if hasattr(self, '_moveController') and self._moveController:
            self._moveController.abort()

        # Also directly call forceStop on stage for blocking moves
        try:
            if hasattr(self.stages, 'forceStop'):
                self.stages.forceStop(gAxis)
                self.__logger.info(f"Force stop called on {gAxis} axis")
        except Exception as e:
            self.__logger.error(f"Error calling forceStop: {e}")

        # Emit signal to notify frontend
        self._commChannel.sigAutoFocusRunning.emit(False)

        return {"status": "stopped", "state": self._getAutofocusState().value}

    @APIExport(runOnUIThread=True)
    def autoFocusHillClimbing(self, initial_step: float = 20.0, min_step: float = 1.0,
                               step_reduction: float = 0.5, max_iterations: int = 50,
                               tSettle: float = 0.1, nCropsize: int = 2048,
                               focusAlgorithm: str = "LAPE", nGauss: int = 0,
                               static_offset: float = 0.0):
        """
        Hill-climbing autofocus using gradient-based contrast detection.

        Instead of sweeping the entire Z range, this method iteratively searches
        for the peak contrast by moving in the direction of increasing focus
        value. When contrast decreases, the direction is reversed and the step
        size is reduced. The search converges when the step size drops below
        min_step.

        Args:
            initial_step: Starting step size in Z units (µm)
            min_step: Minimum step size — convergence criterion
            step_reduction: Factor to reduce step on direction reversal (0 < f < 1)
            max_iterations: Safety limit on total iterations
            tSettle: Settling time between Z moves (seconds)
            nCropsize: Crop size for focus calculation
            focusAlgorithm: Focus measurement method ("LAPE", "GLVA", or "JPEG")
            nGauss: Gaussian blur sigma (0 to disable)
            static_offset: Static offset added to final focus position

        Returns:
            dict with status information
        """
        # Thread-safe state check
        with self._stateLock:
            if self._autofocusState not in [AutofocusState.IDLE, AutofocusState.FINISHED,
                                            AutofocusState.ERROR, AutofocusState.ABORTED]:
                self.__logger.warning(f"Autofocus already running (state: {self._autofocusState.value})")
                return {"status": "error",
                        "message": f"Autofocus already running (state: {self._autofocusState.value})"}
            self._autofocusState = AutofocusState.STARTING
            self.isAutofusRunning = True

        # Validate current position
        current_z, is_valid, error_msg = self._getSafeCurrentZ()
        if not is_valid:
            self.__logger.error(f"Cannot start hill-climbing AF: {error_msg}")
            self._setAutofocusState(AutofocusState.ERROR)
            return {"status": "error", "message": error_msg}

        self._AutofocusThead = threading.Thread(
            target=self._doHillClimbingBackground,
            args=(initial_step, min_step, step_reduction, max_iterations,
                  tSettle, nCropsize, focusAlgorithm, nGauss, static_offset),
            daemon=True
        )
        self._AutofocusThead.start()
        return {"status": "started", "centerZ": current_z, "method": "hill_climbing"}

    def _doHillClimbingBackground(self, initial_step: float = 20.0, min_step: float = 1.0,
                                   step_reduction: float = 0.5, max_iterations: int = 50,
                                   tSettle: float = 0.1, nCropsize: int = 2048,
                                   focusAlgorithm: str = "LAPE", nGauss: int = 0,
                                   static_offset: float = 0.0):
        """
        Hill-climbing autofocus background thread.

        Algorithm:
        1. Grab frame at current position, measure baseline focus F0
        2. Move +initial_step, measure F1 to determine gradient direction
        3. If F1 > F0: keep going in same direction
           Else: reverse direction
        4. Loop:
           a. Move step_size in current direction
           b. Grab frame, measure F_new
           c. If F_new > F_best: update best, continue same direction
           d. If F_new <= F_best: reduce step_size, reverse direction
           e. If step_size < min_step or max_iterations reached: stop
        5. Move to best_position + static_offset (clamped)
        """
        try:
            self._setAutofocusState(AutofocusState.SCANNING)
            self._commChannel.sigAutoFocusRunning.emit(True)
            axis = gAxis

            # Helper to measure focus at current position (uses shared fast pipeline)
            def measure_focus():
                frame = self.grabCameraFrame()
                if frame is None:
                    return None
                return _compute_focus_value_fast(
                    frame,
                    crop_size=nCropsize,
                    binning=1,
                    n_gauss=nGauss,
                    method=focusAlgorithm,
                )

            # Step 1: Measure baseline at current position
            current_z, is_valid, _ = self._getSafeCurrentZ(axis)
            if not is_valid:
                self._setAutofocusState(AutofocusState.ERROR)
                self._commChannel.sigAutoFocusRunning.emit(False)
                return None

            z_start = current_z
            f0 = measure_focus()
            if f0 is None:
                self.__logger.error("Hill-climbing: failed to grab initial frame")
                self._setAutofocusState(AutofocusState.ERROR)
                self._commChannel.sigAutoFocusRunning.emit(False)
                return None

            # Collect data for plotting
            z_history = [current_z]
            f_history = [f0]

            best_z = current_z
            best_f = f0
            step_size = abs(initial_step)

            # Step 2: Probe direction — move +step to determine gradient
            probe_z = self._clampPosition(current_z + step_size, axis)
            self.stages.move(value=probe_z, axis=axis, is_absolute=True, is_blocking=True)
            time.sleep(tSettle)

            if self._getAutofocusState() == AutofocusState.ABORTED:
                self.stages.move(value=z_start, axis=axis, is_absolute=True, is_blocking=True)
                self._commChannel.sigAutoFocusRunning.emit(False)
                return None

            f1 = measure_focus()
            if f1 is None:
                self.__logger.error("Hill-climbing: failed to grab probe frame")
                self.stages.move(value=z_start, axis=axis, is_absolute=True, is_blocking=True)
                self._setAutofocusState(AutofocusState.ERROR)
                self._commChannel.sigAutoFocusRunning.emit(False)
                return None

            current_z = probe_z
            z_history.append(current_z)
            f_history.append(f1)

            if f1 > best_f:
                best_z = current_z
                best_f = f1

            # Determine initial direction
            direction = 1.0 if f1 >= f0 else -1.0
            if f1 < f0:
                # Initial probe worsened — reverse direction
                step_size *= step_reduction

            # Step 3–4: Iterative hill-climb
            for iteration in range(max_iterations):
                # Check abort
                if self._getAutofocusState() == AutofocusState.ABORTED:
                    self.__logger.info(f"Hill-climbing aborted at iteration {iteration}")
                    self.stages.move(value=z_start, axis=axis, is_absolute=True, is_blocking=True)
                    self._commChannel.sigAutoFocusRunning.emit(False)
                    return None

                # Move in current direction
                next_z = self._clampPosition(current_z + direction * step_size, axis)

                # If clamped to same position, reverse direction
                if abs(next_z - current_z) < 1e-6:
                    direction *= -1.0
                    step_size *= step_reduction
                    if step_size < min_step:
                        self.__logger.info(f"Hill-climbing converged at boundary (step={step_size:.3f})")
                        break
                    continue

                self.stages.move(value=next_z, axis=axis, is_absolute=True, is_blocking=True)
                time.sleep(tSettle)
                current_z = next_z

                f_new = measure_focus()
                if f_new is None:
                    self.__logger.warning(f"Hill-climbing: frame grab failed at iteration {iteration}")
                    continue

                z_history.append(current_z)
                f_history.append(f_new)

                if f_new > best_f:
                    # Improving — update best and keep going
                    best_z = current_z
                    best_f = f_new
                else:
                    # Passed the peak — reverse and reduce step
                    direction *= -1.0
                    step_size *= step_reduction
                    if step_size < min_step:
                        self.__logger.info(
                            f"Hill-climbing converged after {iteration + 1} iterations "
                            f"(step={step_size:.3f} < min_step={min_step})"
                        )
                        break

            # Step 5: Move to best position + offset
            self._setAutofocusState(AutofocusState.MOVING_TO_FOCUS)
            best_target = self._clampPosition(best_z + static_offset, axis)
            self.stages.move(value=best_target, axis=axis, is_absolute=True, is_blocking=True)

            # Emit plot data (sorted by Z for clean visualization)
            z_arr = np.array(z_history)
            f_arr = np.array(f_history)
            sort_idx = np.argsort(z_arr)
            try:
                self.sigUpdateFocusPlot.emit(z_arr[sort_idx], f_arr[sort_idx])
            except Exception:
                pass

            self._setAutofocusState(AutofocusState.FINISHED)
            self._commChannel.sigAutoFocusRunning.emit(False)
            self.sigUpdateFocusValue.emit({"bestzpos": best_target})
            self.__logger.info(
                f"Hill-climbing autofocus complete: best_z={best_target:.2f}, "
                f"focus={best_f:.2f}, iterations={len(z_history)}"
            )
            return best_target

        except Exception as e:
            self.__logger.error(f"Hill-climbing autofocus error: {e}")
            self._setAutofocusState(AutofocusState.ERROR)
            self._commChannel.sigAutoFocusRunning.emit(False)
            return None

    @APIExport(runOnUIThread=True)
    def startLiveMonitoring(self, period: float = 0.5, method: str = "LAPE", nCropsize: int = 2048):
        """
        Start continuous live focus value monitoring.

        Args:
            period: Update period in seconds (default 0.5s)
            method: Focus measurement method ("LAPE", "GLVA", or "JPEG")
            nCropsize: Crop size for focus calculation (default 2048)
        """
        if self.isLiveMonitoring:
            self.__logger.warning("Live monitoring already running")
            return {"status": "already_running", "period": self._liveMonitoringPeriod, "method": self._focusMethod}

        self._liveMonitoringPeriod = max(0.1, float(period))  # minimum 0.1s
        self._focusMethod = method if method in ["LAPE", "GLVA", "JPEG"] else "LAPE"
        self._liveMonitoringCropsize = int(nCropsize)

        self.isLiveMonitoring = True
        self._liveMonitoringThread = threading.Thread(
            target=self._doLiveMonitoringBackground,
            daemon=True
        )
        self._liveMonitoringThread.start()

        self.__logger.info(f"Live focus monitoring started with period={self._liveMonitoringPeriod}s, method={self._focusMethod}, cropsize={self._liveMonitoringCropsize}")
        return {"status": "started", "period": self._liveMonitoringPeriod, "method": self._focusMethod, "cropsize": self._liveMonitoringCropsize}

    @APIExport(runOnUIThread=True)
    def stopLiveMonitoring(self):
        """Stop continuous live focus value monitoring."""
        if not self.isLiveMonitoring:
            return {"status": "not_running"}

        self.isLiveMonitoring = False
        if self._liveMonitoringThread and self._liveMonitoringThread.is_alive():
            self._liveMonitoringThread.join(timeout=2.0)

        self.__logger.info("Live focus monitoring stopped")
        return {"status": "stopped"}

    @APIExport(runOnUIThread=True)
    def setLiveMonitoringParameters(self, period: float = None, method: str = None, nCropsize: int = None, nGauss: int = None, binning: int = None):
        """
        Update live monitoring parameters.

        Args:
            period: Update period in seconds (optional)
            method: Focus measurement method ("LAPE", "GLVA", or "JPEG") (optional)
            nCropsize: Crop size for focus calculation (optional)
            nGauss: Gaussian blur sigma (0 = no blur, 7 = match autofocus default) (optional)
            binning: Binning factor (1 = no binning, 3 = match autofocus default) (optional)
        """
        if period is not None:
            self._liveMonitoringPeriod = max(0.1, float(period))
        if method is not None and method in ["LAPE", "GLVA", "JPEG"]:
            self._focusMethod = method
        if nCropsize is not None:
            self._liveMonitoringCropsize = int(nCropsize)
        if nGauss is not None:
            self._liveMonitoringNGauss = max(0, int(nGauss))
        if binning is not None:
            self._liveMonitoringBinning = max(1, int(binning))

        return {
            "status": "updated",
            "period": self._liveMonitoringPeriod,
            "method": self._focusMethod,
            "cropsize": self._liveMonitoringCropsize,
            "nGauss": self._liveMonitoringNGauss,
            "binning": self._liveMonitoringBinning,
            "is_running": self.isLiveMonitoring
        }

    @APIExport(runOnUIThread=True)
    def getLiveMonitoringStatus(self):
        """Get current status of live monitoring."""
        return {
            "is_running": self.isLiveMonitoring,
            "period": self._liveMonitoringPeriod,
            "method": self._focusMethod,
            "cropsize": getattr(self, '_liveMonitoringCropsize', 2048),
            "nGauss": getattr(self, '_liveMonitoringNGauss', 0),
            "binning": getattr(self, '_liveMonitoringBinning', 1)
        }

    def _doLiveMonitoringBackground(self):
        """Background thread for continuous focus value monitoring."""
        self.__logger.info("Live monitoring thread started")

        while self.isLiveMonitoring:
            try:
                t_start = time.time()

                # Grab a fresh frame
                frame = self.grabCameraFrame(frameSync=1)
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Process frame using the unified fast pipeline (RGB-safe, OpenCV-based)
                # so live monitoring values are consistent with autofocus scan results.
                binning = getattr(self, '_liveMonitoringBinning', 1)
                nGauss = getattr(self, '_liveMonitoringNGauss', 0)
                focus_value = _compute_focus_value_fast(
                    frame,
                    crop_size=self._liveMonitoringCropsize,
                    binning=binning,
                    n_gauss=nGauss,
                    method=self._focusMethod,
                )

                # Emit signal with focus value and timestamp
                self._commChannel.sigAutoFocusLiveValue.emit({
                    "focus_value": float(focus_value),
                    "timestamp": time.time(),
                    "method": self._focusMethod
                })

                # Sleep for remaining period time
                elapsed = time.time() - t_start
                sleep_time = max(0, self._liveMonitoringPeriod - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.__logger.error(f"Error in live monitoring: {e}")
                time.sleep(0.1)  # avoid tight loop on repeated errors

        self.__logger.info("Live monitoring thread stopped")

    def grabCameraFrame(self, frameSync: int = 1, returnFrameNumber: bool = False):
        # ensure we get a fresh frame
        timeoutFrameRequest = 1 # seconds # TODO: Make dependent on exposure time
        cTime = time.time()

        lastFrameNumber=-1
        currentFrameNumber = None
        while(1):
            # get frame and frame number to get one that is newer than the one with illumination off eventually
            mFrame, currentFrameNumber = self.camera.getLatestFrame(returnFrameNumber=True)
            self.__logger.debug(f"Grabbed frame with frame number: {currentFrameNumber}, last frame number: {lastFrameNumber}")
            if lastFrameNumber==-1:
                # first round
                lastFrameNumber = currentFrameNumber
            if time.time()-cTime> timeoutFrameRequest:
                # in case exposure time is too long we need break at one point
                if mFrame is None:
                    mFrame = self.camera.getLatestFrame(returnFrameNumber=False)
                break
            if currentFrameNumber <= lastFrameNumber+frameSync:
                time.sleep(0.01) # off-load CPU
            else:
                break

        if returnFrameNumber:
            return mFrame, currentFrameNumber
        return mFrame




    # ---------- Step-scan autofocus with Gaussian fit ----------
    def doAutofocusBackground(self, rangez:float=100, resolutionz:float=10, defocusz:float=0, axis:str=gAxis, tSettle:float=0.1, isDebug:bool=False, nGauss:int=0, nCropsize:int=2048, focusAlgorithm:str="LAPE", static_offset:float=0.0, twoStage:bool=False, twoStageDivisor:int=10):
        try:

            self.__logger.info(f"Starting autofocus - Stage 1: Coarse scan (range=±{rangez}, resolution={resolutionz}), defocus={defocusz}, axis={axis}, tSettle={tSettle}, nGauss={nGauss}, nCropsize={nCropsize}, focusAlgorithm={focusAlgorithm}, static_offset={static_offset}, twoStage={twoStage}")
            self._setAutofocusState(AutofocusState.SCANNING)
            self._commChannel.sigAutoFocusRunning.emit(True)
            # Stage 1: Coarse scan
            best_z_coarse = self._doSingleAutofocusScan(rangez, resolutionz, defocusz, axis, tSettle, isDebug, nGauss, nCropsize, focusAlgorithm, static_offset)

            # Check if aborted
            if self._getAutofocusState() == AutofocusState.ABORTED:
                self.__logger.info("Autofocus aborted during coarse scan")
                self._commChannel.sigAutoFocusRunning.emit(False)
                return None

            if best_z_coarse is None:
                self._setAutofocusState(AutofocusState.ERROR)
                self._commChannel.sigAutoFocusRunning.emit(False)
                return None

            # Stage 2: Fine scan if enabled
            if twoStage and self._getAutofocusState() != AutofocusState.ABORTED:
                # Fine scan with 10x finer parameters around the coarse best position
                twoStageDivisor = max(2, twoStageDivisor)  # Ensure minimum divisor of 2
                fine_rangez = rangez / float(twoStageDivisor)
                fine_resolutionz = resolutionz / float(twoStageDivisor)
                self.__logger.info(f"Starting autofocus - Stage 2: Fine scan (range=±{fine_rangez}, resolution={fine_resolutionz}, divisor={twoStageDivisor}) around z={best_z_coarse}")

                # Move to coarse best position first (clamped)
                clamped_coarse = self._clampPosition(best_z_coarse, axis)
                self.stages.move(value=clamped_coarse, axis=axis, is_absolute=True, is_blocking=True)
                time.sleep(tSettle * 2)

                # Check abort again
                if self._getAutofocusState() == AutofocusState.ABORTED:
                    self.__logger.info("Autofocus aborted before fine scan")
                    self._commChannel.sigAutoFocusRunning.emit(False)
                    return best_z_coarse

                # Perform fine scan centered at best_z_coarse
                best_z_fine = self._doSingleAutofocusScan(fine_rangez, fine_resolutionz, defocusz, axis, tSettle, isDebug, nGauss, nCropsize, focusAlgorithm, static_offset, center_position=clamped_coarse)

                if best_z_fine is None or self._getAutofocusState() == AutofocusState.ABORTED:
                    # If fine scan failed or aborted, use coarse result
                    final_z = best_z_coarse
                else:
                    final_z = best_z_fine
                    self.__logger.info(f"Two-stage autofocus complete: Coarse={best_z_coarse:.2f}, Fine={best_z_fine:.2f}")
            else:
                final_z = best_z_coarse

            self._setAutofocusState(AutofocusState.FINISHED)
            self._commChannel.sigAutoFocusRunning.emit(False)
            self.sigUpdateFocusValue.emit({"bestzpos": final_z})
            self.__logger.info(f"Autofocus Value is: {final_z:.2f}")
            return final_z
        except Exception as e:
            self.__logger.error(f"Autofocus error: {e}")
            self._setAutofocusState(AutofocusState.ERROR)
            self._commChannel.sigAutoFocusRunning.emit(False)
            return None

    def _doSingleAutofocusScan(self, rangez:float, resolutionz:float, defocusz:float, axis:str, tSettle:float, isDebug:bool, nGauss:int, nCropsize:int, focusAlgorithm:str, static_offset:float, center_position:float=None):
        """
        Perform a single autofocus scan with position validation and clamping.

        Args:
            rangez: Z-range to scan (±rangez from center position)
            resolutionz: Step size in Z
            defocusz: Defocus offset (currently unused)
            axis: Axis to scan (default "Z")
            tSettle: Settling time between steps
            isDebug: Save debug images if True
            nGauss: Gaussian blur sigma
            nCropsize: Crop size for focus calculation
            focusAlgorithm: Focus measurement method
            static_offset: Static offset to add to final position
            center_position: Center position for scan (None = current position, validated)

        Returns:
            Best Z position found, or None on error
        """
        try:
            mProcessor = FrameProcessor(nGauss=nGauss, nCropsize=nCropsize, isDebug=isDebug, focusMethod=focusAlgorithm)

            # Get center position for scan with validation
            if center_position is None:
                center_position, is_valid, error_msg = self._getSafeCurrentZ(axis)
                if not is_valid:
                    self.__logger.error(f"Cannot perform autofocus scan: {error_msg}")
                    return None
            else:
                # Validate provided center position
                center_position = self._clampPosition(center_position, axis)

            # Calculate scan positions with clamping
            Nz = int(max(5, np.floor((abs(rangez)) / max(1e-6, abs(resolutionz))) + 1))
            relative_positions = np.linspace(-abs(rangez/2), abs(rangez/2), Nz).astype(float)
            absolute_positions = relative_positions + center_position

            # Clamp all positions to safe bounds
            absolute_positions = np.array([self._clampPosition(p, axis) for p in absolute_positions])

            self.__logger.debug(f"Scan positions: {absolute_positions[0]:.2f} to {absolute_positions[-1]:.2f} ({Nz} steps)")

            # Check abort before starting motion
            if self._getAutofocusState() == AutofocusState.ABORTED:
                mProcessor.stop()
                return None

            # Move to start position (already clamped)
            self.stages.move(value=absolute_positions[0], axis=axis, is_absolute=True, is_blocking=True)
            time.sleep(tSettle)  # allow some settling time

            # Scan through positions
            for iz in range(Nz):
                # Check abort flag
                if self._getAutofocusState() == AutofocusState.ABORTED:
                    self.__logger.info(f"Autofocus scan aborted at step {iz}/{Nz}")
                    mProcessor.stop()
                    # Return to center position on abort
                    try:
                        self.stages.move(value=center_position, axis=axis, is_absolute=True, is_blocking=True)
                    except Exception:
                        pass
                    return center_position

                if iz != 0:
                    self.stages.move(value=absolute_positions[iz], axis=axis, is_absolute=True, is_blocking=True)
                    time.sleep(tSettle)

                frame = self.grabCameraFrame()  # works for mono and RGB; processor handles channels
                if frame is None:
                    self.__logger.warning(f"Failed to grab frame at step {iz}")
                    # Submit a None so the index slot is filled with NaN
                    mProcessor.add_frame(None, iz)
                    continue

                if isDebug:
                    import tifffile as tif
                    # Save raw frames as-is (preserve original datatype)
                    if frame.dtype == np.uint8 or frame.dtype == np.uint16:
                        tif.imwrite("autofocus_frame_z.tif", frame, append=True)
                    else:
                        # For float or other types, convert to float32
                        tif.imwrite("autofocus_frame_z.tif", frame.astype(np.float32), append=True)
                mProcessor.add_frame(frame, iz)

            # Block until all results arrive (event-based, not busy polling)
            allfocusvals = np.array(
                mProcessor.getFocusValueList(Nz, timeout=max(5.0, Nz * (tSettle + 0.5))),
                dtype=float,
            )
            mProcessor.stop()

            # Check abort before fitting
            if self._getAutofocusState() == AutofocusState.ABORTED:
                self.stages.move(value=center_position, axis=axis, is_absolute=True, is_blocking=True)
                return center_position

            # Update state to fitting
            self._setAutofocusState(AutofocusState.FITTING)

            # Move back to start position before fitting
            self.stages.move(value=absolute_positions[0], axis=axis, is_absolute=True, is_blocking=True)

            # Drop NaN entries (failed frames) — alignment is preserved by index
            valid_mask = np.isfinite(allfocusvals)
            n_valid = int(valid_mask.sum())
            if n_valid < 5:
                self.__logger.error(
                    f"Autofocus: only {n_valid}/{Nz} valid focus values — "
                    f"cannot fit. Returning to center."
                )
                self.stages.move(value=center_position, axis=axis, is_absolute=True, is_blocking=True)
                return None

            zs_valid = absolute_positions[valid_mask]
            fv_valid = allfocusvals[valid_mask]

            # Plot data
            try:
                self.sigUpdateFocusPlot.emit(zs_valid, fv_valid)
            except Exception:
                pass

            # Fit Gaussian to find best position
            x0_fit, fit_y = _robust_gaussian_fit(zs_valid, fv_valid)

            # Calculate and clamp best target position
            best_target = self._clampPosition(float(x0_fit) + static_offset, axis)

            # Check abort before final move
            if self._getAutofocusState() == AutofocusState.ABORTED:
                return center_position

            # Update state to moving to focus
            self._setAutofocusState(AutofocusState.MOVING_TO_FOCUS)

            # Move to best position
            self.stages.move(value=best_target, axis=axis, is_absolute=True, is_blocking=True)

            return best_target

        except Exception as e:
            self.__logger.error(f"Single autofocus scan error: {e}")
            return None
    # ---------- Continuous fast-sweep autofocus with time→Z mapping (no continuous Z readback) ----------
    def _doAutofocusFastBackground_timeMapped(self, sweep_range=150.0, speed=None, defocusz=0, axis=gAxis,
                                               nCropsize=2048, focusAlgorithm="LAPE", static_offset=0.0):
        self._setAutofocusState(AutofocusState.SCANNING)
        self._commChannel.sigAutoFocusRunning.emit(True)

        # Get and validate current position
        z0, is_valid, error_msg = self._getSafeCurrentZ(axis)
        if not is_valid:
            self.__logger.error(f"Cannot start fast autofocus: {error_msg}")
            self._setAutofocusState(AutofocusState.ERROR)
            self._commChannel.sigAutoFocusRunning.emit(False)
            return None

        # Calculate and clamp sweep positions
        z_start = self._clampPosition(z0 + abs(sweep_range), axis)
        z_end = self._clampPosition(z0 - abs(sweep_range), axis)
        total_dist = float(z_end - z_start)  # negative for downward sweep

        self.__logger.debug(f"Fast autofocus sweep: {z_start:.2f} to {z_end:.2f} (center: {z0:.2f})")

        # Check abort before starting
        if self._getAutofocusState() == AutofocusState.ABORTED:
            self._commChannel.sigAutoFocusRunning.emit(False)
            return None

        # Move to start
        self.stages.move(value=z_start, axis=axis, is_absolute=True, is_blocking=True)
        time.sleep(0.005)

        # Launch continuous move to end
        self._moveController.move_to_position(z_end, axis=axis, speed=speed, is_absolute=True)
        t_start = time.time()

        # Accumulate timestamps and focus values; no Z readbacks here
        t_rel_list = []
        fvals = []
        last_fn = None

        while not self._moveController.is_target_reached() and self._getAutofocusState() != AutofocusState.ABORTED:
            frame, fn = self.grabCameraFrame(returnFrameNumber=True)
            if frame is None:
                time.sleep(0.001)
                continue

            # dedupe by frame number if available
            if fn is not None:
                if last_fn is None:
                    last_fn = fn
                if fn <= last_fn:
                    time.sleep(0.0005)
                    continue
                last_fn = fn

            # Use the shared fast OpenCV-based pipeline (RGB-safe, ~10x faster)
            f_measure = _compute_focus_value_fast(
                frame,
                crop_size=nCropsize,
                binning=1,
                n_gauss=0,
                method=focusAlgorithm,
            )
            t_rel_list.append(time.time() - t_start)
            fvals.append(f_measure)

            time.sleep(0.0005)  # reduce CPU

        # Ensure motion fully completed; capture total time
        while not self._moveController.is_target_reached() and self._getAutofocusState() != AutofocusState.ABORTED:
            time.sleep(0.001)
        t_end = time.time()
        total_time = max(1e-6, t_end - t_start)

        # Handle abort
        if self._getAutofocusState() == AutofocusState.ABORTED:
            self.__logger.info("Fast autofocus aborted, returning to original position")
            # Abort the movement controller to stop any ongoing motion
            self._moveController.abort()
            self.stages.move(value=z0, axis=axis, is_absolute=True, is_blocking=True)
            self._commChannel.sigAutoFocusRunning.emit(False)
            self.sigUpdateFocusValue.emit({"bestzpos": z0})
            return z0

        # Update state to fitting
        self._setAutofocusState(AutofocusState.FITTING)

        # Map times -> Z using effective constant velocity
        v_eff = total_dist / total_time  # units of stage axis per second (likely µm/s)
        t_rel = np.asarray(t_rel_list, dtype=float)
        zs = z_start + v_eff * t_rel
        fvals = np.asarray(fvals, dtype=float)

        # Clip to [min(z_start,z_end), max(...)] to avoid small drift after finish
        zmin, zmax = (z_end, z_start) if z_end < z_start else (z_start, z_end)
        zs = np.clip(zs, zmin, zmax)

        # Plot raw
        try:
            self.sigUpdateFocusPlot.emit(zs, fvals)
        except Exception:
            pass

        # Fit Gaussian
        if len(zs) >= 5:
            x0_fit, fit_y = _robust_gaussian_fit(zs, fvals)
        else:
            x0_fit, fit_y = (float(zs[np.argmax(fvals)]) if len(zs) else z0, None)


        # Check abort before final move
        if self._getAutofocusState() == AutofocusState.ABORTED:
            self.stages.move(value=z0, axis=axis, is_absolute=True, is_blocking=True)
            self._commChannel.sigAutoFocusRunning.emit(False)
            return z0

        # Update state to moving to focus
        self._setAutofocusState(AutofocusState.MOVING_TO_FOCUS)

        # Move to best focus (absolute) with static offset and clamping
        best_target = self._clampPosition(float(x0_fit) + static_offset, axis)
        self.stages.move(value=best_target, axis=axis, is_absolute=True, is_blocking=True)

        final_z = best_target
        self._setAutofocusState(AutofocusState.FINISHED)
        self._commChannel.sigAutoFocusRunning.emit(False)
        self.sigUpdateFocusValue.emit({"bestzpos": final_z})
        return final_z


class FrameProcessor:
    """
    Background frame processor for autofocus scans.

    Design notes / fixes:
      * Worker thread is robust: any per-frame exception is caught and the
        frame is recorded as NaN so position/value alignment is preserved.
      * Results are stored in an index-keyed dict (``_results[iz] = value``)
        so a missing frame never silently shifts subsequent values.
      * ``getFocusValueList`` blocks on a ``threading.Event`` instead of
        polling, returning as soon as ``nFrameExpected`` results have
        arrived (or timeout).
      * Image processing uses the fast OpenCV pipeline shared with the
        live-monitoring / hill-climbing / fast-sweep paths
        (``_compute_focus_value_fast``).
    """

    def __init__(self, nGauss=0, nCropsize=2048, isDebug=False,
                 focusMethod="LAPE", binning=2, noise_threshold=0.02):
        self._logger = initLogger(self, tryInheritParent=False)
        self.isRunning = True
        self.frame_queue = queue.Queue()
        self._results = {}            # iz -> focus value (or np.nan on failure)
        self._results_lock = threading.Lock()
        self._progress_event = threading.Event()
        self._expected_count = None   # set by getFocusValueList

        self.flatFieldFrame = None
        self.nGauss = int(nGauss)
        self.nCropsize = int(nCropsize)
        self.isDebug = bool(isDebug)
        self.focusMethod = focusMethod
        self.binning = max(1, int(binning))
        self.noise_threshold = noise_threshold

        self.worker_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.worker_thread.start()

    # ----- public API -----
    def setFlatfieldFrame(self, flatfieldFrame):
        self.flatFieldFrame = flatfieldFrame

    def add_frame(self, img, iz):
        self.frame_queue.put((img, iz))

    def getFocusValueList(self, nFrameExpected, timeout=5.0):
        """
        Block until ``nFrameExpected`` results are available or timeout
        expires.  Returns a list of length ``nFrameExpected`` ordered by
        ``iz``.  Missing entries are filled with ``np.nan``.
        """
        self._expected_count = int(nFrameExpected)
        # Quick path: already done
        with self._results_lock:
            done = len(self._results) >= nFrameExpected
        if done:
            self._progress_event.set()
        else:
            self._progress_event.wait(timeout=timeout)

        with self._results_lock:
            ordered = [self._results.get(i, float('nan'))
                       for i in range(nFrameExpected)]
        return ordered

    def stop(self):
        self.isRunning = False
        # Unblock any waiter
        self._progress_event.set()
        try:
            self.worker_thread.join(timeout=0.5)
        except Exception:
            pass

    # ----- worker -----
    def _process_frames(self):
        while self.isRunning:
            try:
                item = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            img, iz = item
            try:
                value = self._process_frame(img)
            except Exception as e:
                # Critical: never let the worker die silently.  Record NaN
                # so the index alignment is preserved.
                self._logger.warning(f"Autofocus frame {iz} processing failed: {e}")
                value = float('nan')

            with self._results_lock:
                self._results[iz] = value
                if (self._expected_count is not None
                        and len(self._results) >= self._expected_count):
                    self._progress_event.set()

    def _process_frame(self, img):
        """
        Fast focus-measure pipeline (handles mono + RGB).
        Order chosen for speed: crop -> grayscale -> bin -> blur -> measure.
        """
        if img is None:
            return float('nan')

        # Optional flat-field correction (uses float math)
        if self.flatFieldFrame is not None:
            ff = self.flatFieldFrame
            img_f = img.astype(np.float32, copy=False)
            img = (img_f / (ff + 1e-6))

        focus_value = _compute_focus_value_fast(
            img,
            crop_size=self.nCropsize,
            binning=self.binning,
            n_gauss=self.nGauss,
            method=self.focusMethod,
        )

        if self.isDebug:
            try:
                import tifffile as tif
                tif.imwrite("autofocus_proc_frame.tif",
                            np.asarray(img).astype(np.float32, copy=False),
                            append=True)
            except Exception:
                pass

        return focus_value

    # ----- helpers retained for backward compatibility -----
    @staticmethod
    def extract(marray, crop_size):
        h, w = marray.shape[0], marray.shape[1]
        cs = int(min(crop_size, h, w))
        cx, cy = w // 2, h // 2
        x0 = max(0, cx - cs // 2)
        y0 = max(0, cy - cs // 2)
        return marray[y0:y0 + cs, x0:x0 + cs]

    @staticmethod
    def calculate_focus_measure_static(image, method="LAPE"):
        """
        Backward-compatible static focus-measure used by hill-climbing,
        live-monitoring and fast-sweep code paths.  Internally delegates
        to the fast pipeline if a raw frame is provided, otherwise
        operates on the already-prepared 2D array.
        """
        # Convert 3-channel -> grayscale via cv2 (much faster than np.mean)
        if image.ndim == 3:
            if image.dtype == np.uint8:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.dtype == np.uint16:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # float input -> normalize for cv2
                tmp = np.clip(image * 255.0, 0, 255).astype(np.uint8) \
                    if image.max() <= 1.0 else image.astype(np.uint8)
                gray = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Normalize dtype for OpenCV operators
        if gray.dtype == np.float32 or gray.dtype == np.float64:
            if gray.size and gray.max() <= 1.0:
                img_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
            else:
                img_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        elif gray.dtype == np.uint16:
            img_u8 = (gray >> 8).astype(np.uint8)
        elif gray.dtype != np.uint8:
            img_u8 = gray.astype(np.uint8)
        else:
            img_u8 = gray

        if method == "LAPE":
            lap = cv2.Laplacian(img_u8, cv2.CV_32F)
            return float(np.mean(lap * lap)), lap
        elif method == "GLVA":
            v = float(np.std(img_u8))
            return v, img_u8
        elif method == "JPEG":
            ok, enc = cv2.imencode('.jpg', img_u8,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            return (float(len(enc)) if ok else 0.0), (enc if ok else img_u8)
        else:
            return float(np.std(img_u8)), img_u8


def _compute_focus_value_fast(img, crop_size=2048, binning=1, n_gauss=0,
                               method="LAPE"):
    """
    Single fast pipeline used by FrameProcessor, hill-climbing, fast-sweep
    and live-monitoring paths.

    Steps:
      1. Center-crop (reduces pixels first → all later ops are cheap).
      2. RGB→gray with cv2.cvtColor (≫ faster than np.mean for uint8).
      3. Optional cv2.resize for binning.
      4. Optional cv2.GaussianBlur (≫ faster than skimage.gaussian).
      5. Focus measure on uint8 (OpenCV-native, no float round-trip).
    """
    if img is None:
        return float('nan')

    # 1) Crop first
    cropped = FrameProcessor.extract(img, crop_size)

    # 2) Grayscale
    if cropped.ndim == 3:
        if cropped.shape[2] >= 3:
            if cropped.dtype == np.uint8 or cropped.dtype == np.uint16:
                gray = cv2.cvtColor(cropped[..., :3], cv2.COLOR_RGB2GRAY)
            else:
                gray = np.mean(cropped[..., :3], axis=-1)
        else:
            gray = cropped[..., 0]
    else:
        gray = cropped

    # 3) Binning via cv2.resize (much faster than skimage.transform.resize)
    if binning > 1:
        new_h = max(1, gray.shape[0] // binning)
        new_w = max(1, gray.shape[1] // binning)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4) Convert to uint8 once for OpenCV operators (avoid repeated converts).
    if gray.dtype == np.uint8:
        gray_u8 = gray
    elif gray.dtype == np.uint16:
        gray_u8 = (gray >> 8).astype(np.uint8)
    elif gray.dtype in (np.float32, np.float64):
        # Heuristic: float images may be in [0,1] (normalized) or raw counts.
        gmax = float(gray.max()) if gray.size else 1.0
        if gmax <= 1.0:
            gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        else:
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        gray_u8 = gray.astype(np.uint8)

    # 5) Optional Gaussian blur
    if n_gauss and n_gauss > 0:
        # cv2 expects an odd kernel size; derive from sigma
        ksize = max(3, int(2 * round(3 * n_gauss) + 1))
        if ksize % 2 == 0:
            ksize += 1
        gray_u8 = cv2.GaussianBlur(gray_u8, (ksize, ksize), float(n_gauss))

    # 6) Focus measure
    if method == "LAPE":
        lap = cv2.Laplacian(gray_u8, cv2.CV_32F)
        return float(np.mean(lap * lap))
    elif method == "GLVA":
        return float(np.std(gray_u8))
    elif method == "JPEG":
        ok, enc = cv2.imencode('.jpg', gray_u8,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return float(len(enc)) if ok else 0.0
    else:
        return float(np.std(gray_u8))


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
