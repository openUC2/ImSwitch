from imswitch.imcommon.model import initLogger
from .PositionerManager import PositionerManager
import time
import threading
import numpy as np

MAX_ACCEL = 1000000
PHYS_FACTOR = 1
gTIMEOUT = 100
# Minimum distance (µm) Z is lifted out of the bottom before XY are homed, so the
# stage cannot knock the objective into the sample during the homing sweep.
MIN_SAFE_Z_LIFT_HOMING = 3000
# Default TMC stepper-driver parameters. Applied on launch and used as the value
# the frontend reads back (the firmware has no TMC read-back command we can use —
# /tmc_get currently crashes the HAT, so we never query the device).
#
# These defaults favour MAX TORQUE / SPEED over silent running:
#  - rms_current high (torque scales with current; raise toward the motor's rated
#    RMS — ~1.2 A suits a typical NEMA17; too high overheats the driver/motor).
#  - semin = 0 disables CoolStep, so the driver always delivers the full run
#    current instead of throttling it back at low load (max holding/moving torque).
#  - spreadCycle chopper (toff/blank_time) rather than stealthChop — set in
#    firmware; stealthChop is quieter but gives less torque at speed.
TMC_DEFAULTS = {
    "msteps": 16,         # microsteps (must match the stepSize calibration)
    "rms_current": 1200,  # mA RMS — raise toward the motor's rated current for more torque
    "sgthrs": 10,         # StallGuard threshold (sensorless homing)
    "semin": 0,           # 0 = CoolStep OFF -> always full current (max torque)
    "semax": 0,           # CoolStep upper threshold (unused when semin = 0)
    "blank_time": 24,     # spreadCycle comparator blank time
    "toff": 3,            # spreadCycle off time (chopper enabled)
}
# Firmware stepper ids (uc2-esp32 enum Stepper: A=0, X=1, Y=2, Z=3, B..G=4..9; ids >= 4 are
# usually CAN satellites). A setup adds or remaps names with managerProperties "axisIds",
# e.g. {"B": 4, "T": 5}; the legacy "axisOrder" lists the ids of A, X, Y, Z.
DEFAULT_AXIS_IDS = {"A": 0, "X": 1, "Y": 2, "Z": 3, "B": 4, "C": 5, "D": 6, "E": 7, "F": 8, "G": 9}

class ESP32StageManager(PositionerManager):
    def __init__(self, positionerInfo, name, **lowLevelManagers):
        super().__init__(positionerInfo, name, initialPosition={axis: 0 for axis in positionerInfo.axes}, initialSpeed={axis: 0 for axis in positionerInfo.axes})
        self._rs232manager = lowLevelManagers['rs232sManager'][positionerInfo.managerProperties['rs232device']]
        self._commChannel = lowLevelManagers['commChannel']
        self.__logger = initLogger(self, instanceName=name)
        self._name = name
        # Grab motor object
        self._motor = self._rs232manager._esp32.motor
        self._homeModule = self._rs232manager._esp32.home
        # Strobed sweep (see startStrobeSweep): axis of the running sweep and
        # the frame-converting wrappers handed to uc2rest, keyed by callback.
        self._strobeAxis = "X"
        self._strobeCallbacks = {}
        mp = positionerInfo.managerProperties

        # Axis name -> firmware stepperid. Several stages can share one board (e.g. the
        # XYZ stage and a CAN satellite with axes B-E) as long as their ids differ:
        # every call into uc2rest passes ids, never names.
        self.axisOrder = mp.get('axisOrder', [0, 1, 2, 3])
        self.axisIds = dict(DEFAULT_AXIS_IDS)
        self.axisIds.update(zip(("A", "X", "Y", "Z"), self.axisOrder))
        self.axisIds.update({str(k).upper(): int(v) for k, v in mp.get('axisIds', {}).items()})

        # Dual Axis if we have A and Z to drive the motor
        self.isDualAxis = mp.get("isDualaxis", False)
        # Axes this stage configures: the setup's axes, plus A (the second Z motor) on a dual-axis stage
        self._axes = list(positionerInfo.axes) + (["A"] if self.isDualAxis and "A" not in positionerInfo.axes else [])
        unknown = [ax for ax in self._axes if ax not in self.axisIds]
        if unknown:
            raise ValueError(f"{name}: no stepperid for axes {unknown}; add them to managerProperties.axisIds")

        def perAxis(key, default):
            # per-axis setup values are named <key><axis>, e.g. stepsizeX, homeSpeedB
            return {ax: mp.get(f'{key}{ax}', default) for ax in self._axes}

        # global offset (i.e. difference between stage zero and device zero)
        self.stageOffsetPositions = {ax: positionerInfo.stageOffsets.get(f'stageOffsetPosition{ax}', 0) for ax in self._axes}

        # Calibrated stepsizes in µm/step
        self.stepSizes = perAxis('stepsize', 1)
        if self.isDualAxis and "Z" in self.stepSizes:
            self.stepSizes["A"] = self.stepSizes["Z"]

        # Soft limits in µm, checked by move() when limit<axis>enabled is set
        self.minPos = perAxis('min', -np.inf)
        self.maxPos = perAxis('max', np.inf)
        self.limitEnabled = {ax: mp.get(f'limit{ax}enabled', False) for ax in self._axes}
        # Hardware hard limits - enable physical endstop protection per axis (default True)
        self.hardLimitsEnabled = perAxis('hardLimitsEnabled', True)

        # Calibrated backlash in steps
        self.backlash = perAxis('backlash', 0)

        # maximum / default speed and acceleration per axis
        self.maxSpeed = perAxis('maxSpeed', 10000)
        self._speed = {ax: mp.get(f'speed{ax}', mp.get(f'initialSpeed{ax}', 10000)) for ax in self._axes}
        self.acceleration = {ax: MAX_ACCEL for ax in self._axes}

        self.sampleLoadingPositions = perAxis('sampleLoadingPosition', 0)

        # transportation position - stage is moved here so the locking
        # blocks can be bolted on before shipping. Loaded from the setup JSON and
        # persisted back via PositionerController.setTransportPosition.
        self.transportPositions = {ax: mp.get(f'transportPosition{ax}', {"X": 65000, "Y": 40000}.get(ax, 0)) for ax in self._axes}

        # move z before homing? Sign encodes the safe (away-from-sample) direction.
        # The frame-homing procedure always lifts Z by at least MIN_SAFE_Z_LIFT_HOMING
        # before homing XY so the stage cannot drive the objective into the sample.
        self._safeDistanceZHoming = mp.get('safeDistanceZHoming', 0)

        # frame-homing runtime state (cancellable + observable by the frontend)
        self._frameHomingCancel = threading.Event()
        self._frameHomingThread = None
        self._frameHomingState = {
            "active": False,
            "cancelled": False,
            "phase": "idle",
            "axes": {"Z": "idle", "X": "idle", "Y": "idle", "A": "idle"},
            "message": "",
        }

        # Z-stage synchronisation runtime state. The two Z motors can drift apart
        # if one loses steps; this procedure drives Z into the mechanical stop to
        # mechanically re-sync them. The 'out' direction always follows the Z home
        # direction; only the default travel distance comes from config.
        self._zSyncCancel = threading.Event()
        self._zSyncThread = None
        self._zSyncDefaultSteps = mp.get('zSyncSteps', 10000)
        self._zSyncState = {
            "active": False,
            "cancelled": False,
            "phase": "idle",
            "message": "",
            "steps": self._zSyncDefaultSteps,
        }

        # Homing per axis: endstop homing in the firmware ("enabled") or a blind
        # move of "steps"; "hard" = home, ram the mechanical stop to square dual
        # motors, back off, re-home ("onStart": home when ImSwitch starts)
        self.homeParams = {}
        for ax in self._axes:
            self.setHomeParametersAxis(axis=ax, speed=mp.get(f'homeSpeed{ax}', 15000),
                                       direction=mp.get(f'homeDirection{ax}', -1),
                                       endstoppolarity=mp.get(f'homeEndstoppolarity{ax}', 1),
                                       endposrelease=mp.get(f'homeEndposRelease{ax}', 0),
                                       timeout=mp.get(f'homeTimeout{ax}', 20000))
            self.homeParams[ax].update(enabled=mp.get(f'home{ax}enabled', False),
                                       onStart=mp.get(f'homeOnStart{ax}', 0),
                                       steps=mp.get(f'homeSteps{ax}', 0),
                                       hard=mp.get(f'homeHard{ax}', False))

        # CoreXY geometry: X/Y targets are converted in _moveDevice
        self.isCoreXY = mp.get('isCoreXY', False)
        # uc2rest's own A/X/Y/Z names (used by its name-based helpers, e.g. stage scans)
        if 'axisOrder' in mp:
            self._motor.setMotorAxisOrder(order=self.axisOrder)

        # Enable motors
        self.is_enabled = mp.get('isEnable', True)
        self.enableauto = mp.get('enableauto', True)
        self.enalbeMotors(enable=self.is_enabled, enableauto=self.enableauto)

        # Setup motors (step size, direction, backlash per stepperid in uc2rest)
        for ax in self._axes:
            self.setupMotor(self.minPos[ax], self.maxPos[ax], self.stepSizes[ax], self.backlash[ax], ax)

        # get bootup position and write to GUI
        self._position = self.getPosition()

        # Setup Motor drivers (TMC - if available)
        # Cache the TMC settings per axis (config values override the defaults).
        # This cache is the source of truth the frontend reads back, because the
        # firmware has no reliable TMC read-back command. The values are pushed to
        # the device below for any axis that is TMC-configured.
        self._tmcSettings = {}
        for _ax in self._axes:
            self._tmcSettings[_ax] = {
                key: mp.get(f'{key}{_ax}', default)
                for key, default in (
                    ('msteps', TMC_DEFAULTS['msteps']),
                    ('rms_current', TMC_DEFAULTS['rms_current']),
                    ('sgthrs', TMC_DEFAULTS['sgthrs']),
                    ('semin', TMC_DEFAULTS['semin']),
                    ('semax', TMC_DEFAULTS['semax']),
                    ('blank_time', TMC_DEFAULTS['blank_time']),
                    ('toff', TMC_DEFAULTS['toff']),
                )
            }

        # Apply the cached TMC settings (config merged over TMC_DEFAULTS) to each
        # axis on launch. TMC_DEFAULTS is the single source of defaults — there
        # are no per-call hardcoded fallbacks to drift out of sync.
        try:
            for _ax in self._axes:
                if mp.get(f"msteps{_ax}") is None:
                    continue
                self.setupMotorDriver(axis=_ax, timeout=1, **self._tmcSettings[_ax])
        except Exception as e:
            self.__logger.warning(f"Could not apply TMC settings: {e}")

        # Load joystick-direction inversion per axis from config. The key
        # ``joystickInverted<AXIS>`` maps directly to managerProperties so the
        # same JSON field that holds TMC / homing parameters also holds these.
        self._joystickSettings: dict = {
            _ax: bool(mp.get(f'joystickInverted{_ax}', False)) for _ax in self._axes}
        # Apply only axes that have an explicit entry in the config (same
        # guard used for TMC) so a factory-default config does not spam the
        # device with unnecessary writes.
        try:
            for _ax in self._axes:
                if mp.get(f'joystickInverted{_ax}') is None:
                    continue
                self._motor.set_joystick_direction(
                    axis=self.axisIds[_ax], inverted=self._joystickSettings[_ax], timeout=1
                )
        except Exception as e:
            self.__logger.warning(f"Could not apply joystick direction settings: {e}")

        # Load joystick-jog speed multiplier per axis from config, same
        # pattern as the joystick-direction cache above.
        self._speedMultiplier: dict = perAxis('speedMultiplier', 1)
        try:
            for _ax in self._axes:
                if mp.get(f'speedMultiplier{_ax}') is None:
                    continue
                self._motor.set_speed_multiplier(
                    axis=self.axisIds[_ax], multiplier=self._speedMultiplier[_ax], timeout=1
                )
        except Exception as e:
            self.__logger.warning(f"Could not apply speed multiplier settings: {e}")

        # Dummy move to get the motor to the right position
        for iAxis in positionerInfo.axes:
            self.move(value=-1, speed=1000, axis=iAxis, is_absolute=False, is_blocking=True, isEnable=True, timeout=0.2)
            self.move(value=1, speed=1000, axis=iAxis, is_absolute=False, is_blocking=True, isEnable=True, timeout=0.2)

        # optional: home on startup
        for ax in self._axes:
            if self.homeParams[ax]["onStart"]:
                self._homeAxis(ax)
                time.sleep(0.5)

        # Apply hard-limit settings to hardware for all axes
        try:
            for _axis in self._axes:
                self._motor.set_hard_limits(axis=self.axisIds[_axis], enabled=self.hardLimitsEnabled[_axis])
        except Exception as _e:
            self.__logger.warning(f"Could not apply hard limits on startup: {_e}")

        # save z-position prior to homing
        self._zPositionPriorHoming = self._position.get("Z", 0)

        # try to register the callback
        try:
            # every stage on this board gets the position array (indexed by stepperid)
            self._motor.register_callback(self._name, callbackfct=self.setPositionFromDevice)
        except Exception as e:
            self.__logger.error(f"Could not register callback: {e}")

        # do frame homing if enabled
        if mp.get('frameHomeOnStart', 0):
            self.frameHomingProcedure(False)

    def setHomeParametersAxis(self, axis, speed, direction, endstoppolarity, endposrelease, timeout=None):
        self.homeParams.setdefault(axis, {}).update(
            speed=speed, direction=1 if direction > 0 else -1, endstoppolarity=endstoppolarity,
            endposrelease=endposrelease, timeout=timeout)

    def setAxisOrder(self, order=[0,1,2,3]):
        """Legacy: order lists the stepperids of A, X, Y, Z."""
        self.axisOrder = list(order)
        self.axisIds.update(zip(("A", "X", "Y", "Z"), order))
        self._motor.setMotorAxisOrder(order=order)
        for ax in self._axes:
            self.setupMotor(self.minPos[ax], self.maxPos[ax], self.stepSizes[ax], self.backlash[ax], ax)

    def enalbeMotors(self, enable=None, enableauto=None):
        """
        enable - Enable Motors (i.e. switch on/off power to motors)
        enableauto - Enable automatic motor power off after motors are not used for a while; will be turned on automatically
        """
        self._motor.set_motor_enable(enable=enable, enableauto=enableauto)

    def setupMotor(self, minPos, maxPos, stepSize, backlash, axis):
        self._motor.setup_motor(axis=self.axisIds[axis], minPos=minPos, maxPos=maxPos, stepSize=stepSize, backlash=backlash)

    def setBacklash(self, axis, backlashUm):
        """Set an axis' backlash from a value in microns (e.g. a camera measurement).

        Converts microns to hardware steps with the configured per-axis step size
        and pushes it to the motor, which applies it as a reversal overshoot. The
        cached ``backlash[axis]`` is updated so a later ``setupMotor`` keeps it.
        Note: the value is *not* written back to the saved config here.
        """
        axis = str(axis).upper()
        stepSize = self.stepSizes.get(axis, 1) or 1
        backlashSteps = int(round(float(backlashUm) / stepSize))
        self._motor.set_backlash(axis=self.axisIds[axis], backlash=backlashSteps)
        self.backlash[axis] = backlashSteps
        return {"axis": axis, "backlashUm": float(backlashUm), "backlashSteps": backlashSteps}

    def setupMotorDriver(self, axis="X", msteps=None, rms_current=None, stall_value=None, sgthrs=None, semin=None, semax=None, blank_time=None, toff=None, timeout=0.1):
        axis = axis.upper()
        self._motor.set_tmc_parameters(axis=self.axisIds[axis], msteps=msteps, rms_current=rms_current, stall_value=stall_value, sgthrs=sgthrs, semin=semin, semax=semax, blank_time=blank_time, toff=toff, timeout=timeout)
        # Remember what we applied so the frontend can read it back even though
        # the firmware has no TMC read-back command.
        cache = self._tmcSettings.setdefault(axis, dict(TMC_DEFAULTS))
        for key, value in (('msteps', msteps), ('rms_current', rms_current),
                           ('sgthrs', sgthrs), ('semin', semin), ('semax', semax),
                           ('blank_time', blank_time), ('toff', toff)):
            if value is not None:
                cache[key] = value

    @property
    def combinedAxes(self):
        """Firmware moves X/Y (and X/Y/Z) as one coordinated motion."""
        return ["XY", "XYZ"]

    def move(self, value=0, axis="X", is_absolute=False, is_blocking=True, acceleration=None, speed=None, isEnable=None, timeout=gTIMEOUT, is_reduced=True):
        '''
        Move the motor to a new position
        :param value: The new position (µm); a tuple for a combined axis ("XY", "XYZ")
        :param axis: The axis to move: one of this stage's axes ("X", "B", ...) or a combined axis
        :param is_absolute: If True, the motor will move to the absolute position given by value. If False, the motor will move by the amount given by value.
        :param is_blocking: If True, the function will block until the motor has reached the new position. If False, the function will return immediately.
        :param acceleration: The acceleration to use for the move (one value or a tuple). If None, the default acceleration for the axis will be used.
        :param speed: The speed to use for the move (one value or a tuple). If None, the default speed for the axis will be used.
        :param isEnable: If True, the motor will be enabled before the move. If False, the motor will be disabled before the move.
        :param timeout: The maximum time to wait for the motor to reach the new position. If the motor has not reached the new position after this time, a TimeoutError will be raised.
        '''
        if isEnable is None:
            isEnable = self.is_enabled
        axes = list(axis) if axis in self.combinedAxes else [axis]
        if any(ax not in self._axes for ax in axes):
            self.__logger.error(f'Wrong axis {axis}: this stage has {self._axes} and {self.combinedAxes}')
            return

        def perMove(v, default):
            if v is None:
                return [default[ax] for ax in axes]
            return list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v] * len(axes)

        values = list(value) if len(axes) > 1 else [value]
        speeds = perMove(speed, self.speed)
        accelerations = perMove(acceleration, self.acceleration)
        if any(s <= 0 for s in speeds):
            self.__logger.error(f'Speed for {axis} has to be > 0')
            return
        if not is_absolute and all(v == 0 for v in values):
            return
        # don't move to invalid positions (check both min and max limits)
        for ax, v in zip(axes, values):
            target = v if is_absolute else self._position.get(ax, 0) + v
            if self.limitEnabled[ax] and not self.minPos[ax] <= target <= self.maxPos[ax]:
                self.__logger.warning(f"{ax} move blocked: target {target} outside [{self.minPos[ax]}, {self.maxPos[ax]}]")
                return
        # Apply offset for absolute moves: convert from user position to device position
        device = {ax: v + self.stageOffsetPositions[ax] if is_absolute else v for ax, v in zip(axes, values)}
        self._moveDevice(device, dict(zip(axes, speeds)), dict(zip(axes, accelerations)), is_absolute=is_absolute,
                         is_blocking=is_blocking, isEnable=isEnable, timeout=timeout, is_reduced=is_reduced)
        for ax, v in zip(axes, values):
            self._position[ax] = v if is_absolute else self._position.get(ax, 0) + v

    def _moveDevice(self, targets, speed, acceleration=None, is_absolute=False, is_blocking=True, isEnable=None, timeout=gTIMEOUT, is_reduced=False):
        """One move in device coordinates (no offsets, no soft limits).

        targets: {axis: value}; speed/acceleration: {axis: value} or one value for all.
        Applies the CoreXY transform to X/Y and lets A follow Z on a dual-axis stage.
        """
        if isEnable is None:
            isEnable = self.is_enabled
        targets = dict(targets)
        speed = dict(speed) if isinstance(speed, dict) else {ax: speed for ax in targets}
        acceleration = dict(acceleration) if isinstance(acceleration, dict) else {ax: acceleration for ax in targets}
        if self.isCoreXY and ("X" in targets or "Y" in targets):
            # both belts move for X and for Y
            targets["X"], targets["Y"] = self._motor.cartesian2corexy(targets.get("X", 0), targets.get("Y", 0))
            speed["X"] = speed["Y"] = speed.get("X", speed.get("Y"))
            acceleration["X"] = acceleration["Y"] = acceleration.get("X", acceleration.get("Y"))
        if self.isDualAxis and "Z" in targets:
            targets["A"], speed["A"], acceleration["A"] = targets["Z"], speed["Z"], acceleration.get("Z")
        ids = {ax: self.axisIds[ax] for ax in targets}
        self._motor.move_axes({ids[ax]: v for ax, v in targets.items()},
                              speed={ids[ax]: speed[ax] for ax in targets},
                              acceleration={ids[ax]: acceleration.get(ax) for ax in targets},
                              is_absolute=is_absolute, is_blocking=is_blocking, is_enabled=isEnable,
                              timeout=timeout, is_reduced=is_reduced)

    def measure(self, sensorID=0, NAvg=100):
        return self._motor.read_sensor(sensorID=sensorID, NAvg=NAvg)

    def setupPIDcontroller(self, PIDactive=1, Kp=100, Ki=10, Kd=1, target=500, PID_updaterate=200):
        return self._motor.set_pidcontroller(PIDactive=PIDactive, Kp=Kp, Ki=Ki, Kd=Kd, target=target,
                                             PID_updaterate=PID_updaterate)

    def moveForeverByAxis(self, speed=0, axis="X", is_stop=False):
        self._motor.move_forever(speed={self.axisIds[axis]: speed}, is_stop=is_stop)

    def moveForever(self, speed=(0, 0, 0, 0), is_stop:bool=False):
        """speed: {axis: speed} or the legacy tuple (A, X, Y, Z)."""
        if not isinstance(speed, dict):
            speed = dict(zip(("A", "X", "Y", "Z"), speed))
        self._motor.move_forever(speed={self.axisIds[ax]: v for ax, v in speed.items()}, is_stop=is_stop)

    def setEnabled(self, is_enabled):
        self.is_enabled = is_enabled

    def setSpeed(self, speed, axis=None):
        # TODO: Map that to the JSON!
        if speed is None:
            return
        if axis is None:
            for ax in self._speed:
                self._speed[ax] = speed
        else:
            self._speed[axis] = speed

    def setPosition(self, value, axis):
        # print(f"setPosition - Axis: {axis} -> New Value: {value}")
        self._position[axis] = value

    def setPositionOnDevice(self, value, axis):
        self.setPosition(value, axis)
        self._motor.set_position(self.axisIds[axis], value)

    def setPositionFromDevice(self, positionArray: np.array):
        ''' mostly used for he position callback
        If new positions are coming from the device they will be updated in ImSwitch too.
        positionArray is indexed by stepperid (uc2rest). '''
        posDict = {self._name: {}}
        for axisName in self._axes:
            positionOffsetCorrected = positionArray[self.axisIds[axisName]] - self.getStageOffsetAxis(axisName)
            self.setPosition(positionOffsetCorrected, axisName)
            posDict[self._name][axisName] = positionOffsetCorrected
        self._commChannel.sigUpdateMotorPosition.emit(posDict)

    def closeEvent(self):
        pass

    def _readDevicePositions(self):
        """Device positions as an array indexed by stepperid. Ids 0-3 are the board's own
        axes, which a bare /motor_get lists; other ids (CAN) are asked for explicitly."""
        ids = [self.axisIds[ax] for ax in self._axes]
        return 1. * self._motor.get_position(axis=None if max(ids) < 4 else ids)

    def getPosition(self):
        # load position from device
        try:
            allPositions = self._readDevicePositions()
            return {ax: allPositions[self.axisIds[ax]] - self.getStageOffsetAxis(ax) for ax in self._axes}
        except Exception as e:
            self.__logger.error(e)
            return self._position


    def forceStop(self, axis):
        if axis in self._axes:
            self._motor.stop(axis=self.axisIds[axis])
        else:
            self.stopAll()

    def get_abs(self, axis):
        return self._position[axis]

    def stopAll(self):
        self._motor.stop(axis=[self.axisIds[ax] for ax in self._axes])

    def doHome(self, axis, isBlocking=False, homeDirection=None, homeSpeed=None, homeEndstoppolarity=None, homeEndposRelease=None, homeTimeout=None, hardHome=None):
        hp = self.homeParams.get(axis)
        if hp and (hp["enabled"] or abs(hp["steps"]) > 0):
            self._homeAxis(axis, isBlocking, hardHome)

    def _homeAxis(self, axis, isBlocking=False, hardHome=None):
        hp = self.homeParams[axis]
        # move z prior to homing X/Y?
        if axis in ("X", "Y") and self._safeDistanceZHoming != 0 and "Z" in self.homeParams:
            self.move(value=self._zPositionPriorHoming + self._safeDistanceZHoming, speed=self.homeParams["Z"]["speed"], axis="Z", is_absolute=True, is_blocking=True)
        if abs(hp["steps"]) > 0:
            # no endstop: move by homeSteps, release, call that 0
            self.move(value=hp["steps"], speed=hp["speed"], axis=axis, is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(hp["steps"])*np.abs(hp["endposrelease"]), speed=hp["speed"], axis=axis, is_absolute=False, is_blocking=True)
            self.setPosition(axis=axis, value=0)
            self.setPositionOnDevice(value=0, axis=axis)
        elif hp["enabled"]:
            hard = hp["hard"] if hardHome is None else hardHome
            self._homeModule.home(axis=self.axisIds[axis], speed=hp["speed"], direction=hp["direction"],
                                  endstoppolarity=hp["endstoppolarity"], endposrelease=hp["endposrelease"],
                                  endstoptimeout=hp["timeout"], isBlocking=isBlocking, hardhome=bool(hard))
        else:
            self.__logger.info(f"No homing parameters set for {axis} axis or not enabled in settings.")
            return
        if axis == "Z":
            self._zPositionPriorHoming = 0

    # per-axis shortcuts used by other controllers
    def home_x(self, isBlocking=False, *args, **kwargs):
        self._homeAxis("X", isBlocking)

    def home_y(self, isBlocking=False, *args, **kwargs):
        self._homeAxis("Y", isBlocking)

    def home_z(self, isBlocking=False, *args, hardHome=None, **kwargs):
        self._homeAxis("Z", isBlocking, hardHome)

    def home_a(self, isBlocking=False, *args, **kwargs):
        self._homeAxis("A", isBlocking)

    def startStageScanning(self, nStepsLine=100, dStepsLine=1, nTriggerLine=1, nStepsPixel=100, dStepsPixel=1, nTriggerPixel=1, delayTimeStep=10, nFrames=5, isBlocking=False):
        self._motor.startStageScanning(nStepsLine=nStepsLine, dStepsLine=dStepsLine, nTriggerLine=nTriggerLine,
                                       nStepsPixel=nStepsPixel, dStepsPixel=dStepsPixel, nTriggerPixel=nTriggerPixel,
                                       delayTimeStep=delayTimeStep, nFrames=nFrames, isBlocking=isBlocking)

    def stopStageScanning(self):
        self._motor.stopStageScanning()

    def moveToSampleMountingPosition(self, speed=10000, is_blocking=True):
        value = {ax: self.sampleLoadingPositions[ax] for ax in ("X", "Y", "Z")}
        self._moveDevice(value, speed, is_absolute=True, is_blocking=is_blocking)

    def getDevicePositionAxis(self, axis="X"):
        """Live raw device position for one axis (no offset applied).

        Reads directly from the ESP32 firmware; the firmware preserves position
        across software restarts (steps survive reboot) so this is the stable
        reference for offset calibration.
        """
        try:
            return float(self._readDevicePositions()[self.axisIds[axis]])
        except Exception as e:
            self.__logger.error(f"getDevicePositionAxis({axis}) failed: {e}")
            return float(self._position.get(axis, 0)) + float(self.stageOffsetPositions.get(axis, 0))

    def setStageOffsetAxis(self, knownPosition=0, currentDevicePosition=None,
                            knownOffset=None, axis="X"):
        """Set the stage offset for one axis using the canonical contract.

        ``offset = currentDevicePosition - knownPosition`` so that the user
        coordinate equals ``knownPosition`` at the current physical place.
        ``currentDevicePosition`` is provided by the controller to avoid a
        race with asynchronous position updates; falling back to a live read
        is only a convenience for direct callers.
        """
        if knownOffset is not None:
            offset = float(knownOffset)
        else:
            if currentDevicePosition is None:
                currentDevicePosition = self.getDevicePositionAxis(axis)
            offset = float(currentDevicePosition) - float(knownPosition)
        self.stageOffsetPositions[axis] = offset
        self.__logger.info(
            f"Set offset for {axis} axis to {offset} um "
            f"(device={currentDevicePosition}, known={knownPosition}, knownOffset={knownOffset})."
        )

    def resetStageOffsetAxis(self, axis="X"):
        """Reset the stage offset for the given axis to 0."""
        self.__logger.info(f"Resetting stage offset for {axis} axis.")
        self.stageOffsetPositions[axis] = 0

    def getStageOffsetAxis(self, axis:str="X"):
        """ Get the current stage offset for a given axis.
        If no axis is given, the current stage is used.
        """
        try:
            return self.stageOffsetPositions[axis]
        except KeyError:
            self.__logger.error(f"Axis {axis} not found in stageOffsetPositions.")
            return 0

    def start_stage_scanning(self, xstart=0, xstep=1, nx=100,
                             ystart=0, ystep=1, ny=100,
                             zstart=0, zstep=0, nz=1,
                             tsettle=0.1, tExposure=50, illumination=None, led=None,
                             speed=20000, acceleration=None, tTrig=None):
        """
        Start a stage scanning operation with the given parameters.
        
        :param xstart: Starting position in X direction.
        :param xstep: Step size in X direction.
        :param nx: Number of steps in X direction.
        :param ystart: Starting position in Y direction.
        :param ystep: Step size in Y direction.
        :param ny: Number of steps in Y direction.
        :param zstart: Starting position in Z direction.
        :param zstep: Step size in Z direction (0 = no Z-stacking).
        :param nz: Number of steps in Z direction (1 = single plane).
        :param tsettle: Settle time after each step (ms).
        :param tExposure: Exposure time at each position (ms).
        :param illumination: Optional illumination settings tuple (4 values).
        :param led: Optional LED intensity (0-255).
        :param speed: Motor speed for scanning.
        :param acceleration: Motor acceleration (None = default).
        :param tTrig: Camera trigger pulse width (ms); None = firmware default.
        """
        if illumination is None:
            illumination = (0, 0, 0, 0)  # Default to no illumination
        if led is None:
            led = 0
        r = self._motor.start_stage_scanning(
            xstart=xstart, xstep=xstep, nx=nx,
            ystart=ystart, ystep=ystep, ny=ny,
            zstart=zstart, zstep=zstep, nz=nz,
            tsettle=tsettle, tExposure=tExposure,
            illumination=illumination, led=led,
            speed=speed, acceleration=acceleration, tTrig=tTrig,
        )
        return r

    def stop_stage_scanning(self):
        """
        Stop the current stage scanning operation.
        """
        self._motor.stop_stage_scanning()
        self.__logger.info("Stage scanning stopped.")

    def moveToSampleLoadingPosition(self, speed=10000, is_blocking=True):
        value = {ax: self.sampleLoadingPositions[ax] for ax in ("X", "Y", "Z")}
        self._moveDevice(value, speed, is_absolute=True, is_blocking=is_blocking)


    def _emitHomingState(self, phase=None, axis=None, axisStatus=None, message=None,
                         active=None, cancelled=None):
        """Update the frame-homing state dict and broadcast it to the frontend."""
        if phase is not None:
            self._frameHomingState["phase"] = phase
        if axis is not None and axisStatus is not None:
            self._frameHomingState["axes"][axis] = axisStatus
        if message is not None:
            self._frameHomingState["message"] = message
        if active is not None:
            self._frameHomingState["active"] = active
        if cancelled is not None:
            self._frameHomingState["cancelled"] = cancelled
        try:
            self._commChannel.sigHomingState.emit(dict(self._frameHomingState))
        except Exception as e:
            self.__logger.error(f"Could not emit homing state: {e}")

    def isFrameHomingActive(self):
        return self._frameHomingState.get("active", False)

    def getFrameHomingState(self):
        return dict(self._frameHomingState)

    def cancelFrameHoming(self):
        """Request cancellation of an in-progress frame-homing run."""
        if not self._frameHomingState.get("active", False):
            return
        self.__logger.info("Frame homing cancellation requested.")
        self._frameHomingCancel.set()
        try:
            self.stopAll()
        except Exception as e:
            self.__logger.error(f"stopAll during cancel failed: {e}")

    def frameHomingProcedure(self, is_blocking=False):
        '''
        Collision-safe global homing:
          1. Store current Z
          2. Home Z (establish the bottom reference)
          3. Lift Z by a safe distance (>= MIN_SAFE_Z_LIFT_HOMING) out of the bottom
          4. Home X
          5. Home Y
          6. Restore Z to the previous height (never below the safe lift)

        The run is cancellable via cancelFrameHoming() and emits per-axis progress
        through self._commChannel.sigHomingState. Runs in a background thread unless
        is_blocking=True.
        '''
        if self._frameHomingState.get("active", False):
            self.__logger.warning("Frame homing already in progress; ignoring start request.")
            return

        # Safe lift distance after homing Z. After Z hits its home endstop the
        # device position is 0 and the only valid travel is *away* from the
        # endstop (i.e. opposite to homeDirectionZ). We retract by at least
        # MIN_SAFE_Z_LIFT_HOMING so XY never sweep with the objective near the
        # sample. A non-zero configured safeDistanceZHoming overrides the
        # direction (the operator knows their geometry) but is still floored to
        # the 3 mm minimum magnitude.
        configuredLift = self._safeDistanceZHoming
        if configuredLift != 0:
            liftDirection = 1 if configuredLift > 0 else -1
            safeLift = liftDirection * max(abs(configuredLift), MIN_SAFE_Z_LIFT_HOMING)
        else:
            liftDirection = -1 if self.homeParams["Z"]["direction"] > 0 else 1
            safeLift = liftDirection * MIN_SAFE_Z_LIFT_HOMING

        def homingThreadFunction(self):
            self._frameHomingCancel.clear()
            self._frameHomingState = {
                "active": True,
                "cancelled": False,
                "phase": "starting",
                "axes": {"Z": "pending", "X": "pending", "Y": "pending", "A": "idle"},
                "message": "Starting frame homing",
            }
            self._emitHomingState()

            def aborted():
                if self._frameHomingCancel.is_set():
                    self._emitHomingState(phase="cancelled", active=False, cancelled=True,
                                          message="Frame homing cancelled")
                    self.__logger.info("Frame homing cancelled.")
                    return True
                return False

            try:
                # Step 1: store current Z so we can return to it afterwards
                self._zPositionPriorHoming = self.getPosition()["Z"]
                if aborted():
                    return

                # Step 2: home Z first (so the objective retracts before XY move)
                self._emitHomingState(phase="homing_z", axis="Z", axisStatus="homing",
                                      message="Homing Z axis")
                self.home_z(isBlocking=True)
                self._emitHomingState(axis="Z", axisStatus="done")
                if aborted():
                    return

                # Step 3: lift Z out of the bottom by the safe distance (Z ~ 0 now)
                self._emitHomingState(phase="lifting_z", message="Lifting Z to safe height")
                self.move(value=safeLift, speed=self.homeParams["Z"]["speed"], axis="Z",
                          is_absolute=True, is_blocking=True)
                if aborted():
                    return

                # Step 4: home X
                self._emitHomingState(phase="homing_x", axis="X", axisStatus="homing",
                                      message="Homing X axis")
                self.home_x(isBlocking=True)
                self.move(value=1000, speed=self.homeParams["X"]["speed"], axis="X", is_absolute=True, is_blocking=False)
                self._emitHomingState(axis="X", axisStatus="done")
                if aborted():
                    return

                # Step 5: home Y
                self._emitHomingState(phase="homing_y", axis="Y", axisStatus="homing",
                                      message="Homing Y axis")
                self.home_y(isBlocking=True)
                self.move(value=1000, speed=self.homeParams["Y"]["speed"], axis="Y", is_absolute=True, is_blocking=False)
                self._emitHomingState(axis="Y", axisStatus="done")
                if aborted():
                    return
                
                # Step 6: Moving to a save position (e.g. transport position) after homing XY to avoid collisions during Z restore
                self._emitHomingState(phase="moving_to_safe_xy", message="Moving to safe XY position")
                value = {ax: self.transportPositions[ax] for ax in ("X", "Y")}
                self._moveDevice(value, self.homeParams["X"]["speed"], is_absolute=True, is_blocking=True)

                if aborted():
                    return

                # Step 7: restore Z to its previous height
                self._emitHomingState(phase="restoring_z", message="Restoring Z position")
                self.move(value=self._zPositionPriorHoming, speed=self.homeParams["Z"]["speed"], axis="Z",
                          is_absolute=True, is_blocking=True)

                self._emitHomingState(phase="done", active=False, message="Frame homing complete")
                self.__logger.info("Frame homing procedure completed.")
            except Exception as e:
                self.__logger.error(f"Frame homing failed: {e}")
                self._emitHomingState(phase="error", active=False,
                                      message=f"Frame homing failed: {e}")

        if is_blocking:
            homingThreadFunction(self)
        else:
            self._frameHomingThread = threading.Thread(target=homingThreadFunction, args=(self,))
            self._frameHomingThread.start()

    def register_homing_callback(self, on_homing_state_fct):
        """Register a callback fired on firmware home-status messages.

        The uc2rest Home module maintains an isHomed[] array (indexed A,X,Y,Z) and
        calls the registered function with it whenever a "home" message arrives.
        This is best-effort enrichment; frameHomingProcedure remains the source of
        truth for UI progress.
        """
        try:
            self._homeModule.register_callback(0, on_homing_state_fct)
        except Exception as e:
            self.__logger.error(f"Could not register homing callback: {e}")

    def moveToTransportPosition(self, speed=10000, is_blocking=True, override_endstop_z=True):
        """Move the stage to the stored transportation (locking) position."""
        if override_endstop_z and "Z" in self._axes:
            self.__logger.warning("Overriding Z endstops to move to transport position.")
            self.hardLimitsEnabled["Z"] = False
            self._motor.set_hard_limits(axis=self.axisIds["Z"], enabled=False)

        self._moveDevice({ax: self.transportPositions[ax] for ax in ("X", "Y", "Z") if ax in self._axes},
                         speed, is_absolute=True, is_blocking=is_blocking)
        others = {ax: v for ax, v in self.transportPositions.items() if ax not in ("X", "Y", "Z")}
        if others:
            try:
                self._moveDevice(others, speed, is_absolute=True, is_blocking=is_blocking)
            except Exception as e:
                self.__logger.error(f"Could not move {list(others)} to transport position: {e}")

    def getTransportPositions(self):
        """Return the stored transportation position as an {axis: position} dict."""
        return dict(self.transportPositions)

    def setTransportPositions(self, positions=None):
        """Set the transport position.

        If ``positions`` is None the current stage pose is snapshotted; otherwise
        the provided per-axis values are used (missing axes are left unchanged).
        Returns the updated transport position dict.
        """
        if positions is None:
            current = self.getPosition()
            for ax in self._axes:
                self.transportPositions[ax] = float(current.get(ax, self.transportPositions.get(ax, 0)))
        else:
            for ax in self._axes:
                if positions.get(ax) is not None:
                    self.transportPositions[ax] = float(positions[ax])
        self.__logger.info(f"Transport position set to {self.transportPositions}")
        return dict(self.transportPositions)

    # ========================================================================
    # Z-stage synchronisation (re-sync the two Z motors against a mechanical stop)
    # ========================================================================

    def isZStageSyncActive(self):
        return self._zSyncState.get("active", False)

    def getZStageSyncState(self):
        return dict(self._zSyncState)

    def _emitZSyncState(self, phase=None, message=None, active=None, cancelled=None):
        """Update the Z-sync state dict. The frontend reads this via periodic
        polling of getZStageSyncState (no websocket signal)."""
        if phase is not None:
            self._zSyncState["phase"] = phase
        if message is not None:
            self._zSyncState["message"] = message
        if active is not None:
            self._zSyncState["active"] = active
        if cancelled is not None:
            self._zSyncState["cancelled"] = cancelled

    def cancelZStageSync(self):
        """Stop an in-progress Z-sync run and halt all motors."""
        if not self._zSyncState.get("active", False):
            return
        self.__logger.info("Z-stage sync cancellation requested.")
        self._zSyncCancel.set()
        try:
            self.stopAll()
        except Exception as e:
            self.__logger.error(f"stopAll during Z-sync cancel failed: {e}")

    def zStageSyncProcedure(self, steps=None, speed=None):
        """Re-synchronise the two Z motors against the mechanical stop.

        Two stepper motors push the Z stage up together; if one loses steps they
        desync. This drives Z 'out' (away from the home endstop) by a large
        amount so both motors stall together at the mechanical end, backs off by
        half, restores the limit switch and re-homes Z.

        steps : magnitude (µm) of the 'out' travel (default from config, 10000).
        The Z hard and soft limits are temporarily disabled and always restored,
        even on cancel/error. Runs in a background thread unless is_blocking.
        """
        if self._zSyncState.get("active", False):
            self.__logger.warning("Z-stage sync already in progress; ignoring start request.")
            return
        if steps is None:
            steps = self._zSyncDefaultSteps
        steps = abs(float(steps))
        if speed is None:
            speed = self.homeParams["Z"]["speed"]
        # The 'out' direction follows the Z home direction.
        outSign = 1 if self.homeParams["Z"]["direction"] > 0 else -1
        outMove = outSign * steps

        def threadFn(self):
            self._zSyncCancel.clear()
            # Remember where Z was so we can return there after re-homing.
            try:
                zPosBefore = self.getPosition().get("Z", 0)
            except Exception:
                zPosBefore = self._position.get("Z", 0)
            self._zSyncState = {
                "active": True,
                "cancelled": False,
                "phase": "starting",
                "message": "Starting Z-stage synchronisation",
                "steps": steps,
            }
            self._emitZSyncState()

            def aborted():
                if self._zSyncCancel.is_set():
                    self._emitZSyncState(phase="cancelled", active=False, cancelled=True,
                                         message="Z-stage sync cancelled")
                    self.__logger.info("Z-stage sync cancelled.")
                    return True
                return False

            restoreHard = self.hardLimitsEnabled["Z"]
            restoreSoft = self.limitEnabled["Z"]
            try:
                # 1) disable the Z limits so we can drive into the mechanical stop
                self._emitZSyncState(phase="disabling_limit", message="Disabling Z limit switch")
                self.hardLimitsEnabled["Z"] = False
                self.limitEnabled["Z"] = False
                self._motor.set_hard_limits(axis=self.axisIds["Z"], enabled=False)
                if aborted():
                    return

                # 2) drive Z out by the full amount (both motors stall together)
                self._emitZSyncState(phase="moving_out",
                                     message=f"Moving Z out by {steps:.0f} µm")
                self.move(value=outMove, speed=speed, axis="Z", is_absolute=False, is_blocking=True)
                if aborted():
                    return

                # 3) move back by half
                self._emitZSyncState(phase="moving_back",
                                     message=f"Moving Z back by {steps / 2:.0f} µm")
                self.move(value=-outMove / 2.0, speed=speed, axis="Z", is_absolute=False, is_blocking=True)
                if aborted():
                    return

                # 4) restore the limits before homing
                self._emitZSyncState(phase="enabling_limit", message="Re-enabling Z limit switch")
                self.hardLimitsEnabled["Z"] = restoreHard
                self.limitEnabled["Z"] = restoreSoft
                self._motor.set_hard_limits(axis=self.axisIds["Z"], enabled=restoreHard)
                if aborted():
                    return

                # 5) home Z again to re-establish the reference
                self._emitZSyncState(phase="homing", message="Homing Z axis")
                self.home_z(isBlocking=True)
                if aborted():
                    return

                # 6) return to the Z position we started from
                self._emitZSyncState(phase="restoring",
                                     message=f"Returning to previous Z ({zPosBefore:.0f} um)")
                self.move(value=zPosBefore, speed=speed, axis="Z",
                          is_absolute=True, is_blocking=True)

                self._emitZSyncState(phase="done", active=False,
                                     message="Z-stage synchronisation complete")
                self.__logger.info("Z-stage sync procedure completed.")
            except Exception as e:
                self.__logger.error(f"Z-stage sync failed: {e}")
                self._emitZSyncState(phase="error", active=False,
                                     message=f"Z-stage sync failed: {e}")
            finally:
                # Safety: never leave the Z limits disabled.
                try:
                    self.hardLimitsEnabled["Z"] = restoreHard
                    self.limitEnabled["Z"] = restoreSoft
                    self._motor.set_hard_limits(axis=self.axisIds["Z"], enabled=restoreHard)
                except Exception as e:
                    self.__logger.error(f"Could not restore Z limits after sync: {e}")

        self._zSyncThread = threading.Thread(target=threadFn, args=(self,))
        self._zSyncThread.start()

    def register_stagescan_callback(self, on_stagescan_complete_fct):
        self._motor.register_stagescan_callback(on_stagescan_complete_fct)
        
    
    def unregister_stagescan_callback(self, on_stagescan_complete_fct=None):
        # uc2rest removes one specific callback; calling it without one raised
        # a TypeError that the caller swallowed, so callbacks piled up.
        if on_stagescan_complete_fct is None:
            return
        self._motor.unregister_stagescan_callback(on_stagescan_complete_fct)
    
    def reset_stagescan_complete(self):
        pass

    # ------------------------------------------------------------------ #
    # Strobed sweep: constant-velocity line with a camera trigger, an LED
    # flash and a latched position on every CANopen SYNC (firmware module
    # "strobesweep"). Coordinates here are the user frame, like move() and
    # getPosition(); uc2rest works in the device frame (user + offset).
    # ------------------------------------------------------------------ #

    def hasStrobeSweep(self) -> bool:
        """True when uc2rest and the firmware both support the strobed sweep.

        Old uc2rest (no ``Motor.has_strobe_sweep``) and old firmware (no
        "strobesweep" in /modules_get) both answer False; nothing raises.
        """
        probe = getattr(self._motor, "has_strobe_sweep", None)
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception as e:
            self.__logger.warning(f"Strobe sweep capability query failed: {e}")
            return False

    def startStrobeSweep(self, axis="X", target=0.0, speed=None, period_us=33333,
                         trig_us=100, laser=-1, delay_us=None, width_us=None,
                         latch=True, report=16, max_frames=0) -> bool:
        """Drive ``axis`` to ``target`` (µm, user frame, absolute) at constant speed
        while the firmware sends a SYNC every ``period_us``.

        Each SYNC pulses the camera trigger (``trig_us`` wide), flashes logical
        laser ``laser`` ``delay_us`` after it for ``width_us`` (both given and
        ``laser >= 0``: the firmware enables the strobe for the sweep and
        disables it afterwards; -1 = no flash) and latches the axis position.
        ``speed`` has the same units as move(). ``max_frames > 0`` runs exactly
        that many frames, even without motion. Returns immediately; progress
        arrives through registerStrobeSweepCallback. Returns False when the
        sweep was not sent (uc2rest too old, target outside the soft limits).
        """
        start = getattr(self._motor, "start_strobe_sweep", None)
        if start is None:
            self.__logger.warning("uc2rest has no Motor.start_strobe_sweep; update UC2-REST")
            return False
        axis = str(axis).upper()
        # Same soft limits as move(); a blocked sweep is refused, never forced.
        if axis not in self._axes:
            self.__logger.warning(f"Strobe sweep: {axis} is not an axis of {self._name}")
            return False
        if self.limitEnabled[axis]:
            lo, hi = self.minPos[axis], self.maxPos[axis]
            if not (lo <= target <= hi):
                self.__logger.warning(
                    f"Strobe sweep blocked: {axis} target {target} outside [{lo}, {hi}]")
                return False
        if speed is None:
            speed = self.speed.get(axis, 0)
        self._strobeAxis = axis
        deviceTarget = float(target) + float(self.stageOffsetPositions.get(axis, 0))
        try:
            start(axis=self.axisIds[axis], target=deviceTarget, speed=speed, period_us=int(period_us),
                  trig_us=int(trig_us), laser=int(laser),
                  delay_us=None if delay_us is None else int(round(delay_us)),
                  width_us=None if width_us is None else int(round(width_us)),
                  latch=bool(latch), report=int(report), max_frames=int(max_frames),
                  is_absolute=True)
        except Exception as e:
            self.__logger.error(f"start_strobe_sweep failed: {e}")
            return False
        return True

    def stopStrobeSweep(self):
        """Abort a running strobed sweep (motor stops, strobe off, done event follows)."""
        stop = getattr(self._motor, "stop_strobe_sweep", None)
        if stop is None:
            return False
        try:
            stop()
            return True
        except Exception as e:
            self.__logger.error(f"stop_strobe_sweep failed: {e}")
            return False

    def registerStrobeSweepCallback(self, cb) -> bool:
        """Receive the sweep's report/done events; report ``x`` arrives in the user frame.

        ``cb(event)`` runs on the serial thread (see uc2rest
        Motor.register_strobesweep_callback for the event shapes).
        """
        register = getattr(self._motor, "register_strobesweep_callback", None)
        if register is None:
            return False

        def toUserFrame(event, _cb=cb):
            if isinstance(event, dict) and event.get("type") == "report" and "x" in event:
                offset = float(self.stageOffsetPositions.get(self._strobeAxis, 0))
                event = dict(event, x=[float(v) - offset for v in event["x"]])
            _cb(event)

        self._strobeCallbacks[cb] = toUserFrame
        register(toUserFrame)
        return True

    def unregisterStrobeSweepCallback(self, cb) -> bool:
        wrapper = self._strobeCallbacks.pop(cb, None)
        unregister = getattr(self._motor, "unregister_strobesweep_callback", None)
        if wrapper is None or unregister is None:
            return False
        try:
            unregister(wrapper)
            return True
        except Exception as e:
            self.__logger.debug(f"unregister_strobesweep_callback failed: {e}")
            return False

    # ============================================================================
    # Motor Settings API - Unified configuration interface
    # ============================================================================
    
    def getMotorSettings(self) -> dict:
        """
        Get all motor settings in a unified format.
        Returns a dictionary with settings for all axes plus global settings.
        """
        settings = {
            'global': {
                'axisOrder': self.axisOrder,
                'axisIds': {ax: self.axisIds[ax] for ax in self._axes},
                'isCoreXY': self.isCoreXY,
                'isEnabled': self.is_enabled,
                'enableAuto': self.enableauto,
                'isDualAxis': self.isDualAxis,
            },
            'axes': {}
        }

        for axis in self._axes:
            settings['axes'][axis] = self.getMotorSettingsForAxis(axis)

        return settings

    def getMotorSettingsForAxis(self, axis: str) -> dict:
        """
        Get motor settings for a specific axis.
        """
        axis = axis.upper()
        if axis not in self._axes:
            return {'axis': axis, 'error': f'{self._name} has no axis {axis}'}

        motion = {
            'stepSize': self.stepSizes[axis],
            'maxSpeed': self.maxSpeed[axis],
            'speed': self._speed.get(axis, 10000),
            'acceleration': self.acceleration[axis],
            'minPos': self.minPos[axis] if self.minPos[axis] != float('-inf') else None,
            'maxPos': self.maxPos[axis] if self.maxPos[axis] != float('inf') else None,
            'backlash': self.backlash[axis],
        }
        hp = self.homeParams[axis]
        homing = {
            'enabled': hp['enabled'],
            'speed': hp['speed'],
            'direction': hp['direction'],
            'endstopPolarity': hp['endstoppolarity'],
            'endposRelease': hp['endposrelease'],
            'timeout': hp['timeout'],
            'homeOnStart': hp['onStart'],
            'homeSteps': hp['steps'],
        }
        limits = {'enabled': self.limitEnabled[axis], 'hardLimitsEnabled': self.hardLimitsEnabled[axis]}

        return {
            'axis': axis,
            'stepperId': self.axisIds[axis],
            'motion': motion,
            'homing': homing,
            'limits': limits,
        }

    def setMotorSettingsForAxis(self, axis: str, settings: dict) -> dict:
        """
        Set motor settings for a specific axis.
        Updates both in-memory values and sends to device if applicable.
        """
        axis = axis.upper()
        result = {'axis': axis, 'updated': [], 'errors': []}
        if axis not in self._axes:
            result['success'] = False
            result['errors'].append(f'{self._name} has no axis {axis}')
            return result

        try:
            # Update motion settings
            if 'motion' in settings:
                motion = settings['motion']
                for key, target in (('stepSize', self.stepSizes), ('maxSpeed', self.maxSpeed),
                                    ('acceleration', self.acceleration), ('backlash', self.backlash)):
                    if key in motion:
                        target[axis] = motion[key]
                        result['updated'].append(key)
                if 'speed' in motion:
                    self.setSpeed(motion['speed'], axis)
                    result['updated'].append('speed')
                if 'minPos' in motion:
                    self.minPos[axis] = motion['minPos'] if motion['minPos'] is not None else float('-inf')
                    result['updated'].append('minPos')
                if 'maxPos' in motion:
                    self.maxPos[axis] = motion['maxPos'] if motion['maxPos'] is not None else float('inf')
                    result['updated'].append('maxPos')

                # Update motor setup on device
                if any(k in motion for k in ['minPos', 'maxPos', 'stepSize', 'backlash']):
                    try:
                        self.setupMotor(self.minPos[axis], self.maxPos[axis], self.stepSizes[axis],
                                        self.backlash[axis], axis)
                    except Exception as e:
                        result['errors'].append(f'Failed to update motor setup: {str(e)}')

            # Update homing settings
            if 'homing' in settings:
                homing = settings['homing']
                hp = self.homeParams[axis]
                for key, name in (('enabled', 'enabled'), ('speed', 'speed'), ('endstopPolarity', 'endstoppolarity'),
                                  ('endposRelease', 'endposrelease'), ('timeout', 'timeout'),
                                  ('homeOnStart', 'onStart'), ('homeSteps', 'steps')):
                    if key in homing:
                        hp[name] = homing[key]
                        result['updated'].append(f'homing.{key}')
                if 'direction' in homing:
                    hp['direction'] = 1 if homing['direction'] > 0 else -1
                    result['updated'].append('homing.direction')

            # Update limit settings
            if 'limits' in settings:
                limits = settings['limits']
                if 'enabled' in limits:
                    self.limitEnabled[axis] = limits['enabled']
                    result['updated'].append('limits.enabled')

                if 'hardLimitsEnabled' in limits:
                    enabled = bool(limits['hardLimitsEnabled'])
                    self.hardLimitsEnabled[axis] = enabled
                    # Push the new setting to the hardware immediately
                    try:
                        self._motor.set_hard_limits(axis=self.axisIds[axis], enabled=enabled)
                    except Exception as _e:
                        result['errors'].append(f'Failed to set hard limits on device: {_e}')
                    result['updated'].append('limits.hardLimitsEnabled')

            result['success'] = True
            self.__logger.info(f"Updated motor settings for axis {axis}: {result['updated']}")

        except Exception as e:
            result['success'] = False
            result['errors'].append(str(e))
            self.__logger.error(f"Error updating motor settings for axis {axis}: {e}")

        return result

    def getTMCSettingsForAxis(self, axis: str) -> dict:
        """
        Get TMC stepper driver settings for a specific axis.

        Read from the ImSwitch-side cache ONLY — we push TMC params to the device
        on startup and never query it back, because /tmc_get currently crashes the
        HAT. The cache holds the config values merged over TMC_DEFAULTS.
        """
        axis = axis.upper()
        result = {'axis': axis, 'success': False}

        merged = dict(self._tmcSettings.get(axis, TMC_DEFAULTS))
        try:
            result['settings'] = {
                'msteps': merged.get('msteps'),
                'rmsCurrent': merged.get('rms_current'),
                'sgthrs': merged.get('sgthrs'),
                'semin': merged.get('semin'),
                'semax': merged.get('semax'),
                'blankTime': merged.get('blank_time'),
                'toff': merged.get('toff'),
            }
            result['source'] = 'cache'
            result['success'] = True
            self.__logger.debug(f"Retrieved cached TMC settings for axis {axis}")
        except Exception as e:
            result['error'] = str(e)
            self.__logger.error(f"Error getting TMC settings for axis {axis}: {e}")

        return result
    
    def setTMCSettingsForAxis(self, axis: str, settings: dict) -> dict:
        """
        Set TMC stepper driver settings for a specific axis.
        Sends the settings directly to the device.
        """
        axis = axis.upper()
        result = {'axis': axis, 'success': False}
        
        try:
            self.setupMotorDriver(
                axis=axis,
                msteps=settings.get('msteps'),
                rms_current=settings.get('rmsCurrent'),
                sgthrs=settings.get('sgthrs'),
                semin=settings.get('semin'),
                semax=settings.get('semax'),
                blank_time=settings.get('blankTime'),
                toff=settings.get('toff'),
                timeout=settings.get('timeout', 1)
            )
            result['success'] = True
            self.__logger.info(f"Updated TMC settings for axis {axis}")
        except Exception as e:
            result['error'] = str(e)
            self.__logger.error(f"Error updating TMC settings for axis {axis}: {e}")
        
        return result
    
    def getJoystickDirectionSettings(self) -> dict:
        """Return the in-memory joystick inversion cache: {axis: bool}.

        Read from cache only — no device round-trip required.
        """
        return {ax: bool(self._joystickSettings.get(ax, False)) for ax in self._axes}

    def setJoystickDirectionSettings(self, axis: str, inverted: bool) -> dict:
        """Update the joystick inversion for one axis in-memory and on the device.

        :param axis:     Axis name ("X", "B", ...).
        :param inverted: True to reverse joystick movement for that axis.
        :return:         {success, axis, inverted} or {success, error}.
        """
        axis = axis.upper()
        result = {"axis": axis, "inverted": bool(inverted), "success": False}
        try:
            self._motor.set_joystick_direction(axis=self.axisIds[axis], inverted=bool(inverted), timeout=1)
            self._joystickSettings[axis] = bool(inverted)
            result["success"] = True
            self.__logger.info(f"Joystick direction set: axis={axis}, inverted={inverted}")
        except Exception as e:
            result["error"] = str(e)
            self.__logger.error(f"setJoystickDirectionSettings failed for axis {axis}: {e}")
        return result

    def getSpeedMultiplierSettings(self) -> dict:
        """Return the in-memory joystick-jog speed-multiplier cache: {axis: number}.

        Read from cache only — no device round-trip required.
        """
        return {ax: self._speedMultiplier.get(ax, 1) for ax in self._axes}

    def setSpeedMultiplierSettings(self, axis: str, multiplier) -> dict:
        """Update the joystick-jog speed multiplier for one axis in-memory and on the device.

        :param axis:       Axis name ("X", "B", ...).
        :param multiplier: Speed multiplier to apply for that axis.
        :return:           {success, axis, multiplier} or {success, error}.
        """
        axis = axis.upper()
        result = {"axis": axis, "multiplier": multiplier, "success": False}
        try:
            self._motor.set_speed_multiplier(axis=self.axisIds[axis], multiplier=multiplier, timeout=1)
            self._speedMultiplier[axis] = multiplier
            result["success"] = True
            self.__logger.info(f"Speed multiplier set: axis={axis}, multiplier={multiplier}")
        except Exception as e:
            result["error"] = str(e)
            self.__logger.error(f"setSpeedMultiplierSettings failed for axis {axis}: {e}")
        return result

    def setGlobalMotorSettings(self, settings: dict) -> dict:
        """
        Set global motor settings (axis ids, CoreXY, enable, etc.)
        """
        result = {'updated': [], 'errors': []}

        try:
            if 'axisOrder' in settings:
                self.setAxisOrder(order=settings['axisOrder'])
                result['updated'].append('axisOrder')

            if 'axisIds' in settings:
                self.axisIds.update({str(k).upper(): int(v) for k, v in settings['axisIds'].items()})
                for ax in self._axes:
                    self.setupMotor(self.minPos[ax], self.maxPos[ax], self.stepSizes[ax], self.backlash[ax], ax)
                result['updated'].append('axisIds')

            if 'isCoreXY' in settings:
                self.isCoreXY = settings['isCoreXY']
                result['updated'].append('isCoreXY')

            if 'isEnabled' in settings:
                self.is_enabled = settings['isEnabled']
                result['updated'].append('isEnabled')

            if 'enableAuto' in settings:
                self.enableauto = settings['enableAuto']
                self.enalbeMotors(enable=self.is_enabled, enableauto=self.enableauto)
                result['updated'].append('enableAuto')

            if 'isDualAxis' in settings:
                self.isDualAxis = settings['isDualAxis']
                result['updated'].append('isDualAxis')

            result['success'] = True
            self.__logger.info(f"Updated global motor settings: {result['updated']}")

        except Exception as e:
            result['success'] = False
            result['errors'].append(str(e))
            self.__logger.error(f"Error updating global motor settings: {e}")

        return result

# Copyright (C) 2020, 2021 The imswitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
