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

        # global offset (i.e. difference between stage zero and device zero)
        self.stageOffsetPositions = {}
        self.stageOffsetPositions["X"] = positionerInfo.stageOffsets.get('stageOffsetPositionX',0)
        self.stageOffsetPositions["Y"] = positionerInfo.stageOffsets.get('stageOffsetPositionY',0)
        self.stageOffsetPositions["Z"] = positionerInfo.stageOffsets.get('stageOffsetPositionZ',0)
        self.stageOffsetPositions["A"] = positionerInfo.stageOffsets.get('stageOffsetPositionA',0)

        # Calibrated stepsizes in µm/step
        self.stepSizes = {}
        self.stepSizes["X"] = positionerInfo.managerProperties.get('stepsizeX', 1)
        self.stepSizes["Y"] = positionerInfo.managerProperties.get('stepsizeY', 1)
        self.stepSizes["Z"] = positionerInfo.managerProperties.get('stepsizeZ', 1)
        self.stepSizes["A"] = positionerInfo.managerProperties.get('stepsizeA', 1)

        # Minimum/maximum steps in X
        self.minX = positionerInfo.managerProperties.get('minX', -np.inf)
        self.maxX = positionerInfo.managerProperties.get('maxX', np.inf)

        # Minimum/maximum steps in Y
        self.minY = positionerInfo.managerProperties.get('minY', -np.inf)
        self.maxY = positionerInfo.managerProperties.get('maxY', np.inf)

        # Minimum/maximum steps in Z
        self.minZ = positionerInfo.managerProperties.get('minZ', -np.inf)
        self.maxZ = positionerInfo.managerProperties.get('maxZ', np.inf)

        # Minimum/maximum steps in T
        self.minA = positionerInfo.managerProperties.get('minA', -np.inf)
        self.maxA = positionerInfo.managerProperties.get('maxA', np.inf)

        # Calibrated backlash
        self.backlashX = positionerInfo.managerProperties.get('backlashX', 0)
        self.backlashY = positionerInfo.managerProperties.get('backlashY', 0)
        self.backlashZ = positionerInfo.managerProperties.get('backlashZ', 0)
        self.backlashA = positionerInfo.managerProperties.get('backlashA', 0)

        # maximum speed per Axis
        self.maxSpeed = {}
        self.maxSpeed["X"] = positionerInfo.managerProperties.get('maxSpeedX', 10000)
        self.maxSpeed["Y"] = positionerInfo.managerProperties.get('maxSpeedY', 10000)
        self.maxSpeed["Z"] = positionerInfo.managerProperties.get('maxSpeedZ', 10000)
        self.maxSpeed["A"] = positionerInfo.managerProperties.get('maxSpeedA', 10000)

        self.setSpeed(positionerInfo.managerProperties.get('initialSpeedX', 10000), axis="X")
        self.setSpeed(positionerInfo.managerProperties.get('initialSpeedY', 10000), axis="Y")
        self.setSpeed(positionerInfo.managerProperties.get('initialSpeedZ', 10000), axis="Z")
        self.setSpeed(positionerInfo.managerProperties.get('initialSpeedA', 10000), axis="A")

        self.sampleLoadingPositions = {}
        self.sampleLoadingPositions["X"] = positionerInfo.managerProperties.get('sampleLoadingPositionX', 0)
        self.sampleLoadingPositions["Y"] = positionerInfo.managerProperties.get('sampleLoadingPositionY', 0)
        self.sampleLoadingPositions["Z"] = positionerInfo.managerProperties.get('sampleLoadingPositionZ', 0)

        # transportation position (A/X/Y/Z) - stage is moved here so the locking
        # blocks can be bolted on before shipping. Loaded from the setup JSON and
        # persisted back via PositionerController.setTransportPosition.
        self.transportPositions = {}
        self.transportPositions["A"] = positionerInfo.managerProperties.get('transportPositionA', 0)
        self.transportPositions["X"] = positionerInfo.managerProperties.get('transportPositionX', 65000)
        self.transportPositions["Y"] = positionerInfo.managerProperties.get('transportPositionY', 40000)
        self.transportPositions["Z"] = positionerInfo.managerProperties.get('transportPositionZ', 0)

        # move z before homing? Sign encodes the safe (away-from-sample) direction.
        # The frame-homing procedure always lifts Z by at least MIN_SAFE_Z_LIFT_HOMING
        # before homing XY so the stage cannot drive the objective into the sample.
        self._safeDistanceZHoming = positionerInfo.managerProperties.get('safeDistanceZHoming', 0)

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
        self._zSyncDefaultSteps = positionerInfo.managerProperties.get('zSyncSteps', 10000)
        self._zSyncState = {
            "active": False,
            "cancelled": False,
            "phase": "idle",
            "message": "",
            "steps": self._zSyncDefaultSteps,
        }

        # Setup homing coordinates and speed
        # X
        self.setHomeParametersAxis(axis="X", speed=positionerInfo.managerProperties.get('homeSpeedX', 15000),
                                   direction=positionerInfo.managerProperties.get('homeDirectionX', -1),
                                   endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityX', 1),
                                   endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseX', 0),
                                   timeout=positionerInfo.managerProperties.get('homeTimeoutX', 20000))
        # Y
        self.setHomeParametersAxis(axis="Y", speed=positionerInfo.managerProperties.get('homeSpeedY', 15000),
                                      direction=positionerInfo.managerProperties.get('homeDirectionY', -1),
                                      endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityY', 1),
                                      endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseY', 0),
                                      timeout=positionerInfo.managerProperties.get('homeTimeoutY', 20000))

        # Z
        self.setHomeParametersAxis(axis="Z", speed=positionerInfo.managerProperties.get('homeSpeedZ', 15000),
                                        direction=positionerInfo.managerProperties.get('homeDirectionZ', -1),
                                        endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityZ', 1),
                                        endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseZ', 0),
                                        timeout=positionerInfo.managerProperties.get('homeTimeoutZ', 20000))

        # A
        self.setHomeParametersAxis(axis="A", speed=positionerInfo.managerProperties.get('homeSpeedA', 15000),
                                        direction=positionerInfo.managerProperties.get('homeDirectionA', -1),
                                        endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityA', 1),
                                        endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseA', 0),
                                        timeout=positionerInfo.managerProperties.get('homeTimeoutA', 20000))

        # perform homing on startup?
        self.homeOnStartX = positionerInfo.managerProperties.get('homeOnStartX', 0)
        self.homeOnStartY = positionerInfo.managerProperties.get('homeOnStartY', 0)
        self.homeOnStartZ = positionerInfo.managerProperties.get('homeOnStartZ', 0)
        self.homeOnStartA = positionerInfo.managerProperties.get('homeOnStartA', 0)

        # homing is actually enabled?
        self.homeXenabled = positionerInfo.managerProperties.get('homeXenabled', False)
        self.homeYenabled = positionerInfo.managerProperties.get('homeYenabled', False)
        self.homeZenabled = positionerInfo.managerProperties.get('homeZenabled', False)
        self.homeAenabled = positionerInfo.managerProperties.get('homeAenabled', False)

        # homing steps without endstop
        self.homeStepsX = positionerInfo.managerProperties.get('homeStepsX', 0)
        self.homeStepsY = positionerInfo.managerProperties.get('homeStepsY', 0)
        self.homeStepsZ = positionerInfo.managerProperties.get('homeStepsZ', 0)
        self.homeStepsA = positionerInfo.managerProperties.get('homeStepsA', 0)

        # Limiting is actually enabled - can we go smaller than 0?
        self.limitXenabled = positionerInfo.managerProperties.get('limitXenabled', False)
        self.limitYenabled = positionerInfo.managerProperties.get('limitYenabled', False)
        self.limitZenabled = positionerInfo.managerProperties.get('limitZenabled', False)

        # Hardware hard limits - enable physical endstop protection per axis (default True)
        self.hardLimitsEnabledX = positionerInfo.managerProperties.get('hardLimitsEnabledX', True)
        self.hardLimitsEnabledY = positionerInfo.managerProperties.get('hardLimitsEnabledY', True)
        self.hardLimitsEnabledZ = positionerInfo.managerProperties.get('hardLimitsEnabledZ', True)
        self.hardLimitsEnabledA = positionerInfo.managerProperties.get('hardLimitsEnabledA', True)

        # retreive position coordinates for sample loading
        self.sampleLoadingPositions["X"] = positionerInfo.managerProperties.get('sampleLoadingPositionX', 0)
        self.sampleLoadingPositions["Y"] = positionerInfo.managerProperties.get('sampleLoadingPositionY', 0)
        self.sampleLoadingPositions["Z"] = positionerInfo.managerProperties.get('sampleLoadingPositionZ', 0)

        # Axis order
        self.axisOrder = positionerInfo.managerProperties.get('axisOrder', [0, 1, 2, 3])

        # CoreXY geometry(cont'd)
        self.isCoreXY = positionerInfo.managerProperties.get('isCoreXY', False)

        # Enable motors
        self.is_enabled = positionerInfo.managerProperties.get('isEnable', True)
        self.enableauto = positionerInfo.managerProperties.get('enableauto', True)
        self.enalbeMotors(enable=self.is_enabled, enableauto=self.enableauto)

        # Dual Axis if we have A and Z to drive the motor
        self.isDualAxis = positionerInfo.managerProperties.get("isDualaxis", False)
        if self.isDualAxis:
            self.stepSizes["A"] = self.stepSizes["Z"]
            self.stepSizes["A"] = self.stepSizes["Z"]
        # Acceleration
        self.acceleration = {"X": MAX_ACCEL, "Y": MAX_ACCEL, "Z": MAX_ACCEL, "A": MAX_ACCEL}

        # Set axis order
        self.setAxisOrder(order=self.axisOrder)

        # Set IsCoreXY
        self._motor.setIsCoreXY(isCoreXY=self.isCoreXY)

        # get bootup position and write to GUI
        self._position = self.getPosition()

        # Setup motors
        self.setupMotor(self.minX, self.maxX, self.stepSizes["X"], self.backlashX, "X")
        self.setupMotor(self.minY, self.maxY, self.stepSizes["Y"], self.backlashY, "Y")
        self.setupMotor(self.minZ, self.maxZ, self.stepSizes["Z"], self.backlashZ, "Z")
        self.setupMotor(self.minA, self.maxA, self.stepSizes["A"], self.backlashA, "A")

        # Setup Motor drivers (TMC - if available)
        # Cache the TMC settings per axis (config values override the defaults).
        # This cache is the source of truth the frontend reads back, because the
        # firmware has no reliable TMC read-back command. The values are pushed to
        # the device below for any axis that is TMC-configured.
        self._tmcSettings = {}
        for _ax in ("X", "Y", "Z", "A"):
            self._tmcSettings[_ax] = {
                key: positionerInfo.managerProperties.get(f'{key}{_ax}', default)
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
            for _ax in ("X", "Y", "Z", "A"):
                if positionerInfo.managerProperties.get(f"msteps{_ax}") is None:
                    continue
                self.setupMotorDriver(axis=_ax, timeout=1, **self._tmcSettings[_ax])
        except Exception as e:
            self.__logger.warning(f"Could not apply TMC settings: {e}")

        # Dummy move to get the motor to the right position
        for iAxis in positionerInfo.axes:
            self.move(value=-1, speed=1000, axis=iAxis, is_absolute=False, is_blocking=True, isEnable=True, timeout=0.2)
            self.move(value=1, speed=1000, axis=iAxis, is_absolute=False, is_blocking=True, isEnable=True, timeout=0.2)

        # optional: hom on startup:
        if self.homeOnStartX: self.home_x()
        time.sleep(0.5)
        if self.homeOnStartY: self.home_y()
        time.sleep(0.5)
        if self.homeOnStartZ: self.home_z()
        time.sleep(0.5)
        if self.homeOnStartA: self.home_a()
        time.sleep(0.5)

        # Apply hard-limit settings to hardware for all axes
        try:
            for _axis, _enabled in [
                ('X', self.hardLimitsEnabledX),
                ('Y', self.hardLimitsEnabledY),
                ('Z', self.hardLimitsEnabledZ),
                ('A', self.hardLimitsEnabledA),
            ]:
                self._motor.set_hard_limits(axis=_axis, enabled=_enabled)
        except Exception as _e:
            self.__logger.warning(f"Could not apply hard limits on startup: {_e}")

        # set speed for all axes
        self._speed = {"X": positionerInfo.managerProperties.get('speedX', 10000),
                        "Y": positionerInfo.managerProperties.get('speedY', 10000),
                        "Z": positionerInfo.managerProperties.get('speedZ', 10000),
                        "A": positionerInfo.managerProperties.get('speedA', 10000)}

        # save z-position prior to homing
        self._zPositionPriorHoming = self._position["Z"]

        # try to register the callback
        try:
            # if event "0" is triggered, the callback function to update the stage positions
            # will be called
            self._motor.register_callback(0,callbackfct=self.setPositionFromDevice)
        except Exception as e:
            self.__logger.error(f"Could not register callback: {e}")

        # do frame homing if enabled
        if positionerInfo.managerProperties.get('frameHomeOnStart', 0):
            self.frameHomingProcedure(False)

    def setHomeParametersAxis(self, axis, speed, direction, endstoppolarity, endposrelease, timeout=None):
        if axis == "X":
            self.homeSpeedX = speed
            self.homeDirectionX = 1 if direction > 0 else -1
            self.homeEndstoppolarityX = endstoppolarity
            self.homeEndposReleaseX = endposrelease
            self.homeTimeoutX = timeout
        elif axis == "Y":
            self.homeSpeedY = speed#
            self.homeDirectionY = 1 if direction > 0 else -1
            self.homeEndstoppolarityY = endstoppolarity
            self.homeEndposReleaseY = endposrelease
            self.homeTimeoutY = timeout
        elif axis == "Z":
            self.homeSpeedZ = speed
            self.homeDirectionZ = 1 if direction > 0 else -1
            self.homeEndstoppolarityZ = endstoppolarity
            self.homeEndposReleaseZ = endposrelease
            self.homeTimeoutZ = timeout
        elif axis == "A":
            self.homeSpeedA = speed
            self.homeDirectionA = direction
            self.homeEndstoppolarityA = endstoppolarity
            self.homeEndposReleaseA = endposrelease
            self.homeTimeoutA = timeout


    def setAxisOrder(self, order=[0,1,2,3]):
        self._motor.setMotorAxisOrder(order=order)

    def enalbeMotors(self, enable=None, enableauto=None):
        """
        enable - Enable Motors (i.e. switch on/off power to motors)
        enableauto - Enable automatic motor power off after motors are not used for a while; will be turned on automatically
        """
        self._motor.set_motor_enable(enable=enable, enableauto=enableauto)

    def setupMotor(self, minPos, maxPos, stepSize, backlash, axis):
        self._motor.setup_motor(axis=axis, minPos=minPos, maxPos=maxPos, stepSize=stepSize, backlash=backlash)

    def setBacklash(self, axis, backlashUm):
        """Set an axis' backlash from a value in microns (e.g. a camera measurement).

        Converts microns to hardware steps with the configured per-axis step size
        and pushes it to the motor, which applies it as a reversal overshoot. The
        cached ``backlash<Axis>`` is updated so a later ``setupMotor`` keeps it.
        Note: the value is *not* written back to the saved config here.
        """
        axis = str(axis).upper()
        stepSize = self.stepSizes.get(axis, 1) or 1
        backlashSteps = int(round(float(backlashUm) / stepSize))
        self._motor.set_backlash(axis=axis, backlash=backlashSteps)
        if axis == "X":
            self.backlashX = backlashSteps
        elif axis == "Y":
            self.backlashY = backlashSteps
        elif axis == "Z":
            self.backlashZ = backlashSteps
        elif axis == "A":
            self.backlashA = backlashSteps
        return {"axis": axis, "backlashUm": float(backlashUm), "backlashSteps": backlashSteps}

    def setupMotorDriver(self, axis="X", msteps=None, rms_current=None, stall_value=None, sgthrs=None, semin=None, semax=None, blank_time=None, toff=None, timeout=0.1):
        self._motor.set_tmc_parameters(axis=axis, msteps=msteps, rms_current=rms_current, stall_value=stall_value, sgthrs=sgthrs, semin=semin, semax=semax, blank_time=blank_time, toff=toff, timeout=timeout)
        # Remember what we applied so the frontend can read it back even though
        # the firmware has no TMC read-back command.
        cache = self._tmcSettings.setdefault(axis.upper(), dict(TMC_DEFAULTS))
        for key, value in (('msteps', msteps), ('rms_current', rms_current),
                           ('sgthrs', sgthrs), ('semin', semin), ('semax', semax),
                           ('blank_time', blank_time), ('toff', toff)):
            if value is not None:
                cache[key] = value

    def move(self, value=0, axis="X", is_absolute=False, is_blocking=True, acceleration=None, speed=None, isEnable=None, timeout=gTIMEOUT, is_reduced=True):
        '''
        Move the motor to a new position
        :param value: The new position
        :param axis: The axis to move
        :param is_absolute: If True, the motor will move to the absolute position given by value. If False, the motor will move by the amount given by value.
        :param is_blocking: If True, the function will block until the motor has reached the new position. If False, the function will return immediately.
        :param acceleration: The acceleration to use for the move. If None, the default acceleration for the axis will be used.
        :param speed: The speed to use for the move. If None, the default speed for the axis will be used.
        :param isEnable: If True, the motor will be enabled before the move. If False, the motor will be disabled before the move.
        :param timeout: The maximum time to wait for the motor to reach the new position. If the motor has not reached the new position after this time, a TimeoutError will be raised.
        '''
        #FIXME: for i, iaxis in enumerate(("A","X","Y","Z")):
        #    self._position[iaxis] = self._motor._position[i]
        if isEnable is None:
            isEnable = self.is_enabled
        if speed is None:
            if axis == "X": speed = self.speed["X"]
            if axis == "Y": speed = self.speed["Y"]
            if axis == "Z": speed = self.speed["Z"]
            if axis == "A": speed = self.speed["A"]
            if axis == "XY": speed = (self.speed["X"], self.speed["Y"])
            if axis == "XYZ": speed = (self.speed["X"], self.speed["Y"], self.speed["Z"])
        if acceleration is None:
            if axis == "X": acceleration = self.acceleration["X"]
            if axis == "Y": acceleration = self.acceleration["Y"]
            if axis == "Z": acceleration = self.acceleration["Z"]
            if axis == "A": acceleration = self.acceleration["A"]
            if axis == "XY": acceleration = (self.acceleration["X"], self.acceleration["Y"])
            if axis == "XYZ": acceleration = (self.acceleration["X"], self.acceleration["Y"], self.acceleration["Z"])
        if axis == 'X' and speed >0:
            # don't move to invalid positions (check both min and max limits)
            if not is_absolute and value == 0: return
            target_pos = value if is_absolute else self._position[axis] + value
            if self.limitXenabled:
                if target_pos < self.minX:
                    self.__logger.warning(f"X move blocked: target {target_pos} below minX {self.minX}")
                    return
                if target_pos > self.maxX:
                    self.__logger.warning(f"X move blocked: target {target_pos} above maxX {self.maxX}")
                    return
            # Apply offset for absolute moves: convert from user position to device position
            deviceValue = value + self.stageOffsetPositions["X"] if is_absolute else value
            self._motor.move_x(deviceValue, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout, is_reduced=is_reduced)
            if not is_absolute: self._position[axis] = self._position[axis] + value
            else: self._position[axis] = value
        elif axis == 'Y' and speed >0:
            # don't move to invalid positions (check both min and max limits)
            if not is_absolute and value == 0: return
            target_pos = value if is_absolute else self._position[axis] + value
            if self.limitYenabled:
                if target_pos < self.minY:
                    self.__logger.warning(f"Y move blocked: target {target_pos} below minY {self.minY}")
                    return
                if target_pos > self.maxY:
                    self.__logger.warning(f"Y move blocked: target {target_pos} above maxY {self.maxY}")
                    return
            # Apply offset for absolute moves: convert from user position to device position
            deviceValue = value + self.stageOffsetPositions["Y"] if is_absolute else value
            self._motor.move_y(deviceValue, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout)
            if not is_absolute: self._position[axis] = self._position[axis] + value
            else: self._position[axis] = value
        elif axis == 'Z' and speed >0:
            # don't move to invalid positions (check both min and max limits)
            if not is_absolute and value == 0: return
            target_pos = value if is_absolute else self._position[axis] + value
            if self.limitZenabled:
                if target_pos < self.minZ:
                    self.__logger.warning(f"Z move blocked: target {target_pos} below minZ {self.minZ}")
                    return
                if target_pos > self.maxZ:
                    self.__logger.warning(f"Z move blocked: target {target_pos} above maxZ {self.maxZ}")
                    return
            # Apply offset for absolute moves: convert from user position to device position
            deviceValue = value + self.stageOffsetPositions["Z"] if is_absolute else value
            self._motor.move_z(deviceValue, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, is_dualaxis=self.isDualAxis, timeout=timeout, is_reduced=is_reduced)
            if not is_absolute: self._position[axis] = self._position[axis] + value
            else: self._position[axis] = value
        elif axis == 'A' and speed >0:
            # don't move to negative positions
            #if is_absolute and value < 0: return
            #elif not is_absolute and self._position[axis] + value < 0: return
            if not is_absolute and value == 0: return
            # Apply offset for absolute moves: convert from user position to device position
            deviceValue = value + self.stageOffsetPositions["A"] if is_absolute else value
            self._motor.move_a(deviceValue, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout, is_reduced=is_reduced)
            if not is_absolute: self._position[axis] = self._position[axis] + value
            else: self._position[axis] = value
        elif axis == 'XY':
            # don't move to negative positions
            if (self.limitXenabled and self.limitYenabled) and is_absolute and (value[0] < 0 or value[1] < 0): return
            elif (self.limitXenabled and self.limitYenabled) and not is_absolute and (self._position["X"] + value[0] < 0 or self._position["Y"] + value[1] < 0): return
            # Apply offset for absolute moves: convert from user position to device position
            deviceValue = value
            if is_absolute:
                deviceValue = (value[0] + self.stageOffsetPositions["X"], value[1] + self.stageOffsetPositions["Y"])
            self._motor.move_xy(deviceValue, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout, is_reduced=is_reduced)
            for i, iaxis in enumerate(("X", "Y")):
                if not is_absolute:
                    self._position[iaxis] = self._position[iaxis] + value[i]
                else:
                    self._position[iaxis] = value[i]
        elif axis == 'XYZ':
            # Apply offset for absolute moves: convert from user position to device position
            deviceValue = value
            if is_absolute:
                deviceValue = (value[0] + self.stageOffsetPositions["X"],
                               value[1] + self.stageOffsetPositions["Y"],
                               value[2] + self.stageOffsetPositions["Z"])
            self._motor.move_xyz(deviceValue, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout, is_reduced=is_reduced)
            for i, iaxis in enumerate(("X", "Y", "Z")):
                if not is_absolute: self._position[iaxis] = self._position[iaxis] + value[i]
                else: self._position[iaxis] = value[i]
        else:
            self.__logger.error('Wrong axis, has to be "A", "X" "Y" or "Z" and speed has to be >0')
        #self._commChannel.sigUpdateMotorPosition.emit() # TODO: This is a hacky workaround to force Imswitch to update the motor positions in the gui..

    def measure(self, sensorID=0, NAvg=100):
        return self._motor.read_sensor(sensorID=sensorID, NAvg=NAvg)

    def setupPIDcontroller(self, PIDactive=1, Kp=100, Ki=10, Kd=1, target=500, PID_updaterate=200):
        return self._motor.set_pidcontroller(PIDactive=PIDactive, Kp=Kp, Ki=Ki, Kd=Kd, target=target,
                                             PID_updaterate=PID_updaterate)

    def moveForeverByAxis(self, speed=0, axis="X", is_stop=False):
        speed=(0, 0, 0, 0)
        if axis == "X":
            speed[1]=speed
        elif axis == "Y":
            speed[2]=speed
        elif axis == "Z":
            speed[3]=speed
        elif axis == "A":
            speed[0]=speed
        self.moveForever(speed=speed, is_stop=is_stop)

    def moveForever(self, speed=(0, 0, 0, 0), is_stop:bool=False):
        self._motor.move_forever(speed=speed, is_stop=is_stop)

    def setEnabled(self, is_enabled):
        self.is_enabled = is_enabled

    def setSpeed(self, speed, axis=None):
        # TODO: Map that to the JSON!
        if type(speed) == int and axis == None:
            self._speed["X"] = speed
            self._speed["Y"] = speed
            self._speed["Z"] = speed
            self._speed["A"] = speed
        else:
            self._speed[axis] = speed

    def setPosition(self, value, axis):
        # print(f"setPosition - Axis: {axis} -> New Value: {value}")
        self._position[axis] = value

    def setPositionOnDevice(self, value, axis):
        self.setPosition(value, axis)
        self._motor.set_position(axis, value)

    def setPositionFromDevice(self, positionArray: np.array):
        ''' mostly used for he position callback
        If new positions are coming from the device they will be updated in ImSwitch too'''
        posDict = {"ESP32Stage": {}}
        for iAxis, axisName in enumerate(["A", "X", "Y", "Z"]):
            positionOffsetCorrected = positionArray[iAxis] - self.getStageOffsetAxis(axisName)
            self.setPosition(positionOffsetCorrected, axisName)
            posDict["ESP32Stage"][axisName] = positionOffsetCorrected
        self._commChannel.sigUpdateMotorPosition.emit(posDict)

    def closeEvent(self):
        pass

    def getPosition(self):
        # load position from device
        # t,x,y,z
        posDict = {}
        try:
            allPositions = 1.*self._motor.get_position()
            for i, iaxis in enumerate(("A","X","Y","Z")):
                positionOffsetCorrected = allPositions[i] - self.getStageOffsetAxis(iaxis)
                posDict[iaxis] = positionOffsetCorrected
            return posDict
        except Exception as e:
            self.__logger.error(e)
            return self._position


    def forceStop(self, axis):
        if axis=="X":
            self.stop_x()
        elif axis=="Y":
            self.stop_y()
        elif axis=="Z":
            self.stop_z()
        elif axis=="A":
            self.stop_a()
        else:
            self.stopAll()

    def get_abs(self, axis):
        return self._position[axis]

    def stop_x(self):
        self._motor.stop(axis = "X")

    def stop_y(self):
        self._motor.stop(axis = "Y")

    def stop_z(self):
        self._motor.stop(axis = "Z")

    def stop_a(self):
        self._motor.stop(axis = "A")

    def stopAll(self):
        self._motor.stop()

    def doHome(self, axis, isBlocking=False, homeDirection=None, homeSpeed=None, homeEndstoppolarity=None, homeEndposRelease=None, homeTimeout=None):
        if axis == "X" and (self.homeXenabled or abs(self.homeStepsX)>0):
            self.home_x(isBlocking, homeDirection, homeSpeed, homeEndstoppolarity, homeEndposRelease, homeTimeout)
        if axis == "Y" and (self.homeYenabled or abs(self.homeStepsY)>0):
            self.home_y(isBlocking, homeDirection, homeSpeed, homeEndstoppolarity, homeEndposRelease, homeTimeout)
        if axis == "Z" and (self.homeZenabled or abs(self.homeStepsZ)>0):
            self.home_z(isBlocking, homeDirection, homeSpeed, homeEndstoppolarity, homeEndposRelease, homeTimeout)
        if axis == "A" and (self.homeAenabled or abs(self.homeStepsA)>0):
            self.home_a(isBlocking, homeDirection, homeSpeed, homeEndstoppolarity, homeEndposRelease, homeTimeout)

    def home_x(self, isBlocking=False, homeDirection=None, homeSpeed=None, homeEndstoppolarity=None, homeEndposRelease=None, homeTimeout=None):
        # move z prior to homing?
        if self._safeDistanceZHoming !=0:
            self.move(value=self._zPositionPriorHoming + self._safeDistanceZHoming, speed=self.homeSpeedZ, axis="Z", is_absolute=True, is_blocking=True)
        if abs(self.homeStepsX)>0:
            self.move(value=self.homeStepsX, speed=self.homeSpeedX, axis="X", is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(self.homeStepsX)*np.abs(self.homeEndposReleaseX), speed=self.homeSpeedX, axis="X", is_absolute=False, is_blocking=True)
            self.setPosition(axis="X", value=0)
            self.setPositionOnDevice(value=0, axis="X")
        elif self.homeXenabled:
            self._homeModule.home_x(speed=self.homeSpeedX, direction=self.homeDirectionX, endstoppolarity=self.homeEndstoppolarityX, endposrelease=self.homeEndposReleaseX, isBlocking=isBlocking, timeout=self.homeTimeoutX)
        else:
            self.__logger.info("No homing parameters set for X axis or not enabled in settings.")
            return
        # self.setPosition(axis="X", value=0)  # TODO: Not necessary as we get the position asynchronusly?

    def home_y(self,isBlocking=False, homeDirection=None, homeSpeed=None, homeEndstoppolarity=None, homeEndposRelease=None, homeTimeout=None):
        if self._safeDistanceZHoming !=0:
            self.move(value=self._zPositionPriorHoming + self._safeDistanceZHoming, speed=self.homeSpeedZ, axis="Z", is_absolute=True, is_blocking=True)
        # TODO: Wehave to go back after we are done with the homing
        if abs(self.homeStepsY)>0:
            self.move(value=self.homeStepsY, speed=self.homeSpeedY, axis="Y", is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(self.homeStepsY)*np.abs(self.homeEndposReleaseY), speed=self.homeSpeedY, axis="Y", is_absolute=False, is_blocking=True)
            self.setPosition(axis="Y", value=0)
            self.setPositionOnDevice(value=0, axis="Y")
        elif self.homeYenabled:
            self._homeModule.home_y(speed=self.homeSpeedY, direction=self.homeDirectionY, endstoppolarity=self.homeEndstoppolarityY, endposrelease=self.homeEndposReleaseY, isBlocking=isBlocking, timeout=self.homeTimeoutY)
        else:
            self.__logger.info("No homing parameters set for X axis or not enabled in settings.")
            return
        # self.setPosition(axis="Y", value=0)  # TODO: Not necessary as we get the position asynchronusly?

    def home_z(self,isBlocking=False, homeDirection=None, homeSpeed=None, homeEndstoppolarity=None, homeEndposRelease=None, homeTimeout=None):
        if abs(self.homeStepsZ)>0:
            self.move(value=self.homeStepsZ, speed=self.homeSpeedZ, axis="Z", is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(self.homeStepsZ)*np.abs(self.homeEndposReleaseZ), speed=self.homeSpeedZ, axis="Z", is_absolute=False, is_blocking=True)
            self.setPosition(axis="Z", value=0)
            self.setPositionOnDevice(value=0, axis="Z")
        elif self.homeZenabled:
            self._homeModule.home_z(speed=self.homeSpeedZ, direction=self.homeDirectionZ, endstoppolarity=self.homeEndstoppolarityZ, endposrelease=self.homeEndposReleaseZ, isBlocking=isBlocking, timeout=self.homeTimeoutZ)
        else:
            self.__logger.info("No homing parameters set for X axis or not enabled in settings.")
            return
        # self.setPosition(axis="Z", value=0) # TODO: Not necessary as we get the position asynchronusly?
        self._zPositionPriorHoming = 0

    def home_a(self,isBlocking=False, homeDirection=None, homeSpeed=None, homeEndstoppolarity=None, homeEndposRelease=None, homeTimeout=None):
        if abs(self.homeStepsA)>0:
            self.move(value=self.homeStepsA, speed=self.homeSpeedA, axis="A", is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(self.homeStepsA)*np.abs(self.homeEndposReleaseA), speed=self.homeSpeedA, axis="A", is_absolute=False, is_blocking=True)
            self.setPosition(axis="A", value=0)
            self.setPositionOnDevice(value=0, axis="A")
        elif self.homeAenabled:
            self._homeModule.home_a(speed=self.homeSpeedA, direction=self.homeDirectionA, endstoppolarity=self.homeEndstoppolarityA, endposrelease=self.homeEndposReleaseA, isBlocking=isBlocking, timeout=self.homeTimeoutA)
        else:
            self.__logger.info("No homing parameters set for X axis or not enabled in settings.")
            return
        # self.setPosition(axis="A", value=0) # TODO: Not necessary as we get the position asynchronusly?

    def home_xyz(self):
        if self.homeXenabled and self.homeYenabled and self.homeZenabled:
            self._motor.home_xyz()
            [self.setPosition(axis=axis, value=0) for axis in ["X","Y","Z"]]

    def startStageScanning(self, nStepsLine=100, dStepsLine=1, nTriggerLine=1, nStepsPixel=100, dStepsPixel=1, nTriggerPixel=1, delayTimeStep=10, nFrames=5, isBlocking=False):
        self._motor.startStageScanning(nStepsLine=nStepsLine, dStepsLine=dStepsLine, nTriggerLine=nTriggerLine,
                                       nStepsPixel=nStepsPixel, dStepsPixel=dStepsPixel, nTriggerPixel=nTriggerPixel,
                                       delayTimeStep=delayTimeStep, nFrames=nFrames, isBlocking=isBlocking)

    def stopStageScanning(self):
        self._motor.stopStageScanning()

    def moveToSampleMountingPosition(self, speed=10000, is_blocking=True):
        value = (self.sampleLoadingPositions["X"], self.sampleLoadingPositions["Y"], self.sampleLoadingPositions["Z"])
        self._motor.move_xyz(value, speed, is_absolute=True, is_blocking=is_blocking)

    def getDevicePositionAxis(self, axis="X"):
        """Live raw device position for one axis (no offset applied).

        Reads directly from the ESP32 firmware; the firmware preserves position
        across software restarts (steps survive reboot) so this is the stable
        reference for offset calibration.
        """
        try:
            allPositions = 1. * self._motor.get_position()
            axisIndex = {"A": 0, "X": 1, "Y": 2, "Z": 3}[axis]
            return float(allPositions[axisIndex])
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
                             speed=20000, acceleration=None):
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
            speed=speed, acceleration=acceleration
        )
        return r

    def stop_stage_scanning(self):
        """
        Stop the current stage scanning operation.
        """
        self._motor.stop_stage_scanning()
        self.__logger.info("Stage scanning stopped.")

    def moveToSampleLoadingPosition(self, speed=10000, is_blocking=True):
        value = (self.sampleLoadingPositions["X"], self.sampleLoadingPositions["Y"], self.sampleLoadingPositions["Z"])
        self._motor.move_xyz(value, speed, is_absolute=True, is_blocking=is_blocking)


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
            liftDirection = -1 if self.homeDirectionZ > 0 else 1
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
                self.move(value=safeLift, speed=self.homeSpeedZ, axis="Z",
                          is_absolute=True, is_blocking=True)
                if aborted():
                    return

                # Step 4: home X
                self._emitHomingState(phase="homing_x", axis="X", axisStatus="homing",
                                      message="Homing X axis")
                self.home_x(isBlocking=True)
                self.move(value=1000, speed=self.homeSpeedX, axis="X", is_absolute=True, is_blocking=False)
                self._emitHomingState(axis="X", axisStatus="done")
                if aborted():
                    return

                # Step 5: home Y
                self._emitHomingState(phase="homing_y", axis="Y", axisStatus="homing",
                                      message="Homing Y axis")
                self.home_y(isBlocking=True)
                self.move(value=1000, speed=self.homeSpeedY, axis="Y", is_absolute=True, is_blocking=False)
                self._emitHomingState(axis="Y", axisStatus="done")
                if aborted():
                    return
                
                # Step 6: Moving to a save position (e.g. transport position) after homing XY to avoid collisions during Z restore
                self._emitHomingState(phase="moving_to_safe_xy", message="Moving to safe XY position")
                value = (self.transportPositions["X"], self.transportPositions["Y"], self.transportPositions["Z"])
                self._motor.move_xyz(value, self.homeSpeedX, is_absolute=True, is_blocking=True)

                if aborted():
                    return

                # Step 7: restore Z to its previous height, but never drop it below
                # the safe lift (avoids re-introducing the collision risk at the new XY).
                self._emitHomingState(phase="restoring_z", message="Restoring Z position")
                restoreZ = self._zPositionPriorHoming
                self.move(value=restoreZ, speed=self.homeSpeedZ, axis="Z",
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
        """Move A/X/Y/Z to the stored transportation (locking) position."""
        if override_endstop_z:
            self.__logger.warning("Overriding Z endstops to move to transport position.")
            self.hardLimitsEnabledZ = False
            self._motor.set_hard_limits(axis="Z", enabled=False)

        value = (self.transportPositions["X"], self.transportPositions["Y"], self.transportPositions["Z"])
        self._motor.move_xyz(value, speed, is_absolute=True, is_blocking=is_blocking)
        try:
            self.move(value=self.transportPositions["A"], speed=(speed,speed,speed), axis="A",
                      is_absolute=True, is_blocking=is_blocking)
        except Exception as e:
            self.__logger.error(f"Could not move A to transport position: {e}")

    def getTransportPositions(self):
        """Return the stored transportation position as an A/X/Y/Z dict."""
        return dict(self.transportPositions)

    def setTransportPositions(self, positions=None):
        """Set the transport position.

        If ``positions`` is None the current stage pose is snapshotted; otherwise
        the provided A/X/Y/Z values are used (missing axes are left unchanged).
        Returns the updated transport position dict.
        """
        if positions is None:
            current = self.getPosition()
            for ax in ("A", "X", "Y", "Z"):
                self.transportPositions[ax] = float(current.get(ax, self.transportPositions.get(ax, 0)))
        else:
            for ax in ("A", "X", "Y", "Z"):
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
            speed = self.homeSpeedZ
        # The 'out' direction follows the Z home direction.
        outSign = 1 if self.homeDirectionZ > 0 else -1
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

            restoreHard = self.hardLimitsEnabledZ
            restoreSoft = self.limitZenabled
            try:
                # 1) disable the Z limits so we can drive into the mechanical stop
                self._emitZSyncState(phase="disabling_limit", message="Disabling Z limit switch")
                self.hardLimitsEnabledZ = False
                self.limitZenabled = False
                self._motor.set_hard_limits(axis="Z", enabled=False)
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
                self.hardLimitsEnabledZ = restoreHard
                self.limitZenabled = restoreSoft
                self._motor.set_hard_limits(axis="Z", enabled=restoreHard)
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
                    self.hardLimitsEnabledZ = restoreHard
                    self.limitZenabled = restoreSoft
                    self._motor.set_hard_limits(axis="Z", enabled=restoreHard)
                except Exception as e:
                    self.__logger.error(f"Could not restore Z limits after sync: {e}")

        self._zSyncThread = threading.Thread(target=threadFn, args=(self,))
        self._zSyncThread.start()

    def register_stagescan_callback(self, on_stagescan_complete_fct):
        self._motor.register_stagescan_callback(on_stagescan_complete_fct)
        
    
    def unregister_stagescan_callback(self):
        self._motor.unregister_stagescan_callback()
    
    def reset_stagescan_complete(self):
        pass

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
                'isCoreXY': self.isCoreXY,
                'isEnabled': self.is_enabled,
                'enableAuto': self.enableauto,
                'isDualAxis': self.isDualAxis,
            },
            'axes': {}
        }
        
        for axis in ['X', 'Y', 'Z', 'A']:
            settings['axes'][axis] = self.getMotorSettingsForAxis(axis)
        
        return settings
    
    def getMotorSettingsForAxis(self, axis: str) -> dict:
        """
        Get motor settings for a specific axis.
        """
        axis = axis.upper()
        
        # Motion settings
        motion = {
            'stepSize': self.stepSizes.get(axis, 1),
            'maxSpeed': self.maxSpeed.get(axis, 10000),
            'speed': self._speed.get(axis, 10000),
            'acceleration': self.acceleration.get(axis, MAX_ACCEL),
        }
        
        # Add min/max positions per axis
        if axis == 'X':
            motion['minPos'] = self.minX if self.minX != float('-inf') else None
            motion['maxPos'] = self.maxX if self.maxX != float('inf') else None
            motion['backlash'] = self.backlashX
        elif axis == 'Y':
            motion['minPos'] = self.minY if self.minY != float('-inf') else None
            motion['maxPos'] = self.maxY if self.maxY != float('inf') else None
            motion['backlash'] = self.backlashY
        elif axis == 'Z':
            motion['minPos'] = self.minZ if self.minZ != float('-inf') else None
            motion['maxPos'] = self.maxZ if self.maxZ != float('inf') else None
            motion['backlash'] = self.backlashZ
        elif axis == 'A':
            motion['minPos'] = self.minA if self.minA != float('-inf') else None
            motion['maxPos'] = self.maxA if self.maxA != float('inf') else None
            motion['backlash'] = self.backlashA
        
        # Homing settings
        homing = {}
        if axis == 'X':
            homing = {
                'enabled': self.homeXenabled,
                'speed': self.homeSpeedX,
                'direction': self.homeDirectionX,
                'endstopPolarity': self.homeEndstoppolarityX,
                'endposRelease': self.homeEndposReleaseX,
                'timeout': self.homeTimeoutX,
                'homeOnStart': self.homeOnStartX,
                'homeSteps': self.homeStepsX,
            }
        elif axis == 'Y':
            homing = {
                'enabled': self.homeYenabled,
                'speed': self.homeSpeedY,
                'direction': self.homeDirectionY,
                'endstopPolarity': self.homeEndstoppolarityY,
                'endposRelease': self.homeEndposReleaseY,
                'timeout': self.homeTimeoutY,
                'homeOnStart': self.homeOnStartY,
                'homeSteps': self.homeStepsY,
            }
        elif axis == 'Z':
            homing = {
                'enabled': self.homeZenabled,
                'speed': self.homeSpeedZ,
                'direction': self.homeDirectionZ,
                'endstopPolarity': self.homeEndstoppolarityZ,
                'endposRelease': self.homeEndposReleaseZ,
                'timeout': self.homeTimeoutZ,
                'homeOnStart': self.homeOnStartZ,
                'homeSteps': self.homeStepsZ,
            }
        elif axis == 'A':
            homing = {
                'enabled': self.homeAenabled,
                'speed': self.homeSpeedA,
                'direction': self.homeDirectionA,
                'endstopPolarity': self.homeEndstoppolarityA,
                'endposRelease': self.homeEndposReleaseA,
                'timeout': self.homeTimeoutA,
                'homeOnStart': self.homeOnStartA,
                'homeSteps': self.homeStepsA,
            }
        
        # Limit settings
        if axis == 'X':
            limits = {'enabled': self.limitXenabled, 'hardLimitsEnabled': self.hardLimitsEnabledX}
        elif axis == 'Y':
            limits = {'enabled': self.limitYenabled, 'hardLimitsEnabled': self.hardLimitsEnabledY}
        elif axis == 'Z':
            limits = {'enabled': self.limitZenabled, 'hardLimitsEnabled': self.hardLimitsEnabledZ}
        else:
            limits = {'enabled': False, 'hardLimitsEnabled': self.hardLimitsEnabledA}
        
        return {
            'axis': axis,
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
        
        try:
            # Update motion settings
            if 'motion' in settings:
                motion = settings['motion']
                
                if 'stepSize' in motion:
                    self.stepSizes[axis] = motion['stepSize']
                    result['updated'].append('stepSize')
                
                if 'maxSpeed' in motion:
                    self.maxSpeed[axis] = motion['maxSpeed']
                    result['updated'].append('maxSpeed')
                
                if 'speed' in motion:
                    self.setSpeed(motion['speed'], axis)
                    result['updated'].append('speed')
                
                if 'acceleration' in motion:
                    self.acceleration[axis] = motion['acceleration']
                    result['updated'].append('acceleration')
                
                if 'backlash' in motion:
                    backlash = motion['backlash']
                    if axis == 'X': self.backlashX = backlash
                    elif axis == 'Y': self.backlashY = backlash
                    elif axis == 'Z': self.backlashZ = backlash
                    elif axis == 'A': self.backlashA = backlash
                    result['updated'].append('backlash')
                
                if 'minPos' in motion:
                    minPos = motion['minPos'] if motion['minPos'] is not None else float('-inf')
                    if axis == 'X': self.minX = minPos
                    elif axis == 'Y': self.minY = minPos
                    elif axis == 'Z': self.minZ = minPos
                    elif axis == 'A': self.minA = minPos
                    result['updated'].append('minPos')
                
                if 'maxPos' in motion:
                    maxPos = motion['maxPos'] if motion['maxPos'] is not None else float('inf')
                    if axis == 'X': self.maxX = maxPos
                    elif axis == 'Y': self.maxY = maxPos
                    elif axis == 'Z': self.maxZ = maxPos
                    elif axis == 'A': self.maxA = maxPos
                    result['updated'].append('maxPos')
                
                # Update motor setup on device
                if any(k in motion for k in ['minPos', 'maxPos', 'stepSize', 'backlash']):
                    try:
                        self.setupMotor(
                            getattr(self, f'min{axis}'),
                            getattr(self, f'max{axis}'),
                            self.stepSizes[axis],
                            getattr(self, f'backlash{axis}'),
                            axis
                        )
                    except Exception as e:
                        result['errors'].append(f'Failed to update motor setup: {str(e)}')
            
            # Update homing settings
            if 'homing' in settings:
                homing = settings['homing']
                
                if 'enabled' in homing:
                    if axis == 'X': self.homeXenabled = homing['enabled']
                    elif axis == 'Y': self.homeYenabled = homing['enabled']
                    elif axis == 'Z': self.homeZenabled = homing['enabled']
                    elif axis == 'A': self.homeAenabled = homing['enabled']
                    result['updated'].append('homing.enabled')
                
                if 'speed' in homing:
                    if axis == 'X': self.homeSpeedX = homing['speed']
                    elif axis == 'Y': self.homeSpeedY = homing['speed']
                    elif axis == 'Z': self.homeSpeedZ = homing['speed']
                    elif axis == 'A': self.homeSpeedA = homing['speed']
                    result['updated'].append('homing.speed')
                
                if 'direction' in homing:
                    direction = 1 if homing['direction'] > 0 else -1
                    if axis == 'X': self.homeDirectionX = direction
                    elif axis == 'Y': self.homeDirectionY = direction
                    elif axis == 'Z': self.homeDirectionZ = direction
                    elif axis == 'A': self.homeDirectionA = direction
                    result['updated'].append('homing.direction')
                
                if 'endstopPolarity' in homing:
                    if axis == 'X': self.homeEndstoppolarityX = homing['endstopPolarity']
                    elif axis == 'Y': self.homeEndstoppolarityY = homing['endstopPolarity']
                    elif axis == 'Z': self.homeEndstoppolarityZ = homing['endstopPolarity']
                    elif axis == 'A': self.homeEndstoppolarityA = homing['endstopPolarity']
                    result['updated'].append('homing.endstopPolarity')
                
                if 'endposRelease' in homing:
                    if axis == 'X': self.homeEndposReleaseX = homing['endposRelease']
                    elif axis == 'Y': self.homeEndposReleaseY = homing['endposRelease']
                    elif axis == 'Z': self.homeEndposReleaseZ = homing['endposRelease']
                    elif axis == 'A': self.homeEndposReleaseA = homing['endposRelease']
                    result['updated'].append('homing.endposRelease')
                
                if 'timeout' in homing:
                    if axis == 'X': self.homeTimeoutX = homing['timeout']
                    elif axis == 'Y': self.homeTimeoutY = homing['timeout']
                    elif axis == 'Z': self.homeTimeoutZ = homing['timeout']
                    elif axis == 'A': self.homeTimeoutA = homing['timeout']
                    result['updated'].append('homing.timeout')
                
                if 'homeOnStart' in homing:
                    if axis == 'X': self.homeOnStartX = homing['homeOnStart']
                    elif axis == 'Y': self.homeOnStartY = homing['homeOnStart']
                    elif axis == 'Z': self.homeOnStartZ = homing['homeOnStart']
                    elif axis == 'A': self.homeOnStartA = homing['homeOnStart']
                    result['updated'].append('homing.homeOnStart')
                
                if 'homeSteps' in homing:
                    if axis == 'X': self.homeStepsX = homing['homeSteps']
                    elif axis == 'Y': self.homeStepsY = homing['homeSteps']
                    elif axis == 'Z': self.homeStepsZ = homing['homeSteps']
                    elif axis == 'A': self.homeStepsA = homing['homeSteps']
                    result['updated'].append('homing.homeSteps')
            
            # Update limit settings
            if 'limits' in settings:
                limits = settings['limits']
                if 'enabled' in limits:
                    if axis == 'X': self.limitXenabled = limits['enabled']
                    elif axis == 'Y': self.limitYenabled = limits['enabled']
                    elif axis == 'Z': self.limitZenabled = limits['enabled']
                    result['updated'].append('limits.enabled')

                if 'hardLimitsEnabled' in limits:
                    enabled = bool(limits['hardLimitsEnabled'])
                    if axis == 'X': self.hardLimitsEnabledX = enabled
                    elif axis == 'Y': self.hardLimitsEnabledY = enabled
                    elif axis == 'Z': self.hardLimitsEnabledZ = enabled
                    elif axis == 'A': self.hardLimitsEnabledA = enabled
                    # Push the new setting to the hardware immediately
                    try:
                        self._motor.set_hard_limits(axis=axis, enabled=enabled)
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
    
    def setGlobalMotorSettings(self, settings: dict) -> dict:
        """
        Set global motor settings (axis order, CoreXY, enable, etc.)
        """
        result = {'updated': [], 'errors': []}
        
        try:
            if 'axisOrder' in settings:
                self.axisOrder = settings['axisOrder']
                self.setAxisOrder(order=self.axisOrder)
                result['updated'].append('axisOrder')
            
            if 'isCoreXY' in settings:
                self.isCoreXY = settings['isCoreXY']
                self._motor.setIsCoreXY(isCoreXY=self.isCoreXY)
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
