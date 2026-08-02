from imswitch.imcommon.model import initLogger
from .PositionerManager import PositionerManager
import time
import threading
import numpy as np

try:
    from uc2canopen import OD
except Exception:  # pragma: no cover - dep guarded by the rs232 manager
    OD = None

MAX_ACCEL = 1000000
PHYS_FACTOR = 1
gTIMEOUT = 100
# Minimum distance (steps) Z is lifted out of the bottom before XY are homed, so
# the stage cannot knock the objective into the sample during the homing sweep.
MIN_SAFE_Z_LIFT_HOMING = 3000


class UC2CANOpenStageManager(PositionerManager):
    """ Positioner manager for an openUC2 stage driven over **CANopen**.

    Mirrors :class:`ESP32StageManager` method-for-method so the rest of ImSwitch
    (controllers, REST API, GUI) drives it identically, but the device I/O goes
    through :class:`uc2canopen.UC2Client` instead of uc2rest. Each logical axis
    (X/Y/Z/A) is a separate CAN node; the node ids come from the referenced
    ``UC2CANOpenManager`` (configurable, defaults 11/12/13/14).

    Blocking semantics: the CANopen ``motor.move`` call is fire-and-forget, so a
    blocking move issues the move and then polls ``wait_for_idle`` on the node.
    Combined ``XY``/``XYZ`` moves start every axis first and then wait on all of
    them — motion is concurrent but not firmware-synchronised across nodes.

    Methods that have no CANopen/firmware equivalent (hardware stage scanning,
    analog sensor reads, despeckle) are kept as logged no-ops so controllers that
    call them keep working.
    """

    def __init__(self, positionerInfo, name, **lowLevelManagers):
        super().__init__(positionerInfo, name,
                         initialPosition={axis: 0 for axis in positionerInfo.axes},
                         initialSpeed={axis: 0 for axis in positionerInfo.axes})
        self._rs232manager = lowLevelManagers['rs232sManager'][
            positionerInfo.managerProperties['rs232device']]
        self._commChannel = lowLevelManagers['commChannel']
        self.__logger = initLogger(self, instanceName=name)
        self._name = name

        # The shared CANopen client (uc2canopen.UC2Client)
        self._client = self._rs232manager.client

        # global offset (i.e. difference between stage zero and device zero)
        self.stageOffsetPositions = {}
        self.stageOffsetPositions["X"] = positionerInfo.stageOffsets.get('stageOffsetPositionX', 0)
        self.stageOffsetPositions["Y"] = positionerInfo.stageOffsets.get('stageOffsetPositionY', 0)
        self.stageOffsetPositions["Z"] = positionerInfo.stageOffsets.get('stageOffsetPositionZ', 0)
        self.stageOffsetPositions["A"] = positionerInfo.stageOffsets.get('stageOffsetPositionA', 0)

        # Calibrated stepsizes in steps/µm
        self.stepSizes = {}
        self.stepSizes["X"] = positionerInfo.managerProperties.get('stepsizeX', 1)
        self.stepSizes["Y"] = positionerInfo.managerProperties.get('stepsizeY', 1)
        self.stepSizes["Z"] = positionerInfo.managerProperties.get('stepsizeZ', 1)
        self.stepSizes["A"] = positionerInfo.managerProperties.get('stepsizeA', 1)

        # Minimum/maximum steps per axis
        self.minX = positionerInfo.managerProperties.get('minX', -np.inf)
        self.maxX = positionerInfo.managerProperties.get('maxX', np.inf)
        self.minY = positionerInfo.managerProperties.get('minY', -np.inf)
        self.maxY = positionerInfo.managerProperties.get('maxY', np.inf)
        self.minZ = positionerInfo.managerProperties.get('minZ', -np.inf)
        self.maxZ = positionerInfo.managerProperties.get('maxZ', np.inf)
        self.minA = positionerInfo.managerProperties.get('minA', -np.inf)
        self.maxA = positionerInfo.managerProperties.get('maxA', np.inf)

        # Calibrated backlash (stored only; no CANopen OD for backlash)
        self.backlashX = positionerInfo.managerProperties.get('backlashX', 0)
        self.backlashY = positionerInfo.managerProperties.get('backlashY', 0)
        self.backlashZ = positionerInfo.managerProperties.get('backlashZ', 0)
        self.backlashA = positionerInfo.managerProperties.get('backlashA', 0)

        # maximum speed per axis
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

        # transportation position (A/X/Y/Z)
        self.transportPositions = {}
        self.transportPositions["A"] = positionerInfo.managerProperties.get('transportPositionA', 0)
        self.transportPositions["X"] = positionerInfo.managerProperties.get('transportPositionX', 0)
        self.transportPositions["Y"] = positionerInfo.managerProperties.get('transportPositionY', 0)
        self.transportPositions["Z"] = positionerInfo.managerProperties.get('transportPositionZ', 0)

        # move z before homing? Sign encodes the safe (away-from-sample) direction.
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

        # Setup homing coordinates and speed
        self.setHomeParametersAxis(axis="X", speed=positionerInfo.managerProperties.get('homeSpeedX', 15000),
                                   direction=positionerInfo.managerProperties.get('homeDirectionX', -1),
                                   endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityX', 1),
                                   endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseX', 0),
                                   timeout=positionerInfo.managerProperties.get('homeTimeoutX', 20000))
        self.setHomeParametersAxis(axis="Y", speed=positionerInfo.managerProperties.get('homeSpeedY', 15000),
                                   direction=positionerInfo.managerProperties.get('homeDirectionY', -1),
                                   endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityY', 1),
                                   endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseY', 0),
                                   timeout=positionerInfo.managerProperties.get('homeTimeoutY', 20000))
        self.setHomeParametersAxis(axis="Z", speed=positionerInfo.managerProperties.get('homeSpeedZ', 15000),
                                   direction=positionerInfo.managerProperties.get('homeDirectionZ', -1),
                                   endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityZ', 1),
                                   endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseZ', 0),
                                   timeout=positionerInfo.managerProperties.get('homeTimeoutZ', 20000))
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

        # Axis order (no-op for CANopen — each axis is its own node)
        self.axisOrder = positionerInfo.managerProperties.get('axisOrder', [0, 1, 2, 3])

        # CoreXY geometry (firmware-side; stored only for the settings API)
        self.isCoreXY = positionerInfo.managerProperties.get('isCoreXY', False)

        # Enable motors
        self.is_enabled = positionerInfo.managerProperties.get('isEnable', True)
        self.enableauto = positionerInfo.managerProperties.get('enableauto', True)
        self.enableMotors(enable=self.is_enabled, autoenable=self.enableauto)

        # Dual Axis if we have A and Z to drive the motor
        self.isDualAxis = positionerInfo.managerProperties.get("isDualaxis", False)
        if self.isDualAxis:
            self.stepSizes["A"] = self.stepSizes["Z"]

        # Acceleration
        self.acceleration = {"X": MAX_ACCEL, "Y": MAX_ACCEL, "Z": MAX_ACCEL, "A": MAX_ACCEL}

        # Set axis order / CoreXY (stored; logged no-op on CAN)
        self.setAxisOrder(order=self.axisOrder)
        self.setIsCoreXY(isCoreXY=self.isCoreXY)

        # get bootup position and write to GUI
        self._position = self.getPosition()

        # Setup motors (soft limits via SDO where finite)
        self.setupMotor(self.minX, self.maxX, self.stepSizes["X"], self.backlashX, "X")
        self.setupMotor(self.minY, self.maxY, self.stepSizes["Y"], self.backlashY, "Y")
        self.setupMotor(self.minZ, self.maxZ, self.stepSizes["Z"], self.backlashZ, "Z")
        self.setupMotor(self.minA, self.maxA, self.stepSizes["A"], self.backlashA, "A")

        # Setup Motor drivers (TMC) from config if provided
        try:
            for axis in ("X", "Y", "Z", "A"):
                if positionerInfo.managerProperties.get(f'msteps{axis}') is not None:
                    self.setupMotorDriver(
                        axis=axis,
                        msteps=positionerInfo.managerProperties.get(f'msteps{axis}', 16),
                        rms_current=positionerInfo.managerProperties.get(f'rms_current{axis}', 500),
                        sgthrs=positionerInfo.managerProperties.get(f'sgthrs{axis}', 10),
                        semin=positionerInfo.managerProperties.get(f'semin{axis}', 5),
                        semax=positionerInfo.managerProperties.get(f'semax{axis}', 2),
                        blank_time=positionerInfo.managerProperties.get(f'blank_time{axis}', 24),
                        toff=positionerInfo.managerProperties.get(f'toff{axis}', 3),
                        timeout=1,
                    )
        except Exception as e:
            self.__logger.warning(f"Could not load TMC settings from config: {e}")

        # Optional dummy move on startup (off by default — unlike the ESP32 path,
        # we don't nudge every CAN node on boot unless asked to).
        if positionerInfo.managerProperties.get('dummyMoveOnStart', 0):
            for iAxis in positionerInfo.axes:
                self.move(value=-1, speed=1000, axis=iAxis, is_absolute=False, is_blocking=True, isEnable=True, timeout=0.2)
                self.move(value=1, speed=1000, axis=iAxis, is_absolute=False, is_blocking=True, isEnable=True, timeout=0.2)

        # optional: home on startup
        if self.homeOnStartX: self.home_x()
        time.sleep(0.5)
        if self.homeOnStartY: self.home_y()
        time.sleep(0.5)
        if self.homeOnStartZ: self.home_z()
        time.sleep(0.5)
        if self.homeOnStartA: self.home_a()
        time.sleep(0.5)

        # set speed for all axes
        self._speed = {"X": positionerInfo.managerProperties.get('speedX', 10000),
                       "Y": positionerInfo.managerProperties.get('speedY', 10000),
                       "Z": positionerInfo.managerProperties.get('speedZ', 10000),
                       "A": positionerInfo.managerProperties.get('speedA', 10000)}

        # save z-position prior to homing
        self._zPositionPriorHoming = self._position["Z"]

        # register the motor-done callback to push live positions to the GUI
        try:
            self._client._listener.on_motor_done(self._onMotorDone)
        except Exception as e:
            self.__logger.error(f"Could not register motor-done callback: {e}")

        # do frame homing if enabled
        if positionerInfo.managerProperties.get('frameHomeOnStart', 0):
            self.frameHomingProcedure(False)

    # ------------------------------------------------------------------
    # Low-level CAN helpers
    # ------------------------------------------------------------------
    def _node(self, axis):
        return self._rs232manager.node_for_axis(axis)

    def _blockingTimeout(self, timeout):
        """ Map the uc2rest-style ``timeout`` arg to a wait_for_idle timeout (s). """
        try:
            t = float(timeout)
        except (TypeError, ValueError):
            t = 0
        return t if t > 0 else 30.0

    def _startAxisMove(self, axis, deviceValue, speed, acceleration, is_absolute, is_enabled):
        """ Issue a non-blocking move on one axis-node. Returns the node id. """
        node = self._node(axis)
        if node is None:
            self.__logger.error(f"No CAN node configured for axis {axis}")
            return None
        if speed is None or speed <= 0:
            return None
        try:
            if is_enabled is not None:
                self._client.motor.set_enabled(axis=0, enabled=bool(is_enabled), node_id=node)
            self._client.motor.move(
                axis=0,
                position=int(round(deviceValue)),
                speed=int(speed),
                acceleration=int(acceleration or 0),
                is_absolute=bool(is_absolute),
                node_id=node,
            )
        except Exception as e:
            self.__logger.error(f"move on axis {axis} (node {node}) failed: {e}")
            return None
        return node

    def _waitNode(self, node, timeout):
        if node is None:
            return
        try:
            self._client.motor.wait_for_idle(axis=0, node_id=node, timeout=self._blockingTimeout(timeout))
        except Exception as e:
            self.__logger.error(f"wait_for_idle on node {node} failed: {e}")

    def _onMotorDone(self, node_id, sub, position):
        """ TPDO callback: a motor finished moving — push its position to the GUI. """
        axis = self._rs232manager.axis_for_node(node_id)
        if axis is None:
            return
        corrected = position - self.getStageOffsetAxis(axis)
        self.setPosition(corrected, axis)
        try:
            self._commChannel.sigUpdateMotorPosition.emit({"UC2CANOpenStage": {axis: corrected}})
        except Exception as e:
            self.__logger.error(f"Could not emit motor position update: {e}")

    # ------------------------------------------------------------------
    # Configuration / setup
    # ------------------------------------------------------------------
    def setHomeParametersAxis(self, axis, speed, direction, endstoppolarity, endposrelease, timeout=None):
        if axis == "X":
            self.homeSpeedX = speed
            self.homeDirectionX = 1 if direction > 0 else -1
            self.homeEndstoppolarityX = endstoppolarity
            self.homeEndposReleaseX = endposrelease
            self.homeTimeoutX = timeout
        elif axis == "Y":
            self.homeSpeedY = speed
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

    def setAxisOrder(self, order=[0, 1, 2, 3]):
        # No CANopen equivalent — each axis is a separate node. Stored for the
        # settings API only.
        self.axisOrder = order

    def setIsCoreXY(self, isCoreXY=False):
        # CoreXY mixing is firmware-internal and not exposed over the current OD;
        # store the flag so the settings API round-trips.
        self.isCoreXY = isCoreXY

    def enableMotors(self, enable=None, autoenable=None):
        """ Enable/disable the stepper drivers on every axis node. """
        if enable is None:
            return
        for axis in ("X", "Y", "Z", "A"):
            node = self._node(axis)
            if node is None or OD is None:
                continue
            try:
                self._rs232manager.sdo_write(node, OD.MOTOR_ENABLE, 1, 1 if enable else 0, "u8")
            except Exception as e:
                self.__logger.error(f"enableMotors failed for axis {axis}: {e}")
        if autoenable is not None:
            self.__logger.debug("enableMotors: autoenable has no CANopen equivalent; ignored.")

    # Backwards-compatible alias matching the ESP32 manager's (misspelled) name.
    def enalbeMotors(self, enable=None, enableauto=None):
        self.enableMotors(enable=enable, autoenable=enableauto)

    def setupMotor(self, minPos, maxPos, stepSize, backlash, axis):
        """ Push soft limits to the device (step size / backlash are host-side). """
        node = self._node(axis)
        if node is None or OD is None:
            return
        try:
            if np.isfinite(minPos):
                self._rs232manager.sdo_write(node, OD.MOTOR_MIN_POSITION, 1, int(minPos), "i32")
            if np.isfinite(maxPos):
                self._rs232manager.sdo_write(node, OD.MOTOR_MAX_POSITION, 1, int(maxPos), "i32")
        except Exception as e:
            self.__logger.error(f"setupMotor failed for axis {axis}: {e}")

    def setupMotorDriver(self, axis="X", msteps=None, rms_current=None, stall_value=None,
                         sgthrs=None, semin=None, semax=None, blank_time=None, toff=None, timeout=1):
        node = self._node(axis)
        if node is None or OD is None:
            return
        # (OD index, value, kind) — only written when the value is provided.
        if sgthrs is None and stall_value is not None:
            sgthrs = stall_value
        fields = [
            (OD.TMC_MICROSTEPS, msteps, "u16"),
            (OD.TMC_RMS_CURRENT, rms_current, "u16"),
            (OD.TMC_STALLGUARD_THRESHOLD, sgthrs, "u8"),
            (OD.TMC_COOLSTEP_SEMIN, semin, "u8"),
            (OD.TMC_COOLSTEP_SEMAX, semax, "u8"),
            (OD.TMC_BLANK_TIME, blank_time, "u8"),
            (OD.TMC_TOFF, toff, "u8"),
        ]
        for index, value, kind in fields:
            if value is None:
                continue
            try:
                self._rs232manager.sdo_write(node, index, 1, int(value), kind)
            except Exception as e:
                self.__logger.error(f"setupMotorDriver({axis}) write 0x{index:04X} failed: {e}")

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move(self, value=0, axis="X", is_absolute=False, is_blocking=True, acceleration=None,
             speed=None, isEnable=None, timeout=gTIMEOUT, is_reduced=True):
        """ Move one or more axes. See :class:`ESP32StageManager.move` for the
        contract; ``is_reduced`` is accepted for signature parity (no CAN effect).
        """
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

        if axis in ("X", "Y", "Z", "A"):
            if (not isinstance(speed, (int, float))) or speed <= 0:
                self.__logger.error(f"Invalid speed for axis {axis}: {speed}")
                return
            if not is_absolute and value == 0:
                return
            # Soft-limit guard (host-side), matching the ESP32 manager.
            target_pos = value if is_absolute else self._position[axis] + value
            limitEnabled = getattr(self, f"limit{axis}enabled", False)
            minLimit = getattr(self, f"min{axis}")
            maxLimit = getattr(self, f"max{axis}")
            if limitEnabled:
                if target_pos < minLimit:
                    self.__logger.warning(f"{axis} move blocked: target {target_pos} below min {minLimit}")
                    return
                if target_pos > maxLimit:
                    self.__logger.warning(f"{axis} move blocked: target {target_pos} above max {maxLimit}")
                    return
            deviceValue = value + self.stageOffsetPositions[axis] if is_absolute else value
            node = self._startAxisMove(axis, deviceValue, speed, acceleration, is_absolute, isEnable)
            if is_blocking:
                self._waitNode(node, timeout)
            if not is_absolute:
                self._position[axis] = self._position[axis] + value
            else:
                self._position[axis] = value

        elif axis == "XY":
            axes = ("X", "Y")
            if is_absolute:
                deviceValue = (value[0] + self.stageOffsetPositions["X"],
                               value[1] + self.stageOffsetPositions["Y"])
            else:
                deviceValue = value
            nodes = [self._startAxisMove(axes[i], deviceValue[i], speed[i],
                                         acceleration[i], is_absolute, isEnable) for i in range(2)]
            if is_blocking:
                for node in nodes:
                    self._waitNode(node, timeout)
            for i, iaxis in enumerate(axes):
                if not is_absolute:
                    self._position[iaxis] = self._position[iaxis] + value[i]
                else:
                    self._position[iaxis] = value[i]

        elif axis == "XYZ":
            axes = ("X", "Y", "Z")
            if is_absolute:
                deviceValue = (value[0] + self.stageOffsetPositions["X"],
                               value[1] + self.stageOffsetPositions["Y"],
                               value[2] + self.stageOffsetPositions["Z"])
            else:
                deviceValue = value
            nodes = [self._startAxisMove(axes[i], deviceValue[i], speed[i],
                                         acceleration[i], is_absolute, isEnable) for i in range(3)]
            if is_blocking:
                for node in nodes:
                    self._waitNode(node, timeout)
            for i, iaxis in enumerate(axes):
                if not is_absolute:
                    self._position[iaxis] = self._position[iaxis] + value[i]
                else:
                    self._position[iaxis] = value[i]
        else:
            self.__logger.error('Wrong axis, has to be "A", "X", "Y", "Z", "XY" or "XYZ" and speed >0')

    def moveForeverByAxis(self, speed=0, axis="X", is_stop=False):
        speedTuple = [0, 0, 0, 0]
        idx = {"A": 0, "X": 1, "Y": 2, "Z": 3}.get(axis)
        if idx is not None:
            speedTuple[idx] = speed
        self.moveForever(speed=tuple(speedTuple), is_stop=is_stop)

    def moveForever(self, speed=(0, 0, 0, 0), is_stop=False):
        """ Continuous move. ``speed`` is a 4-tuple in (A, X, Y, Z) order. """
        for axisSpeed, axis in zip(speed, ("A", "X", "Y", "Z")):
            node = self._node(axis)
            if node is None:
                continue
            try:
                if is_stop or axisSpeed == 0:
                    self._client.motor.stop(axis=0, node_id=node)
                else:
                    self._client.motor.move(axis=0, position=0, speed=int(abs(axisSpeed)),
                                            is_forever=True, node_id=node)
            except Exception as e:
                self.__logger.error(f"moveForever failed for axis {axis}: {e}")

    def measure(self, sensorID=0, NAvg=100):
        self.__logger.warning("measure(): analog sensor read not supported over CANopen.")
        return None

    def setupPIDcontroller(self, PIDactive=1, Kp=100, Ki=10, Kd=1, target=500, PID_updaterate=200):
        self.__logger.warning("setupPIDcontroller(): not supported over CANopen (v1).")
        return None

    def setEnabled(self, is_enabled):
        self.is_enabled = is_enabled

    def setSpeed(self, speed, axis=None):
        if type(speed) == int and axis is None:
            self._speed["X"] = speed
            self._speed["Y"] = speed
            self._speed["Z"] = speed
            self._speed["A"] = speed
        else:
            self._speed[axis] = speed

    def setPosition(self, value, axis):
        self._position[axis] = value

    def setPositionOnDevice(self, value, axis):
        """ Update the host position. Setting an arbitrary device-side position
        is not exposed over the current OD (MOTOR_ACTUAL_POSITION is read-only);
        only homing resets it to 0 on the firmware. """
        self.setPosition(value, axis)
        if value != 0:
            self.__logger.debug(
                f"setPositionOnDevice({axis}={value}): device-side set unsupported over CANopen; "
                "host position updated only.")

    def setPositionFromDevice(self, positionArray):
        """ Accept a full [A, X, Y, Z] array (parity with ESP32 manager). """
        posDict = {"UC2CANOpenStage": {}}
        for iAxis, axisName in enumerate(["A", "X", "Y", "Z"]):
            positionOffsetCorrected = positionArray[iAxis] - self.getStageOffsetAxis(axisName)
            self.setPosition(positionOffsetCorrected, axisName)
            posDict["UC2CANOpenStage"][axisName] = positionOffsetCorrected
        self._commChannel.sigUpdateMotorPosition.emit(posDict)

    def closeEvent(self):
        pass

    def getPosition(self):
        """ Read every axis position from the device, offset-corrected. """
        posDict = {}
        for axis in ("A", "X", "Y", "Z"):
            node = self._node(axis)
            raw = float(self._position.get(axis, 0)) + self.getStageOffsetAxis(axis) if hasattr(self, '_position') else 0.0
            if node is not None:
                try:
                    raw = float(self._client.motor.get_position(axis=0, node_id=node))
                except Exception as e:
                    self.__logger.error(f"getPosition({axis}) failed: {e}")
            posDict[axis] = raw - self.getStageOffsetAxis(axis)
        return posDict

    # ------------------------------------------------------------------
    # Stopping
    # ------------------------------------------------------------------
    def forceStop(self, axis):
        if axis == "X":
            self.stop_x()
        elif axis == "Y":
            self.stop_y()
        elif axis == "Z":
            self.stop_z()
        elif axis == "A":
            self.stop_a()
        else:
            self.stopAll()

    def get_abs(self, axis):
        return self._position[axis]

    def _stopAxis(self, axis):
        node = self._node(axis)
        if node is None:
            return
        try:
            self._client.motor.stop(axis=0, node_id=node)
        except Exception as e:
            self.__logger.error(f"stop {axis} failed: {e}")

    def stop_x(self):
        self._stopAxis("X")

    def stop_y(self):
        self._stopAxis("Y")

    def stop_z(self):
        self._stopAxis("Z")

    def stop_a(self):
        self._stopAxis("A")

    def stopAll(self):
        for axis in ("X", "Y", "Z", "A"):
            self._stopAxis(axis)

    # ------------------------------------------------------------------
    # Homing
    # ------------------------------------------------------------------
    def _homeAxisDevice(self, axis, isBlocking=False):
        """ Trigger an endstop homing run on one axis-node via SDO. """
        node = self._node(axis)
        if node is None or OD is None:
            self.__logger.error(f"Cannot home axis {axis}: no node/OD.")
            return
        speed = getattr(self, f"homeSpeed{axis}")
        direction = getattr(self, f"homeDirection{axis}")
        polarity = getattr(self, f"homeEndstoppolarity{axis}")
        release = getattr(self, f"homeEndposRelease{axis}")
        timeout_ms = getattr(self, f"homeTimeout{axis}") or 20000
        try:
            self._rs232manager.sdo_write(node, OD.HOMING_SPEED, 1, int(speed), "u32")
            self._rs232manager.sdo_write(node, OD.HOMING_DIRECTION, 1, int(direction) & 0xFF, "u8")
            self._rs232manager.sdo_write(node, OD.HOMING_ENDSTOP_POLARITY, 1, int(polarity), "u8")
            self._rs232manager.sdo_write(node, OD.HOMING_ENDSTOP_RELEASE, 1, int(release), "i32")
            self._rs232manager.sdo_write(node, OD.HOMING_TIMEOUT, 1, int(timeout_ms), "u32")
            self._rs232manager.sdo_write(node, OD.HOMING_COMMAND, 1, 1, "u8")
        except Exception as e:
            self.__logger.error(f"home {axis} failed: {e}")
            return
        if isBlocking:
            # Homing timeout is in ms; give wait_for_idle a little headroom.
            self._waitNode(node, (timeout_ms / 1000.0) + 5.0)
        self.setPosition(axis=axis, value=0)

    def doHome(self, axis, isBlocking=False, homeDirection=None, homeSpeed=None,
               homeEndstoppolarity=None, homeEndposRelease=None, homeTimeout=None):
        if axis == "X" and (self.homeXenabled or abs(self.homeStepsX) > 0):
            self.home_x(isBlocking)
        if axis == "Y" and (self.homeYenabled or abs(self.homeStepsY) > 0):
            self.home_y(isBlocking)
        if axis == "Z" and (self.homeZenabled or abs(self.homeStepsZ) > 0):
            self.home_z(isBlocking)
        if axis == "A" and (self.homeAenabled or abs(self.homeStepsA) > 0):
            self.home_a(isBlocking)

    def home_x(self, isBlocking=False, *args, **kwargs):
        if self._safeDistanceZHoming != 0:
            self.move(value=self._zPositionPriorHoming + self._safeDistanceZHoming,
                      speed=self.homeSpeedZ, axis="Z", is_absolute=True, is_blocking=True)
        if abs(self.homeStepsX) > 0:
            self.move(value=self.homeStepsX, speed=self.homeSpeedX, axis="X", is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(self.homeStepsX) * np.abs(self.homeEndposReleaseX),
                      speed=self.homeSpeedX, axis="X", is_absolute=False, is_blocking=True)
            self.setPosition(axis="X", value=0)
            self.setPositionOnDevice(value=0, axis="X")
        elif self.homeXenabled:
            self._homeAxisDevice("X", isBlocking)
        else:
            self.__logger.info("No homing parameters set for X axis or not enabled in settings.")

    def home_y(self, isBlocking=False, *args, **kwargs):
        if self._safeDistanceZHoming != 0:
            self.move(value=self._zPositionPriorHoming + self._safeDistanceZHoming,
                      speed=self.homeSpeedZ, axis="Z", is_absolute=True, is_blocking=True)
        if abs(self.homeStepsY) > 0:
            self.move(value=self.homeStepsY, speed=self.homeSpeedY, axis="Y", is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(self.homeStepsY) * np.abs(self.homeEndposReleaseY),
                      speed=self.homeSpeedY, axis="Y", is_absolute=False, is_blocking=True)
            self.setPosition(axis="Y", value=0)
            self.setPositionOnDevice(value=0, axis="Y")
        elif self.homeYenabled:
            self._homeAxisDevice("Y", isBlocking)
        else:
            self.__logger.info("No homing parameters set for Y axis or not enabled in settings.")

    def home_z(self, isBlocking=False, *args, **kwargs):
        if abs(self.homeStepsZ) > 0:
            self.move(value=self.homeStepsZ, speed=self.homeSpeedZ, axis="Z", is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(self.homeStepsZ) * np.abs(self.homeEndposReleaseZ),
                      speed=self.homeSpeedZ, axis="Z", is_absolute=False, is_blocking=True)
            self.setPosition(axis="Z", value=0)
            self.setPositionOnDevice(value=0, axis="Z")
        elif self.homeZenabled:
            self._homeAxisDevice("Z", isBlocking)
        else:
            self.__logger.info("No homing parameters set for Z axis or not enabled in settings.")
        self._zPositionPriorHoming = 0

    def home_a(self, isBlocking=False, *args, **kwargs):
        if abs(self.homeStepsA) > 0:
            self.move(value=self.homeStepsA, speed=self.homeSpeedA, axis="A", is_absolute=False, is_blocking=True)
            self.move(value=-np.sign(self.homeStepsA) * np.abs(self.homeEndposReleaseA),
                      speed=self.homeSpeedA, axis="A", is_absolute=False, is_blocking=True)
            self.setPosition(axis="A", value=0)
            self.setPositionOnDevice(value=0, axis="A")
        elif self.homeAenabled:
            self._homeAxisDevice("A", isBlocking)
        else:
            self.__logger.info("No homing parameters set for A axis or not enabled in settings.")

    def home_xyz(self):
        if self.homeXenabled and self.homeYenabled and self.homeZenabled:
            self.home_x(isBlocking=True)
            self.home_y(isBlocking=True)
            self.home_z(isBlocking=True)
            [self.setPosition(axis=axis, value=0) for axis in ["X", "Y", "Z"]]

    # ------------------------------------------------------------------
    # Stage scanning (hardware-triggered scanning isn't exposed over the
    # current CANopen OD; kept as logged no-ops for controller compatibility).
    # ------------------------------------------------------------------
    def startStageScanning(self, *args, **kwargs):
        self.__logger.warning("startStageScanning(): not supported over CANopen (v1).")

    def stopStageScanning(self):
        self.__logger.warning("stopStageScanning(): not supported over CANopen (v1).")

    def start_stage_scanning(self, *args, **kwargs):
        self.__logger.warning("start_stage_scanning(): not supported over CANopen (v1).")
        return None

    def stop_stage_scanning(self):
        self.__logger.warning("stop_stage_scanning(): not supported over CANopen (v1).")

    def register_stagescan_callback(self, on_stagescan_complete_fct):
        self._stagescan_callback = on_stagescan_complete_fct

    def unregister_stagescan_callback(self):
        self._stagescan_callback = None

    def reset_stagescan_complete(self):
        pass

    # ------------------------------------------------------------------
    # Convenience positions
    # ------------------------------------------------------------------
    def moveToSampleMountingPosition(self, speed=10000, is_blocking=True):
        value = (self.sampleLoadingPositions["X"], self.sampleLoadingPositions["Y"], self.sampleLoadingPositions["Z"])
        self.move(value=value, axis="XYZ", speed=(speed, speed, speed), is_absolute=True, is_blocking=is_blocking)

    def moveToSampleLoadingPosition(self, speed=10000, is_blocking=True):
        value = (self.sampleLoadingPositions["X"], self.sampleLoadingPositions["Y"], self.sampleLoadingPositions["Z"])
        self.move(value=value, axis="XYZ", speed=(speed, speed, speed), is_absolute=True, is_blocking=is_blocking)

    def moveToTransportPosition(self, speed=10000, is_blocking=True):
        value = (self.transportPositions["X"], self.transportPositions["Y"], self.transportPositions["Z"])
        self.move(value=value, axis="XYZ", speed=(speed, speed, speed), is_absolute=True, is_blocking=is_blocking)
        try:
            self.move(value=self.transportPositions["A"], speed=speed, axis="A",
                      is_absolute=True, is_blocking=is_blocking)
        except Exception as e:
            self.__logger.error(f"Could not move A to transport position: {e}")

    def getTransportPositions(self):
        return dict(self.transportPositions)

    def setTransportPositions(self, positions=None):
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

    # ------------------------------------------------------------------
    # Stage offset / calibration
    # ------------------------------------------------------------------
    def getDevicePositionAxis(self, axis="X"):
        """ Live raw device position for one axis (no offset applied). """
        node = self._node(axis)
        if node is not None:
            try:
                return float(self._client.motor.get_position(axis=0, node_id=node))
            except Exception as e:
                self.__logger.error(f"getDevicePositionAxis({axis}) failed: {e}")
        return float(self._position.get(axis, 0)) + float(self.stageOffsetPositions.get(axis, 0))

    def setStageOffsetAxis(self, knownPosition=0, currentDevicePosition=None, knownOffset=None, axis="X"):
        if knownOffset is not None:
            offset = float(knownOffset)
        else:
            if currentDevicePosition is None:
                currentDevicePosition = self.getDevicePositionAxis(axis)
            offset = float(currentDevicePosition) - float(knownPosition)
        self.stageOffsetPositions[axis] = offset
        self.__logger.info(
            f"Set offset for {axis} axis to {offset} um "
            f"(device={currentDevicePosition}, known={knownPosition}, knownOffset={knownOffset}).")

    def resetStageOffsetAxis(self, axis="X"):
        self.__logger.info(f"Resetting stage offset for {axis} axis.")
        self.stageOffsetPositions[axis] = 0

    def getStageOffsetAxis(self, axis="X"):
        try:
            return self.stageOffsetPositions[axis]
        except KeyError:
            self.__logger.error(f"Axis {axis} not found in stageOffsetPositions.")
            return 0

    # ------------------------------------------------------------------
    # Frame homing (collision-safe global homing) — ported from ESP32 manager
    # ------------------------------------------------------------------
    def _emitHomingState(self, phase=None, axis=None, axisStatus=None, message=None,
                         active=None, cancelled=None):
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
        if not self._frameHomingState.get("active", False):
            return
        self.__logger.info("Frame homing cancellation requested.")
        self._frameHomingCancel.set()
        try:
            self.stopAll()
        except Exception as e:
            self.__logger.error(f"stopAll during cancel failed: {e}")

    def frameHomingProcedure(self, is_blocking=False):
        if self._frameHomingState.get("active", False):
            self.__logger.warning("Frame homing already in progress; ignoring start request.")
            return

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
                self._zPositionPriorHoming = self.getPosition()["Z"]
                if aborted():
                    return

                self._emitHomingState(phase="homing_z", axis="Z", axisStatus="homing",
                                      message="Homing Z axis")
                self.home_z(isBlocking=True)
                self._emitHomingState(axis="Z", axisStatus="done")
                if aborted():
                    return

                self._emitHomingState(phase="lifting_z", message="Lifting Z to safe height")
                self.move(value=safeLift, speed=self.homeSpeedZ, axis="Z",
                          is_absolute=True, is_blocking=True)
                if aborted():
                    return

                self._emitHomingState(phase="homing_x", axis="X", axisStatus="homing",
                                      message="Homing X axis")
                self.home_x(isBlocking=True)
                self.move(value=1000, speed=self.homeSpeedX, axis="X", is_absolute=True, is_blocking=False)
                self._emitHomingState(axis="X", axisStatus="done")
                if aborted():
                    return

                self._emitHomingState(phase="homing_y", axis="Y", axisStatus="homing",
                                      message="Homing Y axis")
                self.home_y(isBlocking=True)
                self.move(value=1000, speed=self.homeSpeedY, axis="Y", is_absolute=True, is_blocking=False)
                self._emitHomingState(axis="Y", axisStatus="done")
                if aborted():
                    return

                self._emitHomingState(phase="restoring_z", message="Restoring Z position")
                restoreZ = self._zPositionPriorHoming
                if abs(restoreZ) < abs(safeLift):
                    restoreZ = safeLift
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
        # No separate firmware home-status stream on CANopen; frameHomingProcedure
        # remains the source of truth for UI progress.
        self.__logger.debug("register_homing_callback(): not used on CANopen path.")

    # ------------------------------------------------------------------
    # Motor Settings API — unified configuration interface (parity w/ ESP32)
    # ------------------------------------------------------------------
    def getMotorSettings(self) -> dict:
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
        axis = axis.upper()
        motion = {
            'stepSize': self.stepSizes.get(axis, 1),
            'maxSpeed': self.maxSpeed.get(axis, 10000),
            'speed': self._speed.get(axis, 10000),
            'acceleration': self.acceleration.get(axis, MAX_ACCEL),
        }
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

        homing = {
            'enabled': getattr(self, f'home{axis}enabled', False),
            'speed': getattr(self, f'homeSpeed{axis}'),
            'direction': getattr(self, f'homeDirection{axis}'),
            'endstopPolarity': getattr(self, f'homeEndstoppolarity{axis}'),
            'endposRelease': getattr(self, f'homeEndposRelease{axis}'),
            'timeout': getattr(self, f'homeTimeout{axis}'),
            'homeOnStart': getattr(self, f'homeOnStart{axis}'),
            'homeSteps': getattr(self, f'homeSteps{axis}'),
        }
        if axis in ('X', 'Y', 'Z'):
            limits = {'enabled': getattr(self, f'limit{axis}enabled')}
        else:
            limits = {'enabled': False}
        return {'axis': axis, 'motion': motion, 'homing': homing, 'limits': limits}

    def setMotorSettingsForAxis(self, axis: str, settings: dict) -> dict:
        axis = axis.upper()
        result = {'axis': axis, 'updated': [], 'errors': []}
        try:
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
                    setattr(self, f'backlash{axis}', motion['backlash'])
                    result['updated'].append('backlash')
                if 'minPos' in motion:
                    minPos = motion['minPos'] if motion['minPos'] is not None else float('-inf')
                    setattr(self, f'min{axis}', minPos)
                    result['updated'].append('minPos')
                if 'maxPos' in motion:
                    maxPos = motion['maxPos'] if motion['maxPos'] is not None else float('inf')
                    setattr(self, f'max{axis}', maxPos)
                    result['updated'].append('maxPos')
                if any(k in motion for k in ['minPos', 'maxPos', 'stepSize', 'backlash']):
                    try:
                        self.setupMotor(getattr(self, f'min{axis}'), getattr(self, f'max{axis}'),
                                        self.stepSizes[axis], getattr(self, f'backlash{axis}'), axis)
                    except Exception as e:
                        result['errors'].append(f'Failed to update motor setup: {str(e)}')

            if 'homing' in settings:
                homing = settings['homing']
                mapping = {
                    'enabled': f'home{axis}enabled',
                    'speed': f'homeSpeed{axis}',
                    'endstopPolarity': f'homeEndstoppolarity{axis}',
                    'endposRelease': f'homeEndposRelease{axis}',
                    'timeout': f'homeTimeout{axis}',
                    'homeOnStart': f'homeOnStart{axis}',
                    'homeSteps': f'homeSteps{axis}',
                }
                for key, attr in mapping.items():
                    if key in homing:
                        setattr(self, attr, homing[key])
                        result['updated'].append(f'homing.{key}')
                if 'direction' in homing:
                    setattr(self, f'homeDirection{axis}', 1 if homing['direction'] > 0 else -1)
                    result['updated'].append('homing.direction')

            if 'limits' in settings:
                limits = settings['limits']
                if 'enabled' in limits and axis in ('X', 'Y', 'Z'):
                    setattr(self, f'limit{axis}enabled', limits['enabled'])
                    result['updated'].append('limits.enabled')

            result['success'] = True
            self.__logger.info(f"Updated motor settings for axis {axis}: {result['updated']}")
        except Exception as e:
            result['success'] = False
            result['errors'].append(str(e))
            self.__logger.error(f"Error updating motor settings for axis {axis}: {e}")
        return result

    def getTMCSettingsForAxis(self, axis: str) -> dict:
        axis = axis.upper()
        result = {'axis': axis, 'success': False}
        node = self._node(axis)
        if node is None or OD is None:
            result['error'] = "No CAN node / OD available"
            return result
        try:
            result['settings'] = {
                'msteps': self._rs232manager.sdo_read(node, OD.TMC_MICROSTEPS, 1, "u16"),
                'rmsCurrent': self._rs232manager.sdo_read(node, OD.TMC_RMS_CURRENT, 1, "u16"),
                'sgthrs': self._rs232manager.sdo_read(node, OD.TMC_STALLGUARD_THRESHOLD, 1, "u8"),
                'semin': self._rs232manager.sdo_read(node, OD.TMC_COOLSTEP_SEMIN, 1, "u8"),
                'semax': self._rs232manager.sdo_read(node, OD.TMC_COOLSTEP_SEMAX, 1, "u8"),
                'blankTime': self._rs232manager.sdo_read(node, OD.TMC_BLANK_TIME, 1, "u8"),
                'toff': self._rs232manager.sdo_read(node, OD.TMC_TOFF, 1, "u8"),
            }
            result['success'] = True
        except Exception as e:
            result['error'] = str(e)
            self.__logger.error(f"Error getting TMC settings for axis {axis}: {e}")
        return result

    def setTMCSettingsForAxis(self, axis: str, settings: dict) -> dict:
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
                timeout=settings.get('timeout', 1),
            )
            result['success'] = True
            self.__logger.info(f"Updated TMC settings for axis {axis}")
        except Exception as e:
            result['error'] = str(e)
            self.__logger.error(f"Error updating TMC settings for axis {axis}: {e}")
        return result

    def setGlobalMotorSettings(self, settings: dict) -> dict:
        result = {'updated': [], 'errors': []}
        try:
            if 'axisOrder' in settings:
                self.setAxisOrder(order=settings['axisOrder'])
                result['updated'].append('axisOrder')
            if 'isCoreXY' in settings:
                self.setIsCoreXY(isCoreXY=settings['isCoreXY'])
                result['updated'].append('isCoreXY')
            if 'isEnabled' in settings:
                self.is_enabled = settings['isEnabled']
                result['updated'].append('isEnabled')
            if 'enableAuto' in settings:
                self.enableauto = settings['enableAuto']
                self.enableMotors(enable=self.is_enabled, autoenable=self.enableauto)
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

    def finalize(self):
        # The CAN bus connection is owned by the rs232 UC2CANOpenManager.
        pass


# Copyright (C) 2020-2024 ImSwitch developers
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
