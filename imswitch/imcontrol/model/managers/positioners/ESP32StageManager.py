from imswitch.imcommon.model import initLogger
from .PositionerManager import PositionerManager
import time
import numpy as np
import os
import json

MAX_ACCEL = 1000000
PHYS_FACTOR = 1
gTIMEOUT = 100
class ESP32StageManager(PositionerManager):
    def __init__(self, positionerInfo, name, **lowLevelManagers):
        super().__init__(positionerInfo, name, initialPosition={axis: 0 for axis in positionerInfo.axes}, initialSpeed={axis: 0 for axis in positionerInfo.axes})
        self._rs232manager = lowLevelManagers['rs232sManager'][positionerInfo.managerProperties['rs232device']]
        self._commChannel = lowLevelManagers['commChannel']
        self.__logger = initLogger(self, instanceName=name)

        # Grab motor object
        self._motor = self._rs232manager._esp32.motor
        self._homeModule = self._rs232manager._esp32.home

        # get bootup position and write to GUI
        self._position = self.getPosition()

        # Calibrated stepsizes in steps/µm
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

        self.stageOffsetPositions = {}
        self.stageOffsetPositions["X"] = positionerInfo.stageOffsets.get('stageOffsetPositionX',0)
        self.stageOffsetPositions["Y"] = positionerInfo.stageOffsets.get('stageOffsetPositionY',0)
        self.stageOffsetPositions["Z"] = positionerInfo.stageOffsets.get('stageOffsetPositionZ',0)
        self.stageOffsetPositions["A"] = positionerInfo.stageOffsets.get('stageOffsetPositionA',0)
        # Setup homing coordinates and speed
        # X
        self.setHomeParametersAxis(axis="X", speed=positionerInfo.managerProperties.get('homeSpeedX', 15000),
                                   direction=positionerInfo.managerProperties.get('homeDirectionX', -1),
                                   endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityX', 1),
                                   endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseX', 1),
                                   timeout=positionerInfo.managerProperties.get('homeTimeoutX', 20000))
        # Y
        self.setHomeParametersAxis(axis="Y", speed=positionerInfo.managerProperties.get('homeSpeedY', 15000),
                                      direction=positionerInfo.managerProperties.get('homeDirectionY', -1),
                                      endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityY', 1),
                                      endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseY', 1),
                                      timeout=positionerInfo.managerProperties.get('homeTimeoutY', 20000))

        # Z
        self.setHomeParametersAxis(axis="Z", speed=positionerInfo.managerProperties.get('homeSpeedZ', 15000),
                                        direction=positionerInfo.managerProperties.get('homeDirectionZ', -1),
                                        endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityZ', 1),
                                        endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseZ', 1),
                                        timeout=positionerInfo.managerProperties.get('homeTimeoutZ', 20000))

        # A
        self.setHomeParametersAxis(axis="A", speed=positionerInfo.managerProperties.get('homeSpeedA', 15000),
                                        direction=positionerInfo.managerProperties.get('homeDirectionA', -1),
                                        endstoppolarity=positionerInfo.managerProperties.get('homeEndstoppolarityA', 1),
                                        endposrelease=positionerInfo.managerProperties.get('homeEndposReleaseA', 1),
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

        # Setup motors
        self.setupMotor(self.minX, self.maxX, self.stepSizes["X"], self.backlashX, self.stageOffsetPositions["X"], "X")
        self.setupMotor(self.minY, self.maxY, self.stepSizes["Y"], self.backlashY, self.stageOffsetPositions["Y"], "Y")
        self.setupMotor(self.minZ, self.maxZ, self.stepSizes["Z"], self.backlashZ, self.stageOffsetPositions["Z"], "Z")
        self.setupMotor(self.minA, self.maxA, self.stepSizes["A"], self.backlashA, self.stageOffsetPositions["A"], "A")

        # Setup Motor drivers (TMC - if available)
        #    def set_tmc_parameters(self, axis=0, msteps=None, rms_current=None, stall_value=None, sgthrs=None, semin=None, semax=None, blank_time=None, toff=None, timeout=1):
        if 0:
            if positionerInfo.managerProperties.get('mstepsX', 16) is not None:
                self.setupMotorDriver(axis="X", msteps=positionerInfo.managerProperties.get('mstepsX', 16), rms_current=positionerInfo.managerProperties.get('rms_currentX', 500), sgthrs=positionerInfo.managerProperties.get('sgthrsX', 10), semin=positionerInfo.managerProperties.get('seminX', 5), semax=positionerInfo.managerProperties.get('semaxX', 2), blank_time=positionerInfo.managerProperties.get('blank_timeX', 24), toff=positionerInfo.managerProperties.get('toffX', 3), timeout=1)
            if positionerInfo.managerProperties.get('mstepsY', 16) is not None:
                self.setupMotorDriver(axis="Y", msteps=positionerInfo.managerProperties.get('mstepsY', 16), rms_current=positionerInfo.managerProperties.get('rms_currentY', 500), sgthrs=positionerInfo.managerProperties.get('sgthrsY', 10), semin=positionerInfo.managerProperties.get('seminY', 5), semax=positionerInfo.managerProperties.get('semaxY', 2), blank_time=positionerInfo.managerProperties.get('blank_timeY', 24), toff=positionerInfo.managerProperties.get('toffY', 3), timeout=1)
            if positionerInfo.managerProperties.get('mstepsZ', 16) is not None:
                self.setupMotorDriver(axis="Z", msteps=positionerInfo.managerProperties.get('mstepsZ', 16), rms_current=positionerInfo.managerProperties.get('rms_currentZ', 500), sgthrs=positionerInfo.managerProperties.get('sgthrsZ', 10), semin=positionerInfo.managerProperties.get('seminZ', 5), semax=positionerInfo.managerProperties.get('semaxZ', 2), blank_time=positionerInfo.managerProperties.get('blank_timeZ', 24), toff=positionerInfo.managerProperties.get('toffZ', 3), timeout=1)
            if positionerInfo.managerProperties.get('mstepsA', 16) is not None:
                self.setupMotorDriver(axis="A", msteps=positionerInfo.managerProperties.get('mstepsA', 16), rms_current=positionerInfo.managerProperties.get('rms_currentA', 500), sgthrs=positionerInfo.managerProperties.get('sgthrsA', 10), semin=positionerInfo.managerProperties.get('seminA', 5), semax=positionerInfo.managerProperties.get('semaxA', 2), blank_time=positionerInfo.managerProperties.get('blank_timeA', 24), toff=positionerInfo.managerProperties.get('toffA', 3), timeout=1)

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

        # set speed for all axes
        self._speed = {"X": positionerInfo.managerProperties.get('speedX', 10000),
                        "Y": positionerInfo.managerProperties.get('speedY', 10000),
                        "Z": positionerInfo.managerProperties.get('speedZ', 10000),
                        "A": positionerInfo.managerProperties.get('speedA', 10000)}

        # try to register the callback
        try:
            # if event "0" is triggered, the callback function to update the stage positions
            # will be called
            self._motor.register_callback(0,callbackfct=self.setPositionFromDevice)
        except Exception as e:
            self.__logger.error(f"Could not register callback: {e}")

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

    def setupMotor(self, minPos, maxPos, stepSize, backlash, offset, axis):
        self._motor.setup_motor(axis=axis, minPos=minPos, maxPos=maxPos, stepSize=stepSize, backlash=backlash, offset=offset)

    def setupMotorDriver(self, axis="X", msteps=None, rms_current=None, stall_value=None, sgthrs=None, semin=None, semax=None, blank_time=None, toff=None, timeout=1):
        self._motor.set_tmc_parameters(axis=axis, msteps=msteps, rms_current=rms_current, stall_value=stall_value, sgthrs=sgthrs, semin=semin, semax=semax, blank_time=blank_time, toff=toff, timeout=timeout)

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
            # don't move to negative positions
            if not is_absolute and value == 0: return
            if self.limitXenabled and is_absolute and value < 0: return
            elif self.limitXenabled and not is_absolute and self._position[axis] + value < 0: return
            self._motor.move_x(value, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout, is_reduced=is_reduced)
            if not is_absolute: self._position[axis] = self._position[axis] + value
            else: self._position[axis] = value
        elif axis == 'Y' and speed >0:
            # don't move to negative positions
            if not is_absolute and value == 0: return
            if self.limitYenabled and is_absolute and value < 0: return
            elif self.limitYenabled and not is_absolute and self._position[axis] + value < 0: return
            self._motor.move_y(value, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout)
            if not is_absolute: self._position[axis] = self._position[axis] + value
            else: self._position[axis] = value
        elif axis == 'Z' and speed >0:
            # don't move to negative positions
            if not is_absolute and value == 0: return
            if self.limitZenabled and is_absolute and value < 0: return
            elif self.limitZenabled and not is_absolute and self._position[axis] + value < 0: return
            self._motor.move_z(value, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, is_dualaxis=self.isDualAxis, timeout=timeout, is_reduced=is_reduced)
            if not is_absolute: self._position[axis] = self._position[axis] + value
            else: self._position[axis] = value
        elif axis == 'A' and speed >0:
            # don't move to negative positions
            #if is_absolute and value < 0: return
            #elif not is_absolute and self._position[axis] + value < 0: return
            if not is_absolute and value == 0: return
            self._motor.move_a(value, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout, is_reduced=is_reduced)
            if not is_absolute: self._position[axis] = self._position[axis] + value
            else: self._position[axis] = value
        elif axis == 'XY':
            # don't move to negative positions
            if (self.limitXenabled and self.limitYenabled) and is_absolute and (value[0] < 0 or value[1] < 0): return
            elif (self.limitXenabled and self.limitYenabled) and not is_absolute and (self._position["X"] + value[0] < 0 or self._position["Y"] + value[1] < 0): return
            self._motor.move_xy(value, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout, is_reduced=is_reduced)
            for i, iaxis in enumerate(("X", "Y")):
                if not is_absolute:
                    self._position[iaxis] = self._position[iaxis] + value[i]
                else:
                    self._position[iaxis] = value[i]
        elif axis == 'XYZ':
            self._motor.move_xyz(value, speed, acceleration=acceleration, is_absolute=is_absolute, is_enabled=isEnable, is_blocking=is_blocking, timeout=timeout, is_reduced=is_reduced)
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

    def moveForever(self, speed=(0, 0, 0, 0), is_stop=False):
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
            self.setPosition(positionArray[iAxis] , axisName)
            posDict["ESP32Stage"][axisName] = positionArray[iAxis]
        self._commChannel.sigUpdateMotorPosition.emit(posDict)

    def closeEvent(self):
        pass

    def getPosition(self):
        # load position from device
        # t,x,y,z
        try:
            allPositions = 1.*self._motor.get_position()
            return {"X": allPositions[1], "Y": allPositions[2], "Z": allPositions[3], "A": allPositions[0]}
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

    def doHome(self, axis, isBlocking=False):
        if axis == "X" and (self.homeXenabled or abs(self.homeStepsX)>0):
            self.home_x(isBlocking)
        if axis == "Y" and (self.homeYenabled or abs(self.homeStepsY)>0):
            self.home_y(isBlocking)
        if axis == "Z" and (self.homeZenabled or abs(self.homeStepsZ)>0):
            self.home_z(isBlocking)
        if axis == "A" and (self.homeAenabled or abs(self.homeStepsA)>0):
            self.home_a(isBlocking)

    def home_x(self, isBlocking=False):
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
        self.setPosition(axis="X", value=0)

    def home_y(self,isBlocking=False):
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
        self.setPosition(axis="Y", value=0)

    def home_z(self,isBlocking=False):
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
        self.setPosition(axis="Z", value=0)

    def home_a(self,isBlocking=False):
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
        self.setPosition(axis="A", value=0)

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

    def setStageOffsetAxis(self, knownOffset:float=None, axis="X"):
        try:
            self.stageOffsetPositions[axis] = knownOffset
        except KeyError:
            self.__logger.error(f"Axis {axis} not found in stageOffsetPositions.")
        self.__logger.info(f"Set offset for {axis} axis to {knownOffset} mum.")
        self._motor.set_offset(axis=axis, offset=knownOffset)

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
                             ystart=0, ystep=1, ny=100, tsettle=0.1, tExposure=50, illumination=None, led=None):
        """
        Start a stage scanning operation with the given parameters.
        :param xstart: Starting position in X direction.
        :param xstep: Step size in X direction.
        :param nx: Number of steps in X direction.
        :param ystart: Starting position in Y direction.
        :param ystep: Step size in Y direction.
        :param ny: Number of steps in Y direction.
        :param settle: Settle time after each step.
        :param illumination: Optional illumination settings.
        :param led: Optional LED settings.
        """
        if illumination is None:
            illumination = (0,0,0,0)  # Default to no illumination
        if led is None:
            led = 0
        r = self._motor.start_stage_scanning(xstart=xstart, xstep=xstep, nx=nx,
                                         ystart=ystart, ystep=ystep, ny=ny,
                                         tsettle=tsettle, tExposure=tExposure, illumination=illumination, led=led)
        return r

    def stop_stage_scanning(self):
        """
        Stop the current stage scanning operation.
        """
        self._motor.stop_stage_scanning()
        self.__logger.info("Stage scanning stopped.")


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
