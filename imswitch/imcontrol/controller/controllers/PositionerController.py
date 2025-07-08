from typing import Dict, List

from imswitch import IS_HEADLESS
from imswitch.imcommon.model import APIExport
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import initLogger
from typing import Optional, Union
from imswitch.imcontrol.model import configfiletools


class PositionerController(ImConWidgetController):
    """ Linked to PositionerWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.settingAttr = False

        self.__logger = initLogger(self, tryInheritParent=True)

        # Set up positioners
        for pName, pManager in self._master.positionersManager:
            if not pManager.forPositioning:
                continue

            hasSpeed = hasattr(pManager, 'speed')
            hasHome = hasattr(pManager, 'home')
            hasStop = hasattr(pManager, 'stop')
            if not IS_HEADLESS: self._widget.addPositioner(pName, pManager.axes, hasSpeed, hasHome, hasStop)
            for axis in pManager.axes:
                self.setSharedAttr(pName, axis, _positionAttr, pManager.position[axis])
                if hasSpeed:
                    self.setSharedAttr(pName, axis, _speedAttr, pManager.speed[axis])
                if hasHome:
                    self.setSharedAttr(pName, axis, _homeAttr, pManager.home[axis])
                if hasStop:
                    self.setSharedAttr(pName, axis, _stopAttr, pManager.stop[axis])

        # Connect CommunicationChannel signals
        if 0: #IS_HEADLESS:IS_HEADLESS:
            self._commChannel.sharedAttrs.sigAttributeSet.connect(self.attrChanged, check_nargs=False)
        else:
            self._commChannel.sharedAttrs.sigAttributeSet.connect(self.attrChanged)


        # Connect PositionerWidget signals
        if not IS_HEADLESS:
            self._commChannel.sigUpdateMotorPosition.connect(self.updateAllPositionGUI) # force update position in GUI
            self._widget.sigStepUpClicked.connect(self.stepUp)
            self._widget.sigStepDownClicked.connect(self.stepDown)
            self._widget.sigStepAbsoluteClicked.connect(self.moveAbsolute)
            self._widget.sigHomeAxisClicked.connect(self.homeAxis)
            self._widget.sigStopAxisClicked.connect(self.stopAxis)

    def closeEvent(self):
        self._master.positionersManager.execOnAll(
            lambda p: [p.setPosition(0, axis) for axis in p.axes],
            condition = lambda p: p.resetOnClose
        )

    def getPos(self):
        return self._master.positionersManager.execOnAll(lambda p: p.position)

    def getSpeed(self):
        return self._master.positionersManager.execOnAll(lambda p: p.speed)

    def move(self, positionerName, axis, dist, isAbsolute=None, isBlocking=False, speed=None):
        """ Moves positioner by dist micrometers in the specified axis. """
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]

        # get all speed values from the GUI
        if speed is None:
            if not IS_HEADLESS:
                if axis =="XY":
                    speed = self._widget.getSpeed(positionerName, "X")
                else:
                    speed = self._widget.getSpeed(positionerName, axis)
            else:
                speed = 5000 # FIXME: default speed for headless mode
        # set speed for the positioner
        self.setSpeed(positionerName=positionerName, speed=speed, axis=axis)
        try:
            # special case for UC2 positioner that takes more arguments
            self._master.positionersManager[positionerName].move(dist, axis, isAbsolute, isBlocking)
            if dist is None:
                self.__logger.info(f"Moving {positionerName}, axis {axis}, at speed {str(speed)}")
                self._master.positionersManager[positionerName].moveForeverByAxis(speed=speed, axis=axis, is_stop=~(abs(speed)>0))
        except Exception as e:
            # if the positioner does not have the move method, use the default move method
            self._logger.error(e)
            self._master.positionersManager[positionerName].move(dist, axis)
        if isBlocking: # push signal immediately
            self._commChannel.sigUpdateMotorPosition.emit(self.getPos())
        #self.updatePosition(positionerName, axis)

    def moveForever(self, speed=(0, 0, 0, 0), is_stop=False):
        """ Moves positioner forever. """
        self._master.positionersManager.execOnAll(lambda p: p.moveForever(speed=speed, is_stop=is_stop))

    def setPos(self, positionerName, axis, position):
        """ Moves the positioner to the specified position in the specified axis. """
        self._master.positionersManager[positionerName].setPosition(position, axis)
        self.updatePosition(positionerName, axis)

    def moveAbsolute(self, positionerName, axis):
        self.move(positionerName, axis, self._widget.getAbsPosition(positionerName, axis), isAbsolute=True,
                  isBlocking=False)

    def stepUp(self, positionerName, axis):
        self.move(positionerName, axis, self._widget.getStepSize(positionerName, axis), isAbsolute=False,
                  isBlocking=False)

    def stepDown(self, positionerName, axis):
        self.move(positionerName, axis, -self._widget.getStepSize(positionerName, axis), isAbsolute=False,
                  isBlocking=False)

    def setSpeed(self, positionerName, axis, speed=(1000, 1000, 1000)):
        if positionerName is None or positionerName == "":
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].setSpeed(speed, axis)
        self.setSharedAttr(positionerName, axis, _speedAttr, speed)
        if not IS_HEADLESS: self._widget.setSpeedSize(positionerName, axis, speed)

    def updateAllPositionGUI(self):
        # update all positions for all axes in GUI
        for positionerName in self._master.positionersManager.getAllDeviceNames():
            for axis in self._master.positionersManager[positionerName].axes:
                self.updatePosition(positionerName, axis)
                self.updateSpeed(positionerName, axis)

    def updatePosition(self, positionerName, axis):
        if axis == "XY":
            for axis in (("X", "Y")):
                newPos = self._master.positionersManager[positionerName].position[axis]
                self.setSharedAttr(positionerName, axis, _positionAttr, newPos)
                if not IS_HEADLESS: self._widget.updatePosition(positionerName, axis, newPos)
        else:
            newPos = self._master.positionersManager[positionerName].position[axis]
            self.setSharedAttr(positionerName, axis, _positionAttr, newPos)
            if not IS_HEADLESS: self._widget.updatePosition(positionerName, axis, newPos)

    def updateSpeed(self, positionerName, axis):
        newSpeed = self._master.positionersManager[positionerName].speed[axis]
        self.setSharedAttr(positionerName, axis, _speedAttr, newSpeed)
        if not IS_HEADLESS: self._widget.updateSpeed(positionerName, axis, newSpeed)

    @APIExport(runOnUIThread=True)
    def homeAxis(self, positionerName:str=None, axis:str="X", isBlocking:bool=False):
        self.__logger.debug(f"Homing axis {axis}")
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].doHome(axis, isBlocking=isBlocking)
        self.updatePosition(positionerName, axis)
        self._commChannel.sigUpdateMotorPosition.emit(self.getPos())

    @APIExport()
    def stopAxis(self, positionerName=None, axis="X"):
        self.__logger.debug(f"Stopping axis {axis}")
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].forceStop(axis)

    def attrChanged(self, key, value):
        if self.settingAttr or len(key) != 4 or key[0] != _attrCategory:
            return

        positionerName = key[1]
        axis = key[2]
        if key[3] == _positionAttr:
            self.setPositioner(positionerName, axis, value)

    def setSharedAttr(self, positionerName, axis, attr, value):
        self.settingAttr = True
        try:
            self._commChannel.sharedAttrs[(_attrCategory, positionerName, axis, attr)] = value
        finally:
            self.settingAttr = False

    def setXYPosition(self, x, y):
        positionerX = self.getPositionerNames()[0]
        positionerY = self.getPositionerNames()[1]
        self.__logger.debug(f"Move {positionerX}, axis X, dist {str(x)}")
        self.__logger.debug(f"Move {positionerY}, axis Y, dist {str(y)}")
        # self.move(positionerX, 'X', x)
        # self.move(positionerY, 'Y', y)

    def setZPosition(self, z):
        positionerZ = self.getPositionerNames()[2]
        self.__logger.debug(f"Move {positionerZ}, axis Z, dist {str(z)}")
        # self.move(self.getPositionerNames[2], 'Z', z)

    @APIExport(runOnUIThread=True)
    def enalbeMotors(self, enable=None, enableauto=None):
        try:
            return self._master.positionersManager.enalbeMotors(enable=None, enableauto=None)
        except:
            pass

    @APIExport()
    def getPositionerNames(self) -> List[str]:
        """ Returns the device names of all positioners. These device names can
        be passed to other positioner-related functions. """
        return self._master.positionersManager.getAllDeviceNames()

    @APIExport()
    def getPositionerPositions(self) -> Dict[str, Dict[str, float]]:
        """ Returns the positions of all positioners. """
        return self.getPos()

    @APIExport(runOnUIThread=True)
    def setPositionerStepSize(self, positionerName: str, stepSize: float) -> None:
        """ Sets the step size of the specified positioner to the specified
        number of micrometers. """
        if not IS_HEADLESS: self._widget.setStepSize(positionerName, stepSize)

    @APIExport(runOnUIThread=True)
    def movePositioner(self, positionerName: Optional[str]=None, axis: Optional[str]="X", dist: Optional[float] = None, isAbsolute: bool = False, isBlocking: bool=False, speed: float=None) -> None:
        """ Moves the specified positioner axis by the specified number of
        micrometers. """
        if axis is None or dist is None:
            raise ValueError("Both axis and dist must be specified.")
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        try: # uc2 only
            self.move(positionerName, axis, dist, isAbsolute=isAbsolute, isBlocking=isBlocking, speed=speed)
        except Exception as e:
            self.__logger.error(e)
            self.move(positionerName, axis, dist)

    @APIExport(runOnUIThread=True)
    def movePositionerForever(self, axis="X", speed=0, is_stop=False):
        speed = float(speed)
        if axis == "X": speed = (0, speed, 0, 0)
        elif axis == "Y": speed = (0, 0, speed, 0)
        elif axis == "Z": speed = (0, 0, 0, speed)
        elif axis == "A": speed = (speed, 0, 0, 0)
        else: return
        self.moveForever(speed=speed, is_stop=is_stop)

    @APIExport(runOnUIThread=True)
    def setPositioner(self, positionerName: str, axis: str, position: float) -> None:
        """ Moves the specified positioner axis to the specified position. """
        self.setPos(positionerName, axis, position)

    @APIExport(runOnUIThread=True)
    def setPositionerSpeed(self, positionerName: str, axis: str, speed: float) -> None:
        """ Moves the specified positioner axis to the specified position. """
        self.setSpeed(positionerName, axis, speed)

    @APIExport(runOnUIThread=True)
    def setMotorsEnabled(self, positionerName: str, is_enabled: int) -> None:
        """ Moves the specified positioner axis to the specified position. """
        self._master.positionersManager[positionerName].setEnabled(is_enabled)

    @APIExport(runOnUIThread=True)
    def stepPositionerUp(self, positionerName: str, axis: str) -> None:
        """ Moves the specified positioner axis in positive direction by its
        set step size. """
        self.stepUp(positionerName, axis)

    @APIExport(runOnUIThread=True)
    def stepPositionerDown(self, positionerName: str, axis: str) -> None:
        """ Moves the specified positioner axis in negative direction by its
        set step size. """
        self.stepDown(positionerName, axis)

    @APIExport(runOnUIThread=True)
    def resetStageOffsetAxis(self, positionerName: Optional[str]=None, axis:str="X"):
        """
        Resets the stage offset for the given axis to 0.
        """
        self._logger.debug(f'Resetting stage offset for {axis} axis.')
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].resetStageOffsetAxis(axis=axis)

    @APIExport(runOnUIThread=True)
    def setStageOffsetAxis(self, positionerName: Optional[str]=None, knownPosition:float=0, currentPosition:Optional[float]=None, knownOffset:Optional[float]=None,  axis:str="X"):
        """
        Sets the stage to a known offset aside from the home position.
        knownPosition and currentPosition have to be in physical coordinates (i.e. prior to applying the stepsize)
        """
        self._logger.debug(f'Setting stage offset for {axis} axis.')
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].setStageOffsetAxis(knownPosition=knownPosition, currentPosition=currentPosition, knownOffset=knownOffset, axis=axis)

    @APIExport(runOnUIThread=True)
    def getStageOffsetAxis(self, positionerName: Optional[str]=None, axis:str="X"):
        """
        Returns the stage offset for the given axis.
        """
        self._logger.debug(f'Getting stage offset for {axis} axis.')
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        return self._master.positionersManager[positionerName].getStageOffsetAxis(axis=axis)

    def saveStageOffset(self, positionerName=None, offsetValue=None, axis="X"):
        """ Save the current stage offset to the config file. """
        # This logic is now handled in the manager.
        if positionerName is None:
            positionerName = self._positionerInfo.name if hasattr(self, '_positionerInfo') else None
        if positionerName:
            self._master.positionersManager[positionerName].saveStageOffset(offsetValue=offsetValue, axis=axis)

    @APIExport(runOnUIThread=True, requestType="POST")
    def startStageScan(self, positionerName=None, xstart:float=0, xstep:float=1000, nx:int=20, ystart:float=0,
                       ystep:float=1000, ny:int=10, tsettle:int=5, tExposure:int=50, illumination0: int=0,
                       illumination1: int=0, illumination2: int=0, illumination3: int=0, led:int=0):
        """ Starts a stage scan with the specified parameters.
        Parameters:
            xstart (int): Starting position in X direction.
            xstep (int): Step size in X direction.
            nx (int): Number of steps in X direction.
            ystart (int): Starting position in Y direction.
            ystep (int): Step size in Y direction.
            ny (int): Number of steps in Y direction.
            settle (int): Settle time after each move in seconds.
            illumination (tuple): Illumination settings for the scan.
            led (int): LED index to use for the scan.
        """
        illumination = (illumination0, illumination1, illumination2, illumination3)
        if isinstance(illumination, str):
            # parse from CSV string to float list
            illumination = [float(x) for x in illumination.split(',')]
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self.__logger.debug(f"Starting stage scan with parameters: xstart={xstart}, xstep={xstep}, nx={nx}, "
                            f"ystart={ystart}, ystep={ystep}, ny={ny}, settle={tsettle}, illumination={illumination}, led={led}")

        self._master.positionersManager[positionerName].start_stage_scanning(xstart=xstart, xstep=xstep, nx=nx,
                                                                              ystart=ystart, ystep=ystep, ny=ny,
                                                                              tsettle=tsettle, tExposure=tExposure, illumination=illumination,
                                                                              led=led)
    @APIExport(runOnUIThread=True)
    def stopStageScan(self, positionerName=None):
        """ Stops the current stage scan if one is running. """
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self.__logger.debug(f"Stopping stage scan for positioner {positionerName}")
        self._master.positionersManager[positionerName].stop_stage_scanning()

_attrCategory = 'Positioner'
_positionAttr = 'Position'
_speedAttr = "Speed"
_homeAttr = "Home"
_stopAttr = "Stop"

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
