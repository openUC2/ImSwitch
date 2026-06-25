from typing import Dict, List

from imswitch.imcommon.model import APIExport
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import initLogger
from typing import Optional
from imswitch.imcontrol.model import configfiletools


class PositionerController(ImConWidgetController):
    """ Linked to PositionerWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.settingAttr = False
        self._hasHomedSinceStartup = False
        self._homingRecommendationDismissed = False

        self.__logger = initLogger(self, tryInheritParent=True)

        # Set up positioners
        for pName, pManager in self._master.positionersManager:
            if not pManager.forPositioning:
                continue

            hasSpeed = hasattr(pManager, 'speed')
            hasHome = hasattr(pManager, 'home')
            hasStop = hasattr(pManager, 'stop')
            for axis in pManager.axes:
                self.setSharedAttr(pName, axis, _positionAttr, pManager.position[axis])
                if hasSpeed:
                    self.setSharedAttr(pName, axis, _speedAttr, pManager.speed[axis])
                if hasStop:
                    self.setSharedAttr(pName, axis, _stopAttr, pManager.stop[axis])

        # Connect CommunicationChannel signals
        self._commChannel.sharedAttrs.sigAttributeSet.connect(self.attrChanged)
        
        # Connect position update signal - this updates shared attributes for all modes
        # This is the primary mechanism for keeping metadata in sync with hardware state
        self._commChannel.sigUpdateMotorPosition.connect(self._onMotorPositionUpdate)

        # Register a homing-state callback for managers that support it (ESP32). The
        # firmware "home" messages mark that homing has occurred; the per-axis
        # progress itself is emitted by the manager via sigHomingState.
        for pName, pManager in self._master.positionersManager:
            if hasattr(pManager, 'register_homing_callback'):
                try:
                    pManager.register_homing_callback(self._onHomingDeviceUpdate)
                except Exception as e:
                    self.__logger.error(f"Could not register homing callback for {pName}: {e}")

        # Connect PositionerWidget signals
    def _onHomingDeviceUpdate(self, isHomed):
        """Firmware home-status callback (best-effort).

        Fired by the ESP32 home module with an isHomed[] array whenever a home
        message arrives. Used only to flag that homing has happened; authoritative
        per-axis progress is broadcast by the manager via sigHomingState.
        """
        self._hasHomedSinceStartup = True
        self._homingRecommendationDismissed = False

    def closeEvent(self):
        self._master.positionersManager.execOnAll(
            lambda p: [p.setPosition(0, axis) for axis in p.axes],
            condition = lambda p: p.resetOnClose
        )

    def getPos(self, positionerName:str=None) -> Dict[str, Dict[str, float]]:
        if positionerName is None:
            return self._master.positionersManager.execOnAll(lambda p: p.getPosition())
        else:
            return {positionerName: self._master.positionersManager[positionerName].getPosition()}

    def getSpeed(self):
        return self._master.positionersManager.execOnAll(lambda p: p.speed)

    def move(self, positionerName, axis, dist, isAbsolute=None, isBlocking=False, speed=None):
        """ Moves positioner by dist micrometers in the specified axis. 
        
        For non-blocking moves, the position will be updated asynchronously via 
        sigUpdateMotorPosition signal from the positioner manager.
        """
        if positionerName is None or positionerName == "" or positionerName not in self._master.positionersManager:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]

        # get all speed values from the GUI
        if speed is None:
            speed = 5000 # FIXME: default speed for headless mode
        # set speed for the positioner
        # self.setSpeed(positionerName=positionerName, speed=speed, axis=axis)
                
        try:
            # special case for UC2 positioner that takes more arguments
            self._master.positionersManager[positionerName].move(value=dist, axis=axis, is_absolute=isAbsolute, is_blocking=isBlocking, speed=speed)
            if dist is None:
                self.__logger.info(f"Moving {positionerName}, axis {axis}, at speed {str(speed)}")
                self._master.positionersManager[positionerName].moveForeverByAxis(speed=speed, axis=axis, is_stop=~(abs(speed)>0))
        except Exception as e:
            # if the positioner does not have the move method, use the default move method
            self._logger.error(e)
            self._master.positionersManager[positionerName].move(value=dist, axis=axis)
        # self._commChannel.sigUpdateMotorPosition.emit(self.getPos()) # TODO: Unsure if this is needed - for the ESP motor not as it will update the position itself asynchronously

    def moveForever(self, positionerName: str=None, axis="X", speed=0, is_stop:bool=False):
        """ Moves positioner forever. """
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].moveForever(speed=speed, is_stop=is_stop)

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
        if positionerName is None or positionerName == "" or positionerName not in self._master.positionersManager:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].setSpeed(speed, axis)
        self.setSharedAttr(positionerName, axis, _speedAttr, speed)

    def _onMotorPositionUpdate(self, positionData: Dict = None):
        """
        Handler for sigUpdateMotorPosition signal.
        Updates shared attributes and GUI for all positioners.
        
        This is the central point where motor positions are synced to the metadata system.
        Called both from blocking moves and asynchronous position updates from hardware.
        
        Args:
            positionData: Optional position data dict. If None, positions are read from managers.
        """
        for positionerName in self._master.positionersManager.getAllDeviceNames():
            positioner = self._master.positionersManager[positionerName]
            for axis in positioner.axes:
                self.updatePosition(positionerName, axis)
            # Also update speed if available
            if hasattr(positioner, 'speed'):
                for axis in positioner.axes:
                    self.updateSpeed(positionerName, axis)
    


    def updatePosition(self, positionerName, axis):
        """Update position for a single axis and sync to shared attributes."""
        if axis == "XY":
            # Handle combined XY axis by updating both X and Y
            for single_axis in ("X", "Y"):
                newPos = self._master.positionersManager[positionerName].position[single_axis]
                self.setSharedAttr(positionerName, single_axis, _positionAttr, newPos)
            newPos = self._master.positionersManager[positionerName].position[axis]
            self.setSharedAttr(positionerName, axis, _positionAttr, newPos)
    def updateSpeed(self, positionerName, axis):
        newSpeed = self._master.positionersManager[positionerName].speed[axis]
        self.setSharedAttr(positionerName, axis, _speedAttr, newSpeed)

    @APIExport(runOnUIThread=True)
    def homeAxis(self, positionerName:str=None, axis:str="X", isBlocking:bool=False, homeDirection:int=None, homeSpeed:float=None, homeEndstoppolarity:int=None, homeEndposRelease:float=None, homeTimeout:int=None):
        self.__logger.debug(f"Homing axis {axis}")
        if positionerName is None or positionerName == "" or positionerName not in self._master.positionersManager:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].doHome(axis, 
                                                               isBlocking=isBlocking, 
                                                               homeDirection=homeDirection, 
                                                               homeSpeed=homeSpeed, 
                                                               homeEndstoppolarity=homeEndstoppolarity, 
                                                               homeEndposRelease=homeEndposRelease, 
                                                               homeTimeout=homeTimeout)
        self._hasHomedSinceStartup = True
        self._homingRecommendationDismissed = False
        #self.updatePosition(positionerName, axis)
        #self._commChannel.sigUpdateMotorPosition.emit(self.getPos()) # Not needed as it will be pushed asynchronously from the esp via signal

    @APIExport(runOnUIThread=False)
    def getHomingStatus(self):
        return {
            "hasHomedSinceStartup": self._hasHomedSinceStartup,
            "homingRecommendationDismissed": self._homingRecommendationDismissed
        }

    @APIExport(runOnUIThread=False, requestType="POST")
    def dismissHomingRecommendation(self):
        self._homingRecommendationDismissed = True
        return {
            "success": True,
            "homingRecommendationDismissed": self._homingRecommendationDismissed
        }

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

    def setZPosition(self, z):
        positionerZ = self.getPositionerNames()[2]
        self.__logger.debug(f"Move {positionerZ}, axis Z, dist {str(z)}")

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
    def movePositionerForever(self, positionerName: str=None, axis: str="X", speed: int=0, is_stop: bool=False):
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        speed = float(speed)
        if axis == "X": speed = (0, speed, 0, 0)
        elif axis == "Y": speed = (0, 0, speed, 0)
        elif axis == "Z": speed = (0, 0, 0, speed)
        elif axis == "A": speed = (speed, 0, 0, 0)
        else: return
        self.moveForever(positionerName=positionerName, speed=speed, is_stop=is_stop)

    @APIExport(runOnUIThread=True)
    def movePositionerForeverXYZA(self, positionerName: str=None, speedX: float=0, speedY: float=0, speedZ: float=0, speedA: float=0, is_stop: bool=False):
        """Move all axes simultaneously with individual speed control for X, Y, Z, and A axes."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        # Speed tuple format: (A, X, Y, Z)
        speed = (float(speedA), float(speedX), float(speedY), float(speedZ))
        self.moveForever(positionerName=positionerName, speed=speed, is_stop=is_stop)

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
        """Reset the stage offset for the given axis to 0 and persist."""
        self.__logger.debug(f'Resetting stage offset for {axis} axis.')
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._master.positionersManager[positionerName].resetStageOffsetAxis(axis=axis)
        self.saveStageOffset(positionerName=positionerName, axis=axis)

    @APIExport(runOnUIThread=False)
    def setStageOffsetAxis(
        self,
        positionerName: Optional[str] = None,
        knownPosition: float = 0,
        currentDevicePosition: Optional[float] = None,
        knownOffset: Optional[float] = None,
        axis: str = "X",
    ):
        """Persist a stage offset for one axis.

        Canonical contract::

            offset = device_position_at_known_point - known_user_position

        - ``knownPosition`` (required when ``knownOffset`` is not given):
          desired user coordinate at the current physical position.
        - ``currentDevicePosition`` (optional): raw device position to use.
          If omitted the controller reads it atomically here so the offset is
          based on a single, well-defined device snapshot.
        - ``knownOffset`` (optional): if set, that value is stored verbatim
          and ``knownPosition`` / ``currentDevicePosition`` are ignored.
        """
        self.__logger.debug(
            f'setStageOffsetAxis axis={axis} known={knownPosition} '
            f'currentDevice={currentDevicePosition} knownOffset={knownOffset}'
        )
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        # Snapshot the device position atomically when the frontend did not
        # supply one. This avoids a race with the async position callback.
        if knownOffset is None and currentDevicePosition is None:
            try:
                currentDevicePosition = float(manager.getDevicePositionAxis(axis))
            except Exception as e:
                self.__logger.error(f'getDevicePositionAxis failed: {e}')
                currentDevicePosition = None
        manager.setStageOffsetAxis(
            knownPosition=knownPosition,
            currentDevicePosition=currentDevicePosition,
            knownOffset=knownOffset,
            axis=axis,
        )
        self.saveStageOffset(positionerName=positionerName, axis=axis)
        return {
            "success": True,
            "axis": axis,
            "offset": manager.getStageOffsetAxis(axis=axis),
            "devicePosition": currentDevicePosition,
            "knownPosition": knownPosition,
        }

    @APIExport(runOnUIThread=False)
    def getStageOffsetAxis(self, positionerName: Optional[str]=None, axis:str="X"):
        """Return the persisted stage offset for the given axis."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        return self._master.positionersManager[positionerName].getStageOffsetAxis(axis=axis)

    @APIExport(runOnUIThread=False)
    def getDevicePositionAxis(self, positionerName: Optional[str] = None, axis: str = "X"):
        """Raw device position (no offset applied) for the given axis.

        Use this to obtain a stable physical reference for offset calibration
        - the firmware preserves device steps across software restarts so
        repeated calibrations converge.
        """
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        return self._master.positionersManager[positionerName].getDevicePositionAxis(axis=axis)

    # Back-compat alias for older frontends.
    @APIExport(runOnUIThread=False)
    def getTruePositionerPositionWithoutOffset(self, positionerName: Optional[str] = None, axis: str = "X"):
        return self.getDevicePositionAxis(positionerName=positionerName, axis=axis)

    def saveStageOffset(self, positionerName: str, axis: str = None):
        """Persist the current stage offset(s) to the setup JSON.

        Only axes for which the manager exposes ``stageOffsetPositions`` are
        updated; previously persisted values for other axes are preserved.
        This is critical for managers that do not own a hardware offset
        (Virtual, TANGO, …) so that loading the JSON later does not silently
        zero them.
        """
        try:
            if positionerName is None:
                self.__logger.warning("Cannot save stage offset: positionerName is None")
                return
            if not hasattr(self, '_setupInfo') or self._setupInfo is None:
                self.__logger.warning("Cannot save stage offset: _setupInfo not available")
                return
            if not hasattr(self._setupInfo, 'positioners') or positionerName not in self._setupInfo.positioners:
                self.__logger.warning(f"Positioner {positionerName} not found in setupInfo.positioners")
                return

            manager = self._master.positionersManager[positionerName]
            positionerInfo = self._setupInfo.positioners[positionerName]
            # Start from whatever was last persisted so unrelated axes survive.
            currentOffsets = dict(getattr(positionerInfo, 'stageOffsets', {}) or {})

            if hasattr(manager, 'stageOffsetPositions'):
                for ax in ("X", "Y", "Z", "A"):
                    if ax in manager.stageOffsetPositions:
                        currentOffsets["stageOffsetPosition" + ax] = float(
                            manager.stageOffsetPositions[ax]
                        )
            else:
                self.__logger.info(
                    f"Manager {positionerName} has no stageOffsetPositions; "
                    f"preserving previously persisted offsets unchanged."
                )

            positionerInfo.stageOffsets = currentOffsets
            mOptions, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(mOptions, self._setupInfo)
            self.__logger.info(f"Saved stage offsets for {positionerName}: {currentOffsets}")
        except Exception as e:
            self.__logger.error(f"Could not save stage offset: {e}")
            import traceback
            traceback.print_exc()

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

    @APIExport(runOnUIThread=True)
    def moveToSampleLoadingPosition(self, positionerName=None, speed=10000, is_blocking=True):
        """ Move to sample loading position. """
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self.__logger.debug(f"Moving to sample loading position for positioner {positionerName}")
        self._master.positionersManager[positionerName].moveToSampleLoadingPosition(speed=speed, is_blocking=is_blocking)

    # ========================================================================
    # Frame homing (collision-safe Z-first global homing) and transport position
    # ========================================================================

    @APIExport(runOnUIThread=True)
    def startFrameHoming(self, positionerName: Optional[str] = None, isBlocking: bool = False):
        """Run the collision-safe global homing procedure (Z first, lift, then XY).

        Per-axis progress is pushed to the frontend via the sigHomingState signal.
        """
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if not hasattr(manager, 'frameHomingProcedure'):
            self.__logger.warning(f"{positionerName} does not support frame homing.")
            return {"success": False, "error": "Frame homing not supported"}
        self.__logger.debug(f"Starting frame homing for positioner {positionerName}")
        manager.frameHomingProcedure(is_blocking=isBlocking)
        self._hasHomedSinceStartup = True
        self._homingRecommendationDismissed = False
        return {"success": True}

    @APIExport(runOnUIThread=False)
    def cancelFrameHoming(self, positionerName: Optional[str] = None):
        """Cancel an in-progress frame-homing run."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if hasattr(manager, 'cancelFrameHoming'):
            self.__logger.debug(f"Cancelling frame homing for positioner {positionerName}")
            manager.cancelFrameHoming()
            return {"success": True}
        return {"success": False, "error": "Frame homing not supported"}

    @APIExport(runOnUIThread=False)
    def getFrameHomingState(self, positionerName: Optional[str] = None):
        """Return the current frame-homing progress state."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if hasattr(manager, 'getFrameHomingState'):
            return manager.getFrameHomingState()
        return {"active": False, "axes": {}, "phase": "idle", "message": "", "cancelled": False}

    @APIExport(runOnUIThread=True)
    def moveToTransportPosition(self, positionerName: Optional[str] = None, speed: float = 10000, isBlocking: bool = True):
        """Move the stage to the stored transportation (locking) position."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if not hasattr(manager, 'moveToTransportPosition'):
            return {"success": False, "error": "Transport position not supported"}
        self.__logger.debug(f"Moving to transport position for positioner {positionerName}")
        manager.moveToTransportPosition(speed=speed, is_blocking=isBlocking)
        return {"success": True}

    @APIExport(runOnUIThread=False)
    def getTransportPosition(self, positionerName: Optional[str] = None):
        """Return the stored transportation position (A/X/Y/Z)."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if hasattr(manager, 'getTransportPositions'):
            return manager.getTransportPositions()
        return {}

    @APIExport(runOnUIThread=False, requestType="POST")
    def setTransportPosition(self, positionerName: Optional[str] = None, useCurrent: bool = True,
                             a: Optional[float] = None, x: Optional[float] = None,
                             y: Optional[float] = None, z: Optional[float] = None):
        """Store the transportation position and persist it to the setup JSON.

        With ``useCurrent=True`` the current stage pose is snapshotted; otherwise
        the provided ``a``/``x``/``y``/``z`` values are used.
        """
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if not hasattr(manager, 'setTransportPositions'):
            return {"success": False, "error": "Transport position not supported"}
        if useCurrent:
            positions = manager.setTransportPositions(None)
        else:
            positions = manager.setTransportPositions({"A": a, "X": x, "Y": y, "Z": z})
        self.saveTransportPosition(positionerName=positionerName)
        return {"success": True, "transportPosition": positions}

    @APIExport(runOnUIThread=True)
    def stopAllAxes(self, positionerName: Optional[str] = None):
        """Immediately stop all axes of the positioner."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self.__logger.debug(f"Stopping all axes for positioner {positionerName}")
        self._master.positionersManager[positionerName].forceStop("all")
        return {"success": True}

    # ========================================================================
    # Z-stage synchronisation (re-sync the two Z motors against a mechanical stop)
    # ========================================================================

    @APIExport(runOnUIThread=True)
    def startZStageSync(self, positionerName: Optional[str] = None, steps: Optional[float] = None):
        """Re-synchronise the two Z motors against the mechanical stop.

        Drives Z out by ``steps`` µm (default from config), backs off half,
        restores the Z limit switch and re-homes Z. Progress is pushed to the
        frontend via the sigZStageSyncState signal.
        """
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if not hasattr(manager, 'zStageSyncProcedure'):
            self.__logger.warning(f"{positionerName} does not support Z-stage sync.")
            return {"success": False, "error": "Z-stage sync not supported"}
        self.__logger.debug(f"Starting Z-stage sync for positioner {positionerName} (steps={steps})")
        manager.zStageSyncProcedure(steps=steps)
        return {"success": True}

    @APIExport(runOnUIThread=False)
    def cancelZStageSync(self, positionerName: Optional[str] = None):
        """Stop an in-progress Z-stage sync run and halt all motors."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if hasattr(manager, 'cancelZStageSync'):
            self.__logger.debug(f"Cancelling Z-stage sync for positioner {positionerName}")
            manager.cancelZStageSync()
            return {"success": True}
        return {"success": False, "error": "Z-stage sync not supported"}

    @APIExport(runOnUIThread=False)
    def getZStageSyncState(self, positionerName: Optional[str] = None):
        """Return the current Z-stage sync progress state."""
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        manager = self._master.positionersManager[positionerName]
        if hasattr(manager, 'getZStageSyncState'):
            return manager.getZStageSyncState()
        return {"active": False, "phase": "idle", "message": "", "cancelled": False, "steps": 0}

    def saveTransportPosition(self, positionerName: str):
        """Persist the transport position to the setup JSON managerProperties.

        Mirrors ``saveStageOffset``: writes ``transportPosition{A,X,Y,Z}`` into the
        positioner's managerProperties and re-serialises the setup file.
        """
        try:
            if positionerName is None:
                self.__logger.warning("Cannot save transport position: positionerName is None")
                return
            if not hasattr(self, '_setupInfo') or self._setupInfo is None:
                self.__logger.warning("Cannot save transport position: _setupInfo not available")
                return
            if not hasattr(self._setupInfo, 'positioners') or positionerName not in self._setupInfo.positioners:
                self.__logger.warning(f"Positioner {positionerName} not found in setupInfo.positioners")
                return
            manager = self._master.positionersManager[positionerName]
            positionerInfo = self._setupInfo.positioners[positionerName]
            if hasattr(manager, 'transportPositions'):
                for ax in ("A", "X", "Y", "Z"):
                    if ax in manager.transportPositions:
                        positionerInfo.managerProperties["transportPosition" + ax] = float(
                            manager.transportPositions[ax]
                        )
            mOptions, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(mOptions, self._setupInfo)
            self.__logger.info(
                f"Saved transport position for {positionerName}: {manager.transportPositions}"
            )
        except Exception as e:
            self.__logger.error(f"Could not save transport position: {e}")
            import traceback
            traceback.print_exc()

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
