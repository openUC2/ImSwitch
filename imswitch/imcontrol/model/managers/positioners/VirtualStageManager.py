from imswitch.imcommon.model import initLogger
from .PositionerManager import PositionerManager
import threading
import time

class VirtualStageManager(PositionerManager):
    def __init__(self, positionerInfo, name, **lowLevelManagers):
        super().__init__(positionerInfo, name, initialPosition={axis: 0 for axis in positionerInfo.axes})
        self.__logger = initLogger(self, instanceName=name)
        self._commChannel = lowLevelManagers['commChannel']

        self.stepsizeX = positionerInfo.managerProperties.get("stepsizeX")
        self.stepsizeY = positionerInfo.managerProperties.get("stepsizeY")
        self.stepsizeZ = positionerInfo.managerProperties.get("stepsizeZ")
        self.stepsizeA = positionerInfo.managerProperties.get("stepsizeA")
        # Persisted offsets loaded from the setup JSON. Same key naming as
        # ESP32StageManager so the controller can save/load uniformly.
        self.stageOffsetPositions = {
            "X": positionerInfo.stageOffsets.get('stageOffsetPositionX', 0),
            "Y": positionerInfo.stageOffsets.get('stageOffsetPositionY', 0),
            "Z": positionerInfo.stageOffsets.get('stageOffsetPositionZ', 0),
            "A": positionerInfo.stageOffsets.get('stageOffsetPositionA', 0),
        }
        # Simulated strobed sweep (see startStrobeSweep).
        self._strobeCallbacks = []
        self._strobeThread = None
        self._strobeStop = threading.Event()
        # The simulated camera trigger line: callables(n, x) run at every
        # simulated SYNC, so a simulated camera can expose one frame per pulse.
        self.strobeTriggerListeners = []
        # Arguments of the last startStrobeSweep call, for simulated cameras.
        self.strobeSweepRequest = {}
        try:
            self.VirtualMicroscope = lowLevelManagers["rs232sManager"]["VirtualMicroscope"]
        except:
            return
        # assign the camera from the Virtual Microscope
        self._positioner = self.VirtualMicroscope._positioner

        # get bootup position and write to GUI
        self._position = self.getPosition()

    @property
    def combinedAxes(self):
        """The simulated stage applies X/Y (and X/Y/Z) in one step."""
        return ["XY", "XYZ"]

    def move(self, value=0, axis="X", is_absolute=False, is_blocking=True, acceleration=None, speed=None, isEnable=None, timeout=1):
        # For absolute moves the caller passes user coordinates; the simulated
        # device sees user + offset (same convention as ESP32StageManager).
        ox = self.stageOffsetPositions.get("X", 0)
        oy = self.stageOffsetPositions.get("Y", 0)
        oz = self.stageOffsetPositions.get("Z", 0)
        oa = self.stageOffsetPositions.get("A", 0)
        if axis == "X":
            dv = (value + ox) if is_absolute else value
            self._positioner.move(x=self.stepsizeX*dv, is_absolute=is_absolute)
        if axis == "Y":
            dv = (value + oy) if is_absolute else value
            self._positioner.move(y=self.stepsizeY*dv, is_absolute=is_absolute)
        if axis == "Z":
            dv = (value + oz) if is_absolute else value
            self._positioner.move(z=self.stepsizeZ*dv, is_absolute=is_absolute)
        if axis == "A":
            dv = (value + oa) if is_absolute else value
            self._positioner.move(a=self.stepsizeA*dv, is_absolute=is_absolute)
        if axis == "XYZ":
            self._positioner.move(
                x=self.stepsizeX*((value[0]+ox) if is_absolute else value[0]),
                y=self.stepsizeY*((value[1]+oy) if is_absolute else value[1]),
                z=self.stepsizeZ*((value[2]+oz) if is_absolute else value[2]),
                is_absolute=is_absolute,
            )
        if axis == "XY":
            self._positioner.move(
                x=self.stepsizeX*((value[0]+ox) if is_absolute else value[0]),
                y=self.stepsizeY*((value[1]+oy) if is_absolute else value[1]),
                is_absolute=is_absolute,
            )
        for axes in ["A","X","Y","Z"]:
            self._position[axes] = self._positioner.position[axes]

        self.getPosition() # update position in GUI

    def getDevicePositionAxis(self, axis="X"):
        """Raw device position for one axis on the virtual stage."""
        try:
            all_positions = self._positioner.get_position()
            return float(all_positions.get(axis, 0))
        except Exception:
            return float(self._position.get(axis, 0)) + float(self.stageOffsetPositions.get(axis, 0))

    def setPositionOnDevice(self, axis, value):
        if axis == "X":
            self._positioner.move(x=value, is_absolute=True)
        if axis == "Y":
            self._positioner.move(y=value, is_absolute=True)
        if axis == "Z":
            self._positioner.move(z=value, is_absolute=True)
        if axis == "A":
            self._positioner.move(a=value, is_absolute=True)
        if axis == "XYZ":
            self._positioner.move(x=value[0], y=value[1], z=value[2], is_absolute=True)
        if axis == "XY":
            self._positioner.move(x=value[0], y=value[1], is_absolute=True)
        for axes in ["A","X","Y","Z"]:
            self._position[axes] = self._positioner.position[axes]
        #self._commChannel.sigUpdateMotorPosition.emit()

    def moveForever(self, speed=(0, 0, 0, 0), is_stop=False):
        pass

    def setSpeed(self, speed, axis=None):
        pass

    def setPosition(self, value, axis):
        pass

    def getPosition(self):
        # Returns user (offset-corrected) coordinates so callers see the same
        # frame as ESP32StageManager.getPosition().
        allPositionsDict = self._positioner.get_position()
        corrected = {}
        
        for ax in ("A", "X", "Y", "Z"):
            if ax in allPositionsDict:
                # apply stepsizes 
                if ax == "X":
                    corrected[ax] = float(allPositionsDict[ax]) / self.stepsizeX - float(self.stageOffsetPositions.get(ax, 0))
                elif ax == "Y":
                    corrected[ax] = float(allPositionsDict[ax]) / self.stepsizeY - float(self.stageOffsetPositions.get(ax, 0))
                elif ax == "Z":
                    corrected[ax] = float(allPositionsDict[ax]) / self.stepsizeZ - float(self.stageOffsetPositions.get(ax, 0))
                elif ax == "A":
                    corrected[ax] = float(allPositionsDict[ax]) / self.stepsizeA - float(self.stageOffsetPositions.get(ax, 0))
            else:
                corrected[ax] = float(allPositionsDict[ax]) - float(self.stageOffsetPositions.get(ax, 0))
        emitDict = {"VirtualStage": corrected}
        try: self._commChannel.sigUpdateMotorPosition.emit(emitDict)
        except: pass
        return corrected

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

    def get_abs(self, axis="X"):
        return self._position[axis]

    def stop_x(self):
        pass

    def stop_y(self):
        pass

    def stop_z(self):
        pass

    def stop_a(self):
        pass

    def stopAll(self):
        pass

    def doHome(self, axis, isBlocking=False):
        if axis == "X": self.home_x(isBlocking)
        if axis == "Y": self.home_y(isBlocking)
        if axis == "Z": self.home_z(isBlocking)


    def home_x(self, is_blocking):
        self.move(value=0, axis="X", is_absolute=True)
        self.setPosition(axis="X", value=0)

    def home_y(self,is_blocking):
        self.move(value=0, axis="Y", is_absolute=True)
        self.setPosition(axis="Y", value=0)

    def home_z(self,is_blocking):
        self.move(value=0, axis="Z", is_absolute=True)
        self.setPosition(axis="Z", value=0)

    def home_xyz(self):
        if self.homeXenabled and self.homeYenabled and self.homeZenabled:
            [self.setPosition(axis=axis, value=0) for axis in ["X","Y","Z"]]


    # ------------------------------------------------------------------ #
    # Simulated strobed sweep: same API as ESP32StageManager, user frame.
    # ------------------------------------------------------------------ #

    def hasStrobeSweep(self) -> bool:
        """The simulation always supports it, so the strobed path runs without hardware."""
        return True

    def startStrobeSweep(self, axis="X", target=0.0, speed=None, period_us=33333,
                         trig_us=100, laser=-1, delay_us=None, width_us=None,
                         latch=True, report=16, max_frames=0) -> bool:
        """Move ``axis`` linearly to ``target`` (µm, user frame) at ``speed`` µm/s,
        emitting one simulated SYNC every ``period_us``.

        Every SYNC calls ``strobeTriggerListeners(n, x)`` and latches ``x``;
        latched positions go to the callbacks in "report" batches of
        ``report``, then a "done" event. ``max_frames > 0`` emits exactly that
        many frames (moving towards the target meanwhile, if not there yet).
        """
        if self._strobeThread is not None and self._strobeThread.is_alive():
            self.__logger.warning("Simulated strobe sweep already running")
            return False
        axis = str(axis).upper()
        self.strobeSweepRequest = dict(
            axis=axis, target=float(target), speed=speed, period_us=period_us,
            trig_us=trig_us, laser=laser, delay_us=delay_us, width_us=width_us,
            latch=latch, report=report, max_frames=max_frames)
        self._strobeStop.clear()
        self._strobeThread = threading.Thread(
            target=self._runStrobeSweep, daemon=True, name="VirtualStrobeSweep",
            args=(axis, float(target), float(speed or 1000.0), max(1.0, float(period_us)),
                  bool(laser is not None and laser >= 0 and delay_us is not None
                       and width_us is not None),
                  max(1, int(report)), max(0, int(max_frames))))
        self._strobeThread.start()
        return True

    def _runStrobeSweep(self, axis, target, speed, periodUs, strobe, report, maxFrames):
        start = float(self.getPosition().get(axis, 0.0))
        distance = target - start
        direction = 1.0 if distance >= 0 else -1.0
        period = periodUs / 1e6
        stepSize = float(getattr(self, f"stepsize{axis}", 1) or 1)
        batch = []
        n = 0
        t0 = time.monotonic()
        while not self._strobeStop.is_set():
            n += 1
            travelled = min(abs(distance), speed * (n - 1) * period)
            x = start + direction * travelled
            self.move(value=x, axis=axis, is_absolute=True)
            for listener in list(self.strobeTriggerListeners):
                try:
                    listener(n, x)
                except Exception as e:
                    self.__logger.debug(f"Strobe trigger listener failed: {e}")
            batch.append((n, x))
            if len(batch) >= report:
                self._emitStrobe(self._strobeReport(batch, stepSize))
                batch = []
            reached = travelled >= abs(distance)
            if (maxFrames > 0 and n >= maxFrames) or (maxFrames == 0 and reached):
                break
            time.sleep(max(0.0, t0 + n * period - time.monotonic()))
        if batch:
            self._emitStrobe(self._strobeReport(batch, stepSize))
        aborted = self._strobeStop.is_set()
        self._emitStrobe({
            "type": "done", "frames": n, "camera": n, "flashes": n if strobe else 0,
            "positions": n, "latch": 1, "strobe": int(strobe), "aborted": int(aborted),
            "success": int(not aborted), "qid": 0})

    @staticmethod
    def _strobeReport(batch, stepSize):
        return {"type": "report", "n": [b[0] for b in batch], "x": [b[1] for b in batch],
                "x_steps": [int(round(b[1] * stepSize)) for b in batch]}

    def _emitStrobe(self, event):
        for cb in list(self._strobeCallbacks):
            try:
                cb(event)
            except Exception as e:
                self.__logger.error(f"Strobe sweep callback failed: {e}")

    def stopStrobeSweep(self):
        self._strobeStop.set()
        if self._strobeThread is not None:
            self._strobeThread.join(timeout=2.0)
        return True

    def registerStrobeSweepCallback(self, cb) -> bool:
        self._strobeCallbacks.append(cb)
        return True

    def unregisterStrobeSweepCallback(self, cb) -> bool:
        try:
            self._strobeCallbacks.remove(cb)
            return True
        except ValueError:
            return False

    def start_stage_scanning(self, xstart=0, xstep=1, nx=100,
                             ystart=0, ystep=1, zstart=0, zstep=1, nz=100, ny=100, tsettle=0.1, tExposure=50, illumination=None, led=None,
                             speed=None, acceleration=None, tTrig=None, **kwargs):
        """
        Start a stage scanning operation with the given parameters.
        Virtual implementation that simulates the scanning process.
        
        :param xstart: Starting position in X direction.
        :param xstep: Step size in X direction.
        :param nx: Number of steps in X direction.
        :param ystart: Starting position in Y direction.
        :param ystep: Step size in Y direction.
        :param ny: Number of steps in Y direction.
        :param tsettle: Settle time after each step.
        :param tExposure: Exposure time for each position.
        :param illumination: Optional illumination settings.
        :param led: Optional LED settings.
        """
        self.__logger.info(f"Starting virtual stage scanning: {nx}x{ny} grid")
        self.__logger.info(f"X: start={xstart}, step={xstep}, count={nx}")
        self.__logger.info(f"Y: start={ystart}, step={ystep}, count={ny}")
        self.__logger.info(f"Timing: settle={tsettle}ms, exposure={tExposure}ms")

        # Set default values for optional parameters
        if illumination is None:
            illumination = (0, 0, 0, 0)  # Default to no illumination
        if led is None:
            led = 0

        # For virtual stage, we simulate the scanning by updating positions
        # The actual movement will be handled by the experiment controller
        scan_params = {
            'xstart': xstart,
            'xstep': xstep,
            'nx': nx,
            'ystart': ystart,
            'ystep': ystep,
            'ny': ny,
            'tsettle': tsettle,
            'tExposure': tExposure,
            'illumination': illumination,
            'led': led,
            'status': 'started'
        }

        # Store scan parameters for tracking
        self._scan_params = scan_params

        # Move to starting position
        self.move(value=xstart, axis="X", is_absolute=True)
        self.move(value=ystart, axis="Y", is_absolute=True)
        self.move(value=zstart, axis="Z", is_absolute=True)

        self.__logger.info("Virtual stage scanning started successfully")
        return {"success": True, "message": "Virtual stage scanning started", "params": scan_params}

    def stop_stage_scanning(self):
        """
        Stop the current stage scanning operation.
        Virtual implementation that stops the scanning simulation.
        """
        self.__logger.info("Stopping virtual stage scanning")

        # Clear scan parameters
        if hasattr(self, '_scan_params'):
            self._scan_params['status'] = 'stopped'
            self.__logger.info("Virtual stage scanning stopped successfully")
        else:
            self.__logger.warning("No active stage scanning to stop")

        return {"success": True, "message": "Virtual stage scanning stopped"}

    def get_stage_scan_status(self):
        """
        Get the current status of stage scanning.
        Virtual implementation that returns scan parameters and status.
        """
        if hasattr(self, '_scan_params') and self._scan_params:
            return self._scan_params
        else:
            return {"status": "idle", "message": "No active scanning"}

    def simulate_scan_position(self, tile_x, tile_y):
        """
        Simulate moving to a specific tile position during scanning.
        
        :param tile_x: X tile index (0-based)
        :param tile_y: Y tile index (0-based)
        """
        if hasattr(self, '_scan_params') and self._scan_params:
            x_pos = self._scan_params['xstart'] + tile_x * self._scan_params['xstep']
            y_pos = self._scan_params['ystart'] + tile_y * self._scan_params['ystep']

            self.__logger.debug(f"Simulating move to tile ({tile_x}, {tile_y}) -> position ({x_pos}, {y_pos})")

            # Move to the calculated position
            self.move(value=x_pos, axis="X", is_absolute=True)
            self.move(value=y_pos, axis="Y", is_absolute=True)

            # Simulate settle time
            if self._scan_params['tsettle'] > 0:
                time.sleep(self._scan_params['tsettle'] / 1000.0)  # Convert ms to seconds

            return {"x": x_pos, "y": y_pos, "tile_x": tile_x, "tile_y": tile_y}
        else:
            self.__logger.warning("No active scan parameters for position simulation")
            return None


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
