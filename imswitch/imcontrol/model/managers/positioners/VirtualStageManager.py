from imswitch.imcommon.model import initLogger
from .PositionerManager import PositionerManager
import time
import numpy as np
from imswitch.imcommon.model import APIExport, generateAPI, initLogger
import threading

class VirtualStageManager(PositionerManager):
    def __init__(self, positionerInfo, name, **lowLevelManagers):
        super().__init__(positionerInfo, name, initialPosition={axis: 0 for axis in positionerInfo.axes})
        self.__logger = initLogger(self, instanceName=name)
        self._commChannel = lowLevelManagers['commChannel']

        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
        self.offset_a = 0
        self.stageOffsetPositions = {"X": self.offset_x, "Y": self.offset_y, "Z": self.offset_z, "A": self.offset_a}
        try:
            self.VirtualMicroscope = lowLevelManagers["rs232sManager"]["VirtualMicroscope"]
        except:
            return
        # assign the camera from the Virtual Microscope
        self._positioner = self.VirtualMicroscope._positioner

        # get bootup position and write to GUI
        self._position = self.getPosition()

    def move(self, value=0, axis="X", is_absolute=False, is_blocking=True, acceleration=None, speed=None, isEnable=None, timeout=1):
        if axis == "X":
            self._positioner.move(x=value+self.offset_x, is_absolute=is_absolute)
        if axis == "Y":
            self._positioner.move(y=value+self.offset_y, is_absolute=is_absolute)
        if axis == "Z":
            self._positioner.move(z=value+self.offset_z, is_absolute=is_absolute)
        if axis == "A":
            self._positioner.move(a=value+self.offset_a, is_absolute=is_absolute)
        if axis == "XYZ":
            self._positioner.move(x=value[0]+self.offset_x, y=value[1]+self.offset_y, z=value[2]+self.offset_z, is_absolute=is_absolute)
        if axis == "XY":
            self._positioner.move(x=value[0]+self.offset_x, y=value[1]+self.offset_y, is_absolute=is_absolute)
        for axes in ["A","X","Y","Z"]:
            self._position[axes] = self._positioner.position[axes]

        self.getPosition() # update position in GUI

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
        # load position from device
        # t,x,y,z
        allPositionsDict = self._positioner.get_position()
        posDict= {}
        posDict["VirtualStage"] = allPositionsDict
        try:self._commChannel.sigUpdateMotorPosition.emit(posDict)
        except: pass # Should be a list TODO: This is a hacky workaround to force Imswitch to update the motor positions in the gui..
        return allPositionsDict

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


    def home_x(self, isBlocking):
        self.move(value=0, axis="X", is_absolute=True)
        self.setPosition(axis="X", value=0)

    def home_y(self,isBlocking):
        self.move(value=0, axis="Y", is_absolute=True)
        self.setPosition(axis="Y", value=0)

    def home_z(self,isBlocking):
        self.move(value=0, axis="Z", is_absolute=True)
        self.setPosition(axis="Z", value=0)

    def home_xyz(self):
        if self.homeXenabled and self.homeYenabled and self.homeZenabled:
            [self.setPosition(axis=axis, value=0) for axis in ["X","Y","Z"]]


    def setStageOffset(self, axis, offset):
        if axis == "X": self._positioner.set_stage_offset(x=offset)
        if axis == "Y": self._positioner.set_stage_offset(y=offset)
        if axis == "Z": self._positioner.set_stage_offset(z=offset)
        if axis == "A": self._positioner.set_stage_offset(a=offset)
        if axis == "XYZ": self._positioner.set_stage_offset(xyz=offset)
        if axis == "XY": self._positioner.set_stage_offset(xy=offset)
        #self._commChannel.sigUpdateMotorPosition.emit()

    def start_stage_scanning(self, xstart=0, xstep=1, nx=100,
                             ystart=0, ystep=1, ny=100, tsettle=0.1, tExposure=50, illumination=None, led=None):
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
