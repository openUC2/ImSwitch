import numpy as np
import time
import threading
import collections

from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex, Timer
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger
from ..basecontrollers import ImConWidgetController



class TemperatureController(ImConWidgetController):
    """ Linked to TemperatureWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # Parameters for monitoring the pressure
        self.tMeasure  = 1 # sampling rate of measure pressure
        self.is_measure = True
        self.temperatureValue  = 0
        self.buffer = 0
        self.currPoint = 0
        self.setPointData = np.zeros((self.buffer,2))
        self.timeData = np.zeros(self.buffer)
        self.startTime = time.time()

        # settings for the controller
        self.controlTarget = 37

        # Hard-coded PID values..
        self.Kp = 100
        self.Ki = 0.1
        self.Kd = .5
        self.PIDenabled = False

        # get hold on the Temperature Controller
        try:
            self.temperatureController = self._master.rs232sManager["ESP32"]._esp32.temperature
        except:
            return
        # Connect TemperatureWidget signals
        self._widget.sigPIDToggled.connect(self.setPID)
        self._widget.sigsliderTemperatureValueChanged.connect(self.valueTemperatureValueChanged)
        self.setPID(self._widget.getPIDChecked())

        # Start the temperature display thread
        self.measurementThread = threading.Thread(target=self.updateMeasurements)
        self.measurementThread.start()
        

    def valueTemperatureValueChanged(self, value):
        """ Change setpoint for the temperature. """
        self.controlTarget = value

        # retrieve PID values
        self.Kp, self.Ki, self.Kd = self._widget.getPIDValues()
        
        # get temperature value from GUI
        self.controlTarget = self._widget.getTemperatureValue()
        
        # we actually set the target value with this slider
        self._widget.updateTargetTemperatureValue(self.controlTarget)
        self.temperatureController.set_temperature(active=self.PIDenabled,
                                                       Kp=self.Kp, Ki=self.Ki, Kd=self.Kd, target=self.controlTarget)
        
    def valueRotationSpeedChanged(self, value):
        """ Change magnitude. """
        self.speedRotation = int(value)
        self._widget.updateRotationSpeed(self.speedPump)
        self.tRoundtripRotation = self.stepsPerRotation/(0.001+self.speedRotation) # in s
        self.positioner.moveForever(speed=(self.speedPump,self.speedRotation,0),is_stop=False)

    def __del__(self):
        self.is_measure=False
        self.measurementThread.quit()
        if hasattr(super(), '__del__'):
            super().__del__()

    def setPID(self, enabled):
        """ Show or hide Temperature. """
        self.PIDenabled = enabled
        # retrieve PID values
        self.Kp, self.Ki, self.Kd = self._widget.getPIDValues()
        
        # get temperature value from GUI
        self.controlTarget = float(self._widget.getTemperatureValue())
        
        self.temperatureController.set_temperature(active=enabled
            , Kp=self.Kp, Ki=self.Ki, Kd=self.Kd, target=self.controlTarget)

    def updateSetPointData(self):
        if self.currPoint < self.buffer:
            self.setPointData[self.currPoint,0] = self.temperatureValue
            self.setPointData[self.currPoint,1] = self.controlTarget

            self.timeData[self.currPoint] = time.time() - self.startTime
        else:
            self.setPointData[:-1,0] = self.setPointData[1:,0]
            self.setPointData[-1,0] = self.temperatureValue
            self.setPointData[:-1,1] = self.setPointData[1:,1]
            self.setPointData[-1,1] = self.controlTarget
            self.timeData[:-1] = self.timeData[1:]
            self.timeData[-1] = time.time() - self.startTime
        self.currPoint += 1

    def updateMeasurements(self):
        while self.is_measure:
            self.temperatureValue  = self.temperatureController.get_temperature()
            self._widget.updateTemperature(self.temperatureValue)
            # update plot
            self.updateSetPointData()
            if self.currPoint < self.buffer:
                self._widget.temperaturePlotCurve.setData(self.timeData[1:self.currPoint],
                                                    self.setPointData[1:self.currPoint,0])
            else:
                self._widget.temperaturePlotCurve.setData(self.timeData, self.setPointData[:,0])
            time.sleep(self.tMeasure)



# Copyright (C) 2020-2021 ImSwitch developers
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