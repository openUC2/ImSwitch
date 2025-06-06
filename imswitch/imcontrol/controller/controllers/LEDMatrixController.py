from typing import Dict, List
from functools import partial
import numpy as np

from imswitch import IS_HEADLESS
from imswitch.imcommon.model import APIExport
from ..basecontrollers import ImConWidgetController
from imswitch.imcontrol.view import guitools as guitools
from imswitch.imcommon.model import initLogger, APIExport

class LEDMatrixController(ImConWidgetController):
    """ Linked to LEDMatrixWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        # TODO: This must be easier?
        self.nLedsX = self._master.LEDMatrixsManager._subManagers['ESP32 LEDMatrix'].Nx
        self.nLedsY = self._master.LEDMatrixsManager._subManagers['ESP32 LEDMatrix'].Ny

        self._ledmatrixMode = ""

        # get the name that looks like an LED Matrix
        self.ledMatrix_name = self._master.LEDMatrixsManager.getAllDeviceNames()[0]
        self.ledMatrix = self._master.LEDMatrixsManager[self.ledMatrix_name]

        # initialize matrix

        if IS_HEADLESS:
            return
        self._widget.add_matrix_view(self.nLedsX, self.nLedsY)
        self.connect_leds()
        self.setAllLEDOff()
        self._widget.ButtonAllOn.clicked.connect(self.setAllLEDOn)
        self._widget.ButtonAllOff.clicked.connect(self.setAllLEDOff)
        self._widget.slider.sliderReleased.connect(self.setIntensity)

    @APIExport()
    def setAllLEDOn(self, getReturn=True):
        self.setAllLED(state=(1,1,1), getReturn=getReturn)

    @APIExport()
    def setAllLEDOff(self, getReturn=True):
        self.setAllLED(state=(0,0,0),getReturn=getReturn)

    @APIExport()
    def setAllLED(self, state:int=None, intensity:int=None, getReturn:bool=True):
        if intensity is not None:
            self.setIntensity(intensity=intensity)
        self.ledMatrix.setAll(state=state,getReturn=getReturn)
        if IS_HEADLESS: return
        for coords, btn in self._widget.leds.items():
            if isinstance(btn, guitools.BetterPushButton):
                btn.setChecked(np.sum(state)>0)

    @APIExport()
    def setIntensity(self, intensity:int=None):
        if intensity is None:
            if not IS_HEADLESS: intensity = int(self._widget.slider.value()//1)
        else:
            # this is only if the GUI/API is calling this function
            intensity = int(intensity)

        self.ledMatrix.setLEDIntensity(intensity=(intensity,intensity,intensity))

    @APIExport()
    def setLED(self, LEDid:int, state:int=None):
        self._ledmatrixMode = "single"
        self.ledMatrix.setLEDSingle(indexled=int(LEDid), state=state)
        pattern = self.ledMatrix.getPattern()
        if not IS_HEADLESS: self._widget.leds[str(LEDid)].setChecked(state)

    # GUI functions
    def connect_leds(self):
        """Connect leds (Buttons) to the Sample Pop-Up Method"""
        # Connect signals for all buttons
        if IS_HEADLESS: return
        for coords, btn in self._widget.leds.items():
            # Connect signals
            if isinstance(btn, guitools.BetterPushButton):
                btn.clicked.connect(partial(self.setLED, coords))

    @APIExport()
    def setEnabled(self, enabled:bool) -> None:
        """ Sets the value of the LEDMatrix. """
        self.setAllLED(state=enabled, intensity=None)

    @APIExport()
    def setValue(self, value:int) -> None:
        """ Sets the value of the LEDMatrix. """
        self.setIntensity(intensity=value)
        self.setAllLED(state=(1,1,1), intensity=value)

    @APIExport()
    def setRing(self, ringRadius: int, intensity: int) -> None:
        """ Sets the value of the LEDMatrix. """
        #self.setIntensity(intensity=intensity)
        self.ledMatrix.setRing(radius=ringRadius, intensity=intensity)
        if not IS_HEADLESS: self._widget.leds[str(ringRadius)].setChecked(True)

    @APIExport()
    def setCircle(self, circleRadius: int, intensity: int) -> None:
        """ Sets the value of the LEDMatrix. """
        #self.setIntensity(intensity=intensity)
        self.ledMatrix.setCircle(radius=circleRadius, intensity=intensity)
        if not IS_HEADLESS: self._widget.leds[str(circleRadius)].setChecked(True)

    @APIExport()
    def setHalves(self, intensity: int, direction: str) -> None:
        """ Sets the value of the LEDMatrix. """
        #self.setIntensity(intensity=intensity)
        self.ledMatrix.setHalves(intensity=intensity, region=direction)
        if not IS_HEADLESS: self._widget.leds[str(intensity)].setChecked(True)

    @APIExport()
    def setStatus(self, status:str="idle") -> None:
        """ Sets the value of the LEDMatrix. """
        self.ledMatrix.setStatus(status=status)
        if not IS_HEADLESS: self._widget.leds[str(status)].setChecked(True)

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
