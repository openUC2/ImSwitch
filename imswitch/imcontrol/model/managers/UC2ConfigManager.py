import enum
import glob
import math
import os
import threading

import numpy as np
from PIL import Image
import uc2rest as uc2
import json

from imswitch.imcommon.model import dirtools
from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger


class UC2ConfigManager(SignalInterface):

    def __init__(self, Info, lowLevelManagers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        # TODO: HARDCODED!!
        try:
            self.ESP32 = lowLevelManagers["rs232sManager"]["ESP32"]._esp32
        except:
            return

    def saveState(self, state_general=None, state_pos=None, state_aber=None):
        if state_general is not None:
            self.state_general = state_general
        if state_pos is not None:
            self.state_pos = state_pos
        if state_aber is not None:
            self.state_aber = state_aber

    def setGeneral(self, general_info):
        pass

    def loadDefaultConfig(self):
        return self.ESP32.config.loadDefaultConfig()

    def update(self, maskChange=False, tiltChange=False, aberChange=False):
        pass

    def closeSerial(self):
        return self.ESP32.closeSerial()

    def isConnected(self):
        try:
            return self.ESP32.serial.is_connected
        except:
            return False

    def interruptSerialCommunication(self):
        self.ESP32.serial.interruptCurrentSerialCommunication()

    def initSerial(self, baudrate=None):
        try:
            self.ESP32.serial.reconnect(baudrate=baudrate)
        except:
            self.ESP32.serial.reconnect() # fall back to old version of UC2-REST

    def pairBT(self):
        self.ESP32.state.pairBT()

    def setDebug(self, debug):
        self.ESP32.serial.DEBUG = debug

    def restartESP(self):
        self.ESP32.state.espRestart()

    def restartCANDevice(self, device_id):
        """
        Restart a CAN device by sending a reboot command to the ESP32.

        0 - Master
        10-19 - Motor
        20-29 - Laser
        30-39 - LED
        Args:
            device_id (_type_): _description_
        """
        self.ESP32.can.reboot_remote(can_address=device_id, isBlocking=True, timeout=1)


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
