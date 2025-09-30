import enum
import glob
import cv2
import os

import numpy as np
from PIL import Image
import os
from imswitch.imcommon.model import dirtools

import json

from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger


class ObjectiveManager(SignalInterface):

    def __init__(self, ObjectiveInfo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        if ObjectiveInfo is None:
            # define default
            self.__ObjectiveInfo = {
                "pixelsizes": [0.1, 0.2],
                "NAs": [0.2, 0.1],
                "magnifications": [20, 10],
                "objectiveNames": ["20x", "10x"],
                "objectivePositions": [0,10000],
                "homeDirection": -1,
                "homePolarity": 1,
                "homeSpeed": 20000,
                "homeAcceleration": 20000,
                "calibrateOnStart": True
            }


        self.__ObjectiveInfo = ObjectiveInfo
        self.pixelsizes = self.__ObjectiveInfo.pixelsizes
        self.NAs = self.__ObjectiveInfo.NAs
        self.magnifications = self.__ObjectiveInfo.magnifications
        self.objectiveNames = self.__ObjectiveInfo.objectiveNames
        self.objectivePositions = self.__ObjectiveInfo.objectivePositions
        self.homeDirection = self.__ObjectiveInfo.homeDirection
        self.homePolarity = self.__ObjectiveInfo.homePolarity
        self.homeSpeed = self.__ObjectiveInfo.homeSpeed
        self.homeAcceleration = self.__ObjectiveInfo.homeAcceleration
        self.calibrateOnStart = self.__ObjectiveInfo.calibrateOnStart
        self.isActive = self.__ObjectiveInfo.active



    def update(self):
        return None


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
