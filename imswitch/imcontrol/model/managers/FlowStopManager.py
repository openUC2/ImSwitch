import enum
import glob
import cv2
import os

import numpy as np
from PIL import Image

from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger
from imswitch.imcommon.model import dirtools
import json

class FlowStopManager(SignalInterface):

    def __init__(self, flowStopInfo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        self.flowStopConfigFilename = "config.json"
        self.allParameterKeys = ["wasRunning", "flowRate", "dnumberOfFrames",
                                 "experimentName","frameRate","savePath",
                                 "fileFormat", "axisFocus",
                                 "axisFlow", "delayTimeAfterRestart", "isRecordVideo", "numImages", "uniqueId",
                                 "volumePerImage", "timeToStabilize", "pumpSpeed"]

        # get default configs
        self.defaultConfigPath = os.path.join(dirtools.UserFileDirs.Root, "flowStopController")
        if not os.path.exists(self.defaultConfigPath):
            os.makedirs(self.defaultConfigPath)

        try:
            with open(os.path.join(self.defaultConfigPath, self.flowStopConfigFilename)) as jf:
                # check if all keys are present
                self.defaultConfig = json.load(jf)
                # check if all keys are present
                missing_keys = [key for key in self.allParameterKeys if key not in self.defaultConfig]
                if missing_keys:
                    raise KeyError
                else:
                    pass

        except Exception as e:
            self.__logger.error(f"s {self.defaultConfigPath}: {e}")
            self.defaultConfig = {}
            self.defaultConfig["wasRunning"] = True
            self.defaultConfig["flowRate"] = 100
            self.defaultConfig["numberOfFrames"] = -1
            self.defaultConfig["experimentName"] = "FlowStopExperiment"
            self.defaultConfig["frameRate"] = 1
            self.defaultConfig["savePath"] = "./"
            self.defaultConfig["fileFormat"] = "JPG"
            self.defaultConfig["axisFlow"] = "X"
            self.defaultConfig["axisFocus"] = "Z"
            self.defaultConfig["delayTimeAfterRestart"]=1
            self.defaultConfig["isRecordVideo"]=True
            self.defaultConfig["numImages"]=10
            self.defaultConfig["uniqueId"] = np.random.randint(0, 1000000),
            self.defaultConfig["volumePerImage"] = 1000
            self.defaultConfig["timeToStabilize"] = 1
            self.defaultConfig["pumpSpeed"] = 100
            self.writeConfig(self.defaultConfig)

    def updateConfig(self, parameterName, value):
        with open(os.path.join(self.defaultConfigPath, self.flowStopConfigFilename), "w") as outfile:
            mDict = json.load(outfile)
            mDict[parameterName] = value
            json.dump(mDict, outfile, indent=4)

    def writeConfig(self, data):
        with open(os.path.join(self.defaultConfigPath, self.flowStopConfigFilename), "w") as outfile:
            json.dump(data, outfile, indent=4)

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