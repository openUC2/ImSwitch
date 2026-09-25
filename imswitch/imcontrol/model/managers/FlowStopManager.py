import os
import json

import numpy as np

from imswitch.imcommon.framework import SignalInterface
from imswitch.imcommon.model import initLogger
from imswitch.imcommon.model import dirtools


DEFAULT_CONFIG = {
    "wasRunning": False,
    "flowRate": 100,
    "numberOfFrames": -1,
    "experimentName": "FlowStopExperiment",
    "experimentDescription": "",
    "frameRate": 1,
    "savePath": "./",
    "fileFormat": "JPG",
    "axisFlow": "X",
    "axisFocus": "Z",
    "delayTimeAfterRestart": 1,
    "isRecordVideo": False,
    "numImages": -1,
    "uniqueId": "",
    "volumePerImage": 1000,
    "timeToStabilize": 1,
    "pumpSpeed": 10000,
    "pumpTimeout": 5.0,
    "metadata": {},
}


class FlowStopManager(SignalInterface):

    def __init__(self, flowStopInfo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        self.flowStopConfigFilename = "config.json"
        self.defaultConfigPath = os.path.join(dirtools.UserFileDirs.Root, "flowStopController")
        os.makedirs(self.defaultConfigPath, exist_ok=True)

        # Merge the stored config over the defaults: a config written by an older
        # version is missing keys, and losing the user's settings over that is worse
        # than carrying a default forward.
        self.defaultConfig = dict(DEFAULT_CONFIG)
        if not self.defaultConfig["uniqueId"]:
            self.defaultConfig["uniqueId"] = str(np.random.randint(0, 1000000))
        try:
            with open(os.path.join(self.defaultConfigPath, self.flowStopConfigFilename)) as jf:
                self.defaultConfig.update(json.load(jf))
        except FileNotFoundError:
            self.writeConfig(self.defaultConfig)
        except Exception as e:
            self.__logger.error(f"Could not read {self.flowStopConfigFilename}, using defaults: {e}")

    def updateConfig(self, parameterName, value):
        self.defaultConfig[parameterName] = value
        self.writeConfig(self.defaultConfig)

    def writeConfig(self, data):
        self.defaultConfig = data
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
