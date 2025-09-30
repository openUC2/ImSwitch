import json
import os

from imswitch.imcommon.model import dirtools
from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger


class StresstestManager(SignalInterface):
    """Manager for stress test configuration and default parameters.
    
    Loads and saves default stress test parameters from/to a configuration file,
    similar to HistoScanManager pattern.
    """

    def __init__(self, StresstestInfo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        self.stresstestConfigFilename = "config.json"

        # Get default configs directory
        self.defaultConfigPath = os.path.join(dirtools.UserFileDirs.Root, "stresstestController")
        if not os.path.exists(self.defaultConfigPath):
            os.makedirs(self.defaultConfigPath)

        # Default stress test parameters
        self.defaultParams = {
            "minPosX": 0.0,
            "maxPosX": 10000.0,
            "minPosY": 0.0,
            "maxPosY": 10000.0,
            "numRandomPositions": 10,
            "numCycles": 5,
            "timeInterval": 60.0,
            "illuminationIntensity": 50,
            "exposureTime": 0.1,
            "saveImages": True,
            "outputPath": "",
            "enableImageBasedError": False,
            "numImagesPerPosition": 5,
            "imageRegistrationMethod": "fft",
            "pixelSizeUM": 0.1
        }

        # Load configuration or use defaults
        try:
            with open(os.path.join(self.defaultConfigPath, self.stresstestConfigFilename)) as jf:
                loaded_config = json.load(jf)
                self.__logger.debug(f"Loaded stress test config from {self.defaultConfigPath}")

                # Update default params with loaded values
                for key, value in loaded_config.items():
                    if key in self.defaultParams:
                        setattr(self, key, value)
                        self.defaultParams[key] = value
                    else:
                        setattr(self, key, value)

        except Exception as e:
            self.__logger.debug(f"Could not load default config from {self.defaultConfigPath}: {e}")
            self.__logger.debug("Setting default values, need to save them later once they are set")

            # Set attributes from default params
            for key, value in self.defaultParams.items():
                setattr(self, key, value)

        # Set default output path if not specified
        if not hasattr(self, 'outputPath') or not self.outputPath:
            self.outputPath = os.path.join(dirtools.UserFileDirs.Root, 'stresstest_results')
            self.defaultParams["outputPath"] = self.outputPath

    def getDefaultParams(self):
        """Get all default parameters as a dictionary"""
        params = {}
        for key in self.defaultParams.keys():
            params[key] = getattr(self, key, self.defaultParams[key])
        return params

    def updateParams(self, params_dict):
        """Update parameters and save to config file"""
        for key, value in params_dict.items():
            if key in self.defaultParams:
                setattr(self, key, value)
                self.defaultParams[key] = value
        
        # Save updated configuration
        self.writeConfig(self.getDefaultParams())

    def writeConfig(self, data):
        """Write configuration to file"""
        try:
            with open(os.path.join(self.defaultConfigPath, self.stresstestConfigFilename), "w") as outfile:
                json.dump(data, outfile, indent=4)
            self.__logger.debug(f"Saved stress test config to {self.defaultConfigPath}")
        except Exception as e:
            self.__logger.error(f"Failed to write stress test config: {e}")

    def update(self):
        """Update method for compatibility"""
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