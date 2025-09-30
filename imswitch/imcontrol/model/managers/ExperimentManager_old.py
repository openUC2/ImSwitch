import os
import json
import numpy as np

from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger
from imswitch.imcommon.model import dirtools

class ExperimentManager(SignalInterface):

    def __init__(self, experimentInfo=None, *args, **kwargs):
        self.sigExperimentMaskUpdated = Signal(object)  # (maskCombined)  # (maskCombined)
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        self.tWait = 0.1

    def __init__(self, experimentInfo=None, *args, **kwargs):
        self.sigExperimentMaskUpdated = Signal(object)  # (maskCombined)  # (maskCombined)
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        self.tWait = 0.1
        
        # Store experiment info from setup
        self.experimentInfo = experimentInfo

        # OMERO configuration management
        self.omeroConfigFilename = "omero_config.json"
        self.allOmeroParameterKeys = ["serverUrl", "username", "password", "port", 
                                     "groupId", "projectId", "datasetId", "isEnabled", 
                                     "connectionTimeout", "uploadTimeout"]

        # get default configs for OMERO
        self.defaultConfigPath = os.path.join(dirtools.UserFileDirs.Root, "experimentController")
        if not os.path.exists(self.defaultConfigPath):
            os.makedirs(self.defaultConfigPath)

        # Initialize OMERO configuration
        self._initializeOmeroConfig()

        self.update()

    def _initializeOmeroConfig(self):
        """Initialize OMERO configuration from setup info and/or config file."""
        try:
            with open(os.path.join(self.defaultConfigPath, self.omeroConfigFilename)) as jf:
                # check if all keys are present
                self.omeroConfig = json.load(jf)
                # check if all keys are present
                missing_keys = [key for key in self.allOmeroParameterKeys if key not in self.omeroConfig]
                if missing_keys:
                    raise KeyError(f"Missing keys in OMERO config: {missing_keys}")
                else:
                    self.__logger.info("OMERO configuration loaded successfully from file")

        except Exception as e:
            self.__logger.warning(f"Could not load OMERO config from {self.defaultConfigPath}: {e}")
            
            # Use values from experimentInfo if available, otherwise defaults
            if self.experimentInfo:
                self.omeroConfig = {}
                self.omeroConfig["serverUrl"] = getattr(self.experimentInfo, 'omeroServerUrl', "localhost")
                self.omeroConfig["username"] = getattr(self.experimentInfo, 'omeroUsername', "")
                self.omeroConfig["password"] = getattr(self.experimentInfo, 'omeroPassword', "")
                self.omeroConfig["port"] = getattr(self.experimentInfo, 'omeroPort', 4064)
                self.omeroConfig["groupId"] = getattr(self.experimentInfo, 'omeroGroupId', -1)
                self.omeroConfig["projectId"] = getattr(self.experimentInfo, 'omeroProjectId', -1)
                self.omeroConfig["datasetId"] = getattr(self.experimentInfo, 'omeroDatasetId', -1)
                self.omeroConfig["isEnabled"] = getattr(self.experimentInfo, 'omeroEnabled', False)
                self.omeroConfig["connectionTimeout"] = getattr(self.experimentInfo, 'omeroConnectionTimeout', 30)
                self.omeroConfig["uploadTimeout"] = getattr(self.experimentInfo, 'omeroUploadTimeout', 300)
                self.__logger.info("OMERO configuration initialized from setup info")
            else:
                self.omeroConfig = {}
                self.omeroConfig["serverUrl"] = "localhost"
                self.omeroConfig["username"] = ""
                self.omeroConfig["password"] = ""
                self.omeroConfig["port"] = 4064
                self.omeroConfig["groupId"] = -1
                self.omeroConfig["projectId"] = -1
                self.omeroConfig["datasetId"] = -1
                self.omeroConfig["isEnabled"] = False
                self.omeroConfig["connectionTimeout"] = 30
                self.omeroConfig["uploadTimeout"] = 300
                self.__logger.info("OMERO configuration initialized with defaults")
            
            self.writeOmeroConfig(self.omeroConfig)


    def update(self):
        return None

    def updateOmeroConfig(self, parameterName, value):
        """Update a specific OMERO configuration parameter"""
        if parameterName not in self.allOmeroParameterKeys:
            self.__logger.warning(f"Parameter {parameterName} is not a valid OMERO parameter")
            return False
            
        try:
            with open(os.path.join(self.defaultConfigPath, self.omeroConfigFilename), "r") as infile:
                mDict = json.load(infile)
            
            mDict[parameterName] = value
            
            with open(os.path.join(self.defaultConfigPath, self.omeroConfigFilename), "w") as outfile:
                json.dump(mDict, outfile, indent=4)
            
            # Update internal config
            self.omeroConfig[parameterName] = value
            self.__logger.info(f"Updated OMERO config parameter {parameterName}")
            return True
            
        except Exception as e:
            self.__logger.error(f"Failed to update OMERO config parameter {parameterName}: {e}")
            return False

    def writeOmeroConfig(self, data):
        """Write the complete OMERO configuration to file"""
        try:
            with open(os.path.join(self.defaultConfigPath, self.omeroConfigFilename), "w") as outfile:
                json.dump(data, outfile, indent=4)
            self.__logger.info("OMERO configuration written to file")
        except Exception as e:
            self.__logger.error(f"Failed to write OMERO configuration: {e}")

    def getOmeroConfig(self):
        """Get the current OMERO configuration"""
        return self.omeroConfig.copy()

    def setOmeroConfig(self, config_dict):
        """Set the OMERO configuration from a dictionary"""
        for key, value in config_dict.items():
            if key in self.allOmeroParameterKeys:
                self.updateOmeroConfig(key, value)
            else:
                self.__logger.warning(f"Ignoring unknown OMERO parameter: {key}")

    def isOmeroEnabled(self):
        """Check if OMERO integration is enabled"""
        return self.omeroConfig.get("isEnabled", False)

    def getOmeroConnectionParams(self):
        """Get OMERO connection parameters as a dictionary"""
        if not self.isOmeroEnabled():
            return None
        
        return {
            "serverUrl": self.omeroConfig["serverUrl"],
            "username": self.omeroConfig["username"],
            "password": self.omeroConfig["password"],
            "port": self.omeroConfig["port"],
            "connectionTimeout": self.omeroConfig["connectionTimeout"]
        }

    def getOmeroUploadParams(self):
        """Get OMERO upload parameters as a dictionary"""
        if not self.isOmeroEnabled():
            return None
        
        return {
            "groupId": self.omeroConfig["groupId"],
            "projectId": self.omeroConfig["projectId"],
            "datasetId": self.omeroConfig["datasetId"],
            "uploadTimeout": self.omeroConfig["uploadTimeout"]
        }

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