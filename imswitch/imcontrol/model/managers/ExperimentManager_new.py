from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger

class ExperimentManager(SignalInterface):

    def __init__(self, experimentInfo, *args, **kwargs):
        self.sigExperimentMaskUpdated = Signal(object)  # (maskCombined)  # (maskCombined)
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        
        # Initialize OMERO configuration from experimentInfo (similar to MCTManager pattern)
        if experimentInfo is not None:
            # OMERO configuration from setup
            self.omeroServerUrl = getattr(experimentInfo, 'omeroServerUrl', "localhost")
            self.omeroUsername = getattr(experimentInfo, 'omeroUsername', "")
            self.omeroPassword = getattr(experimentInfo, 'omeroPassword', "")
            self.omeroPort = getattr(experimentInfo, 'omeroPort', 4064)
            self.omeroGroupId = getattr(experimentInfo, 'omeroGroupId', -1)
            self.omeroProjectId = getattr(experimentInfo, 'omeroProjectId', -1)
            self.omeroDatasetId = getattr(experimentInfo, 'omeroDatasetId', -1)
            self.omeroEnabled = getattr(experimentInfo, 'omeroEnabled', False)
            self.omeroConnectionTimeout = getattr(experimentInfo, 'omeroConnectionTimeout', 30)
            self.omeroUploadTimeout = getattr(experimentInfo, 'omeroUploadTimeout', 300)
            self.__logger.info("OMERO configuration loaded from setup info")
        else:
            # Default values if no experimentInfo provided
            self.omeroServerUrl = "localhost"
            self.omeroUsername = ""
            self.omeroPassword = ""
            self.omeroPort = 4064
            self.omeroGroupId = -1
            self.omeroProjectId = -1
            self.omeroDatasetId = -1
            self.omeroEnabled = False
            self.omeroConnectionTimeout = 30
            self.omeroUploadTimeout = 300
            self.__logger.info("OMERO configuration initialized with defaults")

        # General timing parameter (similar to MCTManager's tWait)
        self.tWait = 0.1

        self.update()

    def update(self):
        return None

    def getOmeroConfig(self):
        """Get the current OMERO configuration as a dictionary."""
        return {
            "serverUrl": self.omeroServerUrl,
            "username": self.omeroUsername,
            "password": self.omeroPassword,
            "port": self.omeroPort,
            "groupId": self.omeroGroupId,
            "projectId": self.omeroProjectId,
            "datasetId": self.omeroDatasetId,
            "isEnabled": self.omeroEnabled,
            "connectionTimeout": self.omeroConnectionTimeout,
            "uploadTimeout": self.omeroUploadTimeout
        }

    def setOmeroConfig(self, config_dict):
        """Set OMERO configuration from a dictionary."""
        if "serverUrl" in config_dict:
            self.omeroServerUrl = config_dict["serverUrl"]
        if "username" in config_dict:
            self.omeroUsername = config_dict["username"]
        if "password" in config_dict:
            self.omeroPassword = config_dict["password"]
        if "port" in config_dict:
            self.omeroPort = config_dict["port"]
        if "groupId" in config_dict:
            self.omeroGroupId = config_dict["groupId"]
        if "projectId" in config_dict:
            self.omeroProjectId = config_dict["projectId"]
        if "datasetId" in config_dict:
            self.omeroDatasetId = config_dict["datasetId"]
        if "isEnabled" in config_dict:
            self.omeroEnabled = config_dict["isEnabled"]
        if "connectionTimeout" in config_dict:
            self.omeroConnectionTimeout = config_dict["connectionTimeout"]
        if "uploadTimeout" in config_dict:
            self.omeroUploadTimeout = config_dict["uploadTimeout"]
        
        self.__logger.info("OMERO configuration updated")

    def isOmeroEnabled(self):
        """Check if OMERO integration is enabled."""
        return self.omeroEnabled

    def getOmeroConnectionParams(self):
        """Get OMERO connection parameters as a dictionary."""
        if not self.isOmeroEnabled():
            return None
        
        return {
            "serverUrl": self.omeroServerUrl,
            "username": self.omeroUsername,
            "password": self.omeroPassword,
            "port": self.omeroPort,
            "connectionTimeout": self.omeroConnectionTimeout
        }

    def getOmeroUploadParams(self):
        """Get OMERO upload parameters as a dictionary."""
        if not self.isOmeroEnabled():
            return None
        
        return {
            "groupId": self.omeroGroupId,
            "projectId": self.omeroProjectId,
            "datasetId": self.omeroDatasetId,
            "uploadTimeout": self.omeroUploadTimeout
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
