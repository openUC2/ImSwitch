"""
ImSwitch configuration management.

This module provides a centralized way to manage ImSwitch configuration,
replacing the global variables with a proper configuration object.
"""
import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ImSwitchConfig:
    """Central configuration object for ImSwitch."""
    
    # Core settings
    is_headless: bool = False
    
    # Network settings
    http_port: int = 8001
    socket_port: int = 8002
    ssl: bool = True
    
    # File paths
    default_config: Optional[str] = None
    config_folder: Optional[str] = None
    data_folder: Optional[str] = None
    
    # External drive settings
    scan_ext_data_folder: bool = False
    ext_drive_mount: Optional[str] = None
    
    # Streaming settings
    socket_stream: bool = True
    
    # Jupyter settings
    jupyter_port: int = 8888
    jupyter_url: str = ""
    
    # Version info
    version: str = "2.1.41"
    
    def update_from_args(self, **kwargs) -> None:
        """Update configuration from keyword arguments, ignoring None values."""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    
    def update_from_argparse(self, args) -> None:
        """Update configuration from argparse Namespace object."""
        # Map argparse attribute names to config attribute names
        arg_mapping = {
            'headless': 'is_headless',
            'config_file': 'default_config',
            'http_port': 'http_port',
            'socket_port': 'socket_port',
            'ssl': 'ssl',
            'config_folder': 'config_folder',
            'data_folder': 'data_folder',
            'scan_ext_data_folder': 'scan_ext_data_folder',
            'ext_drive_mount': 'ext_drive_mount'
        }
        
        for arg_name, config_attr in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    setattr(self, config_attr, value)
    
        
    def to_legacy_globals(self, imswitch_module) -> None:
        """Update legacy global variables for backward compatibility."""
        imswitch_module.IS_HEADLESS = self.is_headless
        imswitch_module.__httpport__ = self.http_port
        imswitch_module.__socketport__ = self.socket_port
        imswitch_module.__ssl__ = self.ssl
        imswitch_module.DEFAULT_SETUP_FILE = self.default_config
        imswitch_module.DEFAULT_CONFIG_PATH = self.config_folder
        imswitch_module.DEFAULT_DATA_PATH = self.data_folder
        imswitch_module.SCAN_EXT_DATA_FOLDER = self.scan_ext_data_folder
        imswitch_module.EXT_DRIVE_MOUNT = self.ext_drive_mount
        imswitch_module.SOCKET_STREAM = self.socket_stream
        imswitch_module.jupyternotebookurl = self.jupyter_url


# Global configuration instance
_config: Optional[ImSwitchConfig] = None


def get_config() -> ImSwitchConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ImSwitchConfig()
    return _config


def set_config(config: ImSwitchConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def update_config(**kwargs) -> ImSwitchConfig:
    """Update the global configuration with new values."""
    config = get_config()
    config.update_from_args(**kwargs)
    return config


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
