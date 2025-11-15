import glob
import os
from abc import ABC
from pathlib import Path
from shutil import copy2, disk_usage
from typing import Optional

# Import simplified storage path utilities
from .storage_paths import get_data_path, get_config_path


def getSystemUserDir():
    """ 
    Returns the configuration directory for ImSwitch.
    
    This is now a simple wrapper around get_config_path() for backward compatibility.
    """
    return get_config_path()


# Base directory for program data files (templates, defaults, etc.)
_baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_data')
# Base directory for user configuration files
_baseUserFilesDir = getSystemUserDir()




def is_writable_directory(path: str) -> bool:
    """
    Checks if 'path' is writable by attempting to create and remove a tiny file.
    """
    if not path or not os.path.isdir(path):
        return False
    try:
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception:
        return False


def pick_first_external_folder(default_data_path: str) -> Optional[str]:
    """
    Picks the first subdirectory in 'default_data_path' that is not a system volume and is writable.
    
    Used for external drive detection in Docker environments.
    """
    if not default_data_path or not os.path.exists(default_data_path):
        return None

    SYSTEM_VOLUMES = {"Macintosh HD", "System Volume Information", "Recovery", "Preboot", "VM"}
    
    for d in sorted(os.listdir(default_data_path)):
        full_path = os.path.join(default_data_path, d)
        if not os.path.isdir(full_path):
            continue
        
        # Exclude system volumes and hidden directories
        if d not in SYSTEM_VOLUMES and not d.startswith('.') and is_writable_directory(full_path):
            return full_path
    
    return None


def getDiskusage():
    """
    Checks if the available disk space is above the threshold percentage.
    Returns True if disk is above the threshold occupied.
    """
    # Get the current working directory's drive (cross-platform compatibility)
    current_drive = os.path.abspath(os.sep)

    # Get disk usage statistics
    total, used, free = disk_usage(current_drive)

    # Calculate percentage used
    percent_used = (used / total)

    # Check if it exceeds the threshold
    return percent_used

def initUserFilesIfNeeded():
    """ Initializes all directories that will be used to store user data and
    copies example files. """

    # Initialize directories
    for userFileDir in UserFileDirs.list():
        if userFileDir is not None:
            print(f"Initializing user directory: {userFileDir} by making it if not available already")
            os.makedirs(userFileDir, exist_ok=True)

    # Copy default user files
    for file in glob.glob(os.path.join(DataFileDirs.UserDefaults, '**'), recursive=True):
        filePath = Path(file)

        if not filePath.is_file():
            continue

        if filePath.name.lower() == 'readme.txt':
            continue  # Skip readme.txt files

        relativeFilePath = filePath.relative_to(DataFileDirs.UserDefaults)
        copyDestination = _baseUserFilesDir / relativeFilePath

        if os.path.exists(copyDestination):
            continue  # Don't overwrite existing files

        try:
            os.makedirs(copyDestination.parent, exist_ok=True)
        except FileExistsError:  # Directory path (or part of it) exists as a file
            continue

        copy2(filePath, copyDestination)


class FileDirs(ABC):
    """ Base class for directory catalog classes. """

    @classmethod
    def list(cls):
        """ Returns all directories in the catalog. """
        return [cls.__dict__.get(name) for name in dir(cls)
                if not callable(getattr(cls, name)) and not name.startswith('_')]

class DataFileDirs(FileDirs):
    """ Catalog of directories that contain program data/library/resource
    files. """
    Root = _baseDataFilesDir
    Libs = os.path.join(_baseDataFilesDir, 'libs')
    UserDefaults = os.path.join(_baseDataFilesDir, 'user_defaults')

class UserFileDirs(FileDirs):
    """ 
    Catalog of directories that contain user configuration and data files.
    
    This class now uses the simplified storage_paths module for path resolution.
    Paths are resolved dynamically to support runtime changes via API.
    """
    
    Root = _baseUserFilesDir
    Config = os.path.join(_baseUserFilesDir, 'config')
    Data = get_data_path()  # Dynamic resolution using storage_paths
    
    @classmethod
    def refresh_paths(cls):
        """Refresh paths from current configuration. Call this after runtime path changes."""
        cls.Root = get_config_path()
        cls.Config = os.path.join(cls.Root, 'config')
        cls.Data = get_data_path()




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
