import glob
import os
from abc import ABC
from pathlib import Path
from shutil import copy2, disk_usage
from imswitch import IS_HEADLESS, __file__, DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH, SCAN_EXT_DATA_FOLDER, EXT_DRIVE_MOUNT
import platform
import subprocess

def getSystemUserDir():
    """ Returns the user's documents folder if they are using a Windows system,
    or their home folder if they are using another operating system. """

    if DEFAULT_CONFIG_PATH is not None:
        print("We use the user-provided configuration path: " + DEFAULT_CONFIG_PATH)
        return os.path.join(DEFAULT_CONFIG_PATH)
    else:
        if os.name == 'nt':  # Windows system, try to return documents directory
            try:
                import ctypes.wintypes
                CSIDL_PERSONAL = 5  # Documents
                SHGFP_TYPE_CURRENT = 0  # Current value

                buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
                ctypes.windll.shell32.SHGetFolderPathW(0, CSIDL_PERSONAL, 0, SHGFP_TYPE_CURRENT, buf)

                return buf.value
            except ImportError:
                pass
            #TOOD: How can we ensure that configuration files are updated automatically..
        return os.path.expanduser('~')  # Non-Windows system, return home directory


_baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_data')
_baseUserFilesDir = os.path.join(getSystemUserDir(), 'ImSwitchConfig')




def is_writable_directory(path: str) -> bool:
    # Checks if 'path' is writable by attempting to create and remove a tiny file.
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


def pick_first_external_mount(default_data_path: str):
    # This function picks the first subdirectory in 'default_data_path'
    # that is not obviously a system volume and is writable.
    if not default_data_path or not os.path.exists(default_data_path):
        return None

    for d in sorted(os.listdir(default_data_path)):
        full_path = os.path.join(default_data_path, d)
        if not os.path.isdir(full_path):
            continue
        # Exclude common system volumes
        if d not in ("Macintosh HD", "System Volume Information"):
            # Exclude hidden directories
            if not d.startswith('.') and is_writable_directory(full_path):
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

#TODO: THIS IS A MESS! We need to find a better way to handle the default data path
class UserFileDirs(FileDirs):
    """ Catalog of directories that contain user configuration files. """
    Root = _baseUserFilesDir
    Config = os.path.join(_baseUserFilesDir, 'config')
    Data = os.path.join(_baseUserFilesDir, 'data')
    if DEFAULT_DATA_PATH is not None:
        Data = DEFAULT_DATA_PATH
    if SCAN_EXT_DATA_FOLDER and EXT_DRIVE_MOUNT is not None:
        # TODO: This is a testing workaround
        '''
        Basic idea: We provide ImSwitch (most likely runing inside docker) with the path to the external mounts for external drives (e.g. /media or /Volumes)
        ImSwitch now has to pick the external drive and check if it is mounted and use this as a data storage
        '''
        # If SCAN_EXT_DATA_FOLDER or user sets default_data_path, pick the subfolder
        chosen_mount = pick_first_external_mount(EXT_DRIVE_MOUNT)
        if chosen_mount:
            Data = chosen_mount




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