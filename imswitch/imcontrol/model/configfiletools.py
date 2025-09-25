import glob
import os
from pathlib import Path
import json
from imswitch.imcommon.model import dirtools
from .Options import Options
from imswitch import DEFAULT_SETUP_FILE
import dataclasses

dirtools.initUserFilesIfNeeded()
_setupFilesDir = os.path.join(dirtools.UserFileDirs.Root, 'imcontrol_setups')
os.makedirs(_setupFilesDir, exist_ok=True)
_optionsFilePath = os.path.join(dirtools.UserFileDirs.Config, 'imcontrol_options.json')
_configsFilePath = os.path.join(dirtools.UserFileDirs.Config, 'imcontrol_options.json')

_options = None
_configs = None

def getSetupList():
    return [Path(file).name for file in glob.glob(os.path.join(_setupFilesDir, '*.json'))]

def loadSetupInfo(options, setupInfoType):
    # if options.setupFileName contains absolute path, don't concatenate
    if os.path.isabs(options.setupFileName):
        mPath = options.setupFileName
    else:
        mPath = os.path.join(_setupFilesDir, options.setupFileName)
    print("Loading setup info from: " + mPath)
    '''
    TODO: This is very hacky!! We should implement a proper interface for the different cases
    e.g.:
    - user file does not exist => create new file with default values ? 
    - absolute path to user file that does not exist => error but check if the filename exists in the setup folder
    - file exists but is corrupted => error and create backup of the corrupted file and create new
    - file exists and is valid => load it
    maybe there are more cases - this has to be tested more thoroughly thought through! @GokuGiant
    '''
    
    # check the file exists
    if not os.path.isfile(mPath):
        # test if we can load the file from the setup folder
        if os.path.isfile(os.path.join(_setupFilesDir, os.path.basename(options.setupFileName))):
            mPath = os.path.join(_setupFilesDir, os.path.basename(options.setupFileName))
        else:
            # create a new file with default values
            print("Warning: The setup file does not exist. Creating a new file with default values from virtual microscope from the user defaults .")
            # copy from ./_data/user_defaults/imcontrol_setups/example_virtual_microscope.json 
            defaultSetupFile = os.path.join(dirtools.DataFileDirs.UserDefaults, 'imcontrol_setups', 'example_virtual_microscope.json')
            if os.path.isfile(defaultSetupFile):
                with open(defaultSetupFile, 'r') as src, open(mPath, 'w') as dst:
                    dst.write(src.read())
            else:
                raise FileNotFoundError(f"Default setup file not found: {defaultSetupFile}") 
    
    with open(mPath) as setupFile:
        try:
            mSetupDescription = setupInfoType.from_json(setupFile.read(), infer_missing=True)
        except json.decoder.JSONDecodeError as e:
            # Print the setupFileName to the console
            print("Error: The setup file was corrupted and has been reset to default values.")
            print("Setup file: " + mPath)
            print("Please check the file for errors.")
            print("Using default setup file: " + mPath)
            print("Filecontent:")
            print(setupFile.read())
            print("Error message: " + str(e))
            raise e
        except Exception as e:
            print("Error: Could not load setup file.")
            print("Please check the file for errors.")
            print("Error message: " + str(e))
            raise e
        return mSetupDescription

def saveSetupInfo(options, setupInfo):
    # 1. Make a backup of the current setup file
    # 2. Save the new setup file
    mFilename = os.path.join(_setupFilesDir, options.setupFileName)
    if os.path.isfile(mFilename):
        # make a backup of the current setup file
        backupFileName = mFilename + ".bak"
        if os.path.isfile(backupFileName):
            os.remove(backupFileName)
        os.rename(mFilename, backupFileName)
    try:
        with open(os.path.join(mFilename), 'w') as setupFile:
            setupFile.write(setupInfo.to_json(indent=4))
    except Exception as e:
        print("Error: Could not save setup file.")
        print("Please check the file for errors.")
        print("Error message: " + str(e))
        # revert to the backup file
        if os.path.isfile(backupFileName):
            os.remove(mFilename)
            os.rename(backupFileName, mFilename)
            os.remove(backupFileName)





def loadOptions():
    global _options

    # Check if the options file exists
    if _options is not None:
        return _options, False

    optionsDidNotExist = False
    if DEFAULT_SETUP_FILE is not None:
        _options = Options(
            setupFileName=DEFAULT_SETUP_FILE
        )
        optionsDidNotExist = True
    elif not os.path.isfile(_optionsFilePath):
        _options = Options(
            setupFileName=getSetupList()[0]
        )
        optionsDidNotExist = True
    else:
        try:
            with open(_optionsFilePath, 'r') as optionsFile:
                _options = Options.from_json(optionsFile.read(), infer_missing=True)
                if _options.setupFileName not in getSetupList() and os.path.basename(_options.setupFileName) not in getSetupList():
                    _options = dataclasses.replace(_options, setupFileName="example_virtual_microscope.json")
        except json.decoder.JSONDecodeError:
            # create a warning message as a popup
            print("Warning: The options file was corrupted and has been reset to default values.")

    return _options, optionsDidNotExist

def saveOptions(options):
    global _options

    _options = options
    with open(_optionsFilePath, 'w') as optionsFile:
        optionsFile.write(_options.to_json(indent=4))

def saveConfigs(configs):
    global _configs

    _configs = configs
    with open(_optionsFilePath, 'w') as configsFile:
        configsFile.write(_configs.to_json(indent=4))






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
