import glob
import os
from pathlib import Path
import json
from imswitch.imcommon.model import dirtools
from .Options import Options
from imswitch import DEFAULT_SETUP_FILE
import dataclasses

def getSetupList():
    return [Path(file).name for file in glob.glob(os.path.join(_setupFilesDir, '*.json'))]

def loadSetupInfo(options, setupInfoType):
    # if options.setupFileName contains absolute path, don't concatenate 
    if os.path.isabs(options.setupFileName):
        mPath = options.setupFileName
    else:
        mPath = os.path.join(_setupFilesDir, options.setupFileName)
    print("Loading setup info from: " + mPath)
    with open(mPath) as setupFile:
        return setupInfoType.from_json(setupFile.read(), infer_missing=True)

def saveSetupInfo(options, setupInfo):
    with open(os.path.join(_setupFilesDir, options.setupFileName), 'w') as setupFile:
        setupFile.write(setupInfo.to_json(indent=4))


def loadOptions():
    global _options

    if _options is not None:
        return _options, False

    optionsDidNotExist = False
    if not os.path.isfile(_optionsFilePath):
        _options = Options(
            setupFileName=getSetupList()[0]
        )
        optionsDidNotExist = True
    else:
        try:
            with open(_optionsFilePath, 'r') as optionsFile:
                _options = Options.from_json(optionsFile.read(), infer_missing=True)
                if _options.setupFileName not in getSetupList():
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



dirtools.initUserFilesIfNeeded()
_setupFilesDir = os.path.join(dirtools.UserFileDirs.Root, 'imcontrol_setups')
os.makedirs(_setupFilesDir, exist_ok=True)
_optionsFilePath = os.path.join(dirtools.UserFileDirs.Config, 'imcontrol_options.json')
_configsFilePath = os.path.join(dirtools.UserFileDirs.Config, 'imcontrol_options.json')

_options = None
_configs = None


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
