import imswitch
import dataclasses
import sys
import os
from imswitch import IS_HEADLESS, DEFAULT_DATA_PATH

__imswitch_module__ = True
__title__ = 'Hardware Control'


def getMainViewAndController(moduleCommChannel, *_args,
                             overrideSetupInfo=None, overrideOptions=None, **_kwargs):
    from imswitch.imcommon.model import initLogger
    from .controller import ImConMainController
    from .model import configfiletools
    from .view import ViewSetupInfo, ImConMainView, ImConMainViewNoQt
    logger = initLogger('imcontrol init')

    def pickSetup(options):
        from qtpy import QtWidgets
        from imswitch.imcontrol.view import PickSetupDialog

        # Let user pick the setup to use
        pickSetupDialog = PickSetupDialog()
        pickSetupDialog.setSetups(configfiletools.getSetupList())
        result = pickSetupDialog.exec_()
        setupFileName = pickSetupDialog.getSelectedSetup()
        if result != QtWidgets.QDialog.Accepted or not setupFileName:
            logger.critical('User did not pick a setup to use')
            sys.exit()
        return dataclasses.replace(options, setupFileName=setupFileName)


    '''
    load the options such as imcontrol, imnotebook, etc.
    '''
    if overrideOptions is None:
        options, optionsDidNotExist = configfiletools.loadOptions()
        if not optionsDidNotExist:
            if not IS_HEADLESS: options = pickSetup(options)  # Setup to use not set, let user pick
        configfiletools.saveOptions(options)
    else:
        # force the options to use a specific configuration
        options = overrideOptions

    '''
    load the setup configuration including detectors, stages, etc.
    '''
    if overrideSetupInfo is None: # TODO: This needs a rework!!
        print("imswitch.DEFAULT_SETUP_FILE: ", imswitch.DEFAULT_SETUP_FILE)
        print("imswitch.DEFAULT_SETUP_FILE type: ", type(imswitch.DEFAULT_SETUP_FILE))
        if imswitch.DEFAULT_SETUP_FILE is not None and imswitch.DEFAULT_SETUP_FILE.find("None") == -1:
            try:
                # Attempt to use the file provided via command line arguments
                print(f"Trying to use the file: {imswitch.DEFAULT_SETUP_FILE}")
                setupFileName = imswitch.DEFAULT_SETUP_FILE

                # Check if the file exists before proceeding
                if not os.path.exists(setupFileName):
                    raise FileNotFoundError(f"Setup file not found: {setupFileName}")

                options = dataclasses.replace(options, setupFileName=setupFileName)
                print(f"Loading setup info from: {setupFileName}")
                setupInfo = configfiletools.loadSetupInfo(options, ViewSetupInfo)
            except FileNotFoundError as fnf_error:
                print(f"FileNotFoundError: {fnf_error}")
                raise
            except Exception as e:
                print(f"Error setting default setup file from commandline: {str(e)}")
                print(f"Options: {options}")
                raise KeyError
        else:
            try:
                setupInfo = configfiletools.loadSetupInfo(options, ViewSetupInfo)
            except Exception as e:
                # Have user pick setup anyway
                logger.debug('Setup file not found, let user pick setup')
                logger.debug('Error loading setup file: %s', e)
                if IS_HEADLESS:
                    # load from default _data
                    setupFileName = imswitch.DEFAULT_SETUP_FILE
                else:
                    options = pickSetup(options)
                    configfiletools.saveOptions(options)
                setupInfo = configfiletools.loadSetupInfo(options, ViewSetupInfo)
    # this case is used for pytesting
    elif overrideSetupInfo is not None:
        setupInfo = overrideSetupInfo
    else:
        raise KeyError # FIXME: !!!!
    logger.debug(f'Setup used: {options.setupFileName}')

    # Override Data PAth
    if DEFAULT_DATA_PATH is not None:
        logger.debug("Overriding data save path with: "+DEFAULT_DATA_PATH)
        options_rec = dataclasses.replace(options.recording, outputFolder=DEFAULT_DATA_PATH)
        options = dataclasses.replace(options, recording=options_rec)
    if not IS_HEADLESS:
        view = ImConMainView(options, setupInfo)
    else:
        view = ImConMainViewNoQt(options, setupInfo)
    try:
        controller = ImConMainController(options, setupInfo, view, moduleCommChannel)
    except Exception as e:
        # TODO: To broad exception
        logger.error('Error initializing controller: %s', e)
        if not IS_HEADLESS: view.close()
        raise e

    return view, controller


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
