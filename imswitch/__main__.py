import importlib
import traceback
import logging
import argparse
import os
# TODO: This file needs a heavy re-write! 
import imswitch
from imswitch.config import get_config, update_config

# python -m imswitch --headless 1 --config-file /Users/bene/ImSwitchConfig/imcontrol_setups/FRAME2b.json --scan-ext-drive-mount true --ext-data-folder ~/Downloads --ext-drive-mount /Volumes
# python -m imswitch --headless --http-port 8001 --socket-port 8002 --config-folder /Users/bene --config-file None 


def main(is_headless:bool=None, default_config:str=None, http_port:int=None, socket_port:int=None, ssl:bool=None, config_folder:str=None,
         data_folder: str=None, scan_ext_data_folder:bool=None, ext_drive_mount:str=None):
    '''
    is_headless: bool => start with or without qt
    default_config: str => path to the config file
    http_port: int => port number (default: 8001)
    socket_port: int => port number (default: 8002)
    ssl: bool => use ssl (default: True)
    config_folder: str => path to the config folder (default: None, pointing to Documents/ImSwitch)
    data_folder: str => path to the data folder (default: None, pointing to Documents/ImSwitchConfig)
    scan_ext_data_folder: bool => if True, we will scan the ext_drive_mount for usb drives and use this for data storage
    ext_drive_mount: str => path to the external drive mount point (default: None, optionally pointing to e.g. /Volumes or /media)



    To start imswitch in headless using the arguments, you can call the main file with the following arguments:
        python main.py --headless or
        python -m imswitch --headless 1 --config-file /Users/bene/ImSwitchConfig/imcontrol_setups/FRAME2b.json --scan-ext-drive-mount true --ext-data-folder ~/Downloads --ext-drive-mount /Volumes
    '''
    try:
        # Get the global configuration instance
        config = get_config()
        
        # Update configuration immediately with any provided parameters
        config.update_from_args(
            is_headless=is_headless,
            default_config=default_config,
            http_port=http_port,
            socket_port=socket_port,
            ssl=ssl,
            config_folder=config_folder,
            data_folder=data_folder,
            scan_ext_data_folder=scan_ext_data_folder,
            ext_drive_mount=ext_drive_mount
        )
        
        # Update legacy globals immediately for backward compatibility
        config.to_legacy_globals(imswitch)
        # Only parse command line arguments if no parameters were passed to main()
        # This prevents argparse conflicts when called from test threads
        if (is_headless is None and default_config is None and http_port is None and 
            socket_port is None and ssl is None and config_folder is None and 
            data_folder is None and scan_ext_data_folder is None and ext_drive_mount is None):
            
            try: # Google Colab does not support argparse
                parser = argparse.ArgumentParser(description='Process some integers.')

                # specify if run in headless mode
                parser.add_argument('--headless', dest='headless', default=False, action='store_true',
                                    help='run in headless mode')

                # specify config file name - None for default
                parser.add_argument('--config-file', dest='config_file', type=str, default=None,
                                    help='specify run with config file')

                # specify http port
                parser.add_argument('--http-port', dest='http_port', type=int, default=8001,
                                    help='specify http port')

                # specify socket port
                parser.add_argument('--socket-port', dest='socket_port', type=int, default=8002,
                                    help='specify socket port')
                # specify ssl
                parser.add_argument('--no-ssl', dest='ssl', default=True, action='store_false',
                                    help='specify ssl')

                # specify the config folder (e.g. if running from a different location / container)
                parser.add_argument('--config-folder', dest='config_folder', type=str, default=None,
                                    help='specify config folder')

                parser.add_argument('--ext-data-folder', dest='data_folder', type=str, default=None,
                                    help='point to a folder to store the data. This is the default location for the data folder. If not specified, the default location will be used.')

                parser.add_argument('--scan-ext-drive-mount', dest='scan_ext_data_folder', default=False, action='store_true',
                                    help='scan the external mount (linux only) if we have a USB drive to save to')

                parser.add_argument('--ext-drive-mount', dest='ext_drive_mount', type=str, default=None,
                                    help='specify the external drive mount point (e.g. /Volumes or /media)')

                # Add Jupyter/Colab specific arguments to prevent errors
                parser.add_argument('-f', '--connection-file', dest='connection_file', type=str, default=None,
                                    help='Jupyter connection file (ignored)')

                # Parse known arguments only, ignore unknown ones (important for Jupyter environments)
                args, unknown = parser.parse_known_args()

                # Log unknown arguments for debugging
                if unknown:
                    print(f"Ignoring unknown arguments: {unknown}")

                # Update configuration from parsed arguments
                config.update_from_argparse(args)
                
                # Handle special config file validation
                if hasattr(args, 'config_file') and args.config_file:
                    if isinstance(args.config_file, str) and args.config_file.find("json") >= 0:
                        config.default_config = args.config_file
                    else:
                        config.default_config = None

                # Validate directories exist before setting them
                if hasattr(args, 'config_folder') and args.config_folder and os.path.isdir(args.config_folder):
                    config.config_folder = args.config_folder
                if hasattr(args, 'data_folder') and args.data_folder and os.path.isdir(args.data_folder):
                    config.data_folder = args.data_folder
                
                # Update legacy globals
                config.to_legacy_globals(imswitch)

            except Exception as e:
                print(f"Argparse error: {e}")
                pass
        
        # Apply final configuration update to legacy globals (ensures consistency)
        config.to_legacy_globals(imswitch)

        # FIXME: !!!! This is because the headless flag is loaded after commandline input
        from imswitch.imcommon import prepareApp, launchApp
        from imswitch.imcommon.controller import ModuleCommunicationChannel, MultiModuleWindowController
        from imswitch.imcommon.model import modulesconfigtools, pythontools, initLogger

        logger = initLogger('main')
        logger.info(f'Starting ImSwitch {config.version}')
        logger.info(f'Headless mode: {config.is_headless}')
        logger.info(f'SSL: {config.ssl}')
        logger.info(f'Config file: {config.default_config}')
        logger.info(f'Config folder: {config.config_folder}')
        logger.info(f'Data folder: {config.data_folder}')

        # TODO: check if port is already in use
        
        if config.is_headless:
            app = None
        else:
            app = prepareApp()
        enabledModuleIds = modulesconfigtools.getEnabledModuleIds()

        if 'imscripting' in enabledModuleIds:
            if config.is_headless:
                enabledModuleIds.remove('imscripting')
            else:
                # Ensure that imscripting is added last
                enabledModuleIds.append(enabledModuleIds.pop(enabledModuleIds.index('imscripting')))

        if 'imnotebook' in enabledModuleIds:
            # Ensure that imnotebook is added last
            try:
                enabledModuleIds.append(enabledModuleIds.pop(enabledModuleIds.index('imnotebook')))
            except ImportError:
                logger.error('QtWebEngineWidgets not found, disabling imnotebook')
                enabledModuleIds.remove('imnotebook')

        modulePkgs = [importlib.import_module(pythontools.joinModulePath('imswitch', moduleId))
                    for moduleId in enabledModuleIds]

        # connect the different controllers through the communication channel
        moduleCommChannel = ModuleCommunicationChannel()

        # only create the GUI if necessary
        if not config.is_headless:
            from imswitch.imcommon.view import MultiModuleWindow, ModuleLoadErrorView
            multiModuleWindow = MultiModuleWindow('ImSwitch')
            multiModuleWindowController = MultiModuleWindowController.create(
                multiModuleWindow, moduleCommChannel
            )
            multiModuleWindow.show(showLoadingScreen=True)
            app.processEvents()  # Draw window before continuing
        else:
            multiModuleWindow = None
            multiModuleWindowController = None

        # Register modules
        for modulePkg in modulePkgs:
            moduleCommChannel.register(modulePkg)

        # Load modules
        moduleMainControllers = dict()

        for i, modulePkg in enumerate(modulePkgs):
            moduleId = modulePkg.__name__
            moduleId = moduleId[moduleId.rindex('.') + 1:]  # E.g. "imswitch.imcontrol" -> "imcontrol"

            # The displayed module name will be the module's __title__, or alternatively its ID if
            # __title__ is not set
            moduleName = modulePkg.__title__ if hasattr(modulePkg, '__title__') else moduleId
            # we load all the controllers, managers and widgets here:
            try:
                view, controller = modulePkg.getMainViewAndController(
                    moduleCommChannel=moduleCommChannel,
                    multiModuleWindowController=multiModuleWindowController,
                    moduleMainControllers=moduleMainControllers
                )
                logger.info(f'initialize module {moduleId}')

            except Exception as e:
                logger.error(f'Failed to initialize module {moduleId}')
                logger.error(e)
                logger.error(traceback.format_exc())
                moduleCommChannel.unregister(modulePkg)
                if not config.is_headless:
                    from imswitch.imcommon.view import ModuleLoadErrorView
                    multiModuleWindow.addModule(moduleId, moduleName, ModuleLoadErrorView(e))
            else:
                # Add module to window
                if not config.is_headless:
                    multiModuleWindow.addModule(moduleId, moduleName, view)
                moduleMainControllers[moduleId] = controller

                # in case of the imnotebook, spread the notebook url
                if moduleId == 'imnotebook':
                    config.jupyter_url = controller.webaddr
                    # Update legacy global for backward compatibility
                    imswitch.jupyternotebookurl = controller.webaddr

                # Update loading progress
                if not config.is_headless:
                    multiModuleWindow.updateLoadingProgress(i / len(modulePkgs))
                    app.processEvents()  # Draw window before continuing
        logger.info(f'init done')
        launchApp(app, multiModuleWindow, moduleMainControllers.values())
    except Exception as e:
        logging.error(traceback.format_exc())


if __name__ == '__main__':
    main()

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
