import dataclasses
import pkg_resources
import h5py
from imswitch import IS_HEADLESS
from imswitch.imcommon.controller import MainController, PickDatasetsController
from imswitch.imcommon.model import (
    ostools, initLogger, generateAPI, generateUI, generateShortcuts, SharedAttributes
)
from imswitch.imcommon.framework import Thread
from .server import ImSwitchServer
from imswitch.imcontrol.model import configfiletools
from imswitch.imcontrol.view import guitools
#from . import controllers
from .CommunicationChannel import CommunicationChannel
from .MasterController import MasterController
from .PickSetupController import PickSetupController
from .basecontrollers import ImConWidgetControllerFactory
import threading
import importlib
import logging

class ImConMainController(MainController):
    def __init__(self, options, setupInfo, mainView, moduleCommChannel):
        self.__logger = initLogger(self)
        self.__logger.debug('Initializing')

        self.__options = options
        self.__setupInfo = setupInfo
        self.__mainView = mainView
        self._moduleCommChannel = moduleCommChannel

        # Init communication channel and master controller
        self.__commChannel = CommunicationChannel(self, self.__setupInfo)
        self.__masterController = MasterController(self.__setupInfo, self.__commChannel,
                                                   self._moduleCommChannel)

        # List of Controllers for the GUI Widgets
        self.__factory = ImConWidgetControllerFactory(
            self.__setupInfo, self.__masterController, self.__commChannel, self._moduleCommChannel
        )

        if not IS_HEADLESS:
            # Connect view signals
            self.__mainView.sigLoadParamsFromHDF5.connect(self.loadParamsFromHDF5)
            self.__mainView.sigPickSetup.connect(self.pickSetup)
            self.__mainView.sigClosing.connect(self.closeEvent)
            self.pickSetupController = self.__factory.createController(
                PickSetupController, self.__mainView.pickSetupDialog
            )
            self.pickDatasetsController = self.__factory.createController(
                PickDatasetsController, self.__mainView.pickDatasetsDialog
            )

        self.controllers = {}

        for widgetKey, widget in self.__mainView.widgets.items():
            self.__logger.info(f'Creating controller for widget {widgetKey}')

            controller_name = f'{widgetKey}Controller'
            if widgetKey == 'Scan':
                controller_name = f'{widgetKey}Controller{self.__setupInfo.scan.scanWidgetType}'
            if widgetKey == 'ImSwitchServer':
                continue

            controller_class = None
            # Try to get controller from controllers module (static import)
            #if hasattr(controllers, controller_name):
            #    controller_class = getattr(controllers, controller_name)
            try:
                module = importlib.import_module(f'imswitch.imcontrol.controller.controllers.{controller_name}')
                controller_class = getattr(module, controller_name)
                #module = importlib.import_module(f'.{controller_name}', package='imswitch.imcontrol.controller.controllers')
            except Exception as e:
                self.__logger.warning(f"Could not dynamically import {controller_name}: {e}")
                continue
            if controller_class is not None:
                try:self.controllers[widgetKey] = self.__factory.createController(controller_class, widget)
                except:continue
            else:
                try:
                    mPlugin = self.loadPlugin(widgetKey)
                    if mPlugin is None:
                        raise ValueError(f'No controller found for widget {widgetKey}')
                    self.controllers[widgetKey] = self.__factory.createController(mPlugin, widget)
                except Exception as e:
                    self.__logger.debug(e)


        # Add WiFiController in any way # TODO: Better would be to add this to the widget dict 
        try: 
            self.__logger.info(f'Creating controller for widget WiFi')
            controller_name = "WiFiController"
            module = importlib.import_module(f'imswitch.imcontrol.controller.controllers.WiFiController')
            controller_class = getattr(module, controller_name)
            if controller_class is not None:
                self.controllers["WiFi"] = self.__factory.createController(controller_class, widget)
        except Exception as e:
            self.__logger.warning(f"Could not dynamically import {controller_name}: {e}")
        
        
        # Generate API
        self.__api = None
        apiObjs = list(self.controllers.values()) + [self.__commChannel]
        self.__api = generateAPI(
            apiObjs,
            missingAttributeErrorMsg=lambda attr: f'The imcontrol API does either not have any'
                                                f' method {attr}, or the widget that defines it'
                                                f' is not included in your currently active'
                                                f' hardware setup file.'
        )
        self.__apiui = None
        if IS_HEADLESS:
            uiObjs = mainView.widgets
            self.__apiui = generateUI(
                uiObjs,
                missingAttributeErrorMsg=lambda attr: f'The imcontrol API does either not have any'
                                                    f' method {attr}, or the widget that defines it'
                                                    f' is not included in your currently active'
                                                    f' hardware setup file.'
            )
            
            
        # Generate Shorcuts
        if not IS_HEADLESS:
            self.__shortcuts = None
            shorcutObjs = list(self.__mainView.widgets.values())
            self.__shortcuts = generateShortcuts(shorcutObjs)
            self.__mainView.addShortcuts(self.__shortcuts)

        self.__logger.debug("Start ImSwitch Server")
        self._serverWorker = ImSwitchServer(self.__api, self.__apiui, setupInfo)
        self._thread = threading.Thread(target=self._serverWorker.run)
        self._thread.start()


    def loadPlugin(self, widgetKey):
        # try to get it from the plugins
        foundPluginController = False
        for entry_point in pkg_resources.iter_entry_points(f'imswitch.implugins'):
            if entry_point.name == f'{widgetKey}_controller':
                packageController = entry_point.load()
                return packageController
        self.__logger.error(f'No controller found for widget {widgetKey}')
        return None

    @property
    def api(self):
        return self.__api

    @property
    def shortcuts(self):
        return self.__shortcuts

    def loadParamsFromHDF5(self):
        """ Set detector, positioner, laser etc. params from values saved in a
        user-picked HDF5 snap/recording. """

        filePath = guitools.askForFilePath(self.__mainView, 'Open HDF5 file', nameFilter='*.hdf5')
        if not filePath:
            return

        with h5py.File(filePath) as file:
            datasetsInFile = file.keys()
            if len(datasetsInFile) < 1:
                # File does not contain any datasets
                return
            elif len(datasetsInFile) == 1:
                datasetToLoad = list(datasetsInFile)[0]
            else:
                # File contains multiple datasets
                self.pickDatasetsController.setDatasets(filePath, datasetsInFile)
                if not self.__mainView.showPickDatasetsDialogBlocking():
                    return

                datasetsSelected = self.pickDatasetsController.getSelectedDatasets()
                if len(datasetsSelected) != 1:
                    return

                datasetToLoad = datasetsSelected[0]

            attrs = SharedAttributes.fromHDF5File(file, datasetToLoad)
            self.__commChannel.sharedAttrs.update(attrs)

    def pickSetup(self):
        """ Let the user change which setup is used. """

        options, _ = configfiletools.loadOptions()

        self.pickSetupController.setSetups(configfiletools.getSetupList())
        self.pickSetupController.setSelectedSetup(options.setupFileName)
        if not self.__mainView.showPickSetupDialogBlocking():
            return
        setupFileName = self.pickSetupController.getSelectedSetup()
        if not setupFileName:
            return

        proceed = guitools.askYesNoQuestion(self.__mainView, 'Warning',
                                            'The software will restart. Continue?')
        if not proceed:
            return

        options = dataclasses.replace(options, setupFileName=setupFileName)
        configfiletools.saveOptions(options)
        ostools.restartSoftware()

    def closeEvent(self):
        self.__logger.debug('Shutting down')
        self.__factory.closeAllCreatedControllers()
        self.__masterController.closeEvent()

        # seems like the imswitchserver is not closing from the closing event, need to hard kill it
        self._serverWorker.stop()
        self._thread.join()

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
