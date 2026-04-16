from dataclasses import dataclass
from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import initLogger
try:
    from importlib.metadata import entry_points
except ImportError:
    entry_points = None
import importlib
import importlib.util

class ImConMainViewNoQt(object):
    def __init__(self, options, viewSetupInfo, *args, **kwargs):
        self.__logger = initLogger(self)
        self.__logger.debug('Initializing')

        super().__init__(*args, **kwargs)
        self.sigClosing = Signal()
        self.docks = {}
        self.widgets = {}
        self.shortcuts = {}

        self.viewSetupInfo = viewSetupInfo

        # Dock area
        # add widgets that are enabled but not in allDockKeys
        enabledDockKeys = self.viewSetupInfo.availableWidgets
        disabledKeys = ["Image"]
        widget_keys = {key: _DockInfo(name=key, yPosition=1) for key in enabledDockKeys if key not in disabledKeys}
        self._addWidgetNoQt(widget_keys)
        self.__logger.debug(f"Added widgets: {self.widgets.keys()}")

    def closeEvent(self, event):
        self.sigClosing.emit()
        event.accept()

    def _addWidgetNoQt(self, dockInfoDict):
        # Preload all available plugins for widgets
        try:
            eps = entry_points(group='imswitch.implugins')
        except Exception:
            eps = []
        availablePlugins = {
            entry_point.name: entry_point
            for entry_point in eps
        }

        for widgetKey, dockInfo in dockInfoDict.items():
            if widgetKey == "ImSwitchServer":
                continue
            # try if there is a react widget under ImSwitch.imswitch.imcontrol.view.widgets.*
            # if not, try to load it from the plugins

            # Case 1: there is a react widget under ImSwitch.imswitch.imcontrol.view.widgets.*
            module_name = f'imswitch.imcontrol.view.widgets.{widgetKey}ReactWidget'
            module_spec = importlib.util.find_spec(module_name)
            if module_spec is not None:
                # The module exists, so we can import it
                try:
                    mWidgetModule = importlib.import_module(module_name)
                    # load the class from the module
                    mWidgetClass = getattr(mWidgetModule, f'{widgetKey}ReactWidget')
                    self.widgets[widgetKey] = (widgetKey, mWidgetModule, mWidgetClass())
                    continue
                except ImportError as e:
                    self.__logger.error(f"Could not load widget {widgetKey} from imswitch.imcontrol.view.widgets", e)
                    continue

            # Case 2: Check if there is a plugin for the widget
            plugin_name = f'{widgetKey}_widget'
            if plugin_name in availablePlugins:
                try:
                    packageWidget = availablePlugins[plugin_name].load()
                    # load the class from the module
                    mWidgetClass = getattr(packageWidget, f'{widgetKey}ReactWidget')
                    self.widgets[widgetKey] = (widgetKey, packageWidget, mWidgetClass)
                    continue
                except Exception as e:
                    self.__logger.error(f"Could not load plugin widget {widgetKey}: {e}")
                    self.widgets[widgetKey] = (widgetKey, None, None)
                    continue
            # Case 3: There is no react widget, so we create a default one
            try:
                self.widgets[widgetKey] = (widgetKey, None, None)
            except Exception as e:
                self.__logger.error(f"Could not load widget {widgetKey} from imswitch.imcontrol.view.widgets", e)
                continue

@dataclass
class _DockInfo:
    name: str
    yPosition: int


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
