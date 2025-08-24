"""
Example Plugin for ImSwitch - Demonstrates the new centralized plugin system

This is a minimal example showing how to create plugins that work with
the new ImSwitchPluginManager system.
"""

from imswitch.imcommon.model import PluginInterface
from typing import Dict, Any


class ExamplePluginInfo:
    """Info class for configuration data"""
    
    def __init__(self):
        self.enabled = True
        self.settings = {
            "parameter1": "value1",
            "parameter2": 42
        }


class ExamplePluginManager(PluginInterface):
    """Example manager plugin"""
    
    def __init__(self, module_info):
        self.module_info = module_info
        self.initialized = True
    
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "name": "Example Plugin",
            "version": "1.0.0",
            "description": "Demonstrates the new plugin system",
            "type": "manager"
        }
    
    def do_something(self):
        """Example method"""
        return "Manager is working!"


class ExamplePluginController:
    """Example controller plugin"""
    
    def __init__(self, widget, setupInfo, masterController, commChannel):
        self.widget = widget
        self.setupInfo = setupInfo
        self.masterController = masterController
        self.commChannel = commChannel
    
    def initialize(self):
        """Initialize the controller"""
        print("Example plugin controller initialized")
    
    def on_action(self):
        """Example action handler"""
        return "Controller action executed"


class ExamplePluginWidget:
    """Example Qt widget plugin"""
    
    def __init__(self):
        self.initialized = True
        print("Example plugin widget created")
    
    def show_message(self):
        """Example widget method"""
        return "Widget is active"


class ExamplePluginReactWidget:
    """Example React widget for headless mode"""
    
    # UI metadata for React component serving
    _ui_meta = {
        "name": "example_plugin",
        "icon": "example-icon",
        "path": "/path/to/react/build/directory"
    }
    
    def __init__(self):
        self.initialized = True
        print("Example React widget created")
    
    def get_react_props(self):
        """Return props for React component"""
        return {
            "title": "Example Plugin",
            "data": [1, 2, 3, 4, 5]
        }


# Example setup.py entry points would be:
"""
setup(
    name="imswitch-example-plugin",
    version="1.0.0",
    entry_points={
        'imswitch.implugins': [
            'example_plugin_manager = example_plugin:ExamplePluginManager',
            'example_plugin_controller = example_plugin:ExamplePluginController', 
            'example_plugin_widget = example_plugin:ExamplePluginWidget',
            'example_plugin_info = example_plugin:ExamplePluginInfo',
        ],
    },
)
"""


# Demo usage with the new plugin manager
def demo_plugin_usage():
    """Demonstrate how to use plugins with the new system"""
    from imswitch.imcommon.model import get_plugin_manager
    
    pm = get_plugin_manager()
    
    # Check if plugin is available
    if pm.is_plugin_available('example_plugin', 'manager'):
        print("Example plugin manager is available")
        
        # Load the plugin
        manager_class = pm.load_plugin('example_plugin', 'manager')
        if manager_class:
            # Get info class if available
            info_class = pm.get_plugin_info_class('example_plugin')
            info = info_class() if info_class else None
            
            # Create manager instance
            manager = manager_class(info)
            
            # Use the plugin
            print(f"Plugin info: {manager.get_plugin_info()}")
            print(f"Manager result: {manager.do_something()}")
    
    # Check widget plugins for React components
    react_plugins = pm.get_react_widget_plugins()
    for plugin in react_plugins:
        if plugin.name == 'example_plugin':
            manifest = pm.get_plugin_manifest_data(plugin.name)
            if manifest:
                print(f"React component available at: {manifest['url']}")


if __name__ == "__main__":
    demo_plugin_usage()


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