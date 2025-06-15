"""
Centralized Plugin Management System for ImSwitch

This module provides a unified interface for plugin discovery, loading, and management
across the entire ImSwitch application.
"""

import pkg_resources
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from imswitch.imcommon.model import initLogger


@dataclass
class PluginInfo:
    """Information about a discovered plugin"""
    name: str
    entry_point_name: str
    plugin_type: str  # 'manager', 'controller', 'widget', 'info'
    entry_point: pkg_resources.EntryPoint
    loaded_class: Optional[Type] = None
    is_loaded: bool = False


class PluginInterface(ABC):
    """Base interface for all ImSwitch plugins"""
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        pass


class ImSwitchPluginManager:
    """
    Centralized manager for all ImSwitch plugins.
    
    Handles discovery, loading, and lifecycle management of plugins
    across managers, controllers, widgets, and info classes.
    """
    
    def __init__(self):
        self.__logger = initLogger(self)
        self._plugins: Dict[str, PluginInfo] = {}
        self._loaded_plugins: Dict[str, Any] = {}
        self._plugin_types = ['manager', 'controller', 'widget', 'info']
        self._discover_plugins()
    
    def _discover_plugins(self):
        """Discover all available plugins from entry points"""
        self.__logger.debug("Discovering plugins...")
        
        try:
            for entry_point in pkg_resources.iter_entry_points('imswitch.implugins'):
                plugin_info = self._parse_entry_point(entry_point)
                if plugin_info:
                    self._plugins[plugin_info.name] = plugin_info
                    self.__logger.debug(f"Discovered plugin: {plugin_info.name} ({plugin_info.plugin_type})")
        except Exception as e:
            self.__logger.error(f"Error discovering plugins: {e}")
    
    def _parse_entry_point(self, entry_point: pkg_resources.EntryPoint) -> Optional[PluginInfo]:
        """Parse entry point to extract plugin information"""
        name = entry_point.name
        
        # Determine plugin type based on naming convention
        plugin_type = None
        plugin_name = None
        
        if name.endswith('_manager'):
            plugin_type = 'manager'
            plugin_name = name.replace('_manager', '')
        elif name.endswith('_controller'):
            plugin_type = 'controller'
            plugin_name = name.replace('_controller', '')
        elif name.endswith('_widget'):
            plugin_type = 'widget'
            plugin_name = name.replace('_widget', '')
        elif name.endswith('_info'):
            plugin_type = 'info'
            plugin_name = name.replace('_info', '')
        else:
            # Skip unknown plugin types
            return None
        
        return PluginInfo(
            name=plugin_name,
            entry_point_name=name,
            plugin_type=plugin_type,
            entry_point=entry_point
        )
    
    def get_available_plugins(self, plugin_type: Optional[str] = None) -> List[PluginInfo]:
        """Get list of available plugins, optionally filtered by type"""
        plugins = list(self._plugins.values())
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        return plugins
    
    def get_plugin(self, plugin_name: str, plugin_type: str) -> Optional[PluginInfo]:
        """Get specific plugin info"""
        key = plugin_name
        plugin = self._plugins.get(key)
        if plugin and plugin.plugin_type == plugin_type:
            return plugin
        return None
    
    def load_plugin(self, plugin_name: str, plugin_type: str, info_class: Optional[Type] = None) -> Optional[Type]:
        """
        Load a specific plugin class
        
        Args:
            plugin_name: Name of the plugin (without type suffix)
            plugin_type: Type of plugin ('manager', 'controller', 'widget', 'info')
            info_class: Optional info class for managers that require it
            
        Returns:
            Loaded plugin class or None if not found/failed to load
        """
        plugin = self.get_plugin(plugin_name, plugin_type)
        if not plugin:
            self.__logger.debug(f"Plugin {plugin_name} of type {plugin_type} not found")
            return None
        
        if plugin.is_loaded:
            return plugin.loaded_class
        
        try:
            if plugin_type == 'manager' and info_class:
                # Some managers require an info class parameter
                loaded_class = plugin.entry_point.load(info_class)
            else:
                loaded_class = plugin.entry_point.load()
            
            plugin.loaded_class = loaded_class
            plugin.is_loaded = True
            self._loaded_plugins[f"{plugin_name}_{plugin_type}"] = loaded_class
            
            self.__logger.debug(f"Successfully loaded plugin: {plugin_name} ({plugin_type})")
            return loaded_class
            
        except Exception as e:
            self.__logger.error(f"Failed to load plugin {plugin_name} ({plugin_type}): {e}")
            return None
    
    def get_loaded_plugin(self, plugin_name: str, plugin_type: str) -> Optional[Type]:
        """Get already loaded plugin class"""
        key = f"{plugin_name}_{plugin_type}"
        return self._loaded_plugins.get(key)
    
    def is_plugin_available(self, plugin_name: str, plugin_type: str) -> bool:
        """Check if a plugin is available"""
        return self.get_plugin(plugin_name, plugin_type) is not None
    
    def get_plugin_info_class(self, plugin_name: str) -> Optional[Type]:
        """Get the info class for a plugin (if available)"""
        return self.load_plugin(plugin_name, 'info')
    
    def get_react_widget_plugins(self) -> List[PluginInfo]:
        """Get plugins that provide React widgets for UI"""
        return [p for p in self.get_available_plugins('widget') 
                if self._has_react_component(p)]
    
    def _has_react_component(self, plugin: PluginInfo) -> bool:
        """Check if a widget plugin has React components"""
        # For now, assume all widget plugins may have React components
        # This could be enhanced to check for specific markers
        return True
    
    def get_plugin_manifest_data(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get UI manifest data for a plugin (for React components)"""
        # This method can be extended to extract manifest data from plugins
        # For now, return basic structure compatible with existing system
        plugin = self.get_plugin(plugin_name, 'widget')
        if not plugin:
            return None
        
        # Try to get UI metadata from the loaded plugin
        try:
            loaded_class = self.load_plugin(plugin_name, 'widget')
            if loaded_class and hasattr(loaded_class, '_ui_meta'):
                return loaded_class._ui_meta
        except Exception as e:
            self.__logger.debug(f"Could not get manifest data for {plugin_name}: {e}")
        
        return None


# Global plugin manager instance
_plugin_manager_instance = None


def get_plugin_manager() -> ImSwitchPluginManager:
    """Get the global plugin manager instance"""
    global _plugin_manager_instance
    if _plugin_manager_instance is None:
        _plugin_manager_instance = ImSwitchPluginManager()
    return _plugin_manager_instance


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