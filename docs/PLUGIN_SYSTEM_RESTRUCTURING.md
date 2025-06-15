# Plugin System Restructuring Guide

## Overview

This restructuring centralizes ImSwitch's plugin management system to make it more organized, maintainable, and easier to use for plugin developers.

## What Changed

### Before (Chaotic System)
- Plugin loading code scattered across multiple files
- Different loading patterns for managers, controllers, widgets
- No centralized registry or lifecycle management
- Inconsistent error handling

### After (Centralized System)
- Single `ImSwitchPluginManager` handles all plugin operations
- Standardized interfaces for all plugin types
- Centralized discovery, loading, and lifecycle management
- Consistent error handling and logging
- Full backwards compatibility

## For Plugin Developers

### Plugin Types Supported
1. **Managers** - Hardware/logic management (e.g., `my_plugin_manager`)
2. **Controllers** - UI business logic (e.g., `my_plugin_controller`)
3. **Widgets** - UI components (e.g., `my_plugin_widget`)
4. **Info Classes** - Configuration data classes (e.g., `my_plugin_info`)

### Example Plugin Structure

```python
# setup.py for your plugin
setup(
    name="my-imswitch-plugin",
    entry_points={
        'imswitch.implugins': [
            'my_plugin_manager = my_plugin.manager:MyPluginManager',
            'my_plugin_controller = my_plugin.controller:MyPluginController',
            'my_plugin_widget = my_plugin.widget:MyPluginWidget',
            'my_plugin_info = my_plugin.info:MyPluginInfo',
        ],
    },
)
```

### Plugin Implementation

```python
# my_plugin/manager.py
from imswitch.imcommon.model import PluginInterface

class MyPluginManager(PluginInterface):
    def __init__(self, module_info):
        self.module_info = module_info
    
    def get_plugin_info(self):
        return {
            "name": "My Plugin",
            "version": "1.0.0",
            "description": "Example plugin"
        }

# my_plugin/controller.py
class MyPluginController:
    def __init__(self, widget, setupInfo, masterController, commChannel):
        self.widget = widget
        self.setupInfo = setupInfo
        self.masterController = masterController
        self.commChannel = commChannel

# my_plugin/widget.py
class MyPluginWidget:
    def __init__(self):
        pass

# For React components (headless mode)
class MyPluginReactWidget:
    _ui_meta = {
        "name": "my_plugin",
        "icon": "plugin-icon",
        "path": "/path/to/react/build"
    }
```

## For ImSwitch Core Developers

### Using the Plugin Manager

```python
from imswitch.imcommon.model import get_plugin_manager

# Get the global plugin manager instance
pm = get_plugin_manager()

# Check if a plugin is available
if pm.is_plugin_available('my_plugin', 'manager'):
    # Load the plugin
    manager_class = pm.load_plugin('my_plugin', 'manager')
    manager = manager_class(module_info)

# Get all available plugins of a type
widget_plugins = pm.get_available_plugins('widget')
for plugin in widget_plugins:
    print(f"Found widget plugin: {plugin.name}")

# Get React widget plugins for UI
react_plugins = pm.get_react_widget_plugins()
for plugin in react_plugins:
    manifest = pm.get_plugin_manifest_data(plugin.name)
    if manifest:
        print(f"React plugin: {manifest['name']}")
```

## Benefits

1. **Centralized Management**: All plugin operations in one place
2. **Better Organization**: Clear separation of concerns
3. **Type Safety**: Proper plugin classification and validation
4. **Performance**: On-demand loading with caching
5. **Error Handling**: Graceful failure for missing/broken plugins
6. **Backwards Compatible**: Existing plugins continue to work
7. **React Support**: Automatic React component discovery and serving

## Migration Guide

**For existing plugins**: No changes required! The new system maintains full backwards compatibility.

**For new plugins**: You can optionally implement the `PluginInterface` for better integration, but it's not required.

## API Reference

### ImSwitchPluginManager

- `get_available_plugins(plugin_type=None)` - List available plugins
- `is_plugin_available(name, type)` - Check plugin availability  
- `load_plugin(name, type, info_class=None)` - Load a plugin class
- `get_plugin_manifest_data(name)` - Get React manifest data
- `get_react_widget_plugins()` - Get React-capable widget plugins

### PluginInterface (Optional Base Class)

- `get_plugin_info()` - Return plugin metadata (required method)

## Testing

The new system includes comprehensive testing to ensure:
- Plugin discovery works correctly
- Plugin loading handles errors gracefully
- Backwards compatibility is maintained
- React components are properly served

## Future Enhancements

- Plugin dependency management
- Plugin versioning and updates
- Hot reloading for development
- Plugin marketplace integration
- Enhanced React component integration