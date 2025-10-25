# Inter-Controller Communication Pattern

## Overview

This document describes how controllers can communicate with each other in ImSwitch, allowing direct method calls between controllers without using signals.

## Problem

Previously, controllers could only communicate through:
1. **Signals** - Asynchronous, loosely coupled, but adds complexity
2. **Managers** - Good for hardware abstraction, but not for controller-level logic

For certain use cases (e.g., calling autofocus from another controller), direct method calls are more appropriate.

## Solution: Controller Registry in MasterController

The `MasterController` now maintains a registry of all controllers, allowing any controller to access another controller's public methods.

### Architecture

```
ImConMainController
    └─> MasterController
            ├─> detectorsManager
            ├─> lasersManager
            ├─> positionersManager
            └─> _controllersRegistry  ← NEW!
                    ├─> 'Autofocus' → AutofocusController instance
                    ├─> 'Arkitekt' → ArkitektController instance
                    ├─> 'Experiment' → ExperimentController instance
                    └─> ...
```

## Usage

### 1. Accessing Another Controller

Any controller can access another controller via `self._master.getController(name)`:

```python
class ArkitektController(ImConWidgetController):
    def runTileScan(self, performAutofocus: bool = False):
        # Get the autofocus controller
        autofocusController = self._master.getController('Autofocus')
        
        if autofocusController is None:
            self._logger.warning("AutofocusController not available")
            return
        
        # Call autofocus methods directly - no signals needed!
        autofocusController.autoFocus(
            rangez=100,
            resolutionz=10,
            defocusz=0
        )
```

### 2. Available Controller Names

Controllers are registered using their widget key name:

| Widget Key | Controller Class | Access via |
|------------|-----------------|------------|
| `'Autofocus'` | `AutofocusController` | `getController('Autofocus')` |
| `'Arkitekt'` | `ArkitektController` | `getController('Arkitekt')` |
| `'Experiment'` | `ExperimentController` | `getController('Experiment')` |
| `'LiveView'` | `LiveViewController` | `getController('LiveView')` |
| `'Laser'` | `LaserController` | `getController('Laser')` |
| `'Positioner'` | `PositionerController` | `getController('Positioner')` |

### 3. Best Practices

#### ✅ DO:
- **Check for None**: Always check if the controller exists before using it
- **Use for controller logic**: Call methods that orchestrate controller-level operations
- **Document dependencies**: Add comments about which controllers you depend on
- **Use @APIExport methods**: Only call methods decorated with `@APIExport` for stable API

```python
# Good example
autofocusController = self._master.getController('Autofocus')
if autofocusController is not None:
    autofocusController.autoFocus(rangez=100, resolutionz=10)
else:
    self._logger.warning("Autofocus not available")
```

#### ❌ DON'T:
- **Access private methods**: Don't call `_private_method()` from other controllers
- **Create circular dependencies**: Controller A calls B calls A
- **Replace managers**: Still use managers for hardware access
- **Bypass signals when broadcasting**: Use signals for 1-to-many notifications

```python
# Bad example - accessing private state
autofocusController = self._master.getController('Autofocus')
autofocusController._AutofocusThead.start()  # DON'T!
```

### 4. When to Use What?

| Communication Need | Solution | Example |
|-------------------|----------|---------|
| Call another controller's function | **Controller Registry** | Arkitekt calls autofocus |
| Broadcast event to multiple listeners | **Signals** | "Scan started" event |
| Access hardware | **Managers** | Get camera frame |
| Expose REST API | **@APIExport** | External API calls |

## Implementation Details

### MasterController

```python
class MasterController:
    def __init__(self, setupInfo, commChannel, moduleCommChannel):
        # ...
        self._controllersRegistry = {}
    
    def registerController(self, name, controller):
        """Register a controller for inter-controller communication."""
        self._controllersRegistry[name] = controller
        self.__logger.debug(f"Registered controller: {name}")
    
    def getController(self, name):
        """Get a registered controller by name."""
        return self._controllersRegistry.get(name, None)
```

### ImConMainController

```python
class ImConMainController(MainController):
    def __init__(self, ...):
        # ...
        for widgetKey, widget in self.__mainView.widgets.items():
            controller = self.__factory.createController(controller_class, widget)
            self.controllers[widgetKey] = controller
            
            # Register in MasterController
            self.__masterController.registerController(widgetKey, controller)
```

## Examples

### Example 1: Autofocus in Tile Scan

```python
class ArkitektController(ImConWidgetController):
    def runTileScan(self, performAutofocus: bool = False):
        autofocusController = self._master.getController('Autofocus')
        
        for position in tiles:
            # Move to position
            self.moveToPosition(position)
            
            # Autofocus if requested
            if performAutofocus and autofocusController:
                autofocusController.autoFocus(
                    rangez=100,
                    resolutionz=10
                )
            
            # Capture image
            self.captureImage()
```

### Example 2: Coordinating Experiment with Live View

```python
class ExperimentController(ImConWidgetController):
    def startExperiment(self):
        # Pause live view during experiment
        liveViewController = self._master.getController('LiveView')
        if liveViewController:
            liveViewController.stopLiveView()
        
        # Run experiment
        self.runExperiment()
        
        # Resume live view
        if liveViewController:
            liveViewController.startLiveView()
```

### Example 3: Checking if a Controller is Available

```python
class MyController(ImConWidgetController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Check optional dependencies at init
        self.hasAutofocus = False
        
    def someMethod(self):
        # Check at runtime
        autofocusController = self._master.getController('Autofocus')
        if autofocusController:
            self.hasAutofocus = True
            # Use it
```

## Migration Guide

### Before (using signals):

```python
# In AutofocusController
self._commChannel.sigAutoFocusComplete.connect(self.onAutoFocusComplete)

# In ArkitektController  
self._commChannel.sigAutoFocus.emit(100, 10, 0)
# ... wait for signal ...
```

### After (direct call):

```python
# In ArkitektController
autofocusController = self._master.getController('Autofocus')
if autofocusController:
    result = autofocusController.autoFocus(rangez=100, resolutionz=10)
    # Result is available immediately!
```

## Thread Safety

⚠️ **Important**: When calling methods on another controller:

1. **Check threading**: Some methods may need `runOnUIThread=True`
2. **Avoid deadlocks**: Don't wait for callbacks from the controller you're calling
3. **Use signals for async**: If the operation is long-running, consider signals

```python
# If the method needs UI thread:
@APIExport(runOnUIThread=True)
def autoFocus(self, ...):
    # This will run on UI thread
    pass

# You can still call it from another thread - it will be marshalled automatically
```

## Debugging

To see which controllers are registered:

```python
# In any controller
registered = self._master._controllersRegistry.keys()
self._logger.debug(f"Available controllers: {list(registered)}")
```

## Future Improvements

Potential enhancements:

1. **Type hints**: Add proper typing for controller registry
2. **Lazy loading**: Only instantiate controllers when needed
3. **Dependency injection**: Declare dependencies in controller constructor
4. **Auto-discovery**: Automatically discover and document controller APIs

## See Also

- `imswitch/imcontrol/controller/MasterController.py` - Controller registry implementation
- `imswitch/imcontrol/controller/ImConMainController.py` - Controller registration
- `imswitch/imcontrol/controller/controllers/ArkitektController.py` - Example usage
- `imswitch/imcontrol/controller/controllers/AutofocusController.py` - Example target

---

**Last updated**: 2025-01-25  
**Author**: ImSwitch Development Team
