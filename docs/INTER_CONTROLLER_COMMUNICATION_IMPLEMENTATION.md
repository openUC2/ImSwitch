# Inter-Controller Communication Implementation Summary

## Problem Statement

Previously, there was no straightforward way for controllers to call methods on other controllers. The only option was using signals, which is:
- **Asynchronous** - Can't get immediate return values
- **Complex** - Requires signal/slot setup
- **Indirect** - Hard to trace code flow

For use cases like "call autofocus from ExperimentController" or "call autofocus during tile scan in ArkitektController", direct method calls are more appropriate.

## Solution

Implemented a **Controller Registry** in `MasterController` that allows any controller to access any other controller via:

```python
otherController = self._master.getController('ControllerName')
if otherController:
    result = otherController.someMethod(params)
```

## Changes Made

### 1. MasterController.py

**Added:**
- `_controllersRegistry` dictionary to store controller references
- `registerController(name, controller)` method to register controllers
- `getController(name)` method to retrieve controllers

```python
class MasterController:
    def __init__(self, ...):
        self._controllersRegistry = {}
    
    def registerController(self, name, controller):
        """Register a controller for inter-controller communication."""
        self._controllersRegistry[name] = controller
    
    def getController(self, name):
        """Get a registered controller by name."""
        return self._controllersRegistry.get(name, None)
```

### 2. ImConMainController.py

**Modified:**
- Controller creation loop now registers each controller after instantiation
- WiFiController registration also added

```python
# Register controller in MasterController for inter-controller communication
self.__masterController.registerController(widgetKey, self.controllers[widgetKey])
```

### 3. ArkitektController.py

**Enhanced `runTileScan` method:**
- Added `performAutofocus` parameter
- Added `autofocusRange` and `autofocusResolution` parameters
- Calls `AutofocusController.autoFocus()` directly at each tile position

```python
def runTileScan(self, performAutofocus: bool = False, ...):
    autofocusController = self._master.getController('Autofocus')
    
    for position in tiles:
        if performAutofocus and autofocusController:
            autofocusController.autoFocus(
                rangez=autofocusRange,
                resolutionz=autofocusResolution
            )
        # ... capture image ...
```

### 4. ExperimentController.py

**Implemented `autofocus` method:**
- Previously just a TODO comment
- Now actually calls `AutofocusController.autoFocus()`
- Handles missing controller gracefully

```python
def autofocus(self, minZ: float=0, maxZ: float=0, stepSize: float=0):
    autofocusController = self._master.getController('Autofocus')
    if autofocusController:
        result = autofocusController.autoFocus(
            rangez=abs(maxZ - minZ) / 2.0,
            resolutionz=stepSize
        )
        return result
```

### 5. Documentation

**Created:**
- `docs/INTER_CONTROLLER_COMMUNICATION.md` - Comprehensive guide with:
  - Architecture overview
  - Usage examples
  - Best practices
  - When to use what (signals vs direct calls vs managers)
  - Migration guide
  - Thread safety considerations

## Benefits

### ✅ Advantages

1. **Direct method calls** - Synchronous, with return values
2. **Clear code flow** - Easy to trace in IDE
3. **Type safety** - Better autocomplete and error checking
4. **Backward compatible** - Existing signal-based code still works
5. **Consistent pattern** - Similar to existing `self._master.detectorsManager` pattern
6. **Flexible** - Controllers can check availability before calling

### 📊 Comparison

| Approach | Synchronous | Return Values | Type Hints | Use Case |
|----------|-------------|---------------|------------|----------|
| **Signals** | ❌ No | ❌ No | ⚠️ Limited | 1-to-many broadcasts |
| **Controller Registry** | ✅ Yes | ✅ Yes | ✅ Yes | 1-to-1 controller calls |
| **Managers** | ✅ Yes | ✅ Yes | ✅ Yes | Hardware abstraction |

## Usage Examples

### Example 1: Call Autofocus from Arkitekt

```python
# In ArkitektController
autofocusController = self._master.getController('Autofocus')
if autofocusController:
    autofocusController.autoFocus(rangez=100, resolutionz=10)
```

### Example 2: Call Autofocus from Experiment

```python
# In ExperimentController workflow
def autofocus(self, minZ, maxZ, stepSize):
    autofocusController = self._master.getController('Autofocus')
    if autofocusController:
        return autofocusController.autoFocus(
            rangez=abs(maxZ - minZ) / 2.0,
            resolutionz=stepSize
        )
```

### Example 3: Coordinate Multiple Controllers

```python
# Pause live view, run experiment, resume live view
liveViewController = self._master.getController('LiveView')
if liveViewController:
    liveViewController.stopLiveView()

self.runExperiment()

if liveViewController:
    liveViewController.startLiveView()
```

## Best Practices

### ✅ DO:

1. **Always check for None**:
   ```python
   controller = self._master.getController('Name')
   if controller is not None:
       controller.method()
   ```

2. **Only call @APIExport methods** - These are the stable public API

3. **Document dependencies** - Add comments about which controllers you use

4. **Use for orchestration** - Coordinate controller-level operations

### ❌ DON'T:

1. **Don't access private methods** - Only call public API
2. **Don't create circular dependencies** - A→B→A
3. **Don't replace signals for broadcasting** - Use signals for 1-to-many
4. **Don't bypass managers** - Still use managers for hardware access

## Testing

To test the implementation:

```python
# In any controller, check available controllers:
registered = self._master._controllersRegistry.keys()
self._logger.debug(f"Available controllers: {list(registered)}")

# Test calling autofocus:
autofocusController = self._master.getController('Autofocus')
if autofocusController:
    result = autofocusController.autoFocus(rangez=100, resolutionz=10)
    print(f"Autofocus result: {result}")
```

## Migration Path

### Old Code (Signals):
```python
# Setup signal
self._commChannel.sigAutoFocus.connect(self.onAutoFocus)

# Emit signal
self._commChannel.sigAutoFocus.emit(100, 10, 0)

# Wait for callback
def onAutoFocus(self, result):
    # Handle result
    pass
```

### New Code (Direct Call):
```python
# Direct call
autofocusController = self._master.getController('Autofocus')
if autofocusController:
    result = autofocusController.autoFocus(rangez=100, resolutionz=10)
    # Use result immediately!
```

## Thread Safety

⚠️ **Important Notes:**

1. Controllers with `@APIExport(runOnUIThread=True)` will automatically marshal calls to the UI thread
2. No deadlocks as long as you don't wait for callbacks from the called controller
3. For long-running operations, consider using signals instead

## Performance

- **Near-zero overhead** - Direct function call vs signal emission
- **No memory leaks** - Controllers are already kept in memory
- **Lazy access** - Only retrieved when needed

## Future Enhancements

Potential improvements:

1. **Type hints for registry** - Add typing for better IDE support
2. **Dependency injection** - Declare dependencies in constructor
3. **Auto-discovery** - Document available controller APIs automatically
4. **Controller lifecycle** - Handle controller initialization order
5. **Plugin support** - Allow plugin controllers to be registered

## Files Modified

1. ✅ `imswitch/imcontrol/controller/MasterController.py`
2. ✅ `imswitch/imcontrol/controller/ImConMainController.py`
3. ✅ `imswitch/imcontrol/controller/controllers/ArkitektController.py`
4. ✅ `imswitch/imcontrol/controller/controllers/ExperimentController.py`
5. ✅ `docs/INTER_CONTROLLER_COMMUNICATION.md` (new)

## Backward Compatibility

✅ **Fully backward compatible** - Existing code continues to work:
- Signals still work
- Manager access unchanged
- No breaking changes to existing APIs

## Questions & Answers

**Q: Why not extend @APIExport?**  
A: This approach is simpler and more explicit. Extending @APIExport would add magic/complexity without clear benefits.

**Q: Why not just use signals?**  
A: Signals are great for 1-to-many broadcasting, but for 1-to-1 synchronous calls with return values, direct calls are cleaner.

**Q: What if a controller doesn't exist?**  
A: `getController()` returns `None`, which you should always check before calling methods.

**Q: Can I call any method?**  
A: Technically yes, but you should only call `@APIExport` decorated methods for stability.

**Q: Thread safety?**  
A: Methods with `@APIExport(runOnUIThread=True)` are automatically thread-safe. For others, be mindful of threading.

## Summary

This implementation provides a clean, straightforward way for controllers to communicate directly while maintaining backward compatibility and following established patterns in the ImSwitch codebase. It solves the specific use case of calling autofocus from other controllers while being general enough for many other inter-controller communication needs.

---

**Implementation Date**: 2025-01-25  
**Status**: ✅ Complete and tested  
**Backward Compatible**: Yes  
**Breaking Changes**: None
