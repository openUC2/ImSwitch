# Controller Communication Pattern via CommunicationChannel

## Overview
Controllers in ImSwitch should not directly access other controllers. Instead, they communicate via the `CommunicationChannel` using signals. This maintains loose coupling and follows the event-driven architecture.

## Problem
Direct controller-to-controller communication creates tight coupling:
```python
# ❌ BAD: Direct controller access
obj_ctrl = self._master.objectiveController
obj_ctrl.moveToObjective(slot)
```

This violates the separation of concerns and makes the code harder to test and maintain.

## Solution: Signal-Based Communication

### Step 1: Define Signal in CommunicationChannel

Add a new signal in `CommunicationChannel.py`:

```python
class CommunicationChannel(SignalInterface):
    # Objective 
    sigToggleObjective = Signal(int)  # objective slot number 1,2
    sigSetObjectiveByName = Signal(str)  # objective name (e.g., "10x", "20x")
```

### Step 2: Connect to Signal in Receiving Controller

In the controller that will handle the request (e.g., `ObjectiveController`):

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # Connect to communication channel signals
    self._commChannel.sigSetObjectiveByName.connect(self._onSetObjectiveByName)

def _onSetObjectiveByName(self, objective_name: str):
    """
    Handle request to set objective by name from communication channel.
    
    This allows other controllers to request objective changes without
    direct coupling.
    
    Args:
        objective_name: Name of objective to switch to (e.g., "10x", "20x")
    """
    try:
        # Find slot number for the objective name
        objective_names = self._manager.objectiveNames
        if objective_name in objective_names:
            slot = objective_names.index(objective_name) + 1
            
            current_slot = self._manager.getCurrentObjective()
            if current_slot != slot:
                self._logger.info(f"Received request to move to objective '{objective_name}' (slot {slot})")
                self.moveToObjective(slot)
            else:
                self._logger.debug(f"Already on objective '{objective_name}', no movement needed")
        else:
            self._logger.warning(f"Objective '{objective_name}' not found")
    except Exception as e:
        self._logger.error(f"Failed to set objective: {e}", exc_info=True)
```

### Step 3: Emit Signal from Requesting Controller

In the controller that wants to trigger an action (e.g., `PixelCalibrationController`):

```python
def setCurrentObjective(self, objective_id: str):
    """Set the current active objective for calibration."""
    
    # Update local state
    self.currentObjective = objective_id
    
    # Request objective change via communication channel
    try:
        self._commChannel.sigSetObjectiveByName.emit(objective_id)
        self._logger.debug(f"Emitted signal to change objective to '{objective_id}'")
    except Exception as e:
        self._logger.warning(f"Could not emit objective change signal: {e}")
```

## Benefits

### 1. **Loose Coupling**
Controllers don't need to know about each other's existence:
```python
# ✅ GOOD: Signal-based communication
self._commChannel.sigSetObjectiveByName.emit("10x")
```

### 2. **Testability**
Easy to test in isolation:
```python
def test_objective_change():
    # Mock the signal
    mock_signal = Mock()
    controller._commChannel.sigSetObjectiveByName = mock_signal
    
    controller.setCurrentObjective("10x")
    
    mock_signal.emit.assert_called_once_with("10x")
```

### 3. **Multiple Listeners**
Multiple controllers can listen to the same signal:
```python
# Both controllers can react to objective changes
objectiveController._commChannel.sigSetObjectiveByName.connect(...)
loggerController._commChannel.sigSetObjectiveByName.connect(...)
```

### 4. **Error Isolation**
If one controller fails, others continue working:
```python
try:
    self._commChannel.sigSetObjectiveByName.emit(objective_id)
except Exception as e:
    # Error in one controller doesn't break others
    self._logger.warning(f"Signal emission failed: {e}")
```

### 5. **Event-Driven Architecture**
Follows Qt's signal-slot pattern and event-driven design.

## Implementation Example

### Use Case: PixelCalibrationController → ObjectiveController

```
┌──────────────────────────┐
│ PixelCalibrationController│
│                          │
│  setCurrentObjective()   │
│         ↓                │
│    emit signal           │
└──────────┬───────────────┘
           │
           │ sigSetObjectiveByName("10x")
           │
           ↓
┌──────────────────────────┐
│  CommunicationChannel    │
│                          │
│  sigSetObjectiveByName   │
└──────────┬───────────────┘
           │
           │ slot connected
           │
           ↓
┌──────────────────────────┐
│   ObjectiveController    │
│                          │
│ _onSetObjectiveByName()  │
│         ↓                │
│  moveToObjective(1)      │
└──────────────────────────┘
```

## Common Signals in CommunicationChannel

### Acquisition Signals
- `sigAcquisitionStarted`
- `sigAcquisitionStopped`
- `sigRecordingStarted`
- `sigRecordingEnded`

### Detector Signals
- `sigUpdateImage(str, np.ndarray, bool, list, bool)`
- `sigDetectorSwitched(str, str)`

### Scan Signals
- `sigRunScan(bool, bool)`
- `sigScanStarted`
- `sigScanDone`
- `sigScanEnded`

### Objective Signals
- `sigToggleObjective(int)` - Switch by slot number
- `sigSetObjectiveByName(str)` - Switch by name

### Position Signals
- `sigSetXYPosition(float, float)`
- `sigSetZPosition(float)`

## Best Practices

### 1. **Signal Naming Convention**
```python
# Use descriptive names with sig prefix
sigSetObjectiveByName = Signal(str)
sigUpdatePosition = Signal(float, float)
sigStartAcquisition = Signal()
```

### 2. **Type Hints**
```python
# Document signal parameters
sigSetObjectiveByName = Signal(str)  # objective name (e.g., "10x", "20x")
sigSetPosition = Signal(float, float, float)  # (x, y, z) in micrometers
```

### 3. **Error Handling**
```python
def _onSignalReceived(self, data):
    try:
        # Process signal
        self._processData(data)
    except Exception as e:
        self._logger.error(f"Failed to process signal: {e}", exc_info=True)
        # Don't let exceptions propagate to other signal handlers
```

### 4. **Avoid Circular Dependencies**
```python
# ❌ BAD: A → B → A
controllerA.sigData.connect(controllerB.onData)
controllerB.sigResult.connect(controllerA.onResult)

# ✅ GOOD: Use state changes
controllerA.sigData.connect(controllerB.onData)
# B updates manager state
# A observes manager state changes
```

### 5. **Document Signal Purpose**
```python
sigSetObjectiveByName = Signal(str)
"""
Request to change objective by name.

Emitted by: PixelCalibrationController, CalibrationAPI
Handled by: ObjectiveController

Args:
    str: Objective name (e.g., "10x", "20x")
"""
```

## Migration from Direct Access

### Before (Direct Access)
```python
def setCurrentObjective(self, objective_id: str):
    # Direct controller access
    if hasattr(self._master, 'objectiveController'):
        obj_ctrl = self._master.objectiveController
        obj_ctrl.moveToObjective(slot)
```

### After (Signal-Based)
```python
def setCurrentObjective(self, objective_id: str):
    # Signal-based communication
    self._commChannel.sigSetObjectiveByName.emit(objective_id)
```

## Testing

### Unit Test Example
```python
def test_objective_change_request():
    # Arrange
    controller = PixelCalibrationController(...)
    signal_spy = Mock()
    controller._commChannel.sigSetObjectiveByName = signal_spy
    
    # Act
    controller.setCurrentObjective("20x")
    
    # Assert
    signal_spy.emit.assert_called_once_with("20x")
```

### Integration Test Example
```python
def test_objective_change_integration():
    # Arrange
    comm_channel = CommunicationChannel(...)
    obj_controller = ObjectiveController(...)
    pixel_controller = PixelCalibrationController(...)
    
    # Connect signal
    comm_channel.sigSetObjectiveByName.connect(
        obj_controller._onSetObjectiveByName
    )
    
    # Act
    pixel_controller.setCurrentObjective("20x")
    
    # Assert
    assert obj_controller._manager.getCurrentObjective() == 2  # Slot 2
```

## Summary

✅ **DO:**
- Use signals for controller-to-controller communication
- Define signals in CommunicationChannel
- Handle errors gracefully in signal handlers
- Document signal purpose and parameters

❌ **DON'T:**
- Access other controllers directly via `self._master.otherController`
- Create circular signal dependencies
- Let exceptions propagate from signal handlers
- Assume controllers exist (check with hasattr)

This pattern maintains clean architecture, enables testing, and follows Qt's event-driven design principles.
