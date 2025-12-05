# Objective Indexing Convention

This document describes the indexing convention used throughout the ImSwitch objective system to avoid confusion between 0-based and 1-based numbering.

## Summary

- **Internal state (ObjectiveManager._currentObjective)**: 0-based (0 or 1)
- **API/External interfaces**: 1-based (1 or 2) for user-facing consistency
- **Hardware (ESP32)**: 1-based (slot 1, 2)

## Component Details

### ObjectiveManager (Model)
The ObjectiveManager stores all objective state and configuration.

**Internal Storage (0-based):**
- `_currentObjective`: 0 or 1 (or None if not set)
- `_pixelsizes[idx]`: Array indexed 0-1
- `_objectiveNames[idx]`: Array indexed 0-1
- `_NAs[idx]`: Array indexed 0-1
- `_magnifications[idx]`: Array indexed 0-1

**Methods:**

| Method | Input | Internal | Output | Notes |
|--------|-------|----------|--------|-------|
| `setCurrentObjective(slot)` | 0-based (0 or 1) | 0-based | - | Stores 0-based index |
| `getCurrentObjective()` | - | 0-based | 0-based (0 or 1) | Returns raw 0-based index |
| `getCurrentObjectiveID()` | - | 0-based | 1-based (1 or 2) | Converts to 1-based for API |
| `getCurrentObjectiveName()` | - | 0-based | string | Uses 0-based index to lookup name |
| `setObjectiveParameters(slot, ...)` | 1-based (1 or 2) | 0-based | - | Converts to 0-based: idx = slot - 1 |
| `getObjectiveParameters(slot)` | 1-based (1 or 2) | 0-based | dict | Converts to 0-based: idx = slot - 1 |

### ObjectiveController (Controller)
The ObjectiveController coordinates between hardware, manager, and UI.

**Methods:**

| Method | Input | Manager Call | Hardware Call | Notes |
|--------|-------|--------------|---------------|-------|
| `moveToObjective(slot)` | 1-based (1 or 2) | 0-based (slot-1) | 1-based (slot) | API expects 1-based |
| `calibrateObjective()` | - | 0-based | 1-based | Converts hardware response to 0-based |
| `getCurrentObjective()` | - | Returns 0-based | - | Returns tuple (0-based, name) |
| `setObjectiveParameters(slot, ...)` | 1-based (1 or 2) | 1-based | - | Passes through to manager |
| `getstatus()` | - | - | - | Returns mixed: currentObjective=0-based, state=1-based |

**Signal Handlers:**

| Handler | Signal Input | Action |
|---------|--------------|--------|
| `_onSetObjectiveByID(objective_id)` | str or int (1 or 2) | Calls `moveToObjective(1-based)` |
| `_onSetObjectiveByName(objective_name)` | string | Looks up name → converts to 1-based → calls `moveToObjective()` |

**Initialization:**
- Hardware status returns 1-based slot → converts to 0-based before storing in manager

### PixelCalibrationController
Uses objective names (strings) as primary identifiers, but needs to work with slots for some operations.

**Methods:**

| Method | Input | Manager/Controller Call | Notes |
|--------|-------|-------------------------|-------|
| `calibrateStageAffine(objectiveId)` | 1-based (1 or 2) | Converts to name → uses name | API expects 1-based slot |
| `setCurrentObjective(objective_id)` | string (name) | Finds name → converts to 1-based → `objectiveController.moveToObjective(1-based)` | Uses direct controller access |
| `_distributePixelSizesToObjectives()` | - | 1-based for `setObjectiveParameters()` | Converts array index to 1-based |
| `_applyCalibrationResults()` | - | Gets 0-based from manager → converts to 1-based for `setObjectiveParameters()` | Explicit conversion |

## Conversion Examples

### Example 1: User wants to move to objective 1
```python
# User/API call (1-based)
objectiveController.moveToObjective(slot=1)

# Inside moveToObjective:
internal_slot = slot - 1  # 1 → 0
self._objective.move(slot=slot)  # Hardware gets 1
self._manager.setCurrentObjective(internal_slot)  # Manager stores 0
```

### Example 2: Reading current objective
```python
# Get 0-based index from manager
current_0based = objectiveManager.getCurrentObjective()  # Returns 0

# Get 1-based ID for API
current_1based = objectiveManager.getCurrentObjectiveID()  # Returns 1

# Get name
name = objectiveManager.getCurrentObjectiveName()  # Returns "10x" (using 0-based index)
```

### Example 3: Setting parameters via API
```python
# API call with 1-based slot
objectiveController.setObjectiveParameters(objectiveSlot=1, pixelsize=0.5)

# Passes to manager (expects 1-based)
objectiveManager.setObjectiveParameters(slot=1, pixelsize=0.5)

# Inside setObjectiveParameters:
idx = slot - 1  # 1 → 0
self._pixelsizes[idx] = pixelsize  # Array access uses 0-based
```

### Example 4: Hardware status initialization
```python
# Get hardware status (returns 1-based)
status = self._objective.getstatus()
hardware_slot = status.get("state", 1)  # Returns 1 or 2

# Convert to 0-based for manager
internal_index = hardware_slot - 1  # 1 → 0, 2 → 1
self._manager.setCurrentObjective(internal_index)  # Store 0-based
```

## Design Rationale

### Why 0-based internally?
- **Array indexing**: Python uses 0-based indexing for arrays/lists
- **Simplicity**: Direct mapping between slot index and array index
- **Consistency**: Matches Python conventions

### Why 1-based externally?
- **Hardware interface**: ESP32 firmware uses 1-based slot numbers (1, 2)
- **User expectations**: Physical labels on hardware use 1, 2
- **API clarity**: More intuitive for end users ("objective 1" vs "objective 0")

### Why mixed in getstatus()?
- **Historical reasons**: Existing code expects certain fields in certain formats
- **Transparency**: Shows both internal state and hardware state
- **Compatibility**: Maintains backward compatibility

## Testing Checklist

When modifying objective-related code, verify:

- [ ] `setCurrentObjective()` accepts 0 or 1
- [ ] `getCurrentObjective()` returns 0 or 1
- [ ] `getCurrentObjectiveID()` returns 1 or 2
- [ ] `moveToObjective()` accepts 1 or 2
- [ ] `setObjectiveParameters()` accepts 1 or 2 and converts to 0-based internally
- [ ] Hardware move commands receive 1 or 2
- [ ] Array accesses use 0-based indices
- [ ] Signal handlers convert appropriately
- [ ] Pixel calibration uses correct slot numbers

## Common Pitfalls

1. **Directly using `getCurrentObjective()` for API responses**
   - ❌ Wrong: `return {"slot": manager.getCurrentObjective()}`  # Returns 0
   - ✅ Correct: `return {"slot": manager.getCurrentObjectiveID()}`  # Returns 1

2. **Forgetting to convert when calling setObjectiveParameters**
   - ❌ Wrong: `manager.setObjectiveParameters(slot=current_0based, ...)`
   - ✅ Correct: `manager.setObjectiveParameters(slot=current_0based + 1, ...)`

3. **Assuming hardware returns 0-based**
   - ❌ Wrong: `manager.setCurrentObjective(hardware_status["state"])`  # state is 1-based!
   - ✅ Correct: `manager.setCurrentObjective(hardware_status["state"] - 1)`

4. **Mixing name-based and slot-based access**
   - ❌ Wrong: Assuming objective name matches slot number
   - ✅ Correct: Look up name in `objectiveNames` array, use index for slot

## Future Improvements

Consider these potential improvements:
1. Use an enum or named constants (OBJECTIVE_1, OBJECTIVE_2) to make intent clearer
2. Create explicit conversion functions (to_internal_index, to_api_slot)
3. Type hints to distinguish between internal and external indices
4. Unified slot representation across all interfaces

## See Also

- `ObjectiveManager.py` - State management and storage
- `ObjectiveController.py` - Controller coordinating hardware and UI
- `PixelCalibrationController.py` - Calibration using objective information
- `docs/controller_communication_pattern.md` - Signal-based communication patterns
