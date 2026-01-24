# ObjectiveController Migration to Redux Pattern

## Overview
Successfully migrated `ObjectiveController.js` from the deprecated `WebSocketContext` pattern to the centralized `WebSocketHandler` + Redux architecture.

## Changes Made

### 1. Import Changes
**Removed:**
```javascript
import { useWebSocket } from "../context/WebSocketContext";
```

**Added:**
```javascript
import * as positionSlice from "../state/slices/PositionSlice.js";
```

### 2. Removed WebSocket Connection
**Deleted:**
```javascript
const socket = useWebSocket();
```

**Result:** Component no longer creates its own WebSocket connection. All socket events are handled by the centralized `WebSocketHandler.js`.

### 3. Replaced Local State with Redux
**Removed:**
```javascript
const [positions, setPositions] = useState({});
```

**Added:**
```javascript
const positionState = useSelector(positionSlice.getPositionState);
const positions = {
  X: positionState.x,
  Y: positionState.y,
  Z: positionState.z,
  A: positionState.a,
};
```

**Benefits:**
- Positions are now automatically updated via WebSocketHandler dispatching to positionSlice
- No manual socket listeners needed in component
- Single source of truth for position data across all components

### 4. Removed Socket Event Listeners

#### Listener 1: sigObjectiveChanged (Lines 57-96)
**Removed:**
```javascript
useEffect(() => {
  if (!socket) return;
  const handleSignal = (data) => {
    try {
      const jdata = JSON.parse(data);
      if (jdata.name === "sigObjectiveChanged") {
        // Objective updates handled in WebSocketHandler
      } else if (jdata.name === "sigUpdateImage") {
        // Image updates handled in WebSocketHandler
      }
    } catch (error) {
      console.error("Error parsing signal data:", error);
    }
  };
  socket.on("signal", handleSignal);
  return () => socket.off("signal", handleSignal);
}, [socket]);
```

**Replaced with comment:**
```javascript
// Remove all WebSocket handlers - now handled by WebSocketHandler.js
// - sigObjectiveChanged: Already handled in WebSocketHandler -> objectiveSlice
// - sigUpdateImage: Already handled in WebSocketHandler -> liveStreamSlice
// - sigUpdateMotorPosition: Already handled in WebSocketHandler -> positionSlice
```

#### Listener 2: Initial Position Fetch (Lines 285-298)
**Removed:**
```javascript
useEffect(() => {
  const fetchPositions = async () => {
    const response = await apiPositionerGetAllPositions();
    if (response && response.positions) {
      setPositions(response.positions);
    }
  };
  fetchPositions();
}, []);
```

**Reason:** Positions are now automatically populated in Redux via WebSocketHandler's `sigUpdateMotorPosition` handler.

#### Listener 3: Real-time Position Updates (Lines 300-313)
**Removed:**
```javascript
useEffect(() => {
  if (!socket) return;
  const handleSignal = (data) => {
    try {
      const jdata = JSON.parse(data);
      if (jdata.name === "sigUpdateMotorPosition") {
        const [axis, pos] = jdata.args;
        setPositions((prev) => ({ ...prev, [axis]: pos }));
      }
    } catch (error) {
      console.error("Error parsing signal data:", error);
    }
  };
  socket.on("signal", handleSignal);
  return () => socket.off("signal", handleSignal);
}, [socket]);
```

**Reason:** WebSocketHandler already dispatches `sigUpdateMotorPosition` to `positionSlice.setPosition()`.

## Architecture Benefits

### Before Migration
```
ObjectiveController
  ↓ (creates own socket via useWebSocket)
WebSocket Connection #1
  ↓ (listens to sigUpdateMotorPosition)
Local State (positions)
```

### After Migration
```
WebSocketHandler (singleton)
  ↓ (receives sigUpdateMotorPosition)
positionSlice (Redux)
  ↓ (via useSelector)
ObjectiveController (reads positions)
```

### Key Improvements
1. **Single WebSocket Connection**: No duplicate connections, prevents race conditions
2. **Centralized State**: Positions available to all components via Redux
3. **Automatic Updates**: Component re-renders when Redux state changes, no manual listener logic
4. **Cleaner Code**: Removed ~60 lines of boilerplate socket handling code
5. **Consistent Pattern**: Matches architecture used by other migrated components

## Testing Checklist
- [x] Component still displays current positions (X, Y, Z, A)
- [x] Position updates from backend are reflected in UI
- [x] No duplicate WebSocket connections created
- [x] No console errors related to socket or Redux
- [x] Objective selection/calibration functionality unchanged

## Related Files Modified
- `src/components/ObjectiveController.js` - Main migration target
- `src/state/slices/PositionSlice.js` - Already had position state management
- `src/middleware/WebSocketHandler.js` - Already had sigUpdateMotorPosition handler

## Next Steps
Continue migrating remaining components using WebSocketContext:
- LiveView.js
- STORMControllerLocal.js
- StageOffsetCalibrationController.jsx
- SelectSetupController.js
- FlowStopController.js
- TimelapseController.js
- JoystickController.js
- STORMController.js

Once all components are migrated, remove `src/context/WebSocketContext.js` entirely.

## References
- See `DOCS/MIGRATION_WEBSOCKET_CONTEXT.md` for full migration guide
- See `DOCS/SOCKETVIEW_MIGRATION.md` for another migration example
- See `src/middleware/WebSocketHandler.js` for centralized signal routing logic
