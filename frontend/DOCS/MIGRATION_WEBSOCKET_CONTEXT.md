# WebSocket Context Migration Guide

## Overview
This project is migrating from using individual `WebSocketContext` instances to a centralized `WebSocketHandler` with Redux state management.

## Why Migrate?
- **Single WebSocket Connection**: Prevents duplicate connections and reduces overhead
- **Centralized State Management**: All socket events handled in one place
- **Redux Integration**: Reactive UI updates through Redux store
- **Better Performance**: Less memory usage and cleaner event handling

## Migration Pattern

### Before (using WebSocketContext):
```javascript
import { useWebSocket } from "../context/WebSocketContext";

function Component() {
  const socket = useWebSocket();
  const [data, setData] = useState(null);
  
  useEffect(() => {
    if (!socket) return;
    const handler = (data) => {
      const parsed = JSON.parse(data);
      if (parsed.name === "someSignal") {
        setData(parsed.args);
      }
    };
    socket.on("signal", handler);
    return () => socket.off("signal", handler);
  }, [socket]);
  
  return <div>{data}</div>;
}
```

### After (using Redux):
```javascript
import { useSelector } from "react-redux";
import * as someSlice from "../state/slices/SomeSlice.js";

function Component() {
  // Read data from Redux (auto-updates when WebSocketHandler dispatches)
  const someState = useSelector(someSlice.getSomeState);
  
  return <div>{someState.data}</div>;
}
```

## Components to Migrate

### ✅ Completed
- [x] `XYZControls.js` - Now uses `positionSlice`
- [x] `AutofocusController.js` - Now uses `autofocusSlice`
- [x] `PositionViewComponent.js` - Already uses `positionSlice`
- [x] `SocketView.js` - Now uses `socketDebugSlice`

### 🔄 Needs Migration
- [ ] `LiveView.js` - Uses WebSocketContext
- [ ] `ObjectiveController.js` - Partially migrated (has TODO comment)
- [ ] `STORMControllerLocal.js` - Uses WebSocketContext
- [ ] `StageOffsetCalibrationController.jsx` - Uses WebSocketContext
- [ ] `SelectSetupController.js` - Uses WebSocketContext
- [ ] `FlowStopController.js` - Uses WebSocketContext
- [ ] `TimelapseController.js` - Uses WebSocketContext
- [ ] `JoystickController.js` - Uses WebSocketContext
- [ ] `STORMController.js` - Uses WebSocketContext

## WebSocketHandler Signal Handlers

Currently implemented in `src/middleware/WebSocketHandler.js`:

| Signal Name | Redux Slice | Action |
|------------|-------------|--------|
| `sigUpdateImage` | `liveStreamSlice` | `setLiveViewImage`, `setImageFormat`, etc. |
| `sigHistogramComputed` | `liveStreamSlice` | `setHistogramData` |
| `sigExperimentWorkflowUpdate` | `experimentStateSlice` | `setStatus`, `setStepID`, etc. |
| `sigExperimentImageUpdate` | `tileStreamSlice` | `setTileViewImage` |
| `sigObjectiveChanged` | `objectiveSlice` | `setPixelSize`, `setNA`, etc. |
| `sigUpdateMotorPosition` | `positionSlice` | `setPosition` |
| `sigUpdateOMEZarrStore` | `omeZarrSlice` | `setZarrUrl`, `tileArrived` |
| `sigFocusValueUpdate` | `focusLockSlice` | `addFocusValue` |
| `sigFocusLockStateChanged` | `focusLockSlice` | `setFocusLocked`, etc. |
| `sigUpdateFocusPlot` | `autofocusSlice` | `setPlotData` |
| `sigGameState` | `mazeGameSlice` | `setGameState` |
| `sigCounterUpdated` | `mazeGameSlice` | `setCounter` |
| `sigPreviewUpdated` | `mazeGameSlice` | `setPreviewImage` |
| STORM signals | `stormSlice` | Various STORM actions |
| **All signals** | `socketDebugSlice` | `addMessage` (for debugging) |

## How to Add New Signal Handlers

1. **Create/Update Redux Slice** (if needed)
   ```javascript
   // src/state/slices/MyFeatureSlice.js
   const myFeatureSlice = createSlice({
     name: "myFeatureState",
     initialState: { data: null },
     reducers: {
       setData: (state, action) => {
         state.data = action.payload;
       }
     }
   });
   ```

2. **Register Slice in Store**
   ```javascript
   // src/state/store.js
   import myFeatureReducer from "./slices/MyFeatureSlice";
   
   const rootReducer = combineReducers({
     // ...
     myFeatureState: myFeatureReducer,
   });
   ```

3. **Add Signal Handler in WebSocketHandler**
   ```javascript
   // src/middleware/WebSocketHandler.js
   } else if (dataJson.name === "sigMyFeature") {
     try {
       dispatch(myFeatureSlice.setData(dataJson.args.p0));
     } catch (error) {
       console.error("Error parsing my feature signal:", error);
     }
   ```

4. **Use in Component**
   ```javascript
   import { useSelector } from "react-redux";
   import * as myFeatureSlice from "../state/slices/MyFeatureSlice.js";
   
   function MyComponent() {
     const myFeatureState = useSelector(myFeatureSlice.getMyFeatureState);
     return <div>{myFeatureState.data}</div>;
   }
   ```

## Testing Checklist

When migrating a component:

- [ ] Remove `useWebSocket` import
- [ ] Remove socket event listeners
- [ ] Add Redux `useSelector` for reading state
- [ ] Verify signal is handled in `WebSocketHandler.js`
- [ ] Test that UI updates when signal is received
- [ ] Check browser console for any errors
- [ ] Verify no duplicate socket connections

## Debugging

Enable detailed logging in browser console:
- `sigUpdateMotorPosition` shows position updates
- `PositionSlice.setPosition` shows Redux state changes
- All signal handlers log errors to console

## Next Steps

1. Complete migration of remaining components
2. Remove `WebSocketContext.js` entirely
3. Remove all commented-out socket code
4. Update documentation
5. Add unit tests for Redux slices

## Related Files

- `src/middleware/WebSocketHandler.js` - Central socket handler
- `src/context/WebSocketContext.js` - **TO BE REMOVED**
- `src/state/store.js` - Redux store configuration
- `src/state/slices/` - All Redux slices
