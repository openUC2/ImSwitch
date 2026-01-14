# Autofocus Live Monitoring Implementation

## Overview
Added continuous live focus value monitoring to the autofocus system, allowing real-time tracking of image sharpness without performing a full autofocus scan.

## Backend Changes

### 1. Communication Channel Signal (`CommunicationChannel.py`)
Added new signal for broadcasting live focus values:
```python
sigAutoFocusLiveValue = Signal(object) # live focus value during monitoring mode {"focus_value": float, "timestamp": float}
```

### 2. AutofocusController (`AutofocusController.py`)

#### New Instance Variables
- `isLiveMonitoring`: Boolean flag for monitoring state
- `_liveMonitoringThread`: Thread for background monitoring
- `_liveMonitoringPeriod`: Update period in seconds (default 0.5s)
- `_focusMethod`: Focus measurement method ("LAPE" or "GLVA")

#### New API Endpoints

**`startLiveMonitoring(period: float = 0.5, method: str = "LAPE")`**
- Starts continuous focus value monitoring
- Parameters:
  - `period`: Update period in seconds (minimum 0.1s)
  - `method`: Focus measurement method ("LAPE" or "GLVA")
- Returns: `{"status": "started", "period": float, "method": string}`

**`stopLiveMonitoring()`**
- Stops continuous focus value monitoring
- Returns: `{"status": "stopped"}`

**`setLiveMonitoringParameters(period: float = None, method: str = None)`**
- Updates monitoring parameters on-the-fly
- Parameters are optional - only specified ones are updated
- Returns: `{"status": "updated", "period": float, "method": string, "is_running": bool}`

**`getLiveMonitoringStatus()`**
- Gets current monitoring status
- Returns: `{"is_running": bool, "period": float, "method": string}`

#### Background Thread Implementation
The `_doLiveMonitoringBackground()` method:
1. Runs continuously while `isLiveMonitoring` is True
2. Grabs fresh camera frames with `grabCameraFrame(frameSync=1)`
3. Extracts and processes image (crops to 2048x2048, converts to grayscale)
4. Calculates focus value using static method from FrameProcessor
5. Emits `sigAutoFocusLiveValue` signal with:
   - `focus_value`: Computed sharpness metric
   - `timestamp`: Current time
   - `method`: Focus measurement method used
6. Sleeps for remaining period time

#### Bug Fixes
- Fixed `grabCameraFrame()` to support optional `returnFrameNumber` parameter
- Removed undefined `flatfieldImage` variable from fast autofocus method

## Frontend Changes

### 1. Redux Slice (`AutofocusSlice.js`)

#### New State Properties
```javascript
isLiveMonitoring: false,          // Monitoring active flag
liveFocusValue: null,              // {focus_value, timestamp, method}
liveMonitoringPeriod: 0.5,         // Update period in seconds
liveMonitoringMethod: "LAPE"       // Focus measurement method
```

#### New Actions
- `setIsLiveMonitoring(bool)`: Set monitoring active state
- `setLiveFocusValue(object)`: Update current live focus value
- `setLiveMonitoringPeriod(number)`: Set update period
- `setLiveMonitoringMethod(string)`: Set focus method

### 2. WebSocket Handler (`WebSocketHandler.js`)

Added signal handler for `sigAutoFocusLiveValue`:
```javascript
else if (dataJson.name === "sigAutoFocusLiveValue") {
  try {
    if (dataJson.args && dataJson.args.p0) {
      dispatch(autofocusSlice.setLiveFocusValue(dataJson.args.p0));
    }
  } catch (error) {
    console.error("Error parsing autofocus live value signal:", error);
  }
}
```

### 3. API Functions

Created four new API helper functions:
- `apiAutofocusControllerStartLiveMonitoring(period, method)`
- `apiAutofocusControllerStopLiveMonitoring()`
- `apiAutofocusControllerSetLiveMonitoringParameters(period, method)`
- `apiAutofocusControllerGetLiveMonitoringStatus()`

All located in `/src/backendapi/` following project conventions.

### 4. UI Component (`AutofocusController.js`)

#### New UI Section: Live Focus Monitoring
Added below the autofocus scan section with a divider:

**Controls:**
- Update Period input (0.1-10s, step 0.1)
  - Disabled during monitoring
  - Updates backend immediately if monitoring is active
- Focus Method dropdown (LAPE/GLVA)
  - Disabled during monitoring
  - Updates backend immediately if monitoring is active
- Start/Stop toggle switch
  - Shows "Monitoring Active" when running

**Live Display (shown only when monitoring):**
- Large display card showing:
  - Current focus value (2 decimal places)
  - Method name in parentheses
  - Last update timestamp
- Centered layout with elevated card style
- Real-time updates via WebSocket signal

#### Handler Functions
- `handleStartLiveMonitoring()`: Starts monitoring and updates Redux
- `handleStopLiveMonitoring()`: Stops monitoring and clears value
- `handlePeriodChange(newPeriod)`: Updates period locally and remotely
- `handleMethodChange(newMethod)`: Updates method locally and remotely

## Usage Example

### Backend (Python)
```python
# Start live monitoring with 0.5s period using LAPE method
response = api.imcontrol.controllers["AutofocusController"].startLiveMonitoring(
    period=0.5, 
    method="LAPE"
)

# Change parameters while running
api.imcontrol.controllers["AutofocusController"].setLiveMonitoringParameters(
    period=1.0, 
    method="GLVA"
)

# Stop monitoring
api.imcontrol.controllers["AutofocusController"].stopLiveMonitoring()
```

### Frontend (React)
```javascript
// Start monitoring
await apiAutofocusControllerStartLiveMonitoring(0.5, "LAPE");

// Access live value from Redux
const liveFocusValue = useSelector(state => 
  state.autofocusState.liveFocusValue
);
// liveFocusValue = { focus_value: 123.45, timestamp: 1234567890.12, method: "LAPE" }

// Stop monitoring
await apiAutofocusControllerStopLiveMonitoring();
```

## Signal Flow

1. **Backend Thread**: `_doLiveMonitoringBackground()` runs every 0.5s
2. **Signal Emission**: `sigAutoFocusLiveValue.emit({focus_value, timestamp, method})`
3. **WebSocket Broadcast**: Signal automatically sent to all connected clients
4. **Frontend Handler**: `WebSocketHandler.js` receives signal
5. **Redux Update**: `setLiveFocusValue` action dispatches to store
6. **UI Update**: React component re-renders with new value

## Technical Notes

### Focus Measurement Methods
- **LAPE** (Laplacian Energy): Calculates variance of Laplacian filter - good for general use
- **GLVA** (Gray-Level Variance): Simple standard deviation - faster but less robust

### Performance Considerations
- Minimum period: 0.1s (prevents CPU overload)
- Frame cropping: 2048x2048 max (reduces computation)
- Grayscale conversion: Reduces data by 3x for RGB images
- Non-blocking: Background thread doesn't interfere with other operations

### Thread Safety
- Uses daemon threads (auto-terminates on program exit)
- Proper cleanup in `__del__` method
- Thread join with timeout to prevent hanging

## Integration Points

Follows project architecture patterns:
- ✅ Signal-based backend communication
- ✅ Redux state management
- ✅ WebSocket for real-time updates
- ✅ API functions in `/backendapi/`
- ✅ @APIExport decorators for endpoints
- ✅ English comments as per project guidelines
- ✅ Proper error handling and logging

## Future Enhancements

Potential improvements:
1. Historical plot of focus values over time
2. Configurable ROI for focus measurement
3. Automatic alarm when focus drops below threshold
4. Integration with stage controller for drift correction
5. Save/load monitoring configurations
