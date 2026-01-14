# WebSocket Stream State Synchronization

## Problem

When the backend restarts or the WebSocket reconnects, the frontend's livestream state can become out of sync with the backend's actual state. This causes:

1. **Incorrect button states**: Play/Stop buttons show the wrong state
2. **Unexpected behavior**: Clicking Stop might start the stream (or vice versa)
3. **State persistence issues**: Frontend state doesn't reflect backend reality after reconnection

## Root Cause

The issue was caused by multiple state management problems:

1. **No state sync on reconnect**: When WebSocket reconnects, the frontend wasn't checking the backend's actual stream status
2. **Duplicate state**: `StreamControls` component maintained its own local state (`isLiveViewActive`) separate from Redux state
3. **No periodic sync**: No mechanism to detect and correct state drift between frontend and backend

## Solution

### 1. WebSocket Connection Handler (`WebSocketHandler.js`)

**Added state synchronization on connect/reconnect:**

```javascript
socket.on("connect", () => {
  console.log(`WebSocket connected with socket id: ${socket.id}`);
  dispatch(webSocketSlice.setConnected(true));
  
  // Sync livestream status with backend on connect/reconnect
  syncLivestreamStatus().then((isActive) => {
    console.log(`[WebSocket] Livestream status synced: ${isActive}`);
  });
});
```

**Added new `syncLivestreamStatus` function:**

```javascript
const syncLivestreamStatus = useCallback(async () => {
  try {
    console.log('[WebSocket] Syncing livestream status with backend...');
    const isActive = await apiViewControllerGetLiveViewActive();
    console.log(`[WebSocket] Backend livestream status: ${isActive}`);
    
    // Update Redux state to match backend
    dispatch(liveViewSlice.setIsStreamRunning(isActive));
    
    return isActive;
  } catch (error) {
    console.error('[WebSocket] Failed to sync livestream status:', error);
    // On error, assume stream is not running to prevent incorrect state
    dispatch(liveViewSlice.setIsStreamRunning(false));
    return false;
  }
}, [dispatch]);
```

**Added disconnect handler:**

```javascript
socket.on("disconnect", (reason) => {
  console.log(`[WebSocket] Disconnected. Reason: ${reason}`);
  dispatch(webSocketSlice.setConnected(false));
  
  // Note: We don't reset isStreamRunning here because the backend might still be running
  // The stream status will be re-synced on reconnect
});
```

### 2. Stream Controls Component (`StreamControls.js`)

**Removed duplicate local state:**

```javascript
// BEFORE: Had its own local state
const [isLiveViewActive, setIsLiveViewActive] = useState(false);

// AFTER: Uses Redux state as single source of truth
const liveViewState = useSelector(liveViewSlice.getLiveViewState);
const isLiveViewActive = liveViewState.isStreamRunning;
```

**Added periodic status verification:**

```javascript
// Periodic status check to keep Redux in sync with backend
const checkLiveViewStatus = useCallback(async () => {
  try {
    const active = await apiViewControllerGetLiveViewActive();
    
    // Only update Redux if state differs from backend
    if (active !== liveViewState.isStreamRunning) {
      console.log(`[StreamControls] Backend status mismatch detected. Backend: ${active}, Frontend: ${liveViewState.isStreamRunning}`);
      dispatch(liveViewSlice.setIsStreamRunning(active));
    }
  } catch (error) {
    console.warn('[StreamControls] Failed to check live view status:', error);
  }
}, [liveViewState.isStreamRunning, dispatch]);

// Periodic sync every 5 seconds to catch any state drift
useEffect(() => {
  const interval = setInterval(checkLiveViewStatus, 5000);
  return () => clearInterval(interval);
}, [checkLiveViewStatus]);
```

**Simplified start/stop handlers:**

```javascript
// BEFORE: Had complex logic with isCheckingStatus flag and delayed checks
const handleStartStream = useCallback(async () => {
  if (!isLiveViewActive && !isCheckingStatus) {
    await onToggleStream();
    setTimeout(checkLiveViewStatus, 500);
  }
}, [isLiveViewActive, isCheckingStatus, onToggleStream, checkLiveViewStatus]);

// AFTER: Simple, relies on Redux state
const handleStartStream = useCallback(async () => {
  if (!isLiveViewActive) {
    await onToggleStream();
    // Status will be updated by the toggleStream function in LiveView.js
  }
}, [isLiveViewActive, onToggleStream]);
```

## State Flow

### Initial Connection
```
1. WebSocket connects
2. syncLivestreamStatus() is called
3. Backend status is fetched via API
4. Redux state is updated to match backend
5. UI reflects correct state
```

### Reconnection After Backend Restart
```
1. Backend restarts (stream may or may not be running)
2. WebSocket reconnects
3. syncLivestreamStatus() is called automatically
4. Backend status is fetched
5. Redux state is corrected
6. UI buttons show correct state
```

### Periodic Verification
```
Every 5 seconds:
1. StreamControls checks backend status
2. Compares with Redux state
3. Updates Redux if mismatch detected
4. Logs warning about state drift
```

### User Action
```
1. User clicks Play/Stop
2. Handler checks current Redux state
3. Only acts if state allows (prevents double-clicks)
4. Calls onToggleStream()
5. LiveView.js updates Redux state
6. UI updates immediately
7. Periodic check verifies after 5s
```

## Benefits

1. **Automatic state recovery**: Frontend automatically syncs with backend on reconnect
2. **Single source of truth**: Redux state is the only state, no duplicates
3. **Drift detection**: Periodic checks catch unexpected state changes
4. **Better UX**: Buttons always show correct state
5. **Robust error handling**: Failed syncs don't crash, just log warnings

## Testing

### Test Case 1: Backend Restart
1. Start stream in frontend
2. Restart backend (stream stops)
3. Frontend reconnects
4. ✅ Verify: Play button is enabled, Stop is disabled

### Test Case 2: Backend Restart with Stream Running
1. Start stream in frontend
2. Backend restarts but auto-starts stream
3. Frontend reconnects
4. ✅ Verify: Stop button is enabled, Play is disabled

### Test Case 3: State Drift Detection
1. Start stream in frontend
2. Manually stop stream via backend API/terminal
3. Wait 5 seconds
4. ✅ Verify: Console shows mismatch detected
5. ✅ Verify: Buttons update to correct state

### Test Case 4: Multiple Tabs
1. Open app in two browser tabs
2. Start stream in Tab A
3. ✅ Verify: Tab B shows stream running (via periodic check)
4. Stop stream in Tab B
5. ✅ Verify: Tab A shows stream stopped (via periodic check)

## Debug Logs

Key console messages to watch:

```
[WebSocket] Syncing livestream status with backend...
[WebSocket] Backend livestream status: true/false
[WebSocket] Livestream status synced: true/false
[StreamControls] Backend status mismatch detected. Backend: X, Frontend: Y
[WebSocket] Disconnected. Reason: transport close
```

## Files Changed

- `src/middleware/WebSocketHandler.js` - Added sync logic on connect/disconnect
- `src/components/StreamControls.js` - Removed local state, added periodic sync
- `DOCS/WEBSOCKET_STREAM_STATE_SYNC.md` - This documentation

## Future Improvements

1. **Server-side status broadcasts**: Backend could emit `stream_status_changed` events
2. **Optimistic UI updates**: Update UI immediately, then verify
3. **Exponential backoff**: For periodic checks when backend is unreachable
4. **Status indicator**: Show "syncing..." state during verification
5. **User notification**: Alert user when state mismatch is detected and corrected
