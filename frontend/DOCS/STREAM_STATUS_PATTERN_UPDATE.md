# Stream Status Detection Pattern Update

## Overview
Updated the stream status detection pattern in `StreamControls.js` to reduce backend API polling and rely on actual frame reception instead. Also improved UX with separate start/stop buttons that are always visible with color-coded states.

## Previous Pattern (High-Frequency Polling)
- Polled `apiViewControllerGetLiveViewActive` every 2 seconds continuously
- High backend load from constant status checks
- Could miss rapid state changes between polls
- Single toggle button that changed between Play/Stop

## New Pattern (Frame-Based Detection)

### 1. Initial Check on Startup
- Call `apiViewControllerGetLiveViewActive` **once** on component mount
- Establish baseline stream status

### 2. Active Frame Detection
- Monitor `liveStreamState.stats.fps` from Redux
- If `fps > 0`, frames are actively coming in → stream is running
- Update `lastFrameTimeRef` timestamp on each frame
- Automatically set `isLiveViewActive = true` when frames detected

### 3. Inactivity Detection
- Check every 2 seconds if frames have stopped
- If no frames for >3 seconds AND we think stream is active:
  - Poll `apiViewControllerGetLiveViewActive` to verify stream is actually off
  - This confirms the stream stopped (not just a temporary network issue)

### 4. Manual Start/Stop Actions
- Separate buttons for Start and Stop (always visible)
- After user clicks start/stop, wait 500ms then check backend status
- Ensures UI reflects actual backend state after user action

## UI/UX Improvements

### Two-Button Design
Instead of a single toggle button, we now have two separate buttons that are always visible:

#### Start Button (Play Icon)
- **Green** when stream is OFF (ready to start) 
- **Gray** (disabled appearance) when stream is ON
- Always clickable, but only acts when stream is off

#### Stop Button (Stop Icon)  
- **Red** when stream is ON (ready to stop)
- **Gray** (disabled appearance) when stream is OFF
- Always clickable, but only acts when stream is on

This provides:
- **Clear visual feedback** of current state
- **No confusion** about which action is available
- **Consistent UI** - buttons don't disappear or change
- **Better accessibility** - state is obvious at a glance

## Benefits
1. **Reduced Backend Load**: Only 1 API call on mount, plus occasional verification calls
2. **Real-Time Response**: Immediately detects stream activity via FPS updates
3. **Accurate Status**: Buttons show correct state based on actual frame reception
4. **Network Resilience**: Distinguishes between temporary network issues and actual stream stop
5. **Better UX**: Two separate buttons with color-coded states are clearer than toggle button

## Technical Implementation

### Key Changes in `StreamControls.js`

#### Stream Status Detection
```javascript
// Get FPS from Redux (updated by LiveViewerGL on each frame)
const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);

// Track last frame time
const lastFrameTimeRef = useRef(Date.now());

// Monitor FPS to detect active streaming
useEffect(() => {
  if (liveStreamState.stats.fps > 0) {
    lastFrameTimeRef.current = Date.now();
    if (!isLiveViewActive) {
      setIsLiveViewActive(true);
    }
  }
}, [liveStreamState.stats.fps, isLiveViewActive]);

// Inactivity detection (verify stream stopped after 3s of no frames)
useEffect(() => {
  const interval = setInterval(() => {
    const timeSinceLastFrame = Date.now() - lastFrameTimeRef.current;
    if (timeSinceLastFrame > 3000 && isLiveViewActive) {
      console.log('No frames for 3s, verifying stream status...');
      checkLiveViewStatus(); // Poll backend to confirm
    }
  }, 2000);
  return () => clearInterval(interval);
}, [isLiveViewActive, checkLiveViewStatus]);
```

#### Button Implementation
```javascript
// Handle start stream
const handleStartStream = useCallback(async () => {
  if (!isLiveViewActive && !isCheckingStatus) {
    await onToggleStream();
    setTimeout(checkLiveViewStatus, 500);
  }
}, [isLiveViewActive, isCheckingStatus, onToggleStream, checkLiveViewStatus]);

// Handle stop stream
const handleStopStream = useCallback(async () => {
  if (isLiveViewActive && !isCheckingStatus) {
    await onToggleStream();
    setTimeout(checkLiveViewStatus, 500);
  }
}, [isLiveViewActive, isCheckingStatus, onToggleStream, checkLiveViewStatus]);

// Start button - green when stream is OFF (can start), gray when ON
<IconButton
  onClick={handleStartStream}
  disabled={isCheckingStatus}
  sx={{
    color: !isLiveViewActive ? 'success.main' : 'action.disabled',
    '&:hover': {
      backgroundColor: !isLiveViewActive ? 'success.light' : 'transparent',
      opacity: !isLiveViewActive ? 0.8 : 0.5
    }
  }}
>
  <PlayArrow />
</IconButton>

// Stop button - red when stream is ON (can stop), gray when OFF
<IconButton
  onClick={handleStopStream}
  disabled={isCheckingStatus}
  sx={{
    color: isLiveViewActive ? 'error.main' : 'action.disabled',
    '&:hover': {
      backgroundColor: isLiveViewActive ? 'error.light' : 'transparent',
      opacity: isLiveViewActive ? 0.8 : 0.5
    }
  }}
>
  <Stop />
</IconButton>
```

### Data Flow
1. **LiveViewerGL** receives frames via WebSocket → updates Redux `stats.fps`
2. **StreamControls** reads `stats.fps` from Redux
3. When `fps > 0`: Stream is active (immediate update)
4. When `fps == 0` for >3s: Poll backend to verify stream stopped

## Visual States

| Stream State | Start Button | Stop Button |
|--------------|--------------|-------------|
| OFF (no frames) | 🟢 Green (active) | ⚪ Gray (disabled look) |
| ON (frames coming) | ⚪ Gray (disabled look) | 🔴 Red (active) |
| Checking... | Both buttons disabled during status check |

## Testing Recommendations
1. **Start stream** → verify Start button turns gray, Stop button turns red when frames start
2. **Stop stream** → verify Stop button turns gray, Start button turns green after ~3 seconds
3. **Network interruption** → verify buttons don't flicker, wait for confirmation
4. **Rapid clicks** → verify only one action executes at a time (isCheckingStatus protection)
5. **Initial load** → verify buttons show correct state based on backend status

## Files Modified
- `/src/components/StreamControls.js`
  - Added Redux selector for `liveStreamState`
  - Replaced continuous polling with frame-based detection
  - Added inactivity detection interval
  - Synced `hudData.stats` with Redux stats
  - Replaced single toggle button with separate Start/Stop buttons
  - Added color-coded button states (green/red/gray)

## Dependencies
- Redux state: `liveStreamState.stats.fps` (updated by `LiveViewerGL`)
- Backend API: `apiViewControllerGetLiveViewActive` (used sparingly - once on mount + verification)
- WebSocket: Frame events continue to update Redux as before
- Material-UI: `success.main`, `error.main`, `action.disabled` theme colors

---
**Date**: 2025-10-25  
**Impact**: Lower backend load, more responsive UI, better stream status accuracy, improved UX with clearer button states
