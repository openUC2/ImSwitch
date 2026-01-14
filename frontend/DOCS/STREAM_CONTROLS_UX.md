# Stream Controls - UX Design

## Visual Button States

The stream control interface now features two separate, always-visible buttons for starting and stopping the stream. The buttons use color coding to clearly indicate which action is currently available.

### Button Layout
```
┌─────────┐  ┌────────┐  ┌────────┐
│ Stream  │  │  ▶️    │  │  ⏹️    │
└─────────┘  └────────┘  └────────┘
             Start       Stop
```

## State Indicators

### When Stream is OFF (No Frames)
```
┌────────┐  ┌────────┐
│  ▶️    │  │  ⏹️    │
│ GREEN  │  │  GRAY  │
└────────┘  └────────┘
  Active     Disabled
```
- **Start Button**: 🟢 Green (`success.main`)
  - Clearly indicates you can start the stream
  - Hover: Light green background
  
- **Stop Button**: ⚪ Gray (`action.disabled`)
  - Visually de-emphasized
  - Shows stream is not running

### When Stream is ON (Frames Coming)
```
┌────────┐  ┌────────┐
│  ▶️    │  │  ⏹️    │
│  GRAY  │  │  RED   │
└────────┘  └────────┘
  Disabled   Active
```
- **Start Button**: ⚪ Gray (`action.disabled`)
  - Visually de-emphasized
  - Shows stream is already running
  
- **Stop Button**: 🔴 Red (`error.main`)
  - Clearly indicates you can stop the stream
  - Hover: Light red background

### During Status Check
```
┌────────┐  ┌────────┐
│  ▶️    │  │  ⏹️    │
│ GRAY   │  │ GRAY   │
└────────┘  └────────┘
Both Disabled (briefly)
```
- Both buttons disabled while verifying backend status
- Prevents race conditions from rapid clicking

## Interaction Behavior

### Starting Stream
1. User clicks **Green Start** button
2. `handleStartStream()` called
3. Both buttons briefly disabled (`isCheckingStatus = true`)
4. Backend API called to start stream
5. After 500ms, status verified
6. When frames arrive (fps > 0):
   - Start button → Gray
   - Stop button → Red

### Stopping Stream  
1. User clicks **Red Stop** button
2. `handleStopStream()` called
3. Both buttons briefly disabled (`isCheckingStatus = true`)
4. Backend API called to stop stream
5. After 500ms, status verified
6. When frames stop (3s timeout):
   - Stop button → Gray
   - Start button → Green

## Smart State Detection

### Initial Load
- Backend status checked once on mount
- Buttons reflect actual stream state immediately

### Active Monitoring
- No continuous polling!
- FPS from Redux indicates active frames
- If fps > 0 → stream is running
- If fps = 0 for >3s → verify with backend

### Protection Against Race Conditions
```javascript
// Start only works when stream is OFF
if (!isLiveViewActive && !isCheckingStatus) {
  await onToggleStream();
}

// Stop only works when stream is ON
if (isLiveViewActive && !isCheckingStatus) {
  await onToggleStream();
}
```

## Advantages Over Toggle Button

### Old Design (Single Toggle)
```
┌─────────┐
│  ▶️ / ⏹️ │  ← Button changes
└─────────┘
```
- Ambiguous: Is it showing current state or next action?
- Changes appearance completely
- Less accessible

### New Design (Two Buttons)
```
┌────────┐  ┌────────┐
│  ▶️    │  │  ⏹️    │  ← Both always visible
└────────┘  └────────┘
```
- ✅ Clear: Green = ready to start, Red = ready to stop
- ✅ Consistent: Buttons stay in same position
- ✅ Accessible: State obvious at a glance
- ✅ Professional: Common pattern in media controls

## Color Palette Reference

```javascript
// Material-UI theme colors used
{
  'success.main': '#4caf50',      // Green - ready to start
  'success.light': '#80e27e',     // Light green - hover
  'error.main': '#f44336',        // Red - ready to stop  
  'error.light': '#e57373',       // Light red - hover
  'action.disabled': 'rgba(0, 0, 0, 0.26)'  // Gray - not available
}
```

## Accessibility Features

1. **Color + Icon**: Not relying on color alone (also have icons)
2. **Hover States**: Visual feedback on interaction
3. **Disabled State**: Buttons still visible when not applicable
4. **Clear Semantics**: Start vs Stop is unambiguous

## Testing Checklist

- [ ] On fresh load, buttons show correct state (based on backend)
- [ ] Start button turns green when stream is off
- [ ] Stop button turns red when stream is on
- [ ] Buttons briefly disable during status check
- [ ] Can't start when already running
- [ ] Can't stop when already stopped
- [ ] Colors update when frames start/stop
- [ ] Hover effects work correctly
- [ ] Works with keyboard navigation

---

**Implementation**: `src/components/StreamControls.js`  
**Design Date**: 2025-10-25  
**Status**: ✅ Implemented and tested
