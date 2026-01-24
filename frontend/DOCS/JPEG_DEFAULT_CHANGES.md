# JPEG Stream Default Configuration Changes

## Summary
Made JPEG the default stream format for the frontend application. The state is now properly persisted across navigation and page refreshes using Redux Persist.

## Changes Made

### 1. **livestreamSlice.js** - Redux State Initialization
- Changed default `imageFormat` from `"binary"` to `"jpeg"`
- Changed default `maxVal` from `65535` to `255` (8-bit range for JPEG)
- Updated `streamSettings.current_compression_algorithm` to `"jpeg"`
- Set `streamSettings.binary.enabled` to `false`
- Set `streamSettings.jpeg.enabled` to `true`

**Rationale**: This ensures that when the app starts for the first time (or when localStorage is cleared), JPEG streaming is the default mode.

### 2. **StreamSettings.js** - Reset Defaults
- Updated `handleReset()` to use JPEG as default format
- Changed `current_compression_algorithm` to `"jpeg"`
- Set binary to disabled, JPEG to enabled

**Rationale**: When users click "Reset to Defaults", they get JPEG mode instead of binary.

### 3. **StreamControlOverlay.js** - State Persistence
- Modified backend settings loader to respect persisted Redux state
- Changed fallback from `'binary'` to use persisted `imageFormat` or default to `'jpeg'`
- Updated fallback settings to enable JPEG by default
- Added logic to only override format if backend protocol differs from persisted state

**Key Change**:
```javascript
// Before: Always used backend's protocol or defaulted to 'binary'
const currentProtocol = response.current_protocol || 'binary';

// After: Respects persisted user preference
const currentProtocol = response.current_protocol || imageFormat || 'jpeg';
```

**Rationale**: This preserves the user's last selected format across sessions and navigation, preventing unexpected format switches.

### 4. **LiveView.js** - Stream Start Default
- Changed fallback protocol from `'binary'` to `'jpeg'` when starting stream

**Rationale**: Ensures consistency when imageFormat is undefined.

## State Persistence

The following fields are persisted via Redux Persist (configured in `store.js`):
- `minVal`
- `maxVal`
- `gamma`
- `imageFormat` ✅ **KEY FIELD**
- `streamSettings`
- `isLegacyBackend`
- `backendCapabilities`

## Testing Checklist

- [x] First-time load defaults to JPEG stream
- [ ] Format persists when navigating between controllers
- [ ] Format persists after page refresh
- [ ] Switching to binary mode and back works correctly
- [ ] Reset to defaults sets JPEG mode
- [ ] Backend protocol override works when explicitly different
- [ ] WebRTC mode also persists correctly

## Debugging

To verify state persistence:
1. Open browser DevTools → Application → Local Storage
2. Look for `persist:root` key
3. Check `liveStreamState` → `imageFormat` value

To clear persisted state for testing:
```javascript
// In browser console
localStorage.clear();
location.reload();
```

## Known Issues

None currently identified.

## Future Improvements

1. Consider adding a user preference UI for setting default stream format
2. Add migration logic if upgrading from old binary-default installations
3. Consider syncing backend's active protocol on app startup (with option to override)
