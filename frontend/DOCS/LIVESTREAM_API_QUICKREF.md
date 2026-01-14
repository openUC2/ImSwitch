# LiveStream API Update - Quick Reference

## API Endpoint Changes

### Old API (deprecated)
```
SettingsController/setStreamParams     → LiveViewController/setStreamParameters
SettingsController/getStreamParams     → LiveViewController/getStreamParameters
ViewController/setLiveViewActive       → LiveViewController/startLiveView
                                          LiveViewController/stopLiveView
```

### New Endpoints

#### Stream Control
```javascript
// Start stream
POST /LiveViewController/startLiveView
  ?detectorName=null
  &protocol=binary
  body: { compression_algorithm: 'lz4', ... }

// Stop stream  
GET /LiveViewController/stopLiveView
  ?detectorName=null

// Check if active
GET /LiveViewController/getLiveViewActive
  → returns boolean
```

#### Parameter Management
```javascript
// Get parameters (all or specific protocol)
GET /LiveViewController/getStreamParameters
  ?protocol=binary  // optional
  
// Set parameters (global defaults)
GET /LiveViewController/setStreamParameters
  ?protocol=binary
  body: { compression_algorithm: 'lz4', compression_level: 0, ... }
  
// Get active streams
GET /LiveViewController/getActiveStreams
  → returns { detector1: { protocol: 'binary', params: {...} } }
```

## Parameter Examples

### Binary Stream
```javascript
{
  compression_algorithm: 'lz4',      // 'lz4', 'zstd', 'none'
  compression_level: 0,              // 0-9
  subsampling_factor: 4,             // 1-8
  throttle_ms: 50                    // 16-1000
}
```

### JPEG Stream
```javascript
{
  jpeg_quality: 80,                  // 1-100
  subsampling_factor: 4,             // 1-8
  throttle_ms: 50                    // 16-1000
}
```

### MJPEG Stream
```javascript
{
  jpeg_quality: 80,                  // 1-100
  throttle_ms: 50                    // 16-1000
}
```

## Code Examples

### Start Stream with Custom Parameters
```javascript
import apiLiveViewControllerStartLiveView from '../backendapi/apiLiveViewControllerStartLiveView';

// Binary stream
await apiLiveViewControllerStartLiveView(null, 'binary', {
  compression_algorithm: 'lz4',
  compression_level: 0,
  subsampling_factor: 4,
  throttle_ms: 50
});

// JPEG stream
await apiLiveViewControllerStartLiveView(null, 'jpeg', {
  jpeg_quality: 80,
  subsampling_factor: 1,
  throttle_ms: 100
});
```

### Update Global Parameters
```javascript
import apiLiveViewControllerSetStreamParameters from '../backendapi/apiLiveViewControllerSetStreamParameters';

// Set binary parameters
await apiLiveViewControllerSetStreamParameters('binary', {
  compression_algorithm: 'zstd',
  compression_level: 3,
  subsampling_factor: 2,
  throttle_ms: 33
});
```

### Get Current Parameters
```javascript
import apiLiveViewControllerGetStreamParameters from '../backendapi/apiLiveViewControllerGetStreamParameters';

// Get all protocols
const allParams = await apiLiveViewControllerGetStreamParameters();
// Returns: { binary: {...}, jpeg: {...}, mjpeg: {...}, webrtc: {...} }

// Get specific protocol
const binaryParams = await apiLiveViewControllerGetStreamParameters('binary');
// Returns: { compression_algorithm: 'lz4', ... }
```

### Stop Stream
```javascript
import apiLiveViewControllerStopLiveView from '../backendapi/apiLiveViewControllerStopLiveView';

// Stop first active stream
await apiLiveViewControllerStopLiveView();

// Stop specific detector
await apiLiveViewControllerStopLiveView('Camera1');
```

## Component Integration

### StreamControlOverlay.js
```javascript
// Load settings on mount
const allParams = await apiLiveViewControllerGetStreamParameters();

// Submit settings
const protocol = isJpegMode ? 'jpeg' : 'binary';
await apiLiveViewControllerSetStreamParameters(protocol, params);
```

### LiveView.js
```javascript
// Toggle stream
const protocol = liveStreamState.currentImageFormat === 'jpeg' ? 'jpeg' : 'binary';
if (shouldStart) {
  await apiLiveViewControllerStartLiveView(null, protocol);
} else {
  await apiLiveViewControllerStopLiveView();
}
```

## Testing Checklist

- [ ] Binary stream starts successfully
- [ ] JPEG stream starts successfully
- [ ] Stream stops successfully
- [ ] Parameters persist after stream restart
- [ ] Parameter changes take effect
- [ ] UI shows correct stream status
- [ ] Compression algorithm changes work (lz4, zstd, none)
- [ ] Compression level changes work (0-9)
- [ ] Subsampling changes work (1-8)
- [ ] Throttle changes work (16-1000ms)
- [ ] JPEG quality changes work (1-100)
- [ ] Error handling works correctly
- [ ] Multiple start/stop cycles work
- [ ] Format switching (binary ↔ JPEG) works

## Troubleshooting

### Stream won't start
- Check backend logs for errors
- Verify detector is available
- Check parameter values are in valid ranges
- Try with default parameters (pass `null`)

### Parameters not applying
- Ensure `setStreamParameters` is called before `startLiveView`
- Check Redux state is updated
- Verify backend returns success response
- Restart stream to apply new parameters

### UI not updating
- Check Redux state in DevTools
- Verify `apiViewControllerGetLiveViewActive` is polling
- Check browser console for errors
- Refresh the page

## Migration Guide

### If you have custom code using old endpoints:

**Old:**
```javascript
await fetch(`${host}/SettingsController/setStreamParams`, {
  method: 'POST',
  body: JSON.stringify(params)
});
```

**New:**
```javascript
await apiLiveViewControllerSetStreamParameters('binary', {
  compression_algorithm: 'lz4',
  compression_level: 0,
  subsampling_factor: 4,
  throttle_ms: 50
});
```

**Old:**
```javascript
await fetch(`${host}/ViewController/setLiveViewActive?active=${n}`);
```

**New:**
```javascript
if (n) {
  await apiLiveViewControllerStartLiveView(null, 'binary');
} else {
  await apiLiveViewControllerStopLiveView();
}
```
