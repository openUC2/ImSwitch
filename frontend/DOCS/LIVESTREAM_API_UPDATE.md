# LiveStream API Update - Implementation Summary

## Overview
Updated the React webapp to use the new `LiveViewController` backend endpoints for managing live streams. The new API provides better separation of concerns and more flexible streaming parameter management.

## Changes Made

### 1. New API Functions Created

All new API functions are located in `src/backendapi/`:

#### `apiLiveViewControllerGetActiveStreams.js`
- **Endpoint**: `GET /LiveViewController/getActiveStreams`
- **Purpose**: Get list of currently active streams
- **Returns**: Dictionary with active stream information per detector

#### `apiLiveViewControllerGetStreamParameters.js`
- **Endpoint**: `GET /LiveViewController/getStreamParameters`
- **Purpose**: Get current streaming parameters for all or specific protocol
- **Parameters**: 
  - `protocol` (optional): 'binary', 'jpeg', 'mjpeg', 'webrtc'
- **Returns**: Dictionary with streaming parameters

#### `apiLiveViewControllerSetStreamParameters.js`
- **Endpoint**: `GET /LiveViewController/setStreamParameters`
- **Purpose**: Configure global streaming parameters for a protocol
- **Parameters**: 
  - `protocol` (required): 'binary', 'jpeg', 'mjpeg', 'webrtc'
  - `params` (body): Protocol-specific parameters
- **Supported params**:
  - **Binary**: `compression_algorithm`, `compression_level`, `subsampling_factor`, `throttle_ms`
  - **JPEG/MJPEG**: `jpeg_quality`, `subsampling_factor`, `throttle_ms`
  - **WebRTC**: `ice_servers`, `media_constraints`

#### `apiLiveViewControllerStartLiveView.js`
- **Endpoint**: `POST /LiveViewController/startLiveView`
- **Purpose**: Start live streaming for a specific detector
- **Parameters**:
  - `detectorName` (optional): Detector name (null = first available)
  - `protocol` (optional): 'binary', 'jpeg', 'mjpeg', 'webrtc' (default: 'binary')
  - `params` (body, optional): Override default parameters for this stream
- **Returns**: Dictionary with status and stream info

#### `apiLiveViewControllerStopLiveView.js`
- **Endpoint**: `GET /LiveViewController/stopLiveView`
- **Purpose**: Stop live streaming for a specific detector
- **Parameters**:
  - `detectorName` (optional): Detector name (null = stop first active)
- **Returns**: Dictionary with status

### 2. Updated Components

#### `src/components/StreamControlOverlay.js`
**Changes**:
- Replaced `apiSettingsControllerSetStreamParams` with `apiLiveViewControllerSetStreamParameters`
- Replaced `apiSettingsControllerGetDetectorGlobalParameters` with `apiLiveViewControllerGetStreamParameters`
- Updated `loadBackendSettings()` to use new endpoint format
- Updated `handleSubmitSettings()` to send protocol-specific parameters:
  - For JPEG mode: sends `jpeg_quality`, `subsampling_factor`, `throttle_ms`
  - For binary mode: sends `compression_algorithm`, `compression_level`, `subsampling_factor`, `throttle_ms`

**Key Implementation Details**:
```javascript
// Loading settings from backend
const allParams = await apiLiveViewControllerGetStreamParameters();

// Submitting JPEG parameters (POST request with protocol in query)
await apiLiveViewControllerSetStreamParameters('jpeg', {
  jpeg_quality: 80,
  subsampling_factor: 4,
  throttle_ms: 50
});

// Submitting binary parameters (POST request with protocol in query)
await apiLiveViewControllerSetStreamParameters('binary', {
  compression_algorithm: 'lz4',
  compression_level: 0,
  subsampling_factor: 4,
  throttle_ms: 50
});
```

**Important**: The `setStreamParameters` endpoint is a **POST** request where:
- `protocol` is passed as a **query parameter**: `?protocol=binary`
- `params` are passed as **JSON in the request body**

#### `src/components/LiveView.js`
**Changes**:
- Added imports for `apiLiveViewControllerStartLiveView` and `apiLiveViewControllerStopLiveView`
- Updated `toggleStream()` function to use new endpoints
- Stream now starts with protocol based on current format (binary or jpeg)

**Key Implementation Details**:
```javascript
const toggleStream = async () => {
  const shouldStart = !isStreamRunning;
  
  if (shouldStart) {
    // Determine protocol from current stream settings
    const protocol = liveStreamState.currentImageFormat === 'jpeg' ? 'jpeg' : 'binary';
    await apiLiveViewControllerStartLiveView(null, protocol);
  } else {
    await apiLiveViewControllerStopLiveView();
  }
  
  dispatch(liveViewSlice.setIsStreamRunning(shouldStart));
};
```

#### `src/components/StreamSettings.js`
**Changes**:
- Replaced `apiSettingsControllerGetStreamParams` with `apiLiveViewControllerGetStreamParameters`
- Replaced `apiSettingsControllerSetStreamParams` with `apiLiveViewControllerSetStreamParameters`
- Removed dependency on `apiSettingsControllerGetDetectorGlobalParameters`
- Updated `loadSettings()` to use new endpoint format
- Updated `handleSubmit()` to send protocol-specific parameters
- Updated `handleRetry()` to use new endpoint format

**Key Implementation Details**:
Similar to `StreamControlOverlay.js`, this component now:
- Loads parameters from `apiLiveViewControllerGetStreamParameters()`
- Submits binary or JPEG parameters separately based on enabled mode
- Transforms backend response to frontend format for consistency

#### `src/backendapi/apiViewControllerGetLiveViewActive.js`
**Changes**:
- Updated documentation to reflect that this endpoint now checks if **any** stream is active
- No functional changes, endpoint path remains the same

### 3. Parameter Mapping

The frontend continues to use its internal structure but now maps to the backend's expected format:

| Frontend (StreamControlOverlay) | Backend (LiveViewController) |
|--------------------------------|------------------------------|
| `binary.compression.algorithm` | `compression_algorithm` |
| `binary.compression.level` | `compression_level` |
| `binary.subsampling.factor` | `subsampling_factor` |
| `binary.throttle_ms` | `throttle_ms` |
| `jpeg.quality` | `jpeg_quality` |
| `jpeg.subsampling.factor` | `subsampling_factor` |
| `jpeg.throttle_ms` | `throttle_ms` |

## UI Flow

### Starting a Stream
1. User clicks "Play" button in `StreamControls`
2. `LiveView.toggleStream()` is called
3. Function determines protocol from `liveStreamState.currentImageFormat`
4. Calls `apiLiveViewControllerStartLiveView(null, protocol)`
5. Backend starts streaming with global parameters for that protocol
6. Redux state updated to reflect stream is running

### Changing Stream Parameters
1. User opens StreamControlOverlay settings tab
2. Parameters loaded from backend via `apiLiveViewControllerGetStreamParameters()`
3. User modifies settings (compression, quality, subsampling, throttle)
4. User clicks "Submit"
5. Parameters sent to backend via `apiLiveViewControllerSetStreamParameters(protocol, params)`
6. Redux state updated
7. If stream is running, parameters take effect on next frame

### Stopping a Stream
1. User clicks "Stop" button in `StreamControls`
2. `LiveView.toggleStream()` is called
3. Calls `apiLiveViewControllerStopLiveView()`
4. Backend stops streaming
5. Redux state updated to reflect stream is stopped

## Backward Compatibility

The old endpoints (`SettingsController/setStreamParams`, `SettingsController/getStreamParams`) are no longer used. The new implementation:
- ✅ Maintains the same UI/UX
- ✅ Supports both binary and JPEG streaming
- ✅ Provides better parameter validation
- ✅ Separates global settings from per-stream settings
- ✅ Allows multiple detectors to stream with different protocols

## Testing Recommendations

1. **Test binary stream parameters**:
   - Change compression algorithm (lz4, zstd, none)
   - Change compression level (0-9)
   - Change subsampling (1-8)
   - Change throttle (16-1000ms)

2. **Test JPEG stream parameters**:
   - Change quality (1-100)
   - Change subsampling (1-8)
   - Change throttle (16-1000ms)

3. **Test stream start/stop**:
   - Start binary stream
   - Stop stream
   - Start JPEG stream
   - Stop stream
   - Toggle multiple times

4. **Test parameter persistence**:
   - Set parameters for binary
   - Start stream
   - Stop stream
   - Verify parameters persisted
   - Switch to JPEG
   - Verify JPEG parameters loaded correctly

5. **Test error handling**:
   - Try starting stream with invalid parameters
   - Try starting stream when backend is unavailable
   - Verify UI shows appropriate error messages

## Migration Notes

### For Developers
- All stream parameter management now goes through `LiveViewController`
- Use `apiLiveViewControllerSetStreamParameters()` to set global defaults
- Use `apiLiveViewControllerStartLiveView()` with custom params to override per stream
- The protocol parameter determines which set of parameters to use

### For Backend Developers
- Ensure `LiveViewController` endpoints return expected formats
- `getStreamParameters()` should return flat dictionaries per protocol
- `setStreamParameters()` should accept protocol name as query param and params as body
- `startLiveView()` should accept protocol and optional params to override defaults

## Known Limitations

1. **WebRTC support**: WebRTC endpoints are implemented but not tested in this update
2. **MJPEG stream**: MJPEG stream endpoint (`/LiveViewController/mjpeg_stream`) is documented but not integrated in the UI yet
3. **Multiple detectors**: The UI currently only supports one detector, though the API supports multiple

## Future Enhancements

1. Add UI for selecting specific detector
2. Add UI for MJPEG streaming
3. Add WebRTC streaming support
4. Add visual indicator of active protocol in StreamControlOverlay
5. Add real-time parameter monitoring from backend
6. Add `apiLiveViewControllerGetActiveStreams()` integration to show all active streams

## Files Modified

- ✅ `src/backendapi/apiLiveViewControllerGetActiveStreams.js` (new)
- ✅ `src/backendapi/apiLiveViewControllerGetStreamParameters.js` (new)
- ✅ `src/backendapi/apiLiveViewControllerSetStreamParameters.js` (new)
- ✅ `src/backendapi/apiLiveViewControllerStartLiveView.js` (new)
- ✅ `src/backendapi/apiLiveViewControllerStopLiveView.js` (new)
- ✅ `src/backendapi/apiViewControllerGetLiveViewActive.js` (updated docs)
- ✅ `src/components/StreamControlOverlay.js` (updated)
- ✅ `src/components/StreamSettings.js` (updated)
- ✅ `src/components/LiveView.js` (updated)

## Summary

The implementation successfully migrates from the old `SettingsController` and `ViewController` endpoints to the new unified `LiveViewController` API. The UI remains unchanged for users while providing better separation of concerns and more flexibility for future enhancements.
