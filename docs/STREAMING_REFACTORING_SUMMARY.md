# Streaming Architecture Refactoring Summary

## Overview
This refactoring centralizes frame encoding in `LiveViewController` and removes encoding logic from `noqt.py`, creating a clean separation of concerns where:
- **LiveViewController**: Handles frame capture, encoding, and message formatting
- **noqt.py**: Only handles socket.io emission of pre-formatted messages

## Changes Made

### 1. LiveViewController (`imswitch/imcontrol/controller/controllers/LiveViewController.py`)

#### Signal Changes
- **Removed**: `sigFrameReady = Signal(str, bytes, dict)` from `StreamWorker` class
- **Added**: `sigStreamFrame = Signal(dict)` to both `StreamWorker` and `LiveViewController` classes
  - Emits pre-formatted socket.io messages ready for emission
  - Message format: `{'type': 'binary_frame' or 'jpeg_frame', 'event': str, 'data': bytes or dict, 'metadata': dict}`

#### BinaryStreamWorker Changes
- Modified `_captureAndEmit()` to create complete message structure:
  ```python
  message = {
      'type': 'binary_frame',
      'event': 'frame',
      'data': packet,  # Compressed binary frame
      'metadata': {
          'server_timestamp': time.time(),
          'image_id': self._image_id,
          'detectorname': detector_name,
          'pixelsize': int(pixel_size),
          'format': 'binary'
      }
  }
  self.sigStreamFrame.emit(message)
  ```

#### JPEGStreamWorker Changes
- Modified `_captureAndEmit()` to:
  - Apply subsampling using `self._params.subsampling_factor`
  - Encode frame as JPEG
  - Base64 encode for JSON transmission
  - Create complete message structure:
  ```python
  message = {
      'type': 'jpeg_frame',
      'event': 'signal',
      'data': {
          'name': 'sigUpdateImage',
          'detectorname': detector_name,
          'pixelsize': int(pixel_size),
          'format': 'jpeg',
          'image': encoded_image,  # base64 encoded
          'server_timestamp': time.time(),
          'image_id': self._image_id
      }
  }
  self.sigStreamFrame.emit(message)
  ```

#### Controller Changes
- **Removed**: `_onFrameReady()` method (no longer needed)
- **Modified**: `startLiveView()` to connect worker signal to controller signal:
  ```python
  worker.sigStreamFrame.connect(self.sigStreamFrame)
  ```
- **Removed**: Direct socket.io imports and emission logic
- **Removed**: TODO comments about signal mapping (now resolved)
- **Cleaned up**: Unnecessary globalDetectorParams updates (streaming config now handled internally)

### 2. noqt.py (`imswitch/imcommon/framework/noqt.py`)

#### SignalInstance Changes
- **Simplified** `emit()` method:
  - Added handler for `sigStreamFrame` signal
  - Made `sigUpdateImage`, `sigExperimentImageUpdate`, `sigImageUpdated` no-ops (streaming centralized)
  
- **Added** `_handle_stream_frame(message: dict)` method:
  - Receives pre-formatted messages from LiveViewController
  - Handles client flow control (acknowledgements)
  - Emits binary frames via socket.io 'frame' event
  - Emits JPEG frames via socket.io 'signal' event
  - Emits metadata for binary frames

- **Removed** legacy image encoding methods:
  - `_handle_image_signal()` - No longer needed
  - `_emit_binary_frame()` - Encoding moved to LiveViewController
  - `_emit_jpeg_frame()` - Encoding moved to LiveViewController
  - `update_binary_config()` - Configuration handled by LiveViewController

- **Removed** class attributes:
  - `IMG_QUALITY` - Now in StreamParams
  - `_binary_encoder` - Created per-worker in LiveViewController

### 3. ImConMainController (`imswitch/imcontrol/controller/ImConMainController.py`)

#### Signal Connection
- Added automatic connection of LiveViewController streaming in headless mode:
  ```python
  if IS_HEADLESS and 'LiveView' in self.controllers:
      liveViewController = self.controllers['LiveView']
      # sigStreamFrame automatically handled by noqt's _handle_stream_frame
  ```

## Architecture Benefits

### Clear Separation of Concerns
1. **LiveViewController**:
   - Frame capture from detector
   - Frame preprocessing (normalization, subsampling)
   - Frame encoding (binary compression, JPEG encoding)
   - Message formatting (complete socket.io message structure)
   - Streaming protocol management

2. **noqt.py**:
   - Signal framework implementation
   - Socket.io server management
   - Client connection tracking
   - Flow control (acknowledgements)
   - Message emission only

### Advantages
- **Maintainability**: Encoding logic in one place (LiveViewController)
- **Testability**: Workers can be tested independently
- **Flexibility**: Easy to add new streaming protocols
- **Performance**: Workers run in dedicated threads with proper throttling
- **Type Safety**: Pre-formatted messages ensure consistent structure
- **No Duplication**: Removed duplicate encoding logic from noqt.py

## Message Flow

```
Detector
   ↓
[BinaryStreamWorker / JPEGStreamWorker]
   ├─ Capture frame
   ├─ Encode frame (LZ4/Zstandard/JPEG)
   ├─ Format metadata
   └─ Create complete message
   ↓
[LiveViewController.sigStreamFrame]
   ↓
[noqt.SignalInstance]
   ├─ Check client readiness
   ├─ Emit to socket.io
   └─ Track acknowledgements
   ↓
[WebSocket Clients]
```

## Configuration

Streaming parameters are now centrally managed in LiveViewController:
- `StreamParams` dataclass defines all parameters
- Parameters can be updated via `setStreamParameters()` API
- Workers use parameters from their `StreamParams` instance
- No reliance on globalDetectorParams for streaming config

## Testing

To test the refactoring:
1. Start ImSwitch in headless mode
2. Use API to start streaming: `/LiveViewController/startLiveView`
3. Connect a WebSocket client to receive frames
4. Verify frame metadata and acknowledgement flow
5. Test parameter updates via `/LiveViewController/setStreamParameters`

## Migration Notes

- Existing WebSocket clients should work without changes
- Frame format and events remain the same
- MJPEG and WebRTC streaming unchanged (use their own mechanisms)
- Legacy non-headless mode still uses original signal paths (backward compatible)

## Future Enhancements

- Remove legacy `sigUpdateImage` handling entirely once confirmed unnecessary
- Add WebRTC integration using same message format
- Consider msgpack for binary metadata instead of JSON
- Add streaming statistics/monitoring
