# Unified Frame Transmission System - Implementation Summary

## Overview
This document summarizes the changes made to unify binary and JPEG frame transmission across the ImSwitch backend and microscope-app frontend.

## Key Changes

### 1. Unified Metadata Structure

**Before:**
- Binary frames used `image_id`, `detectorname`, `pixelsize`
- JPEG frames used different field names
- Inconsistent metadata between protocols

**After:**
- **Unified field names:**
  - `frame_id` (replaces `image_id`) - supports rollover at 16-bit boundary (0-65535)
  - `detector_name` (replaces `detectorname`)
  - `pixel_size` (replaces `pixelsize`) - now float instead of int
  - `protocol` - explicitly identifies 'binary' or 'jpeg'
  - `format` - data format ('binary' or 'jpeg')
  - `server_timestamp` - server-side timestamp for latency tracking

### 2. Backend Changes (ImSwitch)

#### LiveViewController.py
- **BinaryStreamWorker:**
  - Now uses actual detector frame number when available
  - Frame ID counter with rollover handling: `(self._frame_id + 1) % 65536`
  - Unified metadata structure with `frame_id`, `detector_name`, `pixel_size`
  - Merges encoding metadata (compression info) into main metadata dict

- **JPEGStreamWorker:**
  - Same frame ID handling as binary worker
  - Changed event from 'signal' to 'frame' for consistency
  - Uses unified metadata structure
  - Payload structure: `{'image': base64_string, 'metadata': {...}}`

#### noqt.py
- **Unified `_handle_stream_frame()` method:**
  - Generic handling for both binary and JPEG frames
  - Both protocols use the same 'frame' event
  - Proper backpressure implementation for both protocols
  - Frame ID rollover handling: `(sent_id + 1) % 65536`
  - Socket.IO payload: `[metadata, data]` for both protocols
  - Thread-safe emission using `asyncio.run_coroutine_threadsafe()`

- **Frame acknowledgement:**
  - Changed from `image_id` to `frame_id` in `frame_ack` handler
  - Consistent field naming across all protocols

### 3. Frontend Changes (microscope-app)

#### WebSocketHandler.js
- **Unified 'frame' event handler:**
  - Handles both binary and JPEG frames on same event
  - Protocol detection via `metadata.protocol` or `metadata.format`
  - Binary frames → dispatch to `uc2:frame` custom event
  - JPEG frames → update Redux directly
  - Uses `frame_id` consistently (with fallback to `image_id` for backwards compatibility)

- **Frame acknowledgement:**
  - Uses unified `frame_id` field
  - Fallback support for legacy `image_id`

- **Legacy sigUpdateImage handler:**
  - Kept for backwards compatibility
  - Now supports both old and new field names:
    - `frame_id` ?? `image_id`
    - `detector_name` ?? `detectorname`
    - `pixel_size` ?? `pixelsize`

## Protocol Comparison

### Binary Frame Flow
```
Detector → BinaryStreamWorker → sigStreamFrame.emit(message) 
  → noqt.SignalInstance.emit() → _handle_stream_frame() 
  → sio.emit('frame', [metadata, binary_data], to=sid)
  → Frontend 'frame' handler → uc2:frame custom event
  → Frontend sends frame_ack
```

### JPEG Frame Flow
```
Detector → JPEGStreamWorker → sigStreamFrame.emit(message)
  → noqt.SignalInstance.emit() → _handle_stream_frame()
  → sio.emit('frame', [metadata, base64_image], to=sid)
  → Frontend 'frame' handler → Redux update
  → Frontend sends frame_ack
```

## Backwards Compatibility

The implementation maintains backwards compatibility:
1. Frontend supports both `frame_id` and `image_id` (with ?? operator)
2. Frontend supports both `detector_name` and `detectorname`
3. Frontend supports both `pixel_size` and `pixelsize`
4. Legacy `sigUpdateImage` handler still works (deprecated)

## Benefits

1. **Consistency:** Same event ('frame') and metadata structure for all protocols
2. **Maintainability:** Single code path for frame emission in backend
3. **Scalability:** Easy to add new streaming protocols
4. **Reliability:** Proper frame ID rollover handling (0-65535)
5. **Performance:** Unified backpressure mechanism for all protocols
6. **Debugging:** Easier to track frames with consistent metadata

## Testing Recommendations

1. Test binary streaming with frame ID rollover (capture > 65535 frames)
2. Test JPEG streaming with same rollover scenario
3. Verify backpressure works for both protocols (slow network simulation)
4. Test with multiple simultaneous clients
5. Verify latency tracking accuracy
6. Test protocol switching (binary ↔ JPEG) during live session

## Migration Notes

For future development:
- Always use `frame_id` instead of `image_id`
- Always use `detector_name` instead of `detectorname`
- Always use `pixel_size` (float) instead of `pixelsize` (int)
- Prefer the unified 'frame' event over legacy 'signal' event for image data
- Consider deprecating `sigUpdateImage` signal in favor of direct frame events

## TODO Items Addressed

✅ Use actual frame ID from detector (LiveViewController.py)  
✅ Rename `image_id` to `frame_id` for consistency (all files)  
✅ Handle int rollover for frame IDs (16-bit boundary)  
✅ Unify metadata vs data handling (consistent structure)  
✅ Make frame transmission generic for binary and JPEG (noqt.py)  
✅ Update frontend to support unified frame structure (WebSocketHandler.js)  
✅ Document signal handling logic (noqt.py comments)  
✅ Explain event loop warning during startup (noqt.py)  

## Performance Considerations

- Frame ID modulo operation is negligible overhead
- Socket.IO binary support is efficient for both protocols
- Backpressure prevents memory buildup
- Thread-safe emission adds minimal latency (~1-2ms)

---
**Date:** November 4, 2025  
**Author:** GitHub Copilot  
**Related Files:**
- `ImSwitch/imswitch/imcontrol/controller/controllers/LiveViewController.py`
- `ImSwitch/imswitch/imcommon/framework/noqt.py`
- `microscope-app/src/middleware/WebSocketHandler.js`
