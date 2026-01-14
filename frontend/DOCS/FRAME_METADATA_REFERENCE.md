# Frame Metadata Reference

## Quick Reference for Unified Frame Structure

### Frame Metadata Fields (Unified)

All streaming protocols (binary, JPEG, MJPEG, WebRTC) now use the same metadata structure:

```javascript
{
  "frame_id": 12345,           // Frame sequence number (0-65535, rolls over)
  "detector_name": "WidefieldCamera",  // Name of the detector/camera
  "pixel_size": 0.2,           // Pixel size in micrometers (float)
  "server_timestamp": 1699123456.789,  // Server timestamp (seconds since epoch)
  "format": "binary",          // Data format: "binary" or "jpeg"
  "protocol": "binary",        // Streaming protocol: "binary", "jpeg", "mjpeg", "webrtc"
  
  // Protocol-specific fields
  // For binary:
  "compression": "lz4",        // Compression algorithm
  "dtype": "uint16",           // Data type
  "shape": [2048, 2048],       // Image dimensions
  
  // For JPEG:
  "jpeg_quality": 80           // JPEG compression quality (0-100)
}
```

### Socket.IO Event Structure

#### Unified 'frame' Event (Recommended)

**Server → Client:**
```javascript
socket.emit('frame', [metadata, data], to=client_sid)
```

**Payload Structure:**
- Element 0: `metadata` (object) - Frame metadata as shown above
- Element 1: `data` (Buffer or String)
  - For binary: raw binary buffer (compressed UC2F packet)
  - For JPEG: base64-encoded JPEG image string

**Client → Server (Acknowledgement):**
```javascript
socket.emit('frame_ack', { frame_id: 12345 })
```

### Frontend Event Handling

```javascript
// Listen for frames (both binary and JPEG)
socket.on("frame", (payload, ack) => {
  if (Array.isArray(payload) && payload.length === 2) {
    const [metadata, frameData] = payload;
    
    if (metadata.protocol === 'binary') {
      // Handle binary frame
      window.dispatchEvent(new CustomEvent("uc2:frame", { 
        detail: { buffer: frameData, metadata: metadata }
      }));
    } else if (metadata.protocol === 'jpeg') {
      // Handle JPEG frame
      dispatch(liveStreamSlice.setLiveViewImage(frameData));
    }
    
    // Acknowledge frame (enables backpressure)
    if (ack) ack();
    else socket.emit('frame_ack', { frame_id: metadata.frame_id });
  }
});
```

### Backwards Compatibility

The system supports legacy field names for smooth migration:

| New Field | Legacy Field | Type Change |
|-----------|-------------|-------------|
| `frame_id` | `image_id` | None |
| `detector_name` | `detectorname` | None |
| `pixel_size` | `pixelsize` | int → float |

**Frontend code should use null coalescing:**
```javascript
const frameId = metadata.frame_id ?? metadata.image_id;
const detectorName = metadata.detector_name ?? metadata.detectorname;
const pixelSize = metadata.pixel_size ?? metadata.pixelsize;
```

### Frame ID Rollover

Frame IDs roll over at **65,536** (16-bit boundary):
```python
frame_id = (frame_id + 1) % 65536
```

This allows for efficient tracking while preventing integer overflow issues.

### Backpressure Flow Control

1. **Server** sends frame with `frame_id: N`
2. **Client** processes frame
3. **Client** sends `frame_ack` with `frame_id: N`
4. **Server** marks client ready and sends next frame with `frame_id: N+1`

**Important:** Clients that don't acknowledge frames will **not** receive new frames (prevents buffer overflow).

### Example: Adding a New Streaming Protocol

To add a new protocol (e.g., "h264"):

1. **Backend (LiveViewController.py):**
```python
class H264StreamWorker(StreamWorker):
    def _captureAndEmit(self):
        frame, detector_frame_number = self._detector.getLatestFrame(returnFrameNumber=True)
        self._frame_id = detector_frame_number if detector_frame_number else (self._frame_id + 1) % 65536
        
        # Encode as H.264
        h264_packet = encode_h264(frame)
        
        # Create unified metadata
        metadata = {
            'server_timestamp': time.time(),
            'frame_id': self._frame_id,
            'detector_name': self._detector.name,
            'pixel_size': float(pixel_size),
            'format': 'h264',
            'protocol': 'h264'
        }
        
        # Emit on unified 'frame' event
        message = {
            'type': 'h264_frame',
            'event': 'frame',
            'data': h264_packet,
            'metadata': metadata
        }
        self.sigStreamFrame.emit(message)
```

2. **noqt.py (already generic - no changes needed!):**
```python
# The unified _handle_stream_frame() handles all protocols automatically
```

3. **Frontend (WebSocketHandler.js):**
```javascript
socket.on("frame", (payload, ack) => {
  const [metadata, frameData] = payload;
  
  if (metadata.protocol === 'h264') {
    // Handle H.264 frame
    decodeH264(frameData, metadata);
  }
  
  if (ack) ack();
  else socket.emit('frame_ack', { frame_id: metadata.frame_id });
});
```

### Common Pitfalls

❌ **Don't do this:**
```javascript
// Using old field names only
const frameId = metadata.image_id;  // Will break with new backend
```

✅ **Do this instead:**
```javascript
// Support both for compatibility
const frameId = metadata.frame_id ?? metadata.image_id;
```

❌ **Don't do this:**
```javascript
// Not acknowledging frames
socket.on("frame", (payload) => {
  processFrame(payload);
  // Missing acknowledgement!
});
```

✅ **Do this instead:**
```javascript
// Always acknowledge frames
socket.on("frame", (payload, ack) => {
  processFrame(payload);
  if (ack) ack();  // Enable backpressure
});
```

### Debug Tips

**Check frame ID sequence:**
```javascript
let lastFrameId = -1;
socket.on("frame", (payload) => {
  const [metadata] = payload;
  const expectedId = (lastFrameId + 1) % 65536;
  if (metadata.frame_id !== expectedId) {
    console.warn(`Frame skip detected: got ${metadata.frame_id}, expected ${expectedId}`);
  }
  lastFrameId = metadata.frame_id;
});
```

**Monitor latency:**
```javascript
socket.on("frame", (payload) => {
  const [metadata] = payload;
  const latency = (Date.now() / 1000 - metadata.server_timestamp) * 1000;
  console.log(`Frame latency: ${latency.toFixed(1)}ms`);
});
```

**Track backpressure:**
```javascript
let pendingAck = 0;
socket.on("frame", (payload, ack) => {
  pendingAck++;
  processFrame(payload).then(() => {
    if (ack) ack();
    pendingAck--;
    if (pendingAck > 5) {
      console.warn(`High backpressure: ${pendingAck} frames pending`);
    }
  });
});
```

---
**See also:**
- [UNIFIED_FRAME_TRANSMISSION.md](./UNIFIED_FRAME_TRANSMISSION.md) - Full implementation details
- [LIVESTREAM_API_README.md](./LIVESTREAM_API_README.md) - Live streaming API documentation
