# WebRTC Integration Summary

## Overview
WebRTC has been fully integrated into the microscope-app as a third streaming protocol alongside Binary (16-bit) and JPEG (8-bit) streaming.

## Architecture

### Backend (Python)
- **File**: `ImSwitch/imswitch/_imscontrol/controller/controllers/LiveViewController.py`
- **Key Features**:
  - WebRTC peer connection management with aiortc
  - DetectorVideoTrack for frame processing
  - Parameter-aware streaming (max_width, throttle_ms, subsampling_factor)
  - Performance optimizations:
    - Disabled ICE servers for localhost (prevents 5+ second STUN timeouts)
    - Fallback to answer.sdp when pc.localDescription is None
    - Comprehensive timing logs for profiling
  - Connection time reduced from 5565ms to <500ms

### Frontend (React)

#### WebRTCViewer Component
- **File**: `src/axon/WebRTCViewer.jsx`
- **Features**:
  - RTCPeerConnection management
  - Automatic localhost detection (disables ICE servers)
  - SDP offer/answer negotiation
  - Real-time stats monitoring (bitrate, FPS, packets lost)
  - Double-click support for full-screen viewing
  - Automatic reconnection on errors
  - Video element with object-fit: contain (preserves aspect ratio)

#### Redux State Management
- **File**: `src/state/slices/LiveStreamSlice.js`
- **Added WebRTC Settings**:
  ```javascript
  streamSettings: {
    webrtc: {
      enabled: false,
      max_width: 1280,      // Resolution limit (320-1920px)
      throttle_ms: 33,       // Frame interval (~30 FPS)
      subsampling_factor: 1  // Downsampling (1-8x)
    }
  }
  ```

#### Viewer Selection Logic
- **File**: `src/axon/LiveViewControlWrapper.js`
- **Three-Way Format Switching**:
  - `imageFormat === 'webrtc'` → WebRTCViewer
  - `imageFormat === 'jpeg'` → LiveViewComponent (MJPEG)
  - `imageFormat === 'binary'` → LiveViewerGL (WebGL with 16-bit support)

#### Stream Control UI
- **File**: `src/components/StreamControlOverlay.js`
- **Format Selection Dropdown**:
  - Binary (16-bit LZ4/Zstd compressed)
  - JPEG (8-bit legacy)
  - WebRTC (real-time low-latency) ← NEW
- **WebRTC Settings Panel**:
  - Max Width slider (320px - 1920px) with resolution presets
  - Subsampling slider (1x - 8x)
  - Throttle slider (16ms - 1000ms) with FPS indicator
  - Info alert explaining real-time streaming benefits

## User Workflow

### Switching to WebRTC
1. Open Stream Control overlay (Settings tab)
2. Select "WebRTC" from format dropdown
3. Adjust parameters:
   - **Max Width**: Controls resolution (higher = better quality, more bandwidth)
   - **Subsampling**: Reduces frame size (higher = faster, lower quality)
   - **Throttle**: Frame interval (lower = higher FPS, more bandwidth)
4. Click "Submit" to apply settings
5. Stream automatically switches to WebRTC viewer

### Performance Tuning
- **Low Latency**: Set throttle_ms to 16-33ms (30-60 FPS)
- **Low Bandwidth**: Increase subsampling_factor or reduce max_width
- **High Quality**: Set max_width to 1920, subsampling_factor to 1

## Technical Details

### Connection Flow
1. Frontend sends SDP offer via `/api/liveviewcontroller/webrtc_offer`
2. Backend creates RTCPeerConnection with iceServers=[] for localhost
3. Backend adds video track from detector
4. Backend generates SDP answer
5. Frontend sets remote description
6. ICE candidate exchange (skipped for localhost)
7. Video stream established via RTP

### Error Handling
- NoneType errors on pc.localDescription → Fallback to answer.sdp
- STUN timeouts → Disabled ICE servers for localhost connections
- Connection failures → Automatic reconnection in WebRTCViewer
- Parameter validation → Backend enforces min/max ranges

### Performance Optimizations
- **Frontend**: Canvas rendering with object-fit: contain
- **Backend**: Frame throttling with configurable interval
- **Network**: No ICE servers for localhost (eliminates 5s delay)
- **Codec**: VP8 with dynamic bitrate adjustment

## Testing

### Verified Scenarios
✅ WebRTC streaming with 1280px max width @ 30 FPS
✅ Switching between Binary → JPEG → WebRTC without restart
✅ Parameter changes via settings UI
✅ Localhost connection without STUN timeouts
✅ Aspect ratio preservation for non-square sensors
✅ Stats monitoring (bitrate, FPS, packet loss)

### Known Limitations
- aiortc has limited RTCConfiguration support (no bundlePolicy, rtcpMuxPolicy)
- ICE servers must be disabled for localhost connections
- Some browsers may require HTTPS for WebRTC (development uses HTTP)

## API Endpoints

### POST /api/liveviewcontroller/webrtc_offer
**Request**:
```json
{
  "sdp": "v=0\r\no=- ...",
  "type": "offer"
}
```

**Response**:
```json
{
  "sdp": "v=0\r\no=- ...",
  "type": "answer"
}
```

### PUT /api/liveviewcontroller/set_stream_parameters
**Request**:
```json
{
  "mode": "webrtc",
  "max_width": 1280,
  "throttle_ms": 33,
  "subsampling_factor": 1
}
```

## Future Enhancements
- [ ] Add codec selection (VP8/H.264)
- [ ] Bitrate control slider
- [ ] Network quality indicator
- [ ] Recording support
- [ ] Multiple viewer support (multicast)
- [ ] HTTPS support for production deployments

## References
- WebRTC API: https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API
- aiortc library: https://github.com/aiortc/aiortc
- Performance profiling docs: `DOCS/LIVESTREAM_API_PERFORMANCE.md`
