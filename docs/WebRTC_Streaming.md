# WebRTC Streaming with ImSwitch

## Overview

ImSwitch now supports WebRTC streaming for low-latency, real-time video streaming from detectors. This implementation is based on the [aiortc](https://github.com/aiortc/aiortc) library and provides a complete WebRTC signaling and streaming solution.

## Features

- **Low Latency**: WebRTC provides sub-second latency for real-time viewing
- **Browser Native**: Works in any modern browser without plugins
- **Adaptive Bitrate**: Automatically adjusts quality based on network conditions
- **Standards Compliant**: Uses standard WebRTC protocols (STUN, ICE, SDP)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              LiveViewController                         │
├─────────────────────────────────────────────────────────┤
│  WebRTCStreamWorker                                     │
│  ├── Frame Queue (detector frames)                      │
│  └── DetectorVideoTrack (aiortc VideoStreamTrack)       │
│      └── Converts numpy frames to av.VideoFrame         │
│                                                          │
│  WebRTC Signaling:                                       │
│  ├── webrtc_offer() - Handle SDP offer from client      │
│  └── RTCPeerConnection - Manage WebRTC connection       │
└─────────────────────────────────────────────────────────┘
         │
         ▼
    Browser WebRTC Client
    ├── Create offer (SDP)
    ├── Send to server
    ├── Receive answer (SDP)
    └── Display video stream
```

## API Usage

### Start WebRTC Stream

```python
# Start WebRTC streaming
result = api.liveview.startLiveView(
    detectorName="Camera",
    protocol="webrtc"
)
```

### WebRTC Offer Endpoint

The client sends an SDP offer to the server:

```javascript
POST /liveview/webrtc_offer
{
  "detectorName": "Camera",  // optional
  "sdp": "v=0\r\no=...",      // SDP offer
  "type": "offer"
}
```

Server responds with SDP answer:

```javascript
{
  "status": "success",
  "sdp": "v=0\r\no=...",      // SDP answer
  "type": "answer"
}
```

## Browser Client

### Using the HTML Page

1. Open `webrtc_stream.html` in your browser
2. Enter the ImSwitch server URL (default: http://localhost:8001)
3. Optionally specify detector name
4. Click "Start WebRTC Stream"
5. Video will start playing once connection is established

### JavaScript Example

```javascript
// Create peer connection
const pc = new RTCPeerConnection({
    iceServers: [{urls: ['stun:stun.l.google.com:19302']}]
});

// Handle incoming video track
pc.addEventListener('track', (evt) => {
    if (evt.track.kind === 'video') {
        document.getElementById('video').srcObject = evt.streams[0];
    }
});

// Create and send offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

// Wait for ICE gathering
await new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') resolve();
    else pc.addEventListener('icegatheringstatechange', () => {
        if (pc.iceGatheringState === 'complete') resolve();
    });
});

// Send to server
const response = await fetch('http://localhost:8001/liveview/webrtc_offer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        detectorName: "Camera",
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type
    })
});

const answer = await response.json();

// Set remote description
await pc.setRemoteDescription(new RTCSessionDescription({
    sdp: answer.sdp,
    type: answer.type
}));
```

## Implementation Details

### DetectorVideoTrack

Custom `VideoStreamTrack` that reads frames from detector:

```python
class DetectorVideoTrack:
    - Reads frames from queue
    - Converts numpy arrays to av.VideoFrame
    - Handles grayscale to RGB conversion
    - Maintains proper timestamps and time_base
```

### WebRTCStreamWorker

```python
class WebRTCStreamWorker(StreamWorker):
    - Polls detector at configured rate
    - Normalizes frames to uint8
    - Puts frames in queue for video track
    - Manages video track lifecycle
```

### Frame Format

- **Input**: Numpy array from detector (any dtype)
- **Normalization**: Convert to uint8 (0-255 range)
- **Color**: Grayscale converted to RGB by duplicating channels
- **Output**: av.VideoFrame in RGB24 format

## Configuration

### Stream Parameters

```python
api.liveview.setStreamParams("webrtc", {
    "throttle_ms": 33,  # ~30 fps
    "stun_servers": ["stun:stun.l.google.com:19302"],
    "turn_servers": []  # Optional TURN servers
})
```

### ICE Servers

Configure STUN/TURN servers for NAT traversal:

```python
params = {
    "stun_servers": [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302"
    ],
    "turn_servers": [
        {
            "urls": "turn:your-turn-server.com:3478",
            "username": "user",
            "credential": "pass"
        }
    ]
}
```

## Requirements

### Python Dependencies

```bash
pip install aiortc av
```

### Browser Requirements

Any modern browser with WebRTC support:
- Chrome/Edge 56+
- Firefox 44+
- Safari 11+
- Opera 43+

## Troubleshooting

### Connection Fails

1. **Check STUN server**: Ensure STUN server is reachable
2. **Firewall**: Make sure ports are open for WebRTC
3. **HTTPS**: Some browsers require HTTPS for WebRTC
4. **Network**: NAT/firewall may block WebRTC traffic

### No Video

1. **Check detector**: Ensure detector is providing frames
2. **Check stream**: Verify stream is started with `getActiveStreams()`
3. **Browser console**: Check for JavaScript errors
4. **Format**: Ensure frames can be converted to RGB

### Poor Quality

1. **Adjust throttle_ms**: Lower for higher frame rate
2. **Network bandwidth**: Check network conditions
3. **Browser stats**: Use `pc.getStats()` to check metrics

## Performance

- **Latency**: Typically 100-500ms depending on network
- **Frame Rate**: Up to detector frame rate (default ~30 fps)
- **Resolution**: Native detector resolution (configurable via subsampling)
- **Bandwidth**: Varies based on resolution and network (typically 1-5 Mbps)

## Security

- **HTTPS Recommended**: Use HTTPS in production
- **Authentication**: Add authentication to API endpoints
- **CORS**: Configure CORS properly for cross-origin access
- **TURN**: Use authenticated TURN servers for secure relay

## Examples

### Start WebRTC Stream via API

```python
# Python
import requests

response = requests.post('http://localhost:8001/liveview/startLiveView', json={
    'protocol': 'webrtc',
    'params': {'throttle_ms': 33}
})

print(response.json())
```

### Stop Stream

```python
response = requests.post('http://localhost:8001/liveview/stopLiveView', json={
    'detectorName': 'Camera'
})
```

### Get Active Streams

```python
response = requests.get('http://localhost:8001/liveview/getActiveStreams')
print(response.json())
```

## Comparison with Other Protocols

| Feature | WebRTC | MJPEG | Binary |
|---------|--------|-------|--------|
| Latency | Very Low (100-500ms) | Medium (500-1000ms) | Low (200-500ms) |
| Browser Native | Yes | Yes | No |
| Bandwidth | Adaptive | High | Variable |
| Setup Complexity | Medium | Low | Low |
| NAT Traversal | Yes (STUN/TURN) | N/A | N/A |

## Future Enhancements

- [ ] Support audio streaming
- [ ] Data channels for metadata
- [ ] Multiple simultaneous clients
- [ ] Bandwidth adaptation
- [ ] Recording WebRTC streams
- [ ] Screen sharing to detector

## References

- [aiortc Documentation](https://aiortc.readthedocs.io/)
- [WebRTC Specification](https://www.w3.org/TR/webrtc/)
- [MDN WebRTC Guide](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
