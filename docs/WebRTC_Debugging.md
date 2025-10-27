# WebRTC Debugging Guide

## Common Issue: ICE Gathering Gets Stuck

### Symptom
The client shows "Creating offer..." but never progresses to "Sending offer to server..."

### Root Cause
**The peer connection needs at least one media track/transceiver to generate ICE candidates!**

Without any media:
```javascript
pc = new RTCPeerConnection();
await pc.createOffer();  // No media tracks!
// iceGatheringState = "new" forever ❌
```

With media (correct):
```javascript
pc = new RTCPeerConnection();
pc.addTransceiver('video', { direction: 'recvonly' });  // Add media track!
await pc.createOffer();
// iceGatheringState = "gathering" → "complete" ✅
```

### Solution
Always call `addTransceiver` before creating the offer:

```javascript
// This tells the peer connection we want to receive video
pc.addTransceiver('video', { direction: 'recvonly' });

// Now ICE gathering will work properly
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);
```

### Direction Options

- `'recvonly'`: Only receive (client receiving from server) ← **Use this for ImSwitch**
- `'sendonly'`: Only send (client sending to server)
- `'sendrecv'`: Both send and receive
- `'inactive'`: Neither send nor receive

## Debugging Steps

### 1. Check Browser Console
Open Developer Tools (F12) and look for:
```javascript
console.log('ICE gathering state:', pc.iceGatheringState);
// Should be: "new" → "gathering" → "complete"

console.log('ICE connection state:', pc.iceConnectionState);
// Should be: "new" → "checking" → "connected"

console.log('Connection state:', pc.connectionState);
// Should be: "new" → "connecting" → "connected"
```

### 2. Monitor ICE Candidates
```javascript
pc.addEventListener('icecandidate', event => {
    if (event.candidate) {
        console.log('ICE candidate:', event.candidate.type, event.candidate.candidate);
    } else {
        console.log('All ICE candidates gathered');
    }
});
```

For local connections, you should see:
- `host` candidates (local IP addresses)
- Possibly `srflx` candidates (if using STUN)

### 3. Check Offer SDP
```javascript
await pc.setLocalDescription(offer);
console.log('Offer SDP:', pc.localDescription.sdp);
```

The SDP should contain:
- `a=candidate:` lines (ICE candidates)
- `m=video` line (video media description)

### 4. Monitor All State Changes
```javascript
pc.addEventListener('icegatheringstatechange', () => {
    console.log('ICE gathering:', pc.iceGatheringState);
});

pc.addEventListener('iceconnectionstatechange', () => {
    console.log('ICE connection:', pc.iceConnectionState);
});

pc.addEventListener('connectionstatechange', () => {
    console.log('Connection:', pc.connectionState);
});

pc.addEventListener('signalingstatechange', () => {
    console.log('Signaling:', pc.signalingState);
});
```

### 5. Expected State Progression

**Normal flow:**
```
1. Create PC + addTransceiver
   ├─ signalingState: "stable"
   ├─ iceGatheringState: "new"
   ├─ iceConnectionState: "new"
   └─ connectionState: "new"

2. createOffer + setLocalDescription
   ├─ signalingState: "have-local-offer"
   ├─ iceGatheringState: "gathering" → "complete"
   ├─ iceConnectionState: "new"
   └─ connectionState: "new"

3. setRemoteDescription (answer)
   ├─ signalingState: "stable"
   ├─ iceGatheringState: "complete"
   ├─ iceConnectionState: "checking" → "connected"
   └─ connectionState: "connecting" → "connected"
```

## Server-Side Debugging

### Check if Worker is Running
```python
# In LiveViewController
@APIExport()
def getActiveStreams(self):
    # Returns list of active streams
    pass
```

Call: `GET /LiveViewController/getActiveStreams`

Expected response:
```json
{
    "status": "success",
    "active_streams": [
        {
            "detector": "Camera",
            "protocol": "webrtc"
        }
    ]
}
```

### Check Worker Frames
Add logging to `WebRTCStreamWorker._captureAndEmit`:
```python
def _captureAndEmit(self):
    frame = self._detector.getLatestFrame()
    if frame is None:
        self._logger.warning("No frame available!")
        return
    
    self._logger.debug(f"Frame captured: {frame.shape}, dtype: {frame.dtype}")
    # ... rest of method
```

### Check Video Track
```python
def get_video_track(self):
    if self._video_track is None:
        self._logger.info("Creating new video track")
        # ...
    else:
        self._logger.info("Reusing existing video track")
    return self._video_track
```

## Common Errors and Solutions

### 1. RuntimeError: No current event loop
**Error:**
```
RuntimeError: There is no current event loop in thread 'AnyIO worker thread'.
```

**Solution:** Already fixed in `webrtc_offer` - we create and set event loop

### 2. No video appears
**Possible causes:**
- Worker not running → Check `getActiveStreams`
- No frames in queue → Check detector is running
- Track not added → Check server logs

**Debug:**
```javascript
pc.addEventListener('track', evt => {
    console.log('Track received:', evt.track);
    console.log('Streams:', evt.streams);
    video.srcObject = evt.streams[0];
});
```

### 3. Connection fails
**Check:**
1. CORS headers (should be already configured in FastAPI)
2. Firewall settings
3. Server is running on correct port

## Testing Checklist

- [ ] Server running: `http://localhost:8001`
- [ ] Start stream: `POST /LiveViewController/startLiveView` with `protocol=webrtc`
- [ ] Check active: `GET /LiveViewController/getActiveStreams`
- [ ] Open HTML page: `http://localhost:8001/static/imswitch/webrtc_stream.html`
- [ ] Click "Start WebRTC Stream"
- [ ] Check browser console for state changes
- [ ] Verify video appears
- [ ] Check stats update (FPS, frames received)

## Performance Monitoring

### Client-side Stats
```javascript
setInterval(() => {
    pc.getStats(null).then(stats => {
        stats.forEach(report => {
            if (report.type === 'inbound-rtp' && report.kind === 'video') {
                console.log('FPS:', report.framesPerSecond);
                console.log('Frames dropped:', report.framesDropped);
                console.log('Jitter:', report.jitter);
            }
        });
    });
}, 1000);
```

### Expected Performance
- **Latency:** < 500ms (local network)
- **FPS:** Depends on detector and `throttle_ms` parameter
- **Frames dropped:** Should be 0 or very low
- **Bandwidth:** Varies with resolution and frame rate

## Useful Browser Tools

### Chrome
1. Open DevTools (F12)
2. Go to `chrome://webrtc-internals/`
3. See detailed WebRTC statistics and connection info

### Firefox
1. Open DevTools (F12)
2. Go to `about:webrtc`
3. See connection statistics

## Summary

✅ **Always add transceiver before createOffer**  
✅ **Add timeout to ICE gathering wait**  
✅ **Monitor all state changes in console**  
✅ **Check server logs for worker status**  
✅ **Use browser WebRTC internals for debugging**  
