# WebRTC Implementation Notes

## ICE Gathering and STUN Servers

### Do we need STUN servers for local connections?

**No, but it's good practice to include them.**

For **purely local connections** (same machine or same local network):
- WebRTC can establish direct peer-to-peer connections without STUN servers
- The browser can discover local IP addresses automatically
- ICE candidates will include `host` candidates (local IP addresses)

However, it's **recommended to keep STUN servers** configured because:
1. They don't hurt for local connections (they're just ignored if not needed)
2. They enable the system to work if you later want remote access
3. Google's STUN servers are free and reliable: `stun:stun.l.google.com:19302`

### Should the backend react to ICE gathering?

**No, the backend doesn't need to wait for ICE gathering completion.**

Here's how the WebRTC signaling flow works:

```
Client                          Server
------                          ------
1. Create offer
2. Gather ICE candidates
3. Wait for gathering complete
4. Send offer with all ICE    -->   Receive offer
   candidates embedded              Create answer
                                    Add video track
                              <--   Send answer
5. Receive answer
6. Connection established
```

**Important:** The client waits for ICE gathering to complete BEFORE sending the offer to the server. This means:
- The offer SDP already contains all ICE candidates
- The server doesn't need to handle separate ICE candidate messages
- The `webrtc_ice_candidate` endpoint is optional (only needed for trickle ICE)

### Client-side ICE gathering code

**Important:** The peer connection needs at least one transceiver/track to trigger ICE gathering!

```javascript
// Add a transceiver to receive video - this is REQUIRED!
pc.addTransceiver('video', { direction: 'recvonly' });

// Now create the offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

// Wait for ICE gathering to complete (with timeout for safety)
await new Promise((resolve) => {
    const timeout = setTimeout(() => {
        console.log('ICE gathering timeout, proceeding anyway');
        resolve(); // Proceed even if not complete
    }, 5000); // 5 second timeout
    
    if (pc.iceGatheringState === 'complete') {
        clearTimeout(timeout);
        resolve();
    } else {
        const checkState = () => {
            if (pc.iceGatheringState === 'complete') {
                clearTimeout(timeout);
                pc.removeEventListener('icegatheringstatechange', checkState);
                resolve();
            }
        };
        pc.addEventListener('icegatheringstatechange', checkState);
    }
});
```

**Why does ICE gathering get stuck?**

Without `addTransceiver`, the peer connection has no media tracks, so:
- No ICE candidates are generated
- `iceGatheringState` stays at `new` forever
- The promise never resolves

**Solution:** Always add a transceiver before creating the offer!

## The RuntimeError Fix

### Problem
```
RuntimeError: There is no current event loop in thread 'AnyIO worker thread'.
```

This happens because:
1. FastAPI runs the endpoint in a worker thread (AnyIO)
2. `aiortc` requires an asyncio event loop
3. Worker threads don't have event loops by default

### Solution
```python
# Create and set event loop in the thread
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    result = loop.run_until_complete(process_offer())
    return result
finally:
    # Proper cleanup
    loop.close()
    asyncio.set_event_loop(None)
```

The key changes:
1. Create `RTCPeerConnection` **inside** the async function (in the event loop context)
2. Set the event loop for the current thread with `asyncio.set_event_loop(loop)`
3. Properly clean up the event loop when done

## WebRTC Workflow

### 1. Start streaming
```bash
POST /LiveViewController/startLiveView
{
    "protocol": "webrtc",
    "detectorName": "DetectorName"  # optional
}
```

This creates the `WebRTCStreamWorker` which:
- Captures frames from the detector
- Puts them in a queue
- Creates a video track when requested

### 2. Establish WebRTC connection
```bash
POST /LiveViewController/webrtc_offer
{
    "sdp": "v=0\r\no=...",
    "type": "offer",
    "detectorName": "DetectorName"  # optional
}
```

Server responds with:
```json
{
    "status": "success",
    "sdp": "v=0\r\no=...",
    "type": "answer"
}
```

### 3. Client sets remote description
```javascript
await pc.setRemoteDescription(answer);
```

### 4. Connection established!
The video track starts flowing from server to client.

## Testing Locally

For local testing, you can use minimal configuration:

```javascript
const pc = new RTCPeerConnection({
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' }  // Optional for local
    ]
});
```

Or even:

```javascript
const pc = new RTCPeerConnection({
    iceServers: []  // Works for same-machine connections
});
```

## Debugging Tips

### Check ICE candidates
```javascript
pc.addEventListener('icecandidate', event => {
    if (event.candidate) {
        console.log('ICE candidate:', event.candidate);
        console.log('Type:', event.candidate.type);  // host, srflx, relay
    }
});
```

For local connections, you should see mostly `host` type candidates.

### Check connection state
```javascript
pc.addEventListener('connectionstatechange', () => {
    console.log('Connection state:', pc.connectionState);
});

pc.addEventListener('iceConnectionState', () => {
    console.log('ICE connection state:', pc.iceConnectionState);
});
```

Expected flow:
1. `new` → `checking` → `connected` → `completed`

### Common issues

1. **Stuck at "checking"**: ICE candidates aren't matching
   - Check firewall settings
   - Verify STUN server is reachable (if using one)
   - For local: ensure both endpoints on same network

2. **Connection failed**: 
   - Check browser console for errors
   - Verify server is sending correct SDP answer
   - Check that video track is properly added

3. **No video**: Connection works but no video appears
   - Check `DetectorVideoTrack.recv()` is being called
   - Verify frames are in the queue
   - Check video element has `autoplay` attribute

## Summary

✅ **ICE gathering**: Done by client, server receives complete offer  
✅ **STUN servers**: Optional for local, recommended to keep  
✅ **Event loop**: Fixed by creating loop in thread context  
✅ **Workflow**: startLiveView → webrtc_offer → connection established  

The backend is now properly configured for WebRTC streaming!
