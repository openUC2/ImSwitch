# LiveViewController Documentation

## Overview

The `LiveViewController` is a centralized controller for all live streaming functionality in ImSwitch. It provides a clean, extensible architecture for managing multiple streaming protocols with per-detector control.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LiveViewController                       │
├─────────────────────────────────────────────────────────────┤
│  Stream Management:                                         │
│  ├── setStreamParams(protocol, params)                      │
│  ├── getStreamParams(protocol)                              │
│                                                             │
│  Per-Detector Control:                                      │
│  ├── startLiveView(detectorName, protocol, params)         │
│  ├── stopLiveView(detectorName, protocol)                  │
│  └── getActiveStreams() -> List[Stream]                     │
│                                                             │
│  Worker Threads (one per active stream):                    │
│  ├── BinaryStreamWorker                                     │
│  ├── JPEGStreamWorker                                       │
│  ├── MJPEGStreamWorker                                      │
│  └── WebRTCStreamWorker                                     │
│                                                             │
│  Signals:                                                   │
│  ├── sigFrameReady(detectorName, protocol, data)            │
│  ├── sigStreamStarted(detectorName, protocol)               │
│  └── sigStreamStopped(detectorName, protocol)               │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    WebSocket API      HTTP Endpoints      WebRTC Peer
    (binary/JPEG)      (MJPEG feed)       (aiortc)
```

## Key Features

### 1. Unified Stream Configuration

All streaming parameters are managed through the `StreamParams` dataclass:

```python
@dataclass
class StreamParams:
    detector_name: Optional[str] = None
    protocol: str = "binary"
    
    # Binary stream parameters
    compression_algorithm: str = "lz4"
    compression_level: int = 0
    subsampling_factor: int = 4
    throttle_ms: int = 50
    
    # JPEG/MJPEG parameters
    jpeg_quality: int = 80
    
    # WebRTC parameters
    stun_servers: list = ["stun:stun.l.google.com:19302"]
    turn_servers: list = []
```

### 2. Protocol-Specific Workers

Each streaming protocol has a dedicated worker that runs in its own thread:

- **BinaryStreamWorker**: LZ4/Zstandard compressed raw frames
- **JPEGStreamWorker**: JPEG compressed frames
- **MJPEGStreamWorker**: Motion JPEG for HTTP streaming
- **WebRTCStreamWorker**: Real-time communication protocol (foundation)

### 3. Per-Detector Streaming

Each detector can have its own independent stream with different protocols:

```python
# Start binary stream for Camera1
liveViewController.startLiveView("Camera1", "binary")

# Start MJPEG stream for Camera2
liveViewController.startLiveView("Camera2", "mjpeg")
```

### 4. Headless Mode Optimization

In headless mode, the `DetectorsManager` LVWorker timer is not automatically started. Streaming is explicitly managed by `LiveViewController`, avoiding unnecessary resource consumption.

## API Reference

### Start Live Stream

```python
@APIExport()
def startLiveView(detectorName: Optional[str] = None, 
                  protocol: str = "binary", 
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
```

**Parameters:**
- `detectorName`: Name of detector (None = first available)
- `protocol`: Streaming protocol (binary, jpeg, mjpeg, webrtc)
- `params`: Optional parameters to override defaults

**Returns:** Dictionary with status and stream info

**Example:**
```python
result = api.liveview.startLiveView(
    detectorName="Camera",
    protocol="binary",
    params={
        "compression_algorithm": "lz4",
        "subsampling_factor": 4,
        "throttle_ms": 50
    }
)
```

### Stop Live Stream

```python
@APIExport()
def stopLiveView(detectorName: str, protocol: str) -> Dict[str, Any]
```

**Parameters:**
- `detectorName`: Name of detector
- `protocol`: Streaming protocol

**Returns:** Dictionary with status

**Example:**
```python
result = api.liveview.stopLiveView("Camera", "binary")
```

### Set Stream Parameters

```python
@APIExport()
def setStreamParams(protocol: str, params: Dict[str, Any]) -> Dict[str, Any]
```

**Parameters:**
- `protocol`: Streaming protocol
- `params`: Dictionary of parameters to set

**Returns:** Dictionary with status and updated params

**Example:**
```python
result = api.liveview.setStreamParams(
    protocol="binary",
    params={
        "compression_algorithm": "zstandard",
        "compression_level": 3,
        "subsampling_factor": 2
    }
)
```

### Get Stream Parameters

```python
@APIExport()
def getStreamParams(protocol: Optional[str] = None) -> Dict[str, Any]
```

**Parameters:**
- `protocol`: Optional protocol (None = all protocols)

**Returns:** Dictionary with streaming parameters

**Example:**
```python
# Get all protocols
all_params = api.liveview.getStreamParams()

# Get specific protocol
binary_params = api.liveview.getStreamParams(protocol="binary")
```

### Get Active Streams

```python
@APIExport()
def getActiveStreams() -> Dict[str, Any]
```

**Returns:** Dictionary with list of active streams

**Example:**
```python
active = api.liveview.getActiveStreams()
# Returns: {"status": "success", "active_streams": [{"detector": "Camera", "protocol": "binary"}]}
```

### MJPEG Video Feed

```python
@APIExport(runOnUIThread=False)
def video_feeder(startStream: bool = True, 
                 detectorName: Optional[str] = None)
```

HTTP endpoint for MJPEG streaming.

**Parameters:**
- `startStream`: Whether to start streaming
- `detectorName`: Name of detector (None = first available)

**Returns:** StreamingResponse with MJPEG data

**Example:**
```
GET http://localhost:8001/liveview/video_feeder?startStream=true
GET http://localhost:8001/liveview/video_feeder?startStream=true&detectorName=Camera
```

## Usage Examples

### Basic Binary Streaming

```python
# Start binary stream with default params
api.liveview.startLiveView(protocol="binary")

# Customize parameters
api.liveview.setStreamParams("binary", {
    "compression_algorithm": "lz4",
    "subsampling_factor": 4,
    "throttle_ms": 50
})

# Stop streaming
api.liveview.stopLiveView("Camera", "binary")
```

### MJPEG HTTP Streaming

```python
# Start MJPEG stream
api.liveview.startLiveView(protocol="mjpeg")

# Access via HTTP
# GET http://localhost:8001/liveview/video_feeder?startStream=true
```

### Multiple Detectors

```python
# Stream from multiple detectors simultaneously
api.liveview.startLiveView("Camera1", "binary")
api.liveview.startLiveView("Camera2", "jpeg")

# Check active streams
active = api.liveview.getActiveStreams()
```

## Backward Compatibility

The old API endpoints in `SettingsController` and `RecordingController` are maintained for backward compatibility:

```python
# These still work but delegate to LiveViewController
api.settings.setStreamParams(...)
api.settings.getStreamParams()
api.recording.video_feeder()
```

## Migration Guide

### From SettingsController

**Old:**
```python
api.settings.setStreamParams(
    compression={"algorithm": "lz4", "level": 0},
    subsampling={"factor": 4},
    throttle_ms=50
)
```

**New:**
```python
api.liveview.setStreamParams("binary", {
    "compression_algorithm": "lz4",
    "compression_level": 0,
    "subsampling_factor": 4,
    "throttle_ms": 50
})
```

### From RecordingController

**Old:**
```python
# MJPEG streaming was mixed with recording logic
```

**New:**
```python
# Explicit MJPEG streaming
api.liveview.startLiveView(protocol="mjpeg")
# Access at: GET /liveview/video_feeder?startStream=true
```

## Signals

The LiveViewController emits several signals that can be connected to:

```python
# Frame ready signal
sigFrameReady = Signal(str, str, bytes)  # (detectorName, protocol, frameData)

# Stream lifecycle signals
sigStreamStarted = Signal(str, str)  # (detectorName, protocol)
sigStreamStopped = Signal(str, str)  # (detectorName, protocol)
```

**Example:**
```python
def on_frame_ready(detector_name, protocol, frame_data):
    print(f"Frame from {detector_name} via {protocol}: {len(frame_data)} bytes")

liveViewController.sigFrameReady.connect(on_frame_ready)
```

## Performance Considerations

### Frame Polling

Each active stream has its own worker thread that polls frames at the configured rate (`throttle_ms`). This ensures:
- No blocking between different streams
- Independent frame rates per stream
- Efficient resource usage (threads only active when streaming)

### Frame Dropping

Workers implement different strategies for handling backpressure:
- **MJPEGStreamWorker**: Uses a queue with max size 10, drops oldest frames when full
- **BinaryStreamWorker**: Skips frames if processing takes too long
- **JPEGStreamWorker**: Similar to binary, processes latest available frame

### Headless Mode

In headless mode (`IS_HEADLESS=True`):
- LVWorker timer in DetectorsManager is not started automatically
- Streaming is explicitly managed by LiveViewController
- No unnecessary resource consumption when not streaming

## WebRTC Support (Future)

The `WebRTCStreamWorker` provides a foundation for WebRTC streaming. Future implementation will include:
- Signaling endpoints for offer/answer/ICE candidates
- STUN/TURN server configuration
- Custom video track implementation
- Browser compatibility testing

## Testing

Use the provided HTML test page to verify streaming functionality:

```bash
# Open the test page
firefox imswitch/imcontrol/controller/controllers/liveview_test.html

# Or serve it via Python
python -m http.server 8000
# Then open http://localhost:8000/liveview_test.html
```

## Troubleshooting

### Stream won't start

1. Check detector is available: `api.settings.getDetectorNames()`
2. Verify detector is started: `api.liveview.getActiveStreams()`
3. Check logs for errors

### MJPEG stream is choppy

1. Reduce frame rate: `api.liveview.setStreamParams("mjpeg", {"throttle_ms": 100})`
2. Reduce JPEG quality: `api.liveview.setStreamParams("mjpeg", {"jpeg_quality": 60})`
3. Check network bandwidth

### Binary stream is slow

1. Increase subsampling: `api.liveview.setStreamParams("binary", {"subsampling_factor": 8})`
2. Try different compression: `api.liveview.setStreamParams("binary", {"compression_algorithm": "zstandard"})`
3. Increase throttle: `api.liveview.setStreamParams("binary", {"throttle_ms": 100})`

## Future Enhancements

- [ ] WebSocket integration for binary/JPEG streaming
- [ ] WebRTC signaling server
- [ ] H.264/H.265 hardware encoding support
- [ ] Adaptive bitrate streaming
- [ ] Stream recording to file
- [ ] Multi-client support with individual stream parameters
