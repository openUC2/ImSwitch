# Unified Event Loop Design for ImSwitch Server

## Overview

This document describes the architectural design for merging the Socket.IO and FastAPI Uvicorn instances into a single server with a unified event loop. This design ensures thread-safe signal emission from device/camera threads to WebSocket clients.

## Problem Statement

Previously, ImSwitch ran two separate Uvicorn instances:
1. **FastAPI server** (port 8000) - REST API endpoints
2. **Socket.IO server** (port 8001) - WebSocket streaming

This caused several issues:
- Two separate event loops running in different threads
- `RuntimeError: There is no current event loop` when emitting signals from camera threads
- Complexity in managing two server lifecycles
- Potential port conflicts and resource duplication

## Solution Architecture

### 1. Single Unified Server

**Implementation**: Mount Socket.IO app onto the FastAPI app
```python
# In ImSwitchServer.py
from imswitch.imcommon.framework.noqt import get_socket_app

app = FastAPI()
socket_app = get_socket_app()
app.mount('/socket.io', socket_app)  # Socket.IO handles /socket.io/* paths
```

**Benefits**:
- Single port (8000)
- Single event loop
- Simplified deployment
- Better resource management

### 2. Shared Event Loop Management

**Event Loop Creation** (in `ServerThread`):
```python
class ServerThread(threading.Thread):
    def __init__(self):
        self._asyncio_loop = asyncio.new_event_loop()
        imswitch._asyncio_loop_imswitchserver = self._asyncio_loop
        
    def run(self):
        asyncio.set_event_loop(self._asyncio_loop)
        set_shared_event_loop(self._asyncio_loop)  # Share with noqt.py
        # ... run Uvicorn with this loop
```

**Event Loop Reference** (in `noqt.py`):
```python
_shared_event_loop = None  # Global reference

def set_shared_event_loop(loop):
    """Called by ImSwitchServer to set the shared loop"""
    global _shared_event_loop
    _shared_event_loop = loop
```

### 3. Thread-Safe Signal Emission

**Key Pattern**: Use `asyncio.run_coroutine_threadsafe()` for cross-thread calls

#### From Device/Camera Threads → Event Loop

Camera/device threads emit signals that need to reach WebSocket clients:

```python
# In noqt.py - SignalInstance._safe_broadcast_message()
def _safe_broadcast_message(self, mMessage: dict) -> None:
    message_json = json.dumps(mMessage)
    
    if _shared_event_loop and _shared_event_loop.is_running():
        async def emit_signal():
            await sio.emit("signal", message_json)
        
        # Thread-safe: schedule coroutine from any thread
        asyncio.run_coroutine_threadsafe(emit_signal(), _shared_event_loop)
```

**How it works**:
1. Signal emitted from camera thread (e.g., Thread-16)
2. `run_coroutine_threadsafe()` safely schedules the coroutine in the event loop
3. Event loop (running in ServerThread) executes the emit when ready
4. Socket.IO sends data to connected clients

#### Binary Frame Streaming

For high-performance frame streaming with backpressure:

```python
# In noqt.py - SignalInstance._handle_stream_frame()
def _handle_stream_frame(self, message: dict):
    # ... check ready clients ...
    
    if msg_type == 'binary_frame':
        frame_payload = [metadata, data]
        
        async def emit_binary_frame():
            for sid in ready_clients:
                await sio.emit(event, frame_payload, to=sid)
        
        # Thread-safe emission from camera thread
        asyncio.run_coroutine_threadsafe(emit_binary_frame(), _shared_event_loop)
```

### 4. Flow Control and Backpressure

**Client Acknowledgment Pattern**:
```python
# Backend (noqt.py)
@sio.event
async def frame_ack(sid):
    """Client acknowledges frame processing complete"""
    with _client_frame_lock:
        _client_frame_ready[sid] = True

# Frontend (React)
socket.on('frame', (data) => {
    processFrame(data);
    socket.emit('frame_ack');  // Signal ready for next frame
});
```

**Frame Dropping**:
- Only send frames to clients marked as "ready"
- Drop frames if no clients are ready (backpressure)
- Log every 10th dropped frame
- Reset clients to ready state after threshold

## Thread Ownership Model

### Thread Roles

1. **Main Thread**: Application startup, UI initialization
2. **ServerThread**: Runs asyncio event loop, handles all async I/O (HTTP, WebSocket)
3. **Camera/Device Threads**: Capture frames, emit signals

### Ownership Rules

| Resource | Owner Thread | Access Pattern |
|----------|-------------|----------------|
| Event Loop | ServerThread | Created and run in ServerThread |
| Socket.IO Server | ServerThread | Runs in event loop |
| FastAPI App | ServerThread | Runs in event loop |
| Signal Emission | Camera Thread | Uses `run_coroutine_threadsafe()` |
| Frame Buffer | Camera Thread | Producer (camera writes) |
| WebSocket Send | ServerThread | Consumer (event loop sends) |

### Critical Section: Signal Emission

```
Camera Thread                    ServerThread (Event Loop)
-------------                    --------------------------
1. Capture frame
2. Create signal data
3. Call signal.emit()
4. _safe_broadcast_message()
5. run_coroutine_threadsafe() ──> Schedule in loop
                                6. Loop picks up task
                                7. await sio.emit()
                                8. Send to WebSocket
```

## API Changes

### Deprecated Functions (noqt.py)

```python
# DEPRECATED: No longer starts separate server
def run_uvicorn():
    """Socket.IO is now mounted on FastAPI app."""
    pass

def start_websocket_server():
    """Socket.IO server integrated with FastAPI."""
    pass
```

### New Functions (noqt.py)

```python
def set_shared_event_loop(loop):
    """Set the shared event loop for thread-safe signal emission."""
    
def get_socket_app():
    """Returns the Socket.IO ASGI app to be mounted on FastAPI."""
```

## Configuration Changes

### Client Connection URLs

**Before**:
- FastAPI: `http://localhost:8000`
- Socket.IO: `http://localhost:8001`

**After**:
- All services: `http://localhost:8000`
- Socket.IO connects to: `http://localhost:8000/socket.io`

### Frontend Configuration

Update Socket.IO client initialization:
```javascript
// Before
const socket = io('http://localhost:8001');

// After
const socket = io('http://localhost:8000', {
    path: '/socket.io'  // Explicitly specify path (default anyway)
});
```

## Benefits of This Design

### 1. Thread Safety
- No more `RuntimeError: There is no current event loop`
- Proper asyncio patterns throughout
- Safe cross-thread communication

### 2. Performance
- Single event loop reduces context switching
- Better CPU cache utilization
- Reduced memory overhead

### 3. Maintainability
- Clear ownership model
- Single server lifecycle
- Easier debugging (single event loop to inspect)

### 4. Reliability
- Proper backpressure handling
- Graceful frame dropping
- No race conditions

## Testing Considerations

### Unit Tests
- Mock `_shared_event_loop` in tests
- Test signal emission without real event loop
- Verify thread-safe patterns

### Integration Tests
- Test frame streaming under load
- Verify backpressure mechanism
- Test multi-client scenarios

### Performance Tests
- Measure frame drop rate
- Monitor event loop latency
- Profile cross-thread calls

## Migration Path

### For Developers

1. **No code changes needed** for API endpoints (FastAPI routes)
2. **No code changes needed** for signal emission (handled transparently)
3. **Update frontend** Socket.IO connection URL from port 8001 to 8000

### For Deployment

1. **Remove** port 8001 from firewall rules (only need 8000)
2. **Update** Docker compose files to expose only port 8000
3. **Update** reverse proxy configs (nginx, etc.)

## Troubleshooting

### Event Loop Not Available
```
Warning: Event loop not available for signal emission
```
**Cause**: Signal emitted before `set_shared_event_loop()` called  
**Solution**: Ensure ImSwitchServer starts before camera initialization

### RuntimeError in Emit
```
RuntimeError: There is no current event loop in thread 'Thread-16'
```
**Cause**: Using `sio.start_background_task()` instead of `run_coroutine_threadsafe()`  
**Solution**: Already fixed in new design

### Dropped Frames
```
Dropped 100 frames due to client backpressure
```
**Cause**: Frontend not sending `frame_ack` events fast enough  
**Solution**: Optimize frontend rendering, reduce frame rate, or improve client performance

## Future Enhancements

1. **Adaptive Frame Rate**: Automatically adjust based on drop rate
2. **Per-Client Buffers**: Small buffer per client for smoother playback
3. **Priority Queue**: Prioritize control signals over frames
4. **Metrics Dashboard**: Monitor event loop health, drop rates, latency

## References

- [asyncio - Thread Safety](https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading)
- [Socket.IO ASGI App](https://python-socketio.readthedocs.io/en/latest/server.html#asgi-applications)
- [FastAPI Mounting](https://fastapi.tiangolo.com/advanced/sub-applications/)
- [PEP 567 - Context Variables](https://www.python.org/dev/peps/pep-0567/)
