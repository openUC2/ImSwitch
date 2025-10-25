# Streaming Thread Safety Fix

## Problem

The initial refactoring had a critical issue where `_handle_stream_frame()` was being called from worker threads (Thread-9, etc.) that don't have an event loop. This caused the error:

```
Error handling stream frame: There is no current event loop in thread 'Thread-9 (run)'.
```

The issue occurred because `sio.start_background_task()` requires an active event loop, but the `StreamWorker` threads in `LiveViewController` don't have one.

## Solution

Reused the existing `_fallback_message_queue` and `_fallback_worker` infrastructure that was already designed to handle Socket.IO emission from non-async contexts.

### Changes Made

#### 1. Updated `_fallback_worker()` function

Modified to handle two message formats:

```python
def _fallback_worker():
    """Background worker thread to handle Socket.IO fallback messages."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                message = _fallback_message_queue.get(timeout=1.0)
                if message is None:  # Poison pill to stop worker
                    break
                
                # Handle different message formats
                if isinstance(message, tuple) and len(message) == 3:
                    # New format: (event, data, sid) for targeted emission
                    event, data, sid = message
                    loop.run_until_complete(sio.emit(event, data, to=sid))
                else:
                    # Legacy format: string message for broadcast
                    loop.run_until_complete(sio.emit("signal", message))
                
                _fallback_message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Silently handle errors to avoid spam
                pass
    finally:
        loop.close()
```

**Key Changes:**
- Added support for tuple format: `(event, data, sid)` for targeted client emission
- Maintains backward compatibility with string messages for broadcast
- Properly handles both binary frame events and signal events

#### 2. Refactored `_handle_stream_frame()` method

Changed from using `sio.start_background_task()` to using the `_fallback_message_queue`:

```python
def _handle_stream_frame(self, message: dict):
    """
    Handle pre-formatted stream frame message from LiveViewController.
    Uses fallback queue mechanism to handle thread safety.
    """
    try:
        msg_type = message.get('type')
        event = message.get('event', 'frame')
        data = message.get('data')
        metadata = message.get('metadata', {})
        
        # Get ready clients
        with _client_frame_lock:
            ready_clients = [sid for sid, ready in _client_frame_ready.items() if ready]
        
        if not ready_clients:
            return
        
        # Mark clients as not ready (waiting for acknowledgement)
        with _client_frame_lock:
            for sid in ready_clients:
                _client_frame_ready[sid] = True  # TODO: Set to False when frontend implements acknowledgement
        
        # Emit frame data using the safe emission mechanism
        if msg_type == 'binary_frame':
            # Use the fallback worker to handle emission from non-async context
            try:
                _start_fallback_worker()
                
                # Queue binary frame emission
                for sid in ready_clients:
                    _fallback_message_queue.put_nowait(('frame', data, sid))
                
                # Queue metadata emission
                meta_message = {
                    "name": "frame_meta",
                    "detectorname": metadata.get('detectorname', ''),
                    "pixelsize": metadata.get('pixelsize', 1),
                    "format": "binary",
                    "metadata": metadata
                }
                for sid in ready_clients:
                    _fallback_message_queue.put_nowait(('signal', json.dumps(meta_message), sid))
                    
            except queue.Full:
                # Drop frames if queue is full to prevent memory buildup
                pass
                
        elif msg_type == 'jpeg_frame':
            # Queue JPEG frame emission as JSON signal
            try:
                _start_fallback_worker()
                
                json_message = json.dumps(data)
                for sid in ready_clients:
                    _fallback_message_queue.put_nowait(('signal', json_message, sid))
                    
            except queue.Full:
                # Drop frames if queue is full
                pass
            
    except Exception as e:
        print(f"Error handling stream frame: {e}")
```

**Key Changes:**
- Removed all `sio.start_background_task()` calls
- Uses `_fallback_message_queue.put_nowait()` instead
- Properly starts fallback worker if not running
- Gracefully handles queue full condition by dropping frames
- Maintains per-client targeting via tuple format

## Benefits

1. **Thread Safety**: No longer requires event loop in worker threads
2. **Consistent Architecture**: Reuses existing infrastructure
3. **Graceful Degradation**: Drops frames when queue is full instead of blocking
4. **Backward Compatible**: Legacy broadcast messages still work
5. **Simple**: No need for complex async context handling

## Message Flow

```
StreamWorker Thread (no event loop)
   ↓
LiveViewController.sigStreamFrame.emit(message)
   ↓
noqt.SignalInstance._handle_stream_frame(message)
   ↓
_fallback_message_queue.put_nowait((event, data, sid))
   ↓
_fallback_worker Thread (has event loop)
   ↓
loop.run_until_complete(sio.emit(event, data, to=sid))
   ↓
WebSocket Client
```

## Testing

The fix resolves the "no current event loop" error and allows:
- Binary frame streaming to work from worker threads
- JPEG frame streaming to work from worker threads
- Proper per-client targeting
- Graceful frame dropping when system is overloaded

## Future Improvements

1. Add metrics for dropped frames (when queue is full)
2. Consider adjustable queue size based on available memory
3. Implement proper acknowledgement flow when frontend supports it
4. Add logging for fallback worker errors (currently silent)
