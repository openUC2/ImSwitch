# LiveViewController Protocol Tracking API

## New API Endpoints for Protocol Management

### 1. `getCurrentStreamProtocol(detectorName=None)`
Returns the current streaming protocol for a specific detector.

**Parameters:**
- `detectorName` (optional): Name of detector. If None, returns first active stream or first available detector.

**Returns:**
```json
{
    "status": "success",
    "detector": "camera1",
    "protocol": "binary",
    "is_streaming": true,
    "params": {
        "compression_algorithm": "lz4",
        "compression_level": 0,
        "subsampling_factor": 4,
        "throttle_ms": 50
    }
}
```

**Example Usage:**
```python
# Get current protocol for specific detector
result = liveViewController.getCurrentStreamProtocol("camera1")

# Get current protocol for first active detector
result = liveViewController.getCurrentStreamProtocol()
```

### 2. `getStreamStatus()`
Returns comprehensive streaming status for all detectors.

**Returns:**
```json
{
    "status": "success",
    "total_detectors": 2,
    "active_streams": 1,
    "detectors": {
        "camera1": {
            "is_streaming": true,
            "protocol": "binary",
            "params": {...}
        },
        "camera2": {
            "is_streaming": false,
            "protocol": null,
            "params": {}
        }
    },
    "available_protocols": ["binary", "jpeg", "mjpeg", "webrtc"]
}
```

### 3. Enhanced `getStreamParameters(protocol=None, detectorName=None)`
Now includes active protocol tracking and detector-specific information.

**New Response Format:**
```json
{
    "status": "success",
    "current_active_protocols": {
        "camera1": "binary",
        "camera2": "jpeg"
    },
    "current_protocol": "binary",
    "total_active_streams": 2,
    "protocols": {
        "binary": {
            "protocol": "binary",
            "compression_algorithm": "lz4",
            "compression_level": 0,
            "subsampling_factor": 4,
            "throttle_ms": 50,
            "active_detectors": ["camera1"],
            "is_active": true
        },
        "jpeg": {
            "protocol": "jpeg",
            "jpeg_quality": 80,
            "subsampling_factor": 4,
            "throttle_ms": 50,
            "active_detectors": ["camera2"],
            "is_active": true
        }
    }
}
```

### 4. Fixed `setStreamParameters(protocol, params)`
Corrected logic to only restart streams with the **same** protocol when parameters change.

**Before (Bug):**
```python
# This was wrong - restarted streams with DIFFERENT protocols
if active_protocol != protocol:
    detectors_to_restart.append(detector_name)
```

**After (Fixed):**
```python
# Now correctly restarts only streams with SAME protocol
if active_protocol == protocol:
    detectors_to_restart.append(detector_name)
```

## Usage Examples

### Check if detector is streaming and what protocol
```python
status = liveViewController.getCurrentStreamProtocol("camera1")
if status["is_streaming"]:
    print(f"Camera1 is streaming using {status['protocol']} protocol")
else:
    print("Camera1 is not streaming")
```

### Get overview of all streaming activity
```python
overview = liveViewController.getStreamStatus()
print(f"Active streams: {overview['active_streams']}/{overview['total_detectors']}")

for detector, info in overview["detectors"].items():
    if info["is_streaming"]:
        print(f"{detector}: {info['protocol']}")
    else:
        print(f"{detector}: not streaming")
```

### Get parameters with active status
```python
params = liveViewController.getStreamParameters()
print("Currently active protocols:", params["current_active_protocols"])

# Check which detectors use binary protocol
binary_info = params["protocols"]["binary"]
print(f"Binary protocol active on: {binary_info['active_detectors']}")
print(f"Binary protocol is active: {binary_info['is_active']}")
```

### Safe parameter updates
```python
# This will now only restart detectors using 'binary' protocol
result = liveViewController.setStreamParameters("binary", {
    "compression_algorithm": "zstandard",
    "subsampling_factor": 2
})

print(f"Restarted detectors: {result.get('restarted_detectors', [])}")
```

## Benefits

1. **Protocol Tracking**: Always know which protocol each detector is using
2. **Status Overview**: Get complete picture of streaming activity across all detectors
3. **Safe Updates**: Parameter changes only affect streams using that protocol
4. **Detailed Information**: Each protocol shows which detectors are actively using it
5. **Backward Compatibility**: Existing API calls still work, with enhanced information

## API Endpoints Summary

| Method | Purpose | Returns Current Protocol |
|--------|---------|-------------------------|
| `getCurrentStreamProtocol()` | Get protocol for specific detector | ✅ |
| `getStreamStatus()` | Complete streaming overview | ✅ |
| `getStreamParameters()` | Parameters with active status | ✅ |
| `getActiveStreams()` | List active streams | ✅ |
| `setStreamParameters()` | Update params (fixed logic) | ✅ |
| `startLiveView()` | Start stream | ✅ |
| `stopLiveView()` | Stop stream | ✅ |