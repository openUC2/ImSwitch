# LiveStream API Architecture

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐      ┌────────────────────┐             │
│  │  StreamControls  │      │ StreamControlOver- │             │
│  │   (Play/Stop)    │      │   lay (Settings)   │             │
│  └────────┬─────────┘      └─────────┬──────────┘             │
│           │                           │                         │
│           │    ┌──────────────────────┘                        │
│           │    │                                                │
│  ┌────────▼────▼──────────┐    ┌────────────────┐             │
│  │      LiveView.js       │    │ StreamSettings │             │
│  │  (Main Component)      │    │   .js          │             │
│  └────────────┬───────────┘    └────────┬───────┘             │
│               │                         │                      │
└───────────────┼─────────────────────────┼──────────────────────┘
                │                         │
                │   Redux State           │
                │   (LiveStreamSlice)     │
                │                         │
┌───────────────▼─────────────────────────▼──────────────────────┐
│                    API LAYER (backendapi/)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Stream Control APIs                                     │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • apiLiveViewControllerStartLiveView()                  │  │
│  │    → POST /LiveViewController/startLiveView              │  │
│  │                                                           │  │
│  │  • apiLiveViewControllerStopLiveView()                   │  │
│  │    → GET /LiveViewController/stopLiveView                │  │
│  │                                                           │  │
│  │  • apiViewControllerGetLiveViewActive()                  │  │
│  │    → GET /LiveViewController/getLiveViewActive           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Parameter Management APIs                               │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • apiLiveViewControllerGetStreamParameters()            │  │
│  │    → GET /LiveViewController/getStreamParameters         │  │
│  │                                                           │  │
│  │  • apiLiveViewControllerSetStreamParameters()            │  │
│  │    → GET /LiveViewController/setStreamParameters         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Monitoring APIs                                         │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • apiLiveViewControllerGetActiveStreams()               │  │
│  │    → GET /LiveViewController/getActiveStreams            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP/HTTPS
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    BACKEND (ImSwitch)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           LiveViewController                             │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                           │  │
│  │  • Manages stream lifecycle (start/stop)                 │  │
│  │  • Stores global stream parameters per protocol          │  │
│  │  • Supports multiple detectors & protocols               │  │
│  │  • Handles parameter validation                          │  │
│  │                                                           │  │
│  └────────┬─────────────────────────────────────────────────┘  │
│           │                                                     │
│  ┌────────▼─────────────────────────────────────────────────┐  │
│  │        Stream Implementations                            │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • Binary Stream (LZ4/Zstandard compression)             │  │
│  │  • JPEG Stream (JPEG compression)                        │  │
│  │  • MJPEG Stream (Motion JPEG)                            │  │
│  │  • WebRTC Stream (Real-time communication)               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Starting a Stream

```
User clicks "Play"
       │
       ▼
LiveView.toggleStream()
       │
       ├─► Determine protocol (binary/jpeg)
       │   from liveStreamState.currentImageFormat
       │
       ▼
apiLiveViewControllerStartLiveView(null, protocol)
       │
       ▼
POST /LiveViewController/startLiveView?protocol=binary
       │
       ▼
Backend starts stream with global params
       │
       ▼
Redux state updated (isStreamRunning = true)
       │
       ▼
UI updates (Play → Stop button)
```

### Changing Parameters

```
User opens StreamControlOverlay
       │
       ▼
Load settings on mount
       │
       ▼
apiLiveViewControllerGetStreamParameters()
       │
       ▼
GET /LiveViewController/getStreamParameters
       │
       ▼
Transform backend response → frontend format
       │
       ▼
Display in UI (sliders, dropdowns, etc.)
       │
       ▼
User modifies settings
       │
       ▼
User clicks "Submit"
       │
       ▼
Determine protocol (binary/jpeg)
       │
       ▼
apiLiveViewControllerSetStreamParameters(protocol, params)
       │
       ▼
GET /LiveViewController/setStreamParameters?protocol=X
body: { param1: value1, param2: value2, ... }
       │
       ▼
Backend validates & stores params
       │
       ▼
Redux state updated
       │
       ▼
Success message shown to user
```

## Protocol Parameter Mapping

### Binary Protocol
```javascript
Frontend Format:
{
  binary: {
    enabled: true,
    compression: {
      algorithm: 'lz4',    // 'lz4', 'zstd', 'none'
      level: 0             // 0-9
    },
    subsampling: {
      factor: 4            // 1-8
    },
    throttle_ms: 50        // 16-1000
  }
}

Backend Format:
{
  compression_algorithm: 'lz4',
  compression_level: 0,
  subsampling_factor: 4,
  throttle_ms: 50
}
```

### JPEG Protocol
```javascript
Frontend Format:
{
  jpeg: {
    enabled: true,
    quality: 80,           // 1-100
    subsampling: {
      factor: 1            // 1-8
    },
    throttle_ms: 100       // 16-1000
  }
}

Backend Format:
{
  jpeg_quality: 80,
  subsampling_factor: 1,
  throttle_ms: 100
}
```

## State Management

### Redux Store (LiveStreamSlice)
```javascript
{
  // Current format determines protocol
  currentImageFormat: 'binary',  // or 'jpeg'
  
  // Stream state
  isStreamRunning: false,
  
  // Settings (frontend format)
  streamSettings: {
    binary: { ... },
    jpeg: { ... }
  },
  
  // Backend capabilities
  backendCapabilities: {
    binaryStreaming: true,
    webglSupported: true
  },
  
  // Legacy mode flag
  isLegacyBackend: false,
  
  // Display settings
  minVal: 0,
  maxVal: 255,
  gamma: 1.0
}
```

## Error Handling

```
API Call
   │
   ├─► Success
   │   └─► Update Redux state
   │       └─► Update UI
   │
   └─► Error
       │
       ├─► 404 Not Found
       │   └─► Legacy backend detected
       │       └─► Fallback to JPEG-only mode
       │
       ├─► Timeout
       │   └─► Auto-retry (up to 3 times)
       │       └─► Show retry button
       │
       └─► Other Error
           └─► Show error message
               └─► Keep current state
```

## Component Responsibilities

### LiveView.js
- Main component coordinator
- Handles stream start/stop
- Manages recording
- Coordinates child components

### StreamControlOverlay.js
- Advanced settings panel (overlay)
- Parameter configuration
- Format switching
- Live preview of settings

### StreamSettings.js
- Alternative settings panel
- Similar to StreamControlOverlay
- Different UI presentation

### StreamControls.js
- Simple Play/Stop/Snap controls
- Polls backend for stream status
- Triggers stream toggle in LiveView

## API Function Structure

All new API functions follow this pattern:

```javascript
// 1. Import axios instance creator
import createAxiosInstance from "./createAxiosInstance";

// 2. Define async function with clear parameters
const apiFunctionName = async (param1, param2 = defaultValue) => {
  try {
    // 3. Create axios instance
    const axiosInstance = createAxiosInstance();
    
    // 4. Build URL and params
    const url = `/Controller/endpoint`;
    const params = { ... };
    
    // 5. Make request
    const response = await axiosInstance.method(url, params);
    
    // 6. Return data
    return response.data;
  } catch (error) {
    // 7. Log and rethrow
    console.error('Error description:', error);
    throw error;
  }
};

// 8. Export default
export default apiFunctionName;

// 9. Include usage examples in comments
/**
 * Example usage:
 * const result = await apiFunctionName(param1, param2);
 */
```

## Future Architecture Enhancements

### Planned
- [ ] Multi-detector UI support
- [ ] MJPEG stream integration
- [ ] WebRTC stream integration
- [ ] Real-time active stream monitoring
- [ ] Stream quality metrics display

### Considerations
- [ ] Stream reconnection handling
- [ ] Bandwidth adaptation
- [ ] Parameter presets/profiles
- [ ] Stream recording to different formats
- [ ] Multi-protocol simultaneous streaming
