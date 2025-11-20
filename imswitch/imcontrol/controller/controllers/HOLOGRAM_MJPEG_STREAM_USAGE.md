# Hologram MJPEG Streaming Usage Guide

## Overview

Both `InLineHoloController` and `OffAxisHoloController` now support real-time MJPEG streaming of reconstructed hologram results via HTTP. This allows you to visualize the processed holograms directly in a web browser or any MJPEG-compatible client.

## Features

- **Real-time streaming**: Reconstructed holograms are encoded as JPEG and streamed via HTTP
- **Configurable quality**: Adjust JPEG compression quality (0-100)
- **Automatic normalization**: Float arrays are automatically normalized to 8-bit for display
- **Low overhead**: Uses frame queue to prevent blocking processing thread
- **Standards-compliant**: MJPEG multipart stream compatible with all browsers

## API Endpoints

### InLineHoloController

**Start Stream:**
```
GET /holocontroller/mjpeg_stream?startStream=true&jpeg_quality=85
```

**Stop Stream:**
```
GET /holocontroller/mjpeg_stream?startStream=false
```

### OffAxisHoloController

**Start Stream:**
```
GET /offaxisholocontroller/mjpeg_stream?startStream=true&jpeg_quality=90
```

**Stop Stream:**
```
GET /offaxisholocontroller/mjpeg_stream?startStream=false
```

## Parameters

- `startStream` (bool): Whether to start (true) or stop (false) streaming. Default: `true`
- `jpeg_quality` (int): JPEG compression quality (0-100). Default: `85`
  - Higher values = better quality but larger file size
  - Recommended range: 70-95
  - 85 is a good balance for most applications

## Usage Examples

### 1. HTML Viewer

Create a simple HTML page to view the hologram stream:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Hologram Viewer</title>
    <style>
        body { 
            margin: 0; 
            background: #000; 
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        img { 
            max-width: 100%; 
            max-height: 100vh;
            border: 2px solid #fff;
        }
    </style>
</head>
<body>
    <img src="http://localhost:8001/holocontroller/mjpeg_stream?startStream=true&jpeg_quality=85" 
         alt="Hologram Stream">
</body>
</html>
```

### 2. Python Client with OpenCV

```python
import cv2
import numpy as np

# Open MJPEG stream
stream_url = "http://localhost:8001/holocontroller/mjpeg_stream?startStream=true&jpeg_quality=85"
cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display the frame
    cv2.imshow('Hologram Stream', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Stop the stream
import requests
requests.get("http://localhost:8001/holocontroller/mjpeg_stream?startStream=false")
```

### 3. JavaScript/Fetch API

```javascript
// Start streaming and display in an img element
const img = document.getElementById('hologram-stream');
img.src = 'http://localhost:8001/holocontroller/mjpeg_stream?startStream=true&jpeg_quality=85';

// Stop streaming when needed
async function stopStream() {
    await fetch('http://localhost:8001/holocontroller/mjpeg_stream?startStream=false');
    img.src = '';
}
```

### 4. cURL Command Line

```bash
# Start streaming (will output MJPEG data to terminal - not very useful)
curl "http://localhost:8001/holocontroller/mjpeg_stream?startStream=true&jpeg_quality=85"

# Better: Save to file
curl "http://localhost:8001/holocontroller/mjpeg_stream?startStream=true&jpeg_quality=85" > stream.mjpeg

# Stop streaming
curl "http://localhost:8001/holocontroller/mjpeg_stream?startStream=false"
```

## Workflow

1. **Start hologram processing** (if not already running):
   ```
   GET /holocontroller/start_processing
   ```

2. **Start MJPEG stream**:
   ```
   GET /holocontroller/mjpeg_stream?startStream=true&jpeg_quality=85
   ```

3. **View the stream** in your browser or client application

4. **Adjust parameters** while streaming (in separate requests):
   ```
   POST /holocontroller/set_parameters
   {
     "dz": 0.005,
     "wavelength": 488e-9,
     "roi_size": 512
   }
   ```

5. **Stop stream** when done:
   ```
   GET /holocontroller/mjpeg_stream?startStream=false
   ```

## State Information

The streaming state is reflected in the controller state:

```bash
curl http://localhost:8001/holocontroller/get_state
```

Response:
```json
{
  "is_processing": true,
  "is_paused": false,
  "is_streaming": true,
  "last_process_time": 1732108456.789,
  "frame_count": 1234,
  "processed_count": 1234
}
```

## Performance Considerations

### Frame Queue
- Internal queue size: 10 frames
- If processing is faster than network can send, frames are dropped (newest kept)
- If processing is slower than network, client may experience delays

### JPEG Quality vs Performance
| Quality | File Size | Encoding Time | Recommended Use |
|---------|-----------|---------------|-----------------|
| 50-70   | Smallest  | Fastest       | High framerate preview |
| 75-85   | Medium    | Moderate      | General use (default) |
| 90-95   | Large     | Slower        | High quality recording |
| 96-100  | Largest   | Slowest       | Not recommended for streaming |

### Network Bandwidth
Approximate bandwidth requirements (for 512×512 images):
- Quality 70: ~10-20 KB/frame → 0.1-0.2 Mbps @ 10 fps
- Quality 85: ~20-40 KB/frame → 0.2-0.4 Mbps @ 10 fps
- Quality 95: ~40-80 KB/frame → 0.4-0.8 Mbps @ 10 fps

## Troubleshooting

### No frames appearing
1. Check that processing is started: `GET /holocontroller/get_state`
2. Verify camera is providing frames
3. Check logs for encoding errors

### Low framerate
1. Reduce JPEG quality (e.g., 70)
2. Decrease processing update frequency: `set_parameters({"update_freq": 5.0})`
3. Reduce ROI size for faster processing

### Stream disconnects
- Client timeout - normal behavior when no frames for >1 second
- Restart stream by refreshing browser or reconnecting client

### opencv-python not available
Install with:
```bash
pip install opencv-python
```

## Dependencies

- **opencv-python** (cv2): Required for JPEG encoding
- **FastAPI**: Required for HTTP streaming (usually installed with ImSwitch)
- **NanoImagingPack**: Required for hologram processing

## Implementation Details

### Architecture
```
Camera → getLatestFrame() → _process_inline/offaxis() → _add_to_mjpeg_stream()
                                                              ↓
                                                        MJPEG Queue (10 frames)
                                                              ↓
                                                      frame_generator()
                                                              ↓
                                                      StreamingResponse
                                                              ↓
                                                         HTTP Client
```

### Thread Safety
- Processing runs in background thread
- MJPEG queue is thread-safe (queue.Queue)
- State updates are protected by threading.Lock()

### Frame Format
Each MJPEG frame:
```
--frame\r\n
Content-Type: image/jpeg\r\n
Content-Length: <bytes>\r\n
\r\n
<JPEG data>
\r\n
```

## Comparison with PixelCalibrationController

The hologram controllers use a similar approach to the observation camera in PixelCalibrationController, but with key differences:

| Feature | PixelCalibrationController | HoloController |
|---------|---------------------------|----------------|
| Image Source | Raw camera frame | Reconstructed hologram |
| Method | Single PNG via `returnObservationCameraImage()` | MJPEG stream via `mjpeg_stream()` |
| Use Case | Static snapshot | Continuous real-time preview |
| Format | PNG (single request) | MJPEG (streaming) |

## License

Copyright (C) 2020-2024 ImSwitch developers
Licensed under GNU GPL v3 or later.
