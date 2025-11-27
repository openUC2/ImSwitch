# Hologram Processing API Quick Reference

## Base URL
```
http://localhost:8001/holocontroller
```

## Endpoints

### Get Current Parameters
```bash
GET /holocontroller/get_parameters_inlineholo
```

**Response:**
```json
{
  "pixelsize": 3.45e-6,
  "wavelength": 488e-9,
  "na": 0.3,
  "dz": 0.005,
  "roi_center": [512, 512],
  "roi_size": 256,
  "color_channel": "green",
  "flip_x": false,
  "flip_y": false,
  "rotation": 0,
  "update_freq": 10.0,
  "binning": 1,
  "use_scipy_fft": true,
  "fft_workers": 4,
  "use_multiprocessing": false,
  "use_float32": true,
  "enable_benchmarking": false
}
```

### Update Parameters
```bash
POST /holocontroller/set_parameters_inlineholo
Content-Type: application/json

{
  "dz": 0.005,
  "wavelength": 488e-9,
  "roi_size": 256,
  "use_scipy_fft": true,
  "fft_workers": 4,
  "use_float32": true,
  "enable_benchmarking": true
}
```

### Get Processing State
```bash
GET /holocontroller/get_state_inlineholo
```

**Response:**
```json
{
  "is_processing": true,
  "is_paused": false,
  "is_streaming": false,
  "last_process_time": 1234567890.123,
  "frame_count": 1523,
  "processed_count": 1521,
  "dropped_frames": 2,
  "capture_fps": 25.3,
  "processing_fps": 24.8,
  "avg_process_time": 0.0382
}
```

### Start Processing
```bash
GET /holocontroller/start_processing_inlineholo
```

Starts hologram reconstruction with current parameters.

### Stop Processing
```bash
GET /holocontroller/stop_processing_inlineholo
```

Stops all processing and cleanup threads/processes.

### Pause Processing
```bash
GET /holocontroller/pause_processing_inlineholo
```

Pauses processing - continues reconstructing last frame at `update_freq`.

### Resume Processing
```bash
GET /holocontroller/resume_processing_inlineholo
```

Resumes normal processing from camera frames.

### MJPEG Stream
```bash
GET /holocontroller/mjpeg_stream_inlineholo?startStream=true&jpeg_quality=85
```

Returns MJPEG stream of reconstructed holograms.

**Parameters:**
- `startStream`: `true` to start, `false` to stop (default: `true`)
- `jpeg_quality`: JPEG quality 0-100 (default: 85)

**Usage in browser:**
```html
<img src="http://localhost:8001/holocontroller/mjpeg_stream_inlineholo?startStream=true" />
```

## Quick Examples

### Python

```python
import requests

base = "http://localhost:8001/holocontroller"

# Configure for Pi 5
requests.post(f"{base}/set_parameters_inlineholo", json={
    "use_scipy_fft": True,
    "fft_workers": 4,
    "use_float32": True,
    "enable_benchmarking": True,
    "dz": 0.005,
    "roi_size": 256,
    "update_freq": 20.0
})

# Start
requests.get(f"{base}/start_processing_inlineholo")

# Check state
state = requests.get(f"{base}/get_state_inlineholo").json()
print(f"FPS: {state['processing_fps']:.1f}")

# Stop
requests.get(f"{base}/stop_processing_inlineholo")
```

### JavaScript

```javascript
const base = 'http://localhost:8001/holocontroller';

// Set parameters
await fetch(`${base}/set_parameters_inlineholo`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    dz: 0.005,
    roi_size: 256,
    use_scipy_fft: true,
    fft_workers: 4
  })
});

// Start processing
await fetch(`${base}/start_processing_inlineholo`);

// Monitor state
setInterval(async () => {
  const state = await fetch(`${base}/get_state_inlineholo`).then(r => r.json());
  console.log(`Processing FPS: ${state.processing_fps.toFixed(1)}`);
}, 1000);
```

### curl

```bash
# Start optimized processing
curl -X POST http://localhost:8001/holocontroller/set_parameters_inlineholo \
  -H "Content-Type: application/json" \
  -d '{"use_scipy_fft":true,"fft_workers":4,"use_float32":true,"dz":0.005}'

curl http://localhost:8001/holocontroller/start_processing_inlineholo

# Check state
curl http://localhost:8001/holocontroller/get_state_inlineholo | jq .

# Stop
curl http://localhost:8001/holocontroller/stop_processing_inlineholo
```

## Performance Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_scipy_fft` | bool | true | Use SciPy FFT with multi-core support |
| `fft_workers` | int | 4 | Number of FFT worker threads (set to CPU count) |
| `use_multiprocessing` | bool | false | Use separate process for processing (bypass GIL) |
| `use_float32` | bool | true | Use float32 instead of float64 (faster, less memory) |
| `enable_benchmarking` | bool | false | Log performance metrics every second |
| `update_freq` | float | 10.0 | Target processing rate in Hz |

## Recommended Settings

### Raspberry Pi 5
```json
{
  "use_scipy_fft": true,
  "fft_workers": 4,
  "use_multiprocessing": false,
  "use_float32": true,
  "roi_size": 256,
  "update_freq": 20.0
}
```

### High-end Desktop (8+ cores)
```json
{
  "use_scipy_fft": true,
  "fft_workers": 8,
  "use_multiprocessing": false,
  "use_float32": true,
  "roi_size": 512,
  "update_freq": 30.0
}
```

### Low-end Hardware
```json
{
  "use_scipy_fft": false,
  "use_float32": true,
  "roi_size": 128,
  "binning": 2,
  "update_freq": 5.0
}
```
