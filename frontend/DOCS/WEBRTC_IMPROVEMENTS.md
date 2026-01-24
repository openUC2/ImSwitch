# WebRTC Streaming Improvements

## Overview
This update significantly improves the WebRTC streaming functionality in ImSwitch by:

1. **Adding comprehensive parameter controls to the frontend**
2. **Fixing video distortion issues by preserving aspect ratio**  
3. **Optimizing streaming performance for faster connection and reduced latency**
4. **Integrating parameter handling between frontend and backend**

## Changes Made

### Frontend (webrtc_stream.html)

#### Parameter Controls Added:
- **Frame Rate Control**: Slider to adjust throttle_ms (16-200ms, ~60-5 FPS)
- **Resolution Control**: Dropdown for max width (640px, 1280px, 1920px, original)
- **Subsampling Factor**: Slider to reduce resolution (1x to 8x reduction)
- **Compression Level**: Slider for processing speed vs quality trade-off
- **STUN Server Configuration**: Toggle and text area for custom STUN servers
- **Connection Type Detection**: Automatic detection of local vs remote connections

#### Video Display Improvements:
- **Aspect Ratio Preservation**: Video uses `object-fit: contain` to maintain proportions
- **Responsive Sizing**: Video scales to fit container while preserving aspect ratio
- **Better Centering**: Video is centered in the container

#### Performance Optimizations:
- **Faster ICE Gathering**: Reduced timeout for local connections (500ms vs 2000ms)
- **Codec Preferences**: VP8 preferred for local, H.264 for remote connections
- **Optimized PeerConnection Config**: Bundle policy and RTCP mux for better performance
- **Better Status Feedback**: More detailed connection state information

### Backend (LiveViewController.py)

#### Parameter Integration:
- **Extended StreamParams**: Added `max_width` parameter to control resolution limits
- **WebRTCOfferRequest**: Added `params` field to accept streaming parameters
- **Parameter Processing**: Backend now applies parameters from frontend requests

#### Aspect Ratio Preservation:
- **Smart Resizing**: Calculates scaling factor to maintain aspect ratio
- **Even Dimensions**: Ensures width/height are even numbers (required for video codecs)
- **Subsampling Support**: Applies subsampling while maintaining aspect ratio
- **Logging**: Detailed logging of frame transformations for debugging

#### Performance Optimizations:
- **Frame Queue Management**: Clears old frames to keep only latest (reduces latency)
- **Fast Normalization**: Optimized uint8 conversion for high frame rates
- **Efficient Processing**: Different processing paths for high vs low frame rates
- **Memory Optimization**: Contiguous arrays and efficient CV2 operations

## Usage Instructions

### For Local/LAN Connections (Fastest):
1. Set server URL to `localhost` or LAN IP
2. Keep STUN servers disabled
3. Use higher frame rates (lower throttle_ms values)
4. Connection establishes in ~500ms

### For Remote Connections:
1. Enable STUN servers checkbox
2. Configure STUN server URLs (defaults provided)
3. Use moderate frame rates
4. Allow up to 2 seconds for connection

### Parameter Recommendations:

#### High Quality (Local Network):
- Frame Rate: 33ms (~30 FPS)
- Max Width: 1920px
- Subsampling: 1x
- Compression: 0 (Fastest)

#### Balanced (Mixed Network):
- Frame Rate: 50ms (~20 FPS)  
- Max Width: 1280px
- Subsampling: 2x
- Compression: 1

#### Low Bandwidth (Slow Network):
- Frame Rate: 100ms (~10 FPS)
- Max Width: 640px
- Subsampling: 4x
- Compression: 3

## Technical Details

### Aspect Ratio Calculation:
```python
# Calculate scaling while preserving aspect ratio
scale_width = max_width / original_width
scale_height = max_height / original_height
scale = min(scale_width, scale_height)

new_width = int(original_width * scale)
new_height = int(original_height * scale)

# Ensure even dimensions for video codecs
new_width = new_width - (new_width % 2)
new_height = new_height - (new_height % 2)
```

### Performance Optimizations:
- **Queue Management**: Only latest frame kept in queue
- **Fast Conversions**: Bit-shift operations for uint16→uint8
- **Conditional Processing**: Different paths for high/low frame rates
- **WebRTC Optimizations**: Reduced ICE gathering time, codec preferences

### Parameter Flow:
1. User adjusts parameters in HTML frontend
2. Parameters sent with WebRTC offer to backend
3. Backend updates StreamParams for the detector
4. Frame processing uses updated parameters
5. Video maintains aspect ratio automatically

## Testing

Use the included test script to verify functionality:

```bash
python test_webrtc_integration.py
```

Or test manually:
1. Start ImSwitch with a detector
2. Open `http://localhost:8001/webrtc_stream.html`
3. Adjust parameters as needed
4. Click "Start WebRTC Stream"
5. Verify video maintains correct aspect ratio

## Troubleshooting

### Video Still Distorted:
- Check that camera sensor is actually square/rectangular as expected
- Verify max width setting isn't forcing incorrect dimensions
- Check browser developer tools for WebRTC errors

### Connection Slow:
- For local connections, disable STUN servers
- Reduce frame rate (increase throttle_ms)
- Lower resolution (reduce max width)
- Check network connectivity

### No Video:
- Verify ImSwitch detector is running
- Check server URL is correct
- Look at browser console for WebRTC errors
- Try different parameter combinations

## Future Improvements

Potential enhancements that could be added:
- **Adaptive Quality**: Automatically adjust parameters based on connection quality
- **Multiple Detectors**: Parameter sets per detector
- **Recording Integration**: Save parameter sets for different use cases
- **Statistics Display**: Real-time bandwidth and quality metrics
- **Mobile Optimization**: Touch-friendly controls for mobile devices