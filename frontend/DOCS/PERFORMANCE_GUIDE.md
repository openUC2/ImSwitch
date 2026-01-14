# WebRTC Connection Speed Optimization Guide

## 🔍 Identifying the Bottleneck

Based on your logs, the delay between "Sending params" and "ICE connection state: checking" suggests the bottleneck is likely on the **backend** during SDP processing.

### Tools to Profile Performance:

1. **Frontend Profiling** (Browser Console):
   - Look for the detailed timing logs we added
   - Check for lines starting with `🕐 Complete timing breakdown:`
   - Identify which phase takes the longest

2. **Backend Profiling** (ImSwitch Logs):
   - Look for lines starting with `🕐 Backend SDP processing timing:`
   - Check for lines starting with `🕐 Complete backend timing breakdown:`
   - Monitor detector initialization time

3. **Use the Profiler Script**:
   ```bash
   python webrtc_profiler.py
   ```

## ⚡ Immediate Optimizations

### 1. Dynamic Parameter Changes (✅ Now Implemented)
You can now change parameters while streaming! Just adjust the sliders and they'll be applied immediately.

### 2. Reduce Connection Setup Time

**Frontend optimizations:**
- Reduced ICE gathering timeout for local connections (500ms vs 2000ms)
- Optimized codec preferences (VP8 for local, H.264 for remote)
- Better PeerConnection configuration

**Backend optimizations:**
- Added comprehensive timing to identify bottlenecks
- Optimized frame queue management
- Faster frame processing for high frame rates

### 3. Common Bottlenecks and Solutions

| Bottleneck | Symptoms | Solution |
|------------|----------|----------|
| ICE Gathering | Frontend log shows >200ms ICE time | Disable STUN servers for local connections |
| SDP Processing | Backend shows >300ms async processing | Check aiortc version, server load |
| Stream Startup | Backend shows >100ms stream start | Check detector initialization |
| Frame Processing | High CPU usage, frame drops | Reduce resolution, increase throttle_ms |

## 🎯 Recommended Settings by Use Case

### Local Development (Fastest):
```javascript
{
  throttle_ms: 33,          // ~30 FPS
  subsampling_factor: 1,    // Full resolution
  max_width: 1280,          // HD quality
  compression_level: 0,     // Fastest processing
  stun_servers: []          // No STUN needed for local
}
```

### Network Streaming (Balanced):
```javascript
{
  throttle_ms: 50,          // ~20 FPS
  subsampling_factor: 2,    // Half resolution
  max_width: 1280,          // HD quality
  compression_level: 1,     // Light compression
  stun_servers: ["stun:stun.l.google.com:19302"]
}
```

### Slow Network (Optimized):
```javascript
{
  throttle_ms: 100,         // ~10 FPS
  subsampling_factor: 4,    // Quarter resolution
  max_width: 640,           // SD quality
  compression_level: 3,     // More compression
  stun_servers: ["stun:stun.l.google.com:19302"]
}
```

## 🔧 Advanced Debugging

### 1. Detailed Browser Profiling:
```javascript
// Open browser console and run:
console.time('webrtc-connection');
// Then start the connection
// Check console for detailed timing logs
```

### 2. Backend Performance Monitoring:
```bash
# Monitor ImSwitch logs for timing information
tail -f /path/to/imswitch.log | grep -E "(🕐|⚡|✅|❌)"
```

### 3. Network Analysis:
```bash
# Check if it's a network issue
ping localhost  # Should be <1ms for local connections
```

## 🚀 Performance Optimization Checklist

- [ ] **Use Local Connection**: localhost/LAN IP instead of remote
- [ ] **Disable STUN**: For local connections, uncheck "Use STUN servers"  
- [ ] **Optimize Frame Rate**: Start with 50ms (20 FPS), adjust as needed
- [ ] **Monitor Console**: Check browser console for timing breakdowns
- [ ] **Check Backend Logs**: Look for backend timing information
- [ ] **Test Parameter Changes**: Try adjusting parameters while streaming
- [ ] **Profile Regularly**: Use the profiler script to identify issues

## 🔄 Dynamic Parameter Testing

Now you can test different parameter combinations in real-time:

1. Start the WebRTC stream
2. Adjust any parameter slider
3. Parameters are applied immediately
4. Watch the "● LIVE" indicator to confirm active streaming
5. Monitor console logs for performance impact

## 📊 Expected Performance Targets

| Connection Type | Setup Time | Frame Latency | Typical Use |
|----------------|------------|---------------|-------------|
| Local/LAN      | <500ms     | <50ms         | Development, demos |
| Local Network  | <1000ms    | <100ms        | Lab environment |
| Internet       | <2000ms    | <200ms        | Remote access |

If you're seeing higher times, use the profiling tools to identify the specific bottleneck!