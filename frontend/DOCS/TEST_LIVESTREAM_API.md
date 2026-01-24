# Testing LiveStream API Integration

## Quick Test Guide

### 1. Test Stream Parameter Retrieval

Open browser console and run:

```javascript
// Get all stream parameters
const params = await apiLiveViewControllerGetStreamParameters();
console.log('All parameters:', params);

// Get binary parameters only
const binaryParams = await apiLiveViewControllerGetStreamParameters('binary');
console.log('Binary parameters:', binaryParams);

// Get JPEG parameters only
const jpegParams = await apiLiveViewControllerGetStreamParameters('jpeg');
console.log('JPEG parameters:', jpegParams);
```

**Expected Response:**
```javascript
{
  "binary": {
    "compression_algorithm": "lz4",
    "compression_level": 0,
    "subsampling_factor": 4,
    "throttle_ms": 50
  },
  "jpeg": {
    "jpeg_quality": 80,
    "subsampling_factor": 4,
    "throttle_ms": 50
  }
}
```

### 2. Test Setting Stream Parameters

#### Set Binary Parameters
```javascript
const result = await apiLiveViewControllerSetStreamParameters('binary', {
  compression_algorithm: 'lz4',
  compression_level: 0,
  subsampling_factor: 4,
  throttle_ms: 50
});
console.log('Binary params set:', result);
```

**Expected Response:**
```javascript
{
  "status": "success",
  "protocol": "binary",
  "params": {
    "compression_algorithm": "lz4",
    "compression_level": 0,
    "subsampling_factor": 4,
    "throttle_ms": 50
  }
}
```

#### Set JPEG Parameters
```javascript
const result = await apiLiveViewControllerSetStreamParameters('jpeg', {
  jpeg_quality: 85,
  subsampling_factor: 1,
  throttle_ms: 100
});
console.log('JPEG params set:', result);
```

**Expected Response:**
```javascript
{
  "status": "success",
  "protocol": "jpeg",
  "params": {
    "jpeg_quality": 85,
    "subsampling_factor": 1,
    "throttle_ms": 100
  }
}
```

### 3. Test Stream Start/Stop

#### Start Binary Stream
```javascript
const result = await apiLiveViewControllerStartLiveView(null, 'binary');
console.log('Binary stream started:', result);
```

**Expected Response:**
```javascript
{
  "status": "success",
  "detector": "DetectorName",
  "protocol": "binary",
  "stream_info": {...}
}
```

#### Start JPEG Stream
```javascript
const result = await apiLiveViewControllerStartLiveView(null, 'jpeg');
console.log('JPEG stream started:', result);
```

#### Stop Stream
```javascript
const result = await apiLiveViewControllerStopLiveView();
console.log('Stream stopped:', result);
```

**Expected Response:**
```javascript
{
  "status": "success",
  "message": "Stream stopped"
}
```

### 4. Test Stream Status

```javascript
// Check if any stream is active
const isActive = await apiViewControllerGetLiveViewActive();
console.log('Stream active:', isActive); // true or false

// Get all active streams
const activeStreams = await apiLiveViewControllerGetActiveStreams();
console.log('Active streams:', activeStreams);
```

**Expected Response for getActiveStreams:**
```javascript
{
  "Detector1": {
    "protocol": "binary",
    "params": {...}
  }
}
```

## Network Inspector Checks

### For POST /LiveViewController/setStreamParameters

**Request URL:**
```
http://localhost:8001/LiveViewController/setStreamParameters?protocol=binary
```

**Request Method:**
```
POST
```

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "compression_algorithm": "lz4",
  "compression_level": 0,
  "subsampling_factor": 4,
  "throttle_ms": 50
}
```

**Response (Success):**
```json
{
  "status": "success",
  "protocol": "binary",
  "params": {
    "compression_algorithm": "lz4",
    "compression_level": 0,
    "subsampling_factor": 4,
    "throttle_ms": 50
  }
}
```

**Response (Error):**
```json
{
  "status": "error",
  "message": "Error description"
}
```

### For POST /LiveViewController/startLiveView

**Request URL:**
```
http://localhost:8001/LiveViewController/startLiveView?protocol=binary
```

**Request Method:**
```
POST
```

**Request Headers:**
```
Content-Type: application/json
```

**Request Body (Optional):**
```json
{
  "compression_algorithm": "lz4",
  "compression_level": 0,
  "subsampling_factor": 4,
  "throttle_ms": 50
}
```

or `null` for default parameters.

## Common Issues & Solutions

### Issue 1: POST request not going through

**Symptoms:**
- Network error in console
- 400 Bad Request
- Parameters not being set

**Solution:**
Check that:
1. Request method is `POST` (not `GET`)
2. Protocol is in query parameters: `?protocol=binary`
3. Params are in request body as JSON
4. Content-Type header is `application/json`

**Axios Configuration:**
```javascript
await axiosInstance.post(url, params, {
  params: { protocol },
  headers: { 'Content-Type': 'application/json' }
});
```

### Issue 2: Parameters in wrong format

**Symptoms:**
- Backend returns error
- Parameters not applied

**Solution:**
Ensure parameter names match backend expectations:

✅ **Correct:**
```javascript
{
  compression_algorithm: 'lz4',  // snake_case
  compression_level: 0,
  subsampling_factor: 4,
  throttle_ms: 50
}
```

❌ **Incorrect:**
```javascript
{
  compressionAlgorithm: 'lz4',  // camelCase
  compressionLevel: 0,
  subsamplingFactor: 4,
  throttleMs: 50
}
```

### Issue 3: Stream not starting

**Symptoms:**
- `startLiveView` returns success but no stream
- Frontend not receiving frames

**Solution:**
1. Check that parameters were set before starting stream
2. Verify detector is available
3. Check WebSocket connection is active
4. Verify backend logs for errors

**Debugging:**
```javascript
// 1. Set parameters
await apiLiveViewControllerSetStreamParameters('binary', {...});

// 2. Wait a moment
await new Promise(resolve => setTimeout(resolve, 500));

// 3. Start stream
await apiLiveViewControllerStartLiveView(null, 'binary');

// 4. Check status
const isActive = await apiViewControllerGetLiveViewActive();
console.log('Is stream active?', isActive);
```

## Manual Testing Checklist

- [ ] **Stream Parameters**
  - [ ] Get all parameters
  - [ ] Get binary parameters
  - [ ] Get JPEG parameters
  - [ ] Set binary parameters (lz4)
  - [ ] Set binary parameters (zstd)
  - [ ] Set JPEG parameters
  - [ ] Verify parameters persist after set

- [ ] **Stream Control**
  - [ ] Start binary stream
  - [ ] Stop stream
  - [ ] Start JPEG stream
  - [ ] Stop stream
  - [ ] Toggle stream multiple times
  - [ ] Start with custom parameters

- [ ] **UI Integration**
  - [ ] StreamControlOverlay loads parameters
  - [ ] StreamControlOverlay submits parameters
  - [ ] StreamSettings loads parameters
  - [ ] StreamSettings submits parameters
  - [ ] LiveView toggles stream
  - [ ] Stream status updates in UI

- [ ] **Error Handling**
  - [ ] Invalid protocol name
  - [ ] Invalid parameter values
  - [ ] Backend unavailable
  - [ ] Network timeout
  - [ ] Concurrent stream requests

## Network Debugging

### Using Browser DevTools

1. Open DevTools (F12)
2. Go to Network tab
3. Filter by "XHR" or "Fetch"
4. Perform an action (e.g., submit settings)
5. Click on the request
6. Check:
   - Request URL (should include query params)
   - Request Method (should be POST for setStreamParameters)
   - Request Payload (should show JSON params)
   - Response (should show success/error)

### Using curl

Test setStreamParameters:
```bash
curl -X POST "http://localhost:8001/LiveViewController/setStreamParameters?protocol=binary" \
  -H "Content-Type: application/json" \
  -d '{
    "compression_algorithm": "lz4",
    "compression_level": 0,
    "subsampling_factor": 4,
    "throttle_ms": 50
  }'
```

Test startLiveView:
```bash
curl -X POST "http://localhost:8001/LiveViewController/startLiveView?protocol=binary" \
  -H "Content-Type: application/json" \
  -d 'null'
```

Test getStreamParameters:
```bash
curl "http://localhost:8001/LiveViewController/getStreamParameters"
```

Test stopLiveView:
```bash
curl "http://localhost:8001/LiveViewController/stopLiveView"
```

## Integration Test Script

Create a test file `test-livestream-api.js`:

```javascript
import apiLiveViewControllerGetStreamParameters from './src/backendapi/apiLiveViewControllerGetStreamParameters';
import apiLiveViewControllerSetStreamParameters from './src/backendapi/apiLiveViewControllerSetStreamParameters';
import apiLiveViewControllerStartLiveView from './src/backendapi/apiLiveViewControllerStartLiveView';
import apiLiveViewControllerStopLiveView from './src/backendapi/apiLiveViewControllerStopLiveView';
import apiViewControllerGetLiveViewActive from './src/backendapi/apiViewControllerGetLiveViewActive';

async function testLiveStreamAPI() {
  console.log('=== Testing LiveStream API ===\n');
  
  try {
    // Test 1: Get current parameters
    console.log('Test 1: Getting stream parameters...');
    const currentParams = await apiLiveViewControllerGetStreamParameters();
    console.log('✅ Current parameters:', currentParams);
    
    // Test 2: Set binary parameters
    console.log('\nTest 2: Setting binary parameters...');
    const binaryResult = await apiLiveViewControllerSetStreamParameters('binary', {
      compression_algorithm: 'lz4',
      compression_level: 0,
      subsampling_factor: 4,
      throttle_ms: 50
    });
    console.log('✅ Binary parameters set:', binaryResult);
    
    // Test 3: Set JPEG parameters
    console.log('\nTest 3: Setting JPEG parameters...');
    const jpegResult = await apiLiveViewControllerSetStreamParameters('jpeg', {
      jpeg_quality: 80,
      subsampling_factor: 1,
      throttle_ms: 100
    });
    console.log('✅ JPEG parameters set:', jpegResult);
    
    // Test 4: Start binary stream
    console.log('\nTest 4: Starting binary stream...');
    const startResult = await apiLiveViewControllerStartLiveView(null, 'binary');
    console.log('✅ Stream started:', startResult);
    
    // Test 5: Check if active
    console.log('\nTest 5: Checking stream status...');
    const isActive = await apiViewControllerGetLiveViewActive();
    console.log('✅ Stream is active:', isActive);
    
    // Wait a moment
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Test 6: Stop stream
    console.log('\nTest 6: Stopping stream...');
    const stopResult = await apiLiveViewControllerStopLiveView();
    console.log('✅ Stream stopped:', stopResult);
    
    // Test 7: Verify stopped
    console.log('\nTest 7: Verifying stream stopped...');
    const isStillActive = await apiViewControllerGetLiveViewActive();
    console.log('✅ Stream is active:', isStillActive);
    
    console.log('\n=== All tests passed! ===');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

// Run tests
testLiveStreamAPI();
```

Run with:
```bash
npm run test-livestream-api
```
