# LiveStream API Update - Testing Checklist

## Pre-Testing Setup

- [ ] Backend is running and accessible
- [ ] Backend has LiveViewController endpoints implemented
- [ ] Frontend is built and running (`npm start`)
- [ ] Browser console is open for debugging
- [ ] Network tab is open to monitor API calls

## Basic Functionality Tests

### Stream Start/Stop
- [ ] Start binary stream (default)
- [ ] Verify stream starts successfully
- [ ] Verify video is displayed
- [ ] Stop stream
- [ ] Verify stream stops successfully
- [ ] Repeat cycle 3 times to ensure stability

### Format Switching
- [ ] Start with binary stream
- [ ] Stop stream
- [ ] Switch to JPEG format in settings
- [ ] Start stream
- [ ] Verify JPEG stream starts
- [ ] Stop stream
- [ ] Switch back to binary
- [ ] Start stream
- [ ] Verify binary stream starts

## Parameter Testing

### Binary Stream Parameters

#### Compression Algorithm
- [ ] Set to LZ4
  - [ ] Submit settings
  - [ ] Restart stream
  - [ ] Verify stream works
- [ ] Set to Zstandard
  - [ ] Submit settings
  - [ ] Restart stream
  - [ ] Verify stream works
- [ ] Set to None
  - [ ] Submit settings
  - [ ] Restart stream
  - [ ] Verify stream works

#### Compression Level
- [ ] Set level to 0 (fastest)
  - [ ] Submit and verify
- [ ] Set level to 5 (medium)
  - [ ] Submit and verify
- [ ] Set level to 9 (best compression)
  - [ ] Submit and verify

#### Subsampling Factor
- [ ] Set to 1x (full resolution)
  - [ ] Submit and verify image resolution
- [ ] Set to 2x
  - [ ] Submit and verify image resolution
- [ ] Set to 4x
  - [ ] Submit and verify image resolution
- [ ] Set to 8x
  - [ ] Submit and verify image resolution

#### Throttle Time
- [ ] Set to 16ms (~60 FPS)
  - [ ] Submit and verify frame rate
- [ ] Set to 50ms (~20 FPS)
  - [ ] Submit and verify frame rate
- [ ] Set to 100ms (~10 FPS)
  - [ ] Submit and verify frame rate
- [ ] Set to 1000ms (1 FPS)
  - [ ] Submit and verify frame rate

### JPEG Stream Parameters

#### JPEG Quality
- [ ] Set quality to 10 (low)
  - [ ] Submit and verify image quality
  - [ ] Check file size/bandwidth
- [ ] Set quality to 50 (medium)
  - [ ] Submit and verify image quality
  - [ ] Check file size/bandwidth
- [ ] Set quality to 100 (high)
  - [ ] Submit and verify image quality
  - [ ] Check file size/bandwidth

#### Subsampling Factor
- [ ] Set to 1x (full resolution)
  - [ ] Submit and verify
- [ ] Set to 4x
  - [ ] Submit and verify

#### Throttle Time
- [ ] Set to 50ms
  - [ ] Submit and verify frame rate
- [ ] Set to 100ms
  - [ ] Submit and verify frame rate

## UI Tests

### StreamControlOverlay
- [ ] Open overlay (click settings icon)
- [ ] Verify current parameters are displayed correctly
- [ ] Navigate to "Controls" tab
  - [ ] Test Window/Level sliders
  - [ ] Test Gamma slider (binary only)
  - [ ] Test Auto button
- [ ] Navigate to "Settings" tab
  - [ ] Test Stream Format dropdown
  - [ ] Test parameter sliders
  - [ ] Test Submit button
  - [ ] Verify success message appears
  - [ ] Test Reset button
- [ ] Navigate to "Info" tab
  - [ ] Verify performance stats are shown
  - [ ] Verify image info is correct
  - [ ] Verify backend info is correct
- [ ] Close overlay
  - [ ] Verify settings persist

### StreamSettings Component
If your app uses this component:
- [ ] Open settings panel
- [ ] Verify current parameters are loaded
- [ ] Modify parameters
- [ ] Click Submit
- [ ] Verify success message
- [ ] Click Reset
- [ ] Verify defaults are restored

### StreamControls
- [ ] Verify Play button is visible when stopped
- [ ] Verify Stop button is visible when running
- [ ] Verify button states match actual stream state
- [ ] Test rapid clicking (toggle multiple times quickly)
- [ ] Verify no race conditions occur

## Error Handling Tests

### Backend Unavailable
- [ ] Stop backend
- [ ] Try to start stream
- [ ] Verify error message is shown
- [ ] Verify UI remains functional
- [ ] Start backend
- [ ] Retry stream start
- [ ] Verify recovery works

### Invalid Parameters
- [ ] Try to set invalid compression level (e.g., 99)
- [ ] Verify validation or error handling
- [ ] Try to set invalid throttle (e.g., 5ms)
- [ ] Verify validation or error handling

### Network Timeout
- [ ] Simulate slow network (Chrome DevTools throttling)
- [ ] Try to load parameters
- [ ] Verify timeout handling
- [ ] Verify retry mechanism works

### Legacy Backend
- [ ] Use old backend without LiveViewController
- [ ] Verify legacy mode is detected
- [ ] Verify warning message is shown
- [ ] Verify JPEG-only mode is enforced

## Integration Tests

### With Redux State
- [ ] Open Redux DevTools
- [ ] Monitor LiveStreamSlice state
- [ ] Start stream
  - [ ] Verify `isStreamRunning` updates
- [ ] Change format
  - [ ] Verify `currentImageFormat` updates
- [ ] Change parameters
  - [ ] Verify `streamSettings` updates
- [ ] Stop stream
  - [ ] Verify state updates correctly

### With LiveViewerGL
If using WebGL viewer:
- [ ] Start binary stream
- [ ] Verify WebGL rendering works
- [ ] Test window/level adjustments
- [ ] Test gamma adjustments
- [ ] Switch to JPEG
- [ ] Verify Canvas2D fallback works

### With Recording
- [ ] Start stream
- [ ] Start recording
- [ ] Verify stream continues during recording
- [ ] Stop recording
- [ ] Verify stream continues after recording
- [ ] Stop stream

## Performance Tests

### Frame Rate
- [ ] Set throttle to 16ms
- [ ] Check actual FPS in overlay
- [ ] Verify FPS is close to expected (~60)
- [ ] Set throttle to 100ms
- [ ] Verify FPS is close to expected (~10)

### Bandwidth
- [ ] Start binary stream with LZ4
- [ ] Check bandwidth in overlay (Mbps)
- [ ] Switch to no compression
- [ ] Verify bandwidth increases
- [ ] Switch to JPEG
- [ ] Compare bandwidth with binary

### CPU Usage
- [ ] Start binary stream
- [ ] Check browser CPU usage
- [ ] Switch to JPEG
- [ ] Compare CPU usage
- [ ] Note: Binary should be more efficient with WebGL

## Browser Compatibility

### Chrome/Edge
- [ ] All tests pass

### Firefox
- [ ] All tests pass
- [ ] Check for console warnings

### Safari
- [ ] All tests pass
- [ ] Check for compatibility issues

## API Call Verification

### Check Network Requests
- [ ] Open Network tab
- [ ] Start stream
  - [ ] Verify `POST /LiveViewController/startLiveView` is called
  - [ ] Check request payload
  - [ ] Verify response is successful
- [ ] Stop stream
  - [ ] Verify `GET /LiveViewController/stopLiveView` is called
  - [ ] Verify response is successful
- [ ] Change parameters
  - [ ] Verify `GET /LiveViewController/setStreamParameters` is called
  - [ ] Check query param `protocol=binary` or `protocol=jpeg`
  - [ ] Check request body contains parameters
  - [ ] Verify response is successful
- [ ] Load parameters
  - [ ] Verify `GET /LiveViewController/getStreamParameters` is called
  - [ ] Verify response contains expected data

### Check Console Logs
- [ ] No unexpected errors
- [ ] Parameter updates are logged
- [ ] Stream state changes are logged
- [ ] Any warnings are expected and documented

## Regression Tests

### Old Functionality Still Works
- [ ] XYZ stage control works
- [ ] Illumination control works
- [ ] Snap image works
- [ ] Recording works
- [ ] File manager works
- [ ] Other microscope features work

## Documentation Review

- [ ] API function JSDoc comments are clear
- [ ] Example code in comments is correct
- [ ] LIVESTREAM_API_UPDATE.md is accurate
- [ ] LIVESTREAM_API_QUICKREF.md is helpful
- [ ] LIVESTREAM_API_ARCHITECTURE.md is understandable

## Sign-off

### Developer
- [ ] All code changes reviewed
- [ ] All new functions tested locally
- [ ] Documentation is complete
- [ ] No known issues

### QA
- [ ] All test cases executed
- [ ] No critical bugs found
- [ ] Performance is acceptable
- [ ] User experience is smooth

### Product
- [ ] Feature meets requirements
- [ ] User workflow is unchanged
- [ ] Ready for deployment

---

## Notes

Use this section to document any issues found during testing:

```
Issue: [Description]
Severity: [Critical/Major/Minor]
Status: [Open/Fixed/Deferred]
Notes: [Additional context]
```

## Test Results Summary

- Total test cases: ___
- Passed: ___
- Failed: ___
- Skipped: ___
- Pass rate: ___%

Date: _______________
Tester: _______________
Version: _______________
