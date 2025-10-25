# LiveViewController Acquisition Safety

## Problem
When changing streaming protocols in `LiveViewController`, there was a potential race condition where multiple `StreamWorker` instances could simultaneously start/stop the camera hardware. This could lead to:
- Hardware failures
- Acquisition state corruption
- Undefined behavior when multiple workers access the detector

## Solution: Handle-Based Acquisition

The solution leverages the existing `DetectorsManager.startAcquisition()` / `stopAcquisition()` handle-based system, which provides thread-safe camera access through:

### 1. Mutex-Protected Handle Management
```python
# DetectorsManager uses a mutex to protect acquisition state
self._activeAcqsMutex.lock()
try:
    handle = np.random.randint(2 ** 31)
    self._activeAcqHandles.append(handle)
    enableAcq = len(self._activeAcqHandles) + len(self._activeAcqLVHandles) == 1
finally:
    self._activeAcqsMutex.unlock()

# Only start acquisition if this is the first handle
if enableAcq:
    self.execOnAll(lambda c: c.startAcquisition(), ...)
```

### 2. Reference Counting
- Each `StreamWorker` acquires a unique handle when starting
- The detector only stops when **all** handles are released
- This prevents premature stopping when multiple workers exist

### 3. StreamWorker Implementation

**Before (Unsafe):**
```python
def run(self):
    self._was_running = self._detector._running
    if not self._detector._running:
        self._detector.startAcquisition()  # Direct access - no protection!
    
    # ... do work ...
    
    if self._was_running is False:
        self._detector.stopAcquisition()  # Could stop while other worker needs it!
```

**After (Safe):**
```python
def run(self):
    # Acquire handle through DetectorsManager's thread-safe system
    self._acq_handle = self._detector.startAcquisition()
    
    # ... do work ...
    
    # Release handle - only stops if no other handles active
    if self._acq_handle is not None:
        self._detector.stopAcquisition(self._acq_handle)
```

## Protocol Switching Safety

When switching protocols (e.g., from `binary` to `jpeg`), the sequence is now:

1. **Stop old worker**: Releases its acquisition handle
2. **Brief pause**: 100ms to ensure clean shutdown
3. **Start new worker**: Acquires new handle

The `DetectorsManager` ensures:
- If this is the only worker: camera stops then restarts
- If other workers exist: camera stays running, only worker changes

## Example Scenarios

### Scenario A: Single Stream Protocol Change
```
Initial: binary worker (handle=12345) → camera RUNNING
↓ User changes to JPEG
Stop binary worker → releases handle=12345 → camera STOPS
↓ 100ms pause
Start JPEG worker → acquires handle=67890 → camera STARTS
Final: JPEG worker (handle=67890) → camera RUNNING
```

### Scenario B: Multiple Detectors
```
Initial: 
  - detector1: binary (handle=111) → camera1 RUNNING
  - detector2: JPEG (handle=222) → camera2 RUNNING

User changes detector1 to JPEG:
  - Stop detector1 binary → releases 111 → camera1 STOPS
  - detector2 unchanged → camera2 RUNNING
  - Start detector1 JPEG → acquires 333 → camera1 STARTS
  
Final:
  - detector1: JPEG (handle=333) → camera1 RUNNING
  - detector2: JPEG (handle=222) → camera2 RUNNING
```

### Scenario C: Race Condition Prevention
```
Thread 1: Changing protocol for detector A
Thread 2: Changing protocol for detector A (same detector!)

With Mutex Protection:
Thread 1 acquires mutex → stops worker → releases handle → mutex unlocked
Thread 2 waits for mutex → then proceeds safely

Without Mutex (old code):
Thread 1: stops camera
Thread 2: stops camera  ← RACE! Camera already stopped
Thread 1: starts camera
Thread 2: starts camera ← RACE! Double start = undefined behavior
```

## Benefits

1. **Thread Safety**: Mutex-protected handle management prevents race conditions
2. **Reference Counting**: Camera only stops when truly idle (no active handles)
3. **Clean Transitions**: 100ms pause ensures complete shutdown before restart
4. **Reusable Pattern**: Same pattern used throughout ImSwitch for detector access
5. **Hardware Protection**: Prevents simultaneous start/stop that could damage hardware

## Related Files

- `DetectorsManager.py`: Provides handle-based acquisition system
- `LiveViewController.py`: Uses handles in `StreamWorker` instances
- `StreamWorker.run()`: Acquires handle on start, releases on stop

## Testing Recommendations

1. Rapidly switch protocols multiple times
2. Test with multiple detectors streaming simultaneously
3. Monitor acquisition handles: `DetectorsManager.getAcquistionHandles()`
4. Verify camera only stops when all workers terminated
5. Check logs for "Acquired detector with handle" / "Released detector handle" messages
