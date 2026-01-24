# Hardware-based Autofocus Integration for Wellplate Scanning

## Overview

This document describes the integration of a **hardware-based one-shot autofocus** system into ImSwitch's ExperimentController, designed for fast wellplate scanning. This complements the existing PID-based continuous focus lock for long-term drift compensation.

## Motivation

### Problem with Software Autofocus
- **Slow**: Requires full Z-sweep (e.g., 20 steps × 100µm)
- **Many images**: Captures 20+ images per XY position
- **Time-consuming**: ~20-30 seconds per position
- **Inefficient**: For wellplate scanning with many positions

### Hardware Autofocus Solution
- **Fast**: Only ONE frame capture per position
- **Accurate**: Uses pre-calibrated linear relationship
- **Efficient**: ~1-2 seconds per position
- **Dedicated hardware**: Separate autofocus camera with laser spots

## Architecture

### Two Autofocus Modes

ImSwitch now supports **two complementary autofocus systems**:

| Mode | Use Case | Speed | Hardware | Method |
|------|----------|-------|----------|--------|
| **Hardware** | Wellplate scanning | ⚡ Very fast (1-2s) | Dedicated AF camera + laser | One-shot measurement |
| **Software** | General purpose | 🐢 Slow (20-30s) | Main camera only | Z-sweep with focus metric |

**Additionally**, the existing **PID-based continuous focus lock** remains available for:
- Long-term time-lapse imaging
- Drift compensation during acquisition
- Maintaining focus during environmental changes

## Implementation

### 1. FocusLockController Enhancements

#### New Method: `performOneStepAutofocus()`

```python
@APIExport(runOnUIThread=True)
def performOneStepAutofocus(
    self, 
    move_to_focus: bool = True, 
    max_attempts: int = 3,
    threshold_um: float = 0.5
) -> Dict[str, Any]:
    """
    Perform one-shot hardware-based autofocus using calibration data.
    
    Similar to Seafront laser autofocus approach:
    1. Capture single frame from focus camera
    2. Calculate focus metric (e.g., laser peak position)
    3. Use calibration: z_target = f(focus_value)
    4. Move to target Z position
    5. Optionally iterate if error too large
    
    Returns:
        - success: bool
        - target_z_position: float (µm)
        - z_offset: float (µm)
        - final_error_um: float
        - num_attempts: int
    """
```

**Workflow:**
```
┌─────────────────────────────────────┐
│ 1. Capture frame from AF camera     │
│    (single frame, ~50ms)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. Calculate focus metric            │
│    (e.g., laser peak position)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Use calibration to get Z target   │
│    z_target = f(focus_value)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. Move to target Z position         │
│    (if error > threshold)            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. Check error, iterate if needed    │
│    (max 3 attempts)                  │
└─────────────────────────────────────┘
```

#### New Helper Method: `_calculate_z_position_from_focus_value()`

```python
def _calculate_z_position_from_focus_value(
    self, 
    focus_value: float
) -> Optional[float]:
    """
    Calculate absolute Z position from focus metric using calibration.
    
    Uses three methods in priority order:
    1. Lookup table with interpolation (most accurate)
    2. Polynomial fit (good for well-characterized systems)
    3. Linear offset from setpoint (simplest)
    
    Returns:
        Target Z position in µm, or None if calibration unavailable
    """
```

**Algorithm (similar to Seafront):**
```python
# Seafront approach:
offset_mm = (measured_x - reference_x) * um_per_px / 1000

# Our implementation:
z_target = z_reference + (focus_value - reference_value) * sensitivity_nm_per_unit / 1000
```

### 2. ExperimentController Integration

#### New Methods

**1. `autofocus_hardware()`** - Fast one-shot autofocus
```python
def autofocus_hardware(
    self, 
    illuminationChannel: str = ""
) -> Optional[float]:
    """
    Hardware-based autofocus using FocusLockController.
    
    Returns:
        Best focus Z position in µm, or None if failed
    """
```

**2. `autofocus_software()`** - Traditional Z-sweep autofocus
```python
def autofocus_software(
    self, 
    minZ: float = 0, 
    maxZ: float = 0, 
    stepSize: float = 0,
    illuminationChannel: str = ""
) -> Optional[float]:
    """
    Software-based autofocus using AutofocusController (Z-sweep).
    
    Returns:
        Best focus Z position in µm, or None if failed
    """
```

**3. `autofocus()` - Unified interface**
```python
def autofocus(
    self, 
    minZ: float = 0, 
    maxZ: float = 0, 
    stepSize: float = 0,
    illuminationChannel: str = "",
    mode: str = "software"
) -> Optional[float]:
    """
    Perform autofocus using either hardware or software method.
    
    Args:
        mode: "hardware" (fast) or "software" (slow)
    """
```

#### New Parameter in `ParameterValue`

```python
class ParameterValue(BaseModel):
    # ... existing parameters ...
    autoFocusMode: str = "software"  # "software" or "hardware"
```

### 3. Workflow Integration

The autofocus mode is now passed through the entire workflow chain:

```
ExperimentController.startWellplateExperiment()
  ↓
  Extract autoFocusMode from ParameterValue
  ↓
ExperimentNormalMode.execute_experiment()
  ↓
  Pass autofocus_mode to _create_tile_workflow_steps()
  ↓
WorkflowStep creation
  ↓
  main_func = controller.autofocus
  main_params = {..., "mode": autofocus_mode}
```

## Usage

### 1. Prerequisites

Before using hardware autofocus, you must:

1. **Run calibration** on the FocusLockController:
   ```python
   # Via API
   POST /focuslock/runFocusCalibration
   {
     "from_position": 49.0,
     "to_position": 51.0,
     "num_steps": 20,
     "settle_time": 0.5
   }
   ```

2. **Verify calibration** succeeded:
   ```python
   GET /focuslock/getCalibrationStatus
   # Returns: {"calibrated": true, "sensitivity_nm_per_unit": 123.45, ...}
   ```

### 2. Enable Hardware Autofocus in Experiment

When creating an experiment, set `autoFocusMode` to `"hardware"`:

```python
POST /experiment/startWellplateExperiment
{
  "name": "Wellplate Scan",
  "parameterValue": {
    "autoFocus": true,
    "autoFocusMode": "hardware",  // ← Select hardware autofocus
    "autoFocusMin": 0,            // Not used in hardware mode
    "autoFocusMax": 0,            // Not used in hardware mode
    "autoFocusStepSize": 0,       // Not used in hardware mode
    "autoFocusIlluminationChannel": "",
    // ... other parameters ...
  },
  "pointList": [...]
}
```

### 3. Fallback Behavior

If hardware autofocus fails (e.g., no calibration, camera error), the system will:
- Log an error
- Return `None` from `autofocus_hardware()`
- The experiment will continue without autofocus for that position

**Recommendation**: Always test hardware autofocus before large experiments.

## Calibration Process

### How Calibration Works

The calibration establishes a linear (or polynomial) relationship between:
- **Input**: Focus metric value (e.g., laser spot pixel position)
- **Output**: Z stage position (µm)

**Calibration workflow:**

```python
1. Move Z stage over defined range (e.g., ±1mm)
2. At each Z position:
   - Capture frame from AF camera
   - Calculate focus metric (e.g., peak position)
   - Store (z_position, focus_value) pair
3. Perform linear regression:
   z_position = a + b * focus_value
4. Store calibration data:
   - polynomial_coeffs: [a, b]
   - sensitivity_nm_per_unit: b (slope)
   - lookup_table: Optional interpolation table
```

### Calibration Data Structure

```python
@dataclass
class CalibrationData:
    position_data: List[float]           # Z positions during calibration
    focus_data: List[float]              # Focus metric values
    polynomial_coeffs: Optional[List[float]]  # [intercept, slope, ...]
    sensitivity_nm_per_unit: float       # Slope (nm/unit)
    r_squared: float                     # Fit quality
    linear_range: Tuple[float, float]    # Valid focus range
    timestamp: float                     # When calibrated
    lookup_table: Optional[Dict[float, float]]  # focus → z mapping
```

## Performance Comparison

### Software Autofocus (Z-sweep)
```
For each XY position:
  1. Move to position         ~500ms
  2. Z-sweep (20 steps):
     - Move Z                 20 × 200ms = 4s
     - Capture image          20 × 100ms = 2s
     - Calculate metric       20 × 10ms  = 200ms
  3. Find maximum             ~10ms
  4. Move to best Z           ~200ms
  ────────────────────────────────────
  Total per position:         ~7 seconds
```

### Hardware Autofocus (One-shot)
```
For each XY position:
  1. Move to position         ~500ms
  2. Capture AF frame         ~50ms
  3. Calculate metric         ~10ms
  4. Calculate Z target       ~1ms
  5. Move to target Z         ~200ms
  6. Verify (optional)        ~300ms
  ────────────────────────────────────
  Total per position:         ~1 second
```

**Speedup: ~7× faster** 🚀

For a 96-well plate with 9 sites per well (864 positions):
- Software AF: 864 × 7s = **~100 minutes**
- Hardware AF: 864 × 1s = **~14 minutes**

**Time saved: 86 minutes** ⏱️

## Hardware Requirements

### Minimum Requirements
1. **Dedicated autofocus camera** (separate from main imaging camera)
2. **Laser or LED source** for autofocus illumination
3. **Calibrated focus metric** (e.g., astigmatism, laser spot position)
4. **Z stage** with position feedback

### Recommended Setup
- **Camera**: Fast readout (>20 fps) for quick acquisition
- **Laser**: Stable power output, ideally 600-800nm wavelength
- **Optics**: Two-spot laser system for extended range
- **Stage**: Closed-loop Z stage with <1µm repeatability

### Configuration Example

```json
{
  "focusLock": {
    "camera": "AutofocusCamera",
    "positioner": "ZStage",
    "focusLockMetric": "peak",
    "crop_size": 300,
    "crop_center": [512, 512],
    "updateFreq": 10,
    "laserName": "AutofocusLaser",
    "laserValue": 50,
    "piKp": 0.1,
    "piKi": 0.01,
    "piKd": 0.0,
    "scaleUmPerUnit": 100.0
  }
}
```

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "No calibration data available" | Calibration not run | Run calibration first |
| "FocusLockController not available" | Controller not configured | Add focusLock section to config |
| "Frame capture failed" | Camera error | Check camera connection |
| "Could not calculate target Z position" | Invalid calibration | Re-run calibration |
| "Stage movement failed" | Stage error | Check stage connection |

### Safety Limits

Hardware autofocus includes safety limits:
- **Max single move**: 1mm (prevents large jumps)
- **Max attempts**: 3 (prevents infinite loops)
- **Threshold**: 0.5µm (defines success criterion)

## API Reference

### FocusLockController

#### `GET /focuslock/performOneStepAutofocus`
Perform one-shot hardware autofocus.

**Parameters:**
- `move_to_focus` (bool): Whether to move stage to focus (default: true)
- `max_attempts` (int): Max correction iterations (default: 3)
- `threshold_um` (float): Success threshold in µm (default: 0.5)

**Response:**
```json
{
  "success": true,
  "current_focus_value": 123.45,
  "target_z_position": 50.123,
  "z_offset": 0.123,
  "moved": true,
  "num_attempts": 1,
  "final_error_um": 0.123
}
```

#### `GET /focuslock/getCalibrationStatus`
Check calibration status.

**Response:**
```json
{
  "calibrated": true,
  "sensitivity_nm_per_unit": 123.45,
  "r_squared": 0.995,
  "timestamp": 1234567890.0
}
```

#### `POST /focuslock/runFocusCalibration`
Run autofocus calibration.

**Request:**
```json
{
  "from_position": 49.0,
  "to_position": 51.0,
  "num_steps": 20,
  "settle_time": 0.5
}
```

### ExperimentController

#### `POST /experiment/startWellplateExperiment`
Start wellplate experiment with autofocus.

**Request:**
```json
{
  "name": "Experiment",
  "parameterValue": {
    "autoFocus": true,
    "autoFocusMode": "hardware",  // or "software"
    "autoFocusIlluminationChannel": "LED_488",
    // ... other parameters
  },
  "pointList": [...]
}
```

## Comparison with Seafront Laser Autofocus

ImSwitch's hardware autofocus is inspired by the Seafront system:

| Feature | Seafront | ImSwitch |
|---------|----------|----------|
| **Principle** | Laser two-spot system | Flexible focus metric |
| **Calibration** | Linear regression | Linear or polynomial fit |
| **Calculation** | `offset = (x - x_ref) * um_per_px` | `z = f(focus_value)` |
| **Iterations** | Up to 3 attempts | Up to 3 attempts |
| **Threshold** | 0.5µm (configurable) | 0.5µm (configurable) |
| **Fallback** | Return to reference Z | Continue without AF |
| **Integration** | Per-site in protocol | Per-tile in workflow |

### Key Differences

1. **Focus Metric Flexibility**
   - Seafront: Fixed laser peak detection
   - ImSwitch: Configurable (peak, astigmatism, gradient, etc.)

2. **Calibration Storage**
   - Seafront: Linear regression only
   - ImSwitch: Linear + polynomial + lookup table

3. **Workflow Integration**
   - Seafront: Protocol-based (async/await)
   - ImSwitch: Workflow-based (synchronous steps)

## Future Enhancements

### Potential Improvements

1. **Adaptive calibration**
   - Re-calibrate if drift detected
   - Temperature-compensated calibration

2. **Multi-wavelength support**
   - Different calibrations per channel
   - Chromatic aberration correction

3. **Machine learning**
   - Train model to predict Z from focus metric
   - Outlier detection and filtering

4. **Performance monitoring**
   - Track autofocus success rate
   - Log timing statistics

5. **Hybrid mode**
   - Start with hardware AF
   - Fallback to software AF if needed

## Testing

### Unit Tests

```python
# Test one-shot autofocus
def test_one_shot_autofocus():
    result = focuslock.performOneStepAutofocus(
        move_to_focus=False,  # Don't move for testing
        threshold_um=1.0
    )
    assert result['success'] == True
    assert result['target_z_position'] is not None

# Test calibration calculation
def test_calculate_z_from_focus():
    z_target = focuslock._calculate_z_position_from_focus_value(100.0)
    assert z_target is not None
    assert isinstance(z_target, float)
```

### Integration Tests

```python
# Test wellplate experiment with hardware AF
def test_wellplate_with_hardware_af():
    experiment = {
        "name": "Test",
        "parameterValue": {
            "autoFocus": True,
            "autoFocusMode": "hardware"
        },
        "pointList": [
            {"x": 0, "y": 0, "iX": 0, "iY": 0}
        ]
    }
    result = experiment_controller.startWellplateExperiment(experiment)
    assert result['status'] == 'running'
```

## Conclusion

The hardware-based autofocus integration provides:

✅ **Fast focusing** for wellplate scanning (7× faster)  
✅ **Flexible metrics** (peak, astigmatism, etc.)  
✅ **Multiple calibration methods** (linear, polynomial, lookup)  
✅ **Safe operation** with limits and error handling  
✅ **Seamless integration** with existing workflow system  
✅ **Backward compatible** with software autofocus  

This enables efficient high-throughput screening while maintaining the option for traditional software autofocus when needed.

## Related Documentation

- [FocusLock Controller Documentation](focuslock.md)
- [Experiment Controller Documentation](experiment.md)
- [Autofocus Controller Documentation](autofocus.md)
- [Workflow Manager Documentation](workflow.md)

## References

- Seafront laser autofocus implementation
- ImSwitch FocusLockController API
- ImSwitch ExperimentController API
- PID controller implementation
