# Claude Code Instructions: PixelCalibration Refactor

## Goal Summary

Refactor the ImSwitch `PixelCalibrationController` and related files to:

1. Make `PixelCalibrationController` a **mandatory, always-on controller** (not config-gated)
2. Establish **PixelCalibration as the single source of truth** for pixel size and flip settings per detector
3. Store and load calibrations **per detector name** (not per objective) on startup
4. **Remove AprilTag / overview calibration** entirely (too complex, error-prone)
5. **Remove duplicate flip/pixelsize storage** in DetectorManagers and config
6. **Fix the frontend** so calibration results are displayed before applying, with editable fields and an Apply button

---

## Part 1 — Backend: PixelCalibrationController.py

### 1.1 Remove AprilTag / overview calibrator imports and all related methods

At the top of `PixelCalibrationController.py`, remove these imports entirely:

```python
# DELETE these lines:
from imswitch.imcontrol.controller.controllers.pixelcalibration.overview_calibrator import OverviewCalibrator
from imswitch.imcontrol.controller.controllers.pixelcalibration.apriltag_grid_calibrator import (
    AprilTagGridCalibrator, GridConfig
)
```

Delete all methods that reference `overviewCalibrator`, `gridCalibrator`, or AprilTag:
- `overviewIsObservationCameraAvailable`
- `overviewIdentifyAxes`
- `overviewMapIlluminationChannels`
- `overviewVerifyHoming`
- `overviewFixStepSign`
- `overviewStream` / `overviewStreamToggle` (MJPEG stream for observation camera)
- `gridMoveToTag`
- `gridCalibrateTransform`
- `_loadGridCalibration`
- `returnObservationCameraImage`
- Any other method whose body references `self.overviewCalibrator` or `self.gridCalibrator`

Delete all instance attributes set in `__init__` related to those:
```python
# DELETE these blocks in __init__:
self.observationCamera = ...
self.observationCameraName = ...
self.observationFlipX = ...
self.observationFlipY = ...
self.overviewCalibrator = OverviewCalibrator(...)
self.gridCalibrator = ...
self._gridRotated180 = ...
self._aprilTagOverlayEnabled = ...
self._overlay_lock = ...
self.overviewStreamRunning = ...
self.overviewStreamStarted = ...
self.overviewStreamQueue = ...
# and the entire ObservationCamera / ObservationCameraFlip config-reading blocks
```

Also delete the config-reading block for `ObservationCamera` and `ObservationCameraFlip`:
```python
# DELETE this block entirely:
if hasattr(self._setupInfo.PixelCalibration, 'ObservationCamera') and ...:
    ...
self.observationFlipX = True
self.observationFlipY = True
if hasattr(self._setupInfo.PixelCalibration, 'ObservationCameraFlip'):
    flip_settings = ...
    ...
```

---

### 1.2 Make the controller always-on (no PixelCalibration config required)

In `__init__`, replace the early-return guard for missing `PixelCalibration` config:

```python
# BEFORE (in _loadAffineCalibrations):
if not hasattr(self._setupInfo, 'PixelCalibration') or self._setupInfo.PixelCalibration is None:
    self._logger.info("No PixelCalibration in setup configuration - using default identity matrix")
    return

# AFTER: keep the method working even with no config, just log and continue
if not hasattr(self._setupInfo, 'PixelCalibration') or self._setupInfo.PixelCalibration is None:
    self._logger.info("No PixelCalibration config found - all detectors start uncalibrated")
    self.affineCalibrations = {}
    return
```

Ensure the controller registers itself even without explicit config. Wherever in the ImSwitch controller registry/loader the controller availability is gated on a config key, remove that gate. The controller should always be loaded. Check `imswitch/imcontrol/controller/MasterController.py` (or equivalent) for:

```python
# Find patterns like:
if hasattr(setupInfo, 'PixelCalibration') and setupInfo.PixelCalibration is not None:
    controllers['PixelCalibration'] = PixelCalibrationController(...)

# Replace with unconditional registration:
controllers['PixelCalibration'] = PixelCalibrationController(...)
```

---

### 1.3 Change calibration storage key from objective → detector name

Currently `affineCalibrations` is keyed by `objective_id` (e.g. `"10x"`, `"default"`).  
Change it to be keyed by **detector name** (e.g. `"WidefieldCamera"`, `"HikCam0"`) AND `objective_id`

**In `_loadAffineCalibrations`:**

```python
# The config structure in JSON will now be:
# "PixelCalibration": {
#   "affineCalibrations": {
#     "WidefieldCamera": { "affine_matrix": [...], "metrics": {...}, "timestamp": "..." },
#     "HikCam0": { ... }
#   }
# }

# After loading, for each detector, apply its calibration:
for detector_name in self._master.detectorsManager.getAllDeviceNames():
    if detector_name in self.affineCalibrations:
        calib_data = self.affineCalibrations[detector_name]
        self._applyCalibrationToDetector(detector_name, calib_data)
    else:
        self._logger.info(f"No calibration for detector '{detector_name}' - using defaults")
```

**New helper: `_applyCalibrationToDetector(detector_name, calib_data)`**

Extract from `_applyCalibrationResults` and `_distributeFlipsToDetectors`. This new method:
1. Gets the detector by name from `detectorsManager`
2. Reads `scale_x_um_per_pixel` and `scale_y_um_per_pixel` from `calib_data['metrics']`
3. Derives `flipX = scale_x < 0`, `flipY = scale_y < 0`
4. Calls `detector.setFlipImage(flipY, flipX)` if the method exists
5. Calls `detector.setPixelSizeUm(avg_pixel_size)` if it exists
6. Logs what was applied

```python
def _applyCalibrationToDetector(self, detector_name: str, calib_data: dict):
    """Apply a calibration dict to a named detector."""
    try:
        detector = self._master.detectorsManager[detector_name]
    except Exception:
        self._logger.warning(f"Detector '{detector_name}' not found, skipping calibration apply")
        return

    metrics = calib_data.get('metrics', {})
    scale_x = metrics.get('scale_x_um_per_pixel', 1.0)
    scale_y = metrics.get('scale_y_um_per_pixel', 1.0)
    avg_pixel_size = (abs(scale_x) + abs(scale_y)) / 2.0
    flipX = scale_x < 0
    flipY = scale_y < 0

    if hasattr(detector, 'setFlipImage'):
        detector.setFlipImage(flipY, flipX)
        self._logger.info(f"[{detector_name}] flip set: Y={flipY}, X={flipX}")

    if hasattr(detector, 'setPixelSizeUm'):
        detector.setPixelSizeUm(avg_pixel_size)
        self._logger.info(f"[{detector_name}] pixel size set: {avg_pixel_size:.4f} µm/px")
```

**Update `calibrateStageAffineInThread`:** change signature so it accepts `detectorName: str` instead of `objectiveId`. The result should be stored with `detector_name` as key.

**Update `calibrateStageAffine` API endpoint:**
```python
@APIExport(runOnUIThread=True, requestType="POST")
def calibrateStageAffine(self, detectorName: str = None, stepSizeUm: float = 100.0,
                         pattern: str = "cross", nSteps: int = 1,
                         crop_size: int = 1024, isDEBUG: bool = False):
    ...
    # Use first detector if not specified
    if detectorName is None:
        all_detectors = self._master.detectorsManager.getAllDeviceNames()
        detectorName = all_detectors[0] if all_detectors else "default"
    ...
```

---

### 1.4 Add a "pending calibration" approval flow

Instead of auto-applying results, store them in a pending state and expose an API for the frontend to approve/apply.

Add to `__init__`:
```python
self._pendingCalibration: dict = {}  # {detector_name: {affine_matrix, metrics, timestamp}}
```

In `calibrateStageAffineInThread`, at the end instead of calling `_applyCalibrationResults`:

```python
# Store as PENDING — do NOT apply yet
self._pendingCalibration[detectorName] = {
    "affine_matrix": result_serializable.get("affine_matrix", []),
    "metrics": result_serializable.get("metrics", {}),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "detector_name": detectorName,
}
self._logger.info(f"Calibration pending approval for detector '{detectorName}'")

return {
    "success": True,
    "pending": True,
    "detectorName": detectorName,
    "metrics": result_serializable.get("metrics", {}),
    "affineMatrix": result_serializable.get("affine_matrix", []),
    "message": "Calibration complete. Review results and call applyPendingCalibration to apply."
}
```

Add new API endpoint `getPendingCalibration`:
```python
@APIExport()
def getPendingCalibration(self, detectorName: str = None):
    """Return pending (not yet applied) calibration results."""
    if detectorName:
        data = self._pendingCalibration.get(detectorName)
        if data is None:
            return {"success": False, "message": f"No pending calibration for '{detectorName}'"}
        return {"success": True, **data}
    # Return all pending
    return {"success": True, "pending": self._pendingCalibration}
```

Add new API endpoint `applyPendingCalibration`:
```python
@APIExport(requestType="POST")
def applyPendingCalibration(self, detectorName: str, affineMatrix: list = None, metrics: dict = None):
    """
    Apply pending calibration (optionally with user-edited values).
    
    The frontend may pass edited affineMatrix/metrics before applying.
    Saves to config and applies to detector immediately.
    """
    pending = self._pendingCalibration.get(detectorName)
    if pending is None:
        return {"success": False, "message": f"No pending calibration for '{detectorName}'"}

    # Allow frontend overrides
    final_matrix = affineMatrix if affineMatrix is not None else pending["affine_matrix"]
    final_metrics = metrics if metrics is not None else pending["metrics"]

    calib_data = {
        "affine_matrix": final_matrix,
        "metrics": final_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detector_name": detectorName,
    }

    # Persist to config
    self._setupInfo.setAffineCalibration(detectorName, calib_data)
    try:
        import imswitch.imcontrol.model.configfiletools as configfiletools
        options, _ = configfiletools.loadOptions()
        configfiletools.saveSetupInfo(options, self._setupInfo)
    except Exception as e:
        self._logger.warning(f"Could not save calibration to disk: {e}")

    # Apply immediately to detector
    self.affineCalibrations[detectorName] = calib_data
    self._applyCalibrationToDetector(detectorName, calib_data)

    # Clear pending
    del self._pendingCalibration[detectorName]

    return {
        "success": True,
        "detectorName": detectorName,
        "message": f"Calibration applied and saved for '{detectorName}'"
    }
```

Add `discardPendingCalibration`:
```python
@APIExport(requestType="POST")
def discardPendingCalibration(self, detectorName: str):
    """Discard pending calibration without applying."""
    if detectorName in self._pendingCalibration:
        del self._pendingCalibration[detectorName]
        return {"success": True, "message": f"Pending calibration discarded for '{detectorName}'"}
    return {"success": False, "message": f"No pending calibration for '{detectorName}'"}
```

Also update `getCalibrationData` and `setCalibrationData` to use `detectorName` as key consistently.

---

### 1.5 Remove `_distributeFlipsToDetectors` and `_distributePixelSizesToObjectives`

These are now replaced by `_applyCalibrationToDetector`. Delete the old methods. Remove all calls to them.

Also remove `_distributePixelSizesToObjectives` — the objective manager should read pixel size from PixelCalibration when needed, not be pushed to. If `objectiveManager` integration is still desired, a single call in `_applyCalibrationToDetector` can optionally update the matching objective slot, but only if the current detector is the primary acquisition detector.

---

### 1.6 Clean up PixelCalibrationController.__init__

After the above changes, `__init__` should be simplified to roughly:

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._logger = initLogger(self)

    self.affineCalibrations = {}
    self._pendingCalibration = {}

    # Get current objective for backward compat (can stay)
    self.currentObjective = self._getCurrentObjectiveId()

    # Load and apply per-detector calibrations from config
    self._loadAffineCalibrations()
```

---

## Part 2 — Backend: HikCamManager.py (and other DetectorManagers)

### 2.1 Remove hardcoded flip settings from config loading

In `HikCamManager.__init__`, remove this block entirely:

```python
# DELETE:
try:
    flipX = detectorInfo.managerProperties['hikcam']['flipX']
except:
    flipX = False

try:
    flipY = detectorInfo.managerProperties['hikcam']['flipY']
except:
    flipY = False

flipImage = (flipY, flipX)
```

Replace with:
```python
flipImage = (False, False)  # Will be set by PixelCalibrationController on startup
```

The camera should still be constructed with `flipImage=(False, False)` initially. `PixelCalibrationController._loadAffineCalibrations()` will call `setFlipImage()` during startup once the correct values are known.

**Do the same for any other DetectorManager** (TIS, OpenCV, etc.) that reads `flipX`/`flipY` from `managerProperties`.

### 2.2 Remove pixelSize from managerProperties as authoritative source

In `HikCamManager.__init__`, the `'Camera pixel size'` parameter should still exist as a fallback display parameter, but its initial value should be `1.0` (uncalibrated default) rather than reading from `cameraEffPixelsize`:

```python
# Change:
'Camera pixel size': DetectorNumberParameter(group='Miscellaneous', value=pixelSize, ...)

# To:
'Camera pixel size': DetectorNumberParameter(group='Miscellaneous', value=1.0, ...)
# PixelCalibrationController will update this via setPixelSizeUm() on startup
```

The `cameraEffPixelsize` property in the JSON config can be removed or kept as a documentation note, but it must not be used as the runtime pixel size source.

---

## Part 3 — Backend: overview_calibrator.py and apriltag_grid_calibrator.py

### 3.1 Remove both files

Delete (or move to an `_archive` folder):
- `imswitch/imcontrol/controller/controllers/pixelcalibration/overview_calibrator.py`
- `imswitch/imcontrol/controller/controllers/pixelcalibration/apriltag_grid_calibrator.py`
- Their compiled `.pyc` caches (auto-cleaned by Python, but can delete manually)

Update `imswitch/imcontrol/controller/controllers/pixelcalibration/__init__.py` to remove any exports of these modules.

---

## Part 4 — Frontend: PixelCalibrationTab.js

### 4.1 Remove overview stream UI

Remove:
- The "Overview Camera" `Paper` section (the MJPEG stream box)
- `overviewStreamUrl`, `overviewStreamActive` state
- `handleOverviewStreamToggle` handler
- Import of `apiPixelCalibrationControllerOverviewStream`

Keep the "Detector Camera" live view section using `LiveViewControlWrapper`.


### 4.3 Implement the "pending calibration" review UI

The calibration flow should be:

```
[Start Calibration] 
    → backend runs, returns pending result
    → frontend shows "Review Results" panel
        - displays affine matrix (editable JSON or per-field inputs)
        - displays pixel size X, pixel size Y (editable number inputs)  
        - displays flip X, flip Y (derived from sign of pixel sizes, shown as checkboxes, editable)
        - displays rotation_deg, rmse_um, quality (read-only)
    → [Apply] button  → calls applyPendingCalibration
    → [Discard] button → calls discardPendingCalibration
```

**State additions:**
```javascript
const [pendingCalibration, setPendingCalibration] = useState(null);
// pendingCalibration shape: { affineMatrix, metrics, detectorName }

// Editable copies of the pending result
const [editPixelSizeX, setEditPixelSizeX] = useState('');
const [editPixelSizeY, setEditPixelSizeY] = useState('');
const [editFlipX, setEditFlipX] = useState(false);
const [editFlipY, setEditFlipY] = useState(false);
const [editAffineMatrix, setEditAffineMatrix] = useState('');
```

**After calibration completes** (polling or direct response), instead of showing `result` in a read-only pre block:

```javascript
const handleCalibrateAffine = async () => {
  setLoading(true);
  setError('');
  setPendingCalibration(null);
  setStatus('Running calibration...');

  try {
    const response = await apiPixelCalibrationControllerCalibrateStageAffine({
      detectorName: selectedDetector,
      stepSizeUm,
      pattern,
      nSteps,
    });

    // Calibration runs in a thread; poll for pending result
    // OR if the API is synchronous and returns the result directly:
    if (response.pending && response.metrics) {
      const metrics = response.metrics;
      setPendingCalibration(response);
      setEditPixelSizeX(String(metrics.scale_x_um_per_pixel?.toFixed(4) ?? '1.0'));
      setEditPixelSizeY(String(metrics.scale_y_um_per_pixel?.toFixed(4) ?? '1.0'));
      setEditFlipX(metrics.scale_x_um_per_pixel < 0);
      setEditFlipY(metrics.scale_y_um_per_pixel < 0);
      setEditAffineMatrix(JSON.stringify(response.affineMatrix, null, 2));
      setStatus('Calibration complete. Review and apply results below.');
    }
  } catch (err) {
    setError(`Calibration failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
};
```

**Polling** (since calibration runs in a thread and `calibrateStageAffine` returns immediately):

Because the backend returns `{"success": true, "message": "Calibration started in background thread"}` immediately, you need to poll `getPendingCalibration` to know when results are ready:

```javascript
useEffect(() => {
  if (!loading) return;
  
  const interval = setInterval(async () => {
    try {
      const res = await fetch(
        `http://${hostIP}:${hostPort}/PixelCalibrationController/getPendingCalibration?detectorName=${selectedDetector}`
      ).then(r => r.json());

      if (res.success && res.affine_matrix) {
        clearInterval(interval);
        setLoading(false);
        const metrics = res.metrics || {};
        setPendingCalibration(res);
        setEditPixelSizeX(String(Math.abs(metrics.scale_x_um_per_pixel ?? 1).toFixed(4)));
        setEditPixelSizeY(String(Math.abs(metrics.scale_y_um_per_pixel ?? 1).toFixed(4)));
        setEditFlipX((metrics.scale_x_um_per_pixel ?? 1) < 0);
        setEditFlipY((metrics.scale_y_um_per_pixel ?? 1) < 0);
        setEditAffineMatrix(JSON.stringify(res.affine_matrix, null, 2));
        setStatus('Review results and apply or discard.');
      }
    } catch {}
  }, 2000); // poll every 2s

  return () => clearInterval(interval);
}, [loading, hostIP, hostPort, selectedDetector]);
```

**Review panel JSX** (add after the controls Paper, shown only when `pendingCalibration !== null`):

```jsx
{pendingCalibration && (
  <Paper sx={{ p: 2, mb: 2, border: '2px solid orange' }}>
    <Typography variant="h6" gutterBottom>
      ⚠ Review Calibration Results
    </Typography>
    <Alert severity="warning" sx={{ mb: 2 }}>
      These values have NOT been applied yet. Edit if needed, then click Apply.
    </Alert>

    {/* Pixel Size */}
    <Typography variant="subtitle2">Pixel Size (µm/px)</Typography>
    <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
      <TextField
        label="Scale X (µm/px)"
        type="number"
        value={editPixelSizeX}
        onChange={(e) => setEditPixelSizeX(e.target.value)}
        size="small"
        inputProps={{ step: 0.0001 }}
      />
      <TextField
        label="Scale Y (µm/px)"
        type="number"
        value={editPixelSizeY}
        onChange={(e) => setEditPixelSizeY(e.target.value)}
        size="small"
        inputProps={{ step: 0.0001 }}
      />
    </Box>

    {/* Flip */}
    <Typography variant="subtitle2">Flip</Typography>
    <Box sx={{ mb: 2 }}>
      <FormControlLabel
        control={<Checkbox checked={editFlipX} onChange={(e) => setEditFlipX(e.target.checked)} />}
        label="Flip X"
      />
      <FormControlLabel
        control={<Checkbox checked={editFlipY} onChange={(e) => setEditFlipY(e.target.checked)} />}
        label="Flip Y"
      />
    </Box>

    {/* Read-only quality metrics */}
    <Typography variant="subtitle2">Quality Metrics (read-only)</Typography>
    <Box sx={{ mb: 2, backgroundColor: '#f5f5f5', p: 1, borderRadius: 1 }}>
      <Typography variant="body2">
        Rotation: {pendingCalibration.metrics?.rotation_deg?.toFixed(2) ?? 'N/A'}°
      </Typography>
      <Typography variant="body2">
        RMSE: {pendingCalibration.metrics?.rmse_um?.toFixed(3) ?? 'N/A'} µm
      </Typography>
      <Typography variant="body2">
        Quality: {pendingCalibration.metrics?.quality ?? 'N/A'}
      </Typography>
      <Typography variant="body2">
        Mean correlation: {pendingCalibration.metrics?.mean_correlation?.toFixed(3) ?? 'N/A'}
      </Typography>
    </Box>

    {/* Editable affine matrix (advanced) */}
    <Typography variant="subtitle2">Affine Matrix (advanced edit)</Typography>
    <TextField
      multiline
      rows={4}
      fullWidth
      value={editAffineMatrix}
      onChange={(e) => setEditAffineMatrix(e.target.value)}
      sx={{ mb: 2, fontFamily: 'monospace' }}
      helperText="2×3 matrix: [[a, b, tx], [c, d, ty]]"
    />

    {/* Action buttons */}
    <Box sx={{ display: 'flex', gap: 2 }}>
      <Button
        variant="contained"
        color="success"
        fullWidth
        onClick={handleApplyCalibration}
      >
        ✓ Apply Calibration
      </Button>
      <Button
        variant="outlined"
        color="error"
        fullWidth
        onClick={handleDiscardCalibration}
      >
        ✗ Discard
      </Button>
    </Box>
  </Paper>
)}
```

**Apply handler** — rebuilds metrics from edited fields, then sends to backend:

```javascript
const handleApplyCalibration = async () => {
  try {
    // Rebuild affine matrix from text field
    let finalMatrix;
    try {
      finalMatrix = JSON.parse(editAffineMatrix);
    } catch {
      setError('Invalid affine matrix JSON');
      return;
    }

    // Rebuild metrics with edited pixel sizes and flip
    const scaleX = parseFloat(editPixelSizeX) * (editFlipX ? -1 : 1);
    const scaleY = parseFloat(editPixelSizeY) * (editFlipY ? -1 : 1);
    const updatedMetrics = {
      ...pendingCalibration.metrics,
      scale_x_um_per_pixel: scaleX,
      scale_y_um_per_pixel: scaleY,
    };

    const res = await fetch(
      `http://${hostIP}:${hostPort}/PixelCalibrationController/applyPendingCalibration`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          detectorName: selectedDetector,
          affineMatrix: finalMatrix,
          metrics: updatedMetrics,
        }),
      }
    ).then(r => r.json());

    if (res.success) {
      setPendingCalibration(null);
      setStatus('Calibration applied and saved!');
      setResult(res);
    } else {
      setError(res.message || 'Apply failed');
    }
  } catch (err) {
    setError(`Apply failed: ${err.message}`);
  }
};

const handleDiscardCalibration = async () => {
  try {
    await fetch(
      `http://${hostIP}:${hostPort}/PixelCalibrationController/discardPendingCalibration`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detectorName: selectedDetector }),
      }
    );
    setPendingCalibration(null);
    setStatus('Calibration discarded.');
  } catch (err) {
    setError(`Discard failed: ${err.message}`);
  }
};
```

### 4.4 Update the API call for calibrateStageAffine

Update `apiPixelCalibrationControllerCalibrateStageAffine` (or the inline fetch) to send `detectorName` instead of `objectiveId`:

```javascript
const response = await apiPixelCalibrationControllerCalibrateStageAffine({
  detectorName: selectedDetector,  // was: objectiveId
  stepSizeUm,
  pattern,
  nSteps,
  // removed: validate (no longer needed)
});
```

---

## Part 5 — Config / JSON schema changes

### 5.1 PixelCalibration JSON structure change

**Old structure:**
```json
"PixelCalibration": {
  "ObservationCamera": "OverviewCam",
  "ObservationCameraFlip": {"flipX": true, "flipY": true},
  "affineCalibrations": {
    "default": { "affine_matrix": [...], "metrics": {...} }
  }
}
```

**New structure** (backward-compatible read, write in new format):
```json
"PixelCalibration": {
  "affineCalibrations": {
    "WidefieldCamera": { "affine_matrix": [...], "metrics": {...}, "timestamp": "..." },
    "HikCam0": { "affine_matrix": [...], "metrics": {...}, "timestamp": "..." }
  }
}
```

The `ObservationCamera`, `ObservationCameraFlip` keys should be **ignored on read** and not written on save.

In the `SetupInfo` model class (wherever `PixelCalibration` is defined as a dataclass or dict schema), update to remove the `ObservationCamera` and `ObservationCameraFlip` fields.

---

## Part 6 — Verification checklist

After making changes, verify:

- [ ] `PixelCalibrationController` imports cleanly without `overview_calibrator` or `apriltag_grid_calibrator`
- [ ] Starting ImSwitch with no `PixelCalibration` config does not crash — controller loads with empty calibrations
- [ ] Starting with existing `affineCalibrations` config applies flip+pixelsize to each named detector on startup
- [ ] Calling `calibrateStageAffine` returns `{"success": true, "pending": true, ...}` with `affineMatrix` and `metrics` in the response
- [ ] Calling `getPendingCalibration` after calibration returns the pending data
- [ ] Frontend shows the review panel with editable pixel size and flip fields
- [ ] Editing pixel size values and clicking Apply sends the corrected values to `applyPendingCalibration`
- [ ] After Apply, config file on disk contains the new calibration under the detector name key
- [ ] After Apply, `detector.setFlipImage()` and `detector.setPixelSizeUm()` reflect the new values
- [ ] HikCamManager no longer reads `flipX`/`flipY` from `managerProperties['hikcam']`
- [ ] No references to `ObservationCamera` or `ObservationCameraFlip` remain in Python code
- [ ] Frontend no longer shows the overview MJPEG stream section
- [ ] `discardPendingCalibration` clears pending without modifying config or detector state

---

## File map — what to change

| File | Action |
|---|---|
| `PixelCalibrationController.py` | Major refactor per Parts 1 & 4 |
| `HikCamManager.py` | Remove flip/pixelsize from init (Part 2) |
| `overview_calibrator.py` | Delete |
| `apriltag_grid_calibrator.py` | Delete |
| `pixelcalibration/__init__.py` | Remove deleted module exports |
| `PixelCalibrationTab.js` | Refactor UI per Part 4 |
| `MasterController.py` (or equivalent) | Make PixelCalibrationController always register |
| Setup JSON config files | Update structure per Part 5 |
| Other DetectorManager files (TIS, OpenCV, etc.) | Remove hardcoded flip/pixelsize from init |

---

## Notes for Claude Code

- **Work file by file.** Start with `PixelCalibrationController.py` since it is the largest change. 
- **Run the test suite after each file** if one exists (`pytest imswitch/tests/` or similar).
- **Preserve the `PixelCalibrationClass` inner class** and the `calibrate_affine` method — the actual calibration math is correct. Only change where results are stored and when they are applied.
- **Keep `_getCurrentObjectiveId`** for now — it is used in other places and removing it is a separate concern.
- **Do not remove the `validate_calibration` import** from `affine_stage_calibration` yet — it may be used elsewhere. Just stop calling it inside `PixelCalibrationController`.
- **Search for all usages** of `ObservationCamera`, `observationCamera`, `overviewCalibrator`, `gridCalibrator`, `AprilTag`, `OverviewCalibrator`, `AprilTagGridCalibrator` across the entire repo before deleting — there may be other references in widget files or API spec files.
- **Search for `cameraEffPixelsize` and `flipX`/`flipY`** in all DetectorManager files to find others that need the same treatment as `HikCamManager`.
