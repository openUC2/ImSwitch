# Focus Map Feature Documentation

## Table of Contents

- [Overview](#overview)
- [User Guide](#user-guide)
  - [What is Focus Mapping?](#what-is-focus-mapping)
  - [When to Use Focus Mapping](#when-to-use-focus-mapping)
  - [How to Use](#how-to-use)
  - [Configuration Parameters](#configuration-parameters)
  - [Interpreting Results](#interpreting-results)
- [Developer Guide](#developer-guide)
  - [Architecture Overview](#architecture-overview)
  - [Backend Implementation](#backend-implementation)
  - [Frontend Implementation](#frontend-implementation)
  - [API Reference](#api-reference)
  - [Data Flow](#data-flow)
  - [Testing with Virtual Microscope](#testing-with-virtual-microscope)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **Focus Map** feature automatically measures and corrects for focus (Z-height) variations across the scan area during multi-position imaging experiments. It addresses the common problem of sample tilt, stage non-flatness, or coverslip curvature that causes different regions to be at different focal planes.

**Key Benefits:**
- ✅ Automatic Z-correction at every XY position
- ✅ Reduces manual intervention in large-scale imaging
- ✅ Improves image quality across the entire scan area
- ✅ Supports multiple surface fitting methods (spline, RBF, constant)
- ✅ Per-region or global focus mapping strategies

---

## User Guide

### What is Focus Mapping?

Focus mapping is a pre-acquisition calibration step that:

1. **Measures** the optimal focus (Z position) at a grid of points across your scan area
2. **Fits** a mathematical surface to these measurements (Z = f(X, Y))
3. **Applies** the fitted Z-correction automatically during image acquisition

This ensures every position in your scan is in focus, even if the sample is tilted or curved.

### When to Use Focus Mapping

**Recommended for:**
- Large field-of-view scans (multi-well plates, slides)
- Tilted samples or imperfect stage mounting
- High-magnification imaging where depth of field is limited
- Automated screening where manual refocusing is impractical

**Not needed for:**
- Single-position imaging
- Small scan areas where focus variation is negligible
- Samples with sufficient depth of field
- Pre-leveled samples with hardware autofocus

### How to Use

#### Step 1: Enable Focus Mapping

1. Open the **Experiment Designer**
2. Click on the **Focus Map** dimension tab (icon: 🏔️ Landscape)
3. Toggle **Enable Focus Mapping** to ON

#### Step 2: Configure Grid Parameters

- **Grid Rows/Cols**: Number of autofocus measurement points (default: 3×3)
  - More points → better accuracy, longer measurement time
  - Recommended: 3×3 for small areas, 5×5 for large areas
- **Add Margin**: Extends measurement grid slightly beyond scan bounds
  - Recommended: ON for better edge extrapolation
- **Fit per Region**: Creates separate focus maps for each scan area/well
  - ON: Better for multi-well plates with individual tilts
  - OFF: Single global focus map for entire experiment

#### Step 3: Choose Fitting Method

- **Spline** (default): Smooth polynomial surface, best for gradual tilts
- **RBF (Radial Basis Function)**: Handles complex non-linear surfaces
- **Constant**: Uses average Z (fallback for single measurement point)

The system automatically falls back to simpler methods if fitting fails.

#### Step 4: Advanced Settings (Optional)

Expand **Advanced Settings** accordion:

- **Smoothing Factor** (0.0-1.0): Controls spline smoothness
  - 0 = interpolates exactly through points
  - >0 = smooths out measurement noise
- **Z Offset**: Global Z adjustment applied after interpolation (µm)
- **Settle Time**: Delay after Z movement before acquisition (ms)
- **Clamp Z**: Limit interpolated Z to safe range (Z Min/Max)
- **Apply during scan**: Enable/disable Z correction application

#### Step 5: Compute Focus Map

1. Click **Compute All** to measure focus at all grid points
2. Wait for completion (progress shown per group)
3. Review results:
   - Green checkmark ✅ = Ready
   - Spinning icon 🔄 = Computing
   - Red X ❌ = Error
4. Click on a group to visualize the focus map

#### Step 6: Start Experiment

Click **Start** in the main toolbar. The focus map Z-corrections will be applied automatically during acquisition.

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Master enable/disable switch |
| `rows` | int | `3` | Number of grid rows (1-20) |
| `cols` | int | `3` | Number of grid columns (1-20) |
| `add_margin` | bool | `false` | Extend grid beyond scan bounds |
| `fit_by_region` | bool | `true` | Separate fit per scan area |
| `method` | enum | `"spline"` | Fitting method: `spline`, `rbf`, `constant` |
| `smoothing_factor` | float | `0.1` | Spline smoothing (0 = exact fit) |
| `apply_during_scan` | bool | `true` | Apply Z correction during acquisition |
| `z_offset` | float | `0.0` | Global Z offset (µm) |
| `clamp_enabled` | bool | `false` | Enable Z clamping |
| `z_min` / `z_max` | float | `0.0` | Z clamp limits (µm) |
| `settle_ms` | int | `0` | Post-movement settle delay (ms) |
| `autofocus_profile` | string | `null` | Autofocus profile name (uses default if null) |

### Interpreting Results

#### Visualization

The focus map preview shows:

- **Heatmap background**: Fitted Z-surface (blue = low Z, red = high Z)
- **Circles with borders**: Measured autofocus points
- **Color bar** (right): Z-height scale in µm
- **Axes**: XY coordinate ranges

#### Fit Statistics

- **Method**: Which fitting method was used (spline/rbf/constant)
- **Z Range**: Total focus variation across area (µm)
- **MAE (Mean Absolute Error)**: Average difference between measured and fitted Z (µm)
  - <1 µm: Excellent fit 🟢
  - 1-5 µm: Acceptable fit 🟡
  - >5 µm: Poor fit, check configuration 🔴
- **Points**: Number of successfully measured grid points
- **Fallback Warning**: Indicates primary method failed, backup used

---

## Developer Guide

### Architecture Overview

The focus map feature consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
├──────────────────────┬──────────────────────┬───────────────┤
│  FocusMapDimension   │  FocusMapSlice       │  Visualization│
│  (UI Controls)       │  (Redux State)       │  (Canvas)     │
└──────────────────────┴──────────────────────┴───────────────┘
                              ▼ API Calls
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Python/FastAPI)                  │
├──────────────────────┬──────────────────────┬───────────────┤
│ ExperimentController │  FocusMapManager     │  FocusMap     │
│ (API Endpoints)      │  (Storage/Compute)   │  (Core Model) │
└──────────────────────┴──────────────────────┴───────────────┘
                              ▼ Execution
┌─────────────────────────────────────────────────────────────┐
│           ExperimentNormalMode (Workflow Generation)         │
│          → WorkflowStep (Z move commands with focus map)     │
└─────────────────────────────────────────────────────────────┘
```

### Backend Implementation

#### File Structure

```
imswitch/imcontrol/
├── model/
│   └── focus_map.py                    # Core FocusMap class
└── controller/controllers/
    ├── ExperimentController.py         # API + FocusMapManager
    └── experiment_controller/
        └── experiment_normal_mode.py   # Workflow integration
```

#### Core Classes

##### `FocusMap` (model/focus_map.py)

Core data structure and algorithms for focus mapping.

**Responsibilities:**
- Generate grid points within bounding box
- Fit surface using scipy interpolation (spline/RBF)
- Interpolate Z at arbitrary XY positions
- Serialize/deserialize focus map state

**Key Methods:**

```python
# Generate measurement grid
points = FocusMap.generate_grid_points(
    bounds=(x_min, x_max, y_min, y_max),
    rows=3, cols=3,
    add_margin=False
)

# Create and fit surface
focus_map = FocusMap(bounds)
focus_map.fit(
    points=[(x, y, z), ...],
    method="spline",
    smoothing_factor=0.1
)

# Query Z at position
z = focus_map.interpolate(x=100, y=200)

# Get preview grid for visualization
preview = focus_map.get_preview_grid(resolution=30)
# Returns: { "x": [...], "y": [...], "z": [[...]] }
```

**Fitting Methods:**

1. **Spline** (`scipy.interpolate.RectBivariateSpline`)
   - Requires ≥4 points
   - Smooth 2D polynomial surface
   - Supports smoothing factor for noise reduction
   - Falls back to RBF if fitting fails

2. **RBF** (`scipy.interpolate.Rbf`)
   - Handles irregular point distributions
   - More flexible than spline
   - Used as fallback if spline fails

3. **Constant**
   - Uses mean Z of all points
   - Fallback for single-point or failed fits

##### `FocusMapManager` (ExperimentController.py)

Manages focus map lifecycle and storage.

**Responsibilities:**
- Store focus maps per experiment group
- Trigger computation via ExperimentController
- Provide access to stored maps

**Key Methods:**

```python
manager = FocusMapManager()

# Compute focus map for a region
result = manager.compute_for_group(
    group_id="Well_A1",
    bounds=(x_min, x_max, y_min, y_max),
    config=focus_map_config,
    autofocus_controller=af_controller,
    positioner_name="XYStage"
)

# Retrieve focus map
focus_map = manager.get_map("Well_A1")

# Clear all or specific group
manager.clear(group_id="Well_A1")
```

##### `FocusMapConfig` (ExperimentController.py)

Pydantic model for configuration validation.

```python
class FocusMapConfig(BaseModel):
    enabled: bool = False
    rows: int = 3
    cols: int = 3
    add_margin: bool = False
    fit_by_region: bool = True
    method: Literal["spline", "rbf", "constant"] = "spline"
    smoothing_factor: float = 0.1
    apply_during_scan: bool = True
    z_offset: float = 0.0
    clamp_enabled: bool = False
    z_min: float = 0.0
    z_max: float = 0.0
    autofocus_profile: Optional[str] = None
    settle_ms: int = 0
    store_debug_artifacts: bool = True
```

#### Workflow Integration

The focus map is applied in `experiment_normal_mode.py` during workflow generation:

```python
# In _get_workflow_steps_list()
focus_map_config = getattr(self.controller, '_active_focus_map_config', None)

if focus_map_config and focus_map_config.enabled and focus_map_config.apply_during_scan:
    area_id = m_point.get("areaName", m_point.get("wellId", "global"))
    focus_map_obj = self.controller._focus_map_manager.get_map(area_id or "global")
    
    if focus_map_obj:
        focus_z = focus_map_obj.interpolate(m_point["x"], m_point["y"])
        if focus_z is not None:
            focus_z += focus_map_config.z_offset
            
            # Optional clamping
            if focus_map_config.clamp_enabled:
                focus_z = max(config.z_min, min(config.z_max, focus_z))
            
            # Add Z movement step
            workflow_steps.append(
                WorkflowStep(action="move_stage_z", value=focus_z, ...)
            )
            
            # Optional settle delay
            if focus_map_config.settle_ms > 0:
                workflow_steps.append(
                    WorkflowStep(action="wait", value=settle_ms, ...)
                )
```

### Frontend Implementation

#### File Structure

```
frontend/src/
├── state/slices/
│   └── FocusMapSlice.js                # Redux state management
├── backendapi/
│   ├── apiExperimentControllerComputeFocusMap.js
│   ├── apiExperimentControllerGetFocusMap.js
│   ├── apiExperimentControllerGetFocusMapPreview.js
│   └── apiExperimentControllerClearFocusMap.js
└── axon/experiment-designer/
    ├── FocusMapDimension.js            # Main UI component
    └── FocusMapVisualization.js        # Canvas visualization
```

#### Redux State Structure

```javascript
{
  focusMap: {
    config: {
      enabled: false,
      rows: 3,
      cols: 3,
      method: "spline",
      // ... all config parameters
    },
    results: {
      "Well_A1": {
        group_id: "Well_A1",
        group_name: "Well A1",
        points: [{ x, y, z }, ...],
        fit_stats: {
          method: "spline",
          n_points: 9,
          z_range: 5.2,
          mean_abs_error: 0.15,
          fallback_used: false
        },
        status: "ready"  // "ready" | "measuring" | "fitting" | "error"
      }
    },
    ui: {
      isComputing: false,
      computingGroupId: null,
      selectedGroupId: null,
      error: null
    }
  }
}
```

#### Component Hierarchy

```
<ExperimentDesigner>
  └─ <DimensionBar>
       └─ [Focus Map Tab]
  └─ <FocusMapDimension>              ← Active when FOCUS_MAP selected
       ├─ Configuration Controls
       ├─ Compute/Clear Actions
       ├─ Group Results List
       └─ <FocusMapVisualization>     ← Canvas-based preview
```

#### Key React Hooks

```javascript
// FocusMapDimension.js
const focusMapState = useSelector(focusMapSlice.getFocusMapState);
const { config, results, ui } = focusMapState;

// Update dimension summary for bar display
useEffect(() => {
  let summary = config.enabled 
    ? `${readyCount}/${groupCount} groups mapped (${config.method})`
    : "Disabled";
  dispatch(experimentUISlice.setDimensionSummary({ 
    dimension: "focusMap", 
    summary 
  }));
}, [config, results]);

// Compute focus map
const handleComputeAll = async () => {
  dispatch(focusMapSlice.setFocusMapComputing({ isComputing: true }));
  const data = await apiExperimentControllerComputeFocusMap(config);
  dispatch(focusMapSlice.setFocusMapResults(data));
};
```

### API Reference

#### POST `/ExperimentController/computeFocusMap`

Compute focus map for all scan regions.

**Request Body:**
```json
{
  "groupId": "Well_A1",  // optional, null = all groups
  "focusMapConfig": {
    "enabled": true,
    "rows": 3,
    "cols": 3,
    "method": "spline",
    // ... full config
  }
}
```

**Response:**
```json
{
  "Well_A1": {
    "group_id": "Well_A1",
    "points": [
      { "x": 100.0, "y": 200.0, "z": 50.5 },
      // ...
    ],
    "fit_stats": {
      "method": "spline",
      "n_points": 9,
      "z_range": 5.2,
      "mean_abs_error": 0.15,
      "max_abs_error": 0.42,
      "fallback_used": false,
      "fallback_reason": null
    },
    "status": "ready"
  }
}
```

#### GET `/ExperimentController/getFocusMap`

Retrieve stored focus maps.

**Query Parameters:**
- `groupId` (optional): Specific group ID, omit for all groups

**Response:** Same as compute endpoint

#### GET `/ExperimentController/getFocusMapPreview`

Get preview grid for visualization.

**Query Parameters:**
- `groupId` (required): Group to preview
- `resolution` (optional, default=30): Grid resolution

**Response:**
```json
{
  "group_id": "Well_A1",
  "measured_points": [{ "x": 100, "y": 200, "z": 50.5 }, ...],
  "preview_grid": {
    "x": [0, 10, 20, ...],     // X coordinates
    "y": [0, 10, 20, ...],     // Y coordinates
    "z": [[50.1, 50.3, ...],   // Z values [row][col]
          [50.4, 50.6, ...],
          ...]
  },
  "fit_stats": { ... }
}
```

#### POST `/ExperimentController/clearFocusMap`

Clear stored focus maps.

**Request Body:**
```json
{
  "groupId": "Well_A1"  // optional, null = clear all
}
```

**Response:** `{ "message": "Focus maps cleared" }`

### Data Flow

#### Measurement Phase

```
1. User clicks "Compute All" in UI
   ↓
2. Frontend dispatches API call with config
   ↓
3. ExperimentController._run_focus_map_phase()
   ↓
4. For each scan area:
   - Extract XY bounds
   - Generate grid points via FocusMap.generate_grid_points()
   - Move stage to each point
   - Run autofocus via AutofocusController
   - Collect (x, y, z) measurements
   ↓
5. FocusMap.fit() with fallback chain
   ↓
6. Store in FocusMapManager._maps[group_id]
   ↓
7. Return results to frontend
   ↓
8. Frontend updates Redux state and displays visualization
```

#### Application Phase (During Experiment)

```
1. User clicks "Start" experiment
   ↓
2. ExperimentController.startWellplateExperiment()
   - Stores focus_map_config in self._active_focus_map_config
   ↓
3. ExperimentNormalMode generates workflow steps
   ↓
4. For each scan position:
   - Check if focus map enabled
   - Get area_id from point metadata
   - Retrieve FocusMap from manager
   - Interpolate Z for current XY
   - Add WorkflowStep(action="move_stage_z", value=focus_z)
   ↓
5. WorkflowsManager executes steps sequentially
   - Stage moves to focus-corrected Z
   - Optional settle delay
   - Camera acquires image
```

### Testing with Virtual Microscope

The virtual microscope supports simulated focus surfaces for testing.

#### Configuration

In `example_virtual_microscope.json`:

```json
{
  "rs232devices": {
    "VirtualMicroscope": {
      "managerProperties": {
        "focus_surface_": {
          "enabled": true,
          "tilt_x": 0.005,        // µm per µm in X
          "tilt_y": 0.003,        // µm per µm in Y
          "curvature": 0.0001,    // Quadratic term
          "noise_std": 0.1,       // Random noise (µm)
          "center_x": 0.0,        // Tilt center (µm)
          "center_y": 0.0
        }
      }
    }
  }
}
```

Note: Use underscore suffix `focus_surface_` to disable by default (ImSwitch convention).

#### Effective Z Calculation

```python
# VirtualMicroscopeManager.py → Positioner.get_effective_z()
def get_effective_z(self, x, y):
    dx = x - center_x
    dy = y - center_y
    z_surface = (tilt_x * dx + 
                 tilt_y * dy + 
                 curvature * (dx**2 + dy**2) +
                 np.random.normal(0, noise_std))
    return position["Z"] + z_surface
```

This simulates realistic focus variation for testing the focus map feature.

---

## Troubleshooting

### Common Issues

#### Problem: "No focus maps computed"

**Cause:** Focus map feature not enabled or compute not triggered  
**Solution:**
1. Enable "Enable Focus Mapping" toggle
2. Click "Compute All" button
3. Wait for completion (check group status icons)

#### Problem: "Fitting failed" or "Fallback used" warning

**Cause:** Insufficient measurement points or autofocus failures  
**Solution:**
1. Increase grid size (e.g., 3×3 → 5×5)
2. Check autofocus settings (range, resolution)
3. Verify sample has sufficient contrast
4. Review individual point measurements in debug output

#### Problem: "MAE > 5 µm" (poor fit quality)

**Cause:** Complex surface shape or measurement noise  
**Solution:**
1. Try RBF method instead of spline
2. Increase grid density
3. Reduce smoothing factor for exact interpolation
4. Check for stage backlash or mechanical issues

#### Problem: "Z clamping warnings during scan"

**Cause:** Fitted Z exceeds safe limits  
**Solution:**
1. Enable Z clamping in Advanced Settings
2. Set appropriate `z_min` and `z_max` values
3. Review focus map visualization for outliers
4. Consider fit-by-region if global fit is poor

#### Problem: "No Z correction during acquisition"

**Cause:** `apply_during_scan` disabled or focus map not loaded  
**Solution:**
1. Check "Apply Z correction during scan" toggle
2. Ensure focus map computed before starting experiment
3. Verify experiment mode is "normal" (not "performance")
4. Check console logs for `focus_map_obj` retrieval

### Debug Mode

Enable debug artifacts storage:

```python
config = FocusMapConfig(
    enabled=True,
    store_debug_artifacts=True,  # Saves raw measurements
    ...
)
```

This stores detailed logs and measurement data for troubleshooting.

### Backend Logs

Check Python console for focus map operations:

```
[FocusMap] Generating 3×3 grid for bounds (0, 1000, 0, 1000)
[FocusMap] Measured 9/9 points successfully
[FocusMap] Fitting with method=spline, smoothing=0.1
[FocusMap] Fit successful: z_range=4.2µm, MAE=0.18µm
```

### Frontend DevTools

Redux DevTools shows focus map state changes:

```javascript
Action: focusMap/setFocusMapComputing
Payload: { isComputing: true, groupId: null }

Action: focusMap/setFocusMapResults
Payload: { "Well_A1": { ... } }
```

---

## Appendix

### Performance Considerations

- **Grid size**: 3×3 (9 points) ≈ 10-30 seconds per group
- **Autofocus time**: ~1-3 seconds per point
- **Fit computation**: <100ms (negligible)
- **Interpolation overhead**: <1ms per position

**Recommendation:** Balance accuracy vs. time by choosing appropriate grid density.

### Mathematical Details

#### Spline Fitting

Uses `scipy.interpolate.RectBivariateSpline`:

```
Z(x, y) = Σ Σ c[i,j] * B[i](x) * B[j](y)
```

Where `B[i]` are B-spline basis functions and `c[i,j]` are computed coefficients.

Smoothing factor `s` controls the tradeoff:
- `s = 0`: Exact interpolation (passes through all points)
- `s > 0`: Least-squares approximation (smooth fit)

#### RBF Fitting

Uses radial basis functions:

```
Z(x, y) = Σ w[i] * φ(||(x,y) - (x[i],y[i])||)
```

Common kernels: multiquadric, gaussian, thin-plate spline.

### Future Enhancements

Potential improvements for future versions:

- [ ] Hardware autofocus integration (faster measurement)
- [ ] Live preview during measurement
- [ ] Adaptive grid refinement (dense grid near edges/features)
- [ ] Multi-plane focus mapping for 3D samples
- [ ] Export/import focus maps between experiments
- [ ] Focus map interpolation across time (time-lapse drift)
- [ ] Machine learning-based surface prediction

---

**Last Updated:** February 17, 2026  
**Version:** 1.0  
**Contributors:** ImSwitch Development Team
