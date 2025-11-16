# AprilTag Grid Navigation - Quick Reference

## API Endpoints Quick Reference

Base URL: `http://localhost:8001/api/pixelcalibration/`

### Setup & Configuration

```bash
# Set grid configuration (17×25 grid, 40mm pitch)
curl "http://localhost:8001/api/pixelcalibration/gridSetConfig?rows=17&cols=25&start_id=0&pitch_mm=40.0"

# Get current configuration
curl "http://localhost:8001/api/pixelcalibration/gridGetConfig"
```

### Detection & Calibration

```bash
# Detect AprilTags in current view
curl "http://localhost:8001/api/pixelcalibration/gridDetectTags"

# Detect and save annotated image
curl "http://localhost:8001/api/pixelcalibration/gridDetectTags?save_annotated=true"

# Calibrate camera-to-stage transformation (requires ≥3 visible tags)
curl "http://localhost:8001/api/pixelcalibration/gridCalibrateTransform"
```

### Navigation

```bash
# Navigate to tag ID 101
curl "http://localhost:8001/api/pixelcalibration/gridMoveToTag?target_id=101"

# Navigate with custom parameters
curl "http://localhost:8001/api/pixelcalibration/gridMoveToTag?target_id=200&roi_tolerance_px=5&max_iterations=30&step_fraction=0.7"

# Get information about a specific tag
curl "http://localhost:8001/api/pixelcalibration/gridGetTagInfo?tag_id=101"
```

### Observation Camera

```bash
# Get single frame from observation camera
curl "http://localhost:8001/api/pixelcalibration/returnObservationCameraImage" -o frame.png

# Stream MJPEG video
# Open in browser: http://localhost:8001/api/pixelcalibration/overviewStream

# Enable AprilTag detection overlay on stream
curl "http://localhost:8001/api/pixelcalibration/gridSetOverlay?enabled=true"

# Disable overlay
curl "http://localhost:8001/api/pixelcalibration/gridSetOverlay?enabled=false"

# Get current overlay status
curl "http://localhost:8001/api/pixelcalibration/gridGetOverlay"
```

## Python Usage Examples

### AprilTag Overlay on Stream

```python
import requests

base_url = "http://localhost:8001/api/pixelcalibration"

# Enable overlay
overlay_response = requests.get(f"{base_url}/gridSetOverlay", params={"enabled": True})
print("Overlay enabled:", overlay_response.json())

# Now open stream in browser: http://localhost:8001/api/pixelcalibration/overviewStream
# You will see detected AprilTags with ID labels and grid positions

# Check overlay status
status_response = requests.get(f"{base_url}/gridGetOverlay")
print("Overlay status:", status_response.json())
# → {"success": true, "overlay_enabled": true}

# Disable overlay
disable_response = requests.get(f"{base_url}/gridSetOverlay", params={"enabled": False})
print("Overlay disabled:", disable_response.json())
```

### Basic Detection

```python
import requests

# Detect tags
response = requests.get("http://localhost:8001/api/pixelcalibration/gridDetectTags")
data = response.json()

if data["success"]:
    print(f"Found {data['num_tags']} tags:")
    for tag in data["tags"]:
        print(f"  Tag {tag['id']}: ({tag['cx']:.1f}, {tag['cy']:.1f}) px, grid pos {tag['grid_position']}")
```

### Calibration Workflow

```python
import requests
import time

base_url = "http://localhost:8001/api/pixelcalibration"

# 1. Configure grid
config_response = requests.get(f"{base_url}/gridSetConfig", params={
    "rows": 17,
    "cols": 25,
    "start_id": 0,
    "pitch_mm": 40.0
})
print("Grid configured:", config_response.json())

# 2. Check tag detection
detect_response = requests.get(f"{base_url}/gridDetectTags")
detect_data = detect_response.json()
print(f"Detected {detect_data['num_tags']} tags")

if detect_data["num_tags"] < 3:
    print("ERROR: Need at least 3 tags visible for calibration")
    exit(1)

# 3. Run calibration
calib_response = requests.get(f"{base_url}/gridCalibrateTransform")
calib_data = calib_response.json()

if calib_data["success"]:
    print(f"Calibration successful!")
    print(f"  Used {calib_data['num_tags']} tags")
    print(f"  Residual error: {calib_data['residual_um']:.2f} µm")
    print(f"  Transform: {calib_data['T_cam2stage']}")
else:
    print(f"Calibration failed: {calib_data.get('error', 'Unknown error')}")
```

### Navigation Example

```python
import requests

base_url = "http://localhost:8001/api/pixelcalibration"

# Navigate to tag 200
nav_response = requests.get(f"{base_url}/gridMoveToTag", params={
    "target_id": 200,
    "roi_tolerance_px": 8.0,
    "max_iterations": 20
})

nav_data = nav_response.json()

if nav_data["success"]:
    print(f"Navigation successful!")
    print(f"  Final offset: {nav_data['final_offset_px']:.1f} px")
    print(f"  Iterations: {nav_data['iterations']}")
    print(f"  Final tag: {nav_data['final_tag_id']}")
    
    # Print trajectory
    print("\nTrajectory:")
    for step in nav_data["trajectory"]:
        print(f"  Iteration {step['iteration']}: {step['mode']}")
        print(f"    Current tag: {step.get('current_tag', 'N/A')}")
        print(f"    Move: {step['move_um']} µm")
else:
    print(f"Navigation failed: {nav_data.get('error', 'Unknown error')}")
```

### Batch Navigation (Visit Multiple Tags)

```python
import requests
import time

base_url = "http://localhost:8001/api/pixelcalibration"

# Tag IDs to visit (e.g., corners of a region)
target_tags = [50, 74, 150, 174]  # 5x5 grid region

results = []
for tag_id in target_tags:
    print(f"\nNavigating to tag {tag_id}...")
    
    response = requests.get(f"{base_url}/gridMoveToTag", params={
        "target_id": tag_id,
        "roi_tolerance_px": 8.0
    })
    
    data = response.json()
    results.append({
        "tag_id": tag_id,
        "success": data.get("success", False),
        "iterations": data.get("iterations", 0),
        "final_offset": data.get("final_offset_px", None)
    })
    
    if data["success"]:
        print(f"  ✓ Success in {data['iterations']} iterations")
    else:
        print(f"  ✗ Failed: {data.get('error', 'Unknown')}")
    
    # Optional: pause between movements
    time.sleep(0.5)

# Summary
print("\n" + "="*50)
print("NAVIGATION SUMMARY")
print("="*50)
success_count = sum(1 for r in results if r["success"])
print(f"Success rate: {success_count}/{len(results)}")
print(f"Average iterations: {sum(r['iterations'] for r in results) / len(results):.1f}")
```

### Grid Position Calculation

```python
def tag_id_from_position(row, col, cols=25, start_id=0):
    """Calculate tag ID from grid position."""
    return start_id + row * cols + col

def position_from_tag_id(tag_id, cols=25, start_id=0):
    """Calculate grid position from tag ID."""
    offset = tag_id - start_id
    row = offset // cols
    col = offset % cols
    return row, col

# Examples
print(f"Tag at (row=4, col=1): {tag_id_from_position(4, 1)}")  # → 101
print(f"Position of tag 200: {position_from_tag_id(200)}")     # → (8, 0)

# Navigate to specific grid position
target_row, target_col = 10, 15
target_id = tag_id_from_position(target_row, target_col)
print(f"Navigating to position ({target_row}, {target_col}) = tag {target_id}")

response = requests.get(f"http://localhost:8001/api/pixelcalibration/gridMoveToTag", 
                       params={"target_id": target_id})
```

## Common Workflows

### Initial System Setup

1. **Generate and print calibration target**
   ```bash
   cd ImSwitch/ImTools/apriltag
   python generateAprilTag.py
   # Print apriltag_grid.pdf at 100% scale
   ```

2. **Configure grid in ImSwitch**
   ```bash
   curl "http://localhost:8001/api/pixelcalibration/gridSetConfig?rows=17&cols=25&pitch_mm=40.0"
   ```

3. **Position stage and verify detection**
   ```bash
   curl "http://localhost:8001/api/pixelcalibration/gridDetectTags"
   ```

4. **Run calibration**
   ```bash
   curl "http://localhost:8001/api/pixelcalibration/gridCalibrateTransform"
   ```

### Daily Usage

1. **Verify calibration is loaded**
   ```bash
   curl "http://localhost:8001/api/pixelcalibration/gridGetConfig"
   # Check: "calibrated": true
   ```

2. **Navigate to target**
   ```bash
   curl "http://localhost:8001/api/pixelcalibration/gridMoveToTag?target_id=101"
   ```

## Troubleshooting Commands

```bash
# Check if observation camera is working
curl "http://localhost:8001/api/pixelcalibration/returnObservationCameraImage" -o test.png
open test.png  # macOS
# xdg-open test.png  # Linux

# Verify grid configuration
curl "http://localhost:8001/api/pixelcalibration/gridGetConfig" | python -m json.tool

# Check current tag detection
curl "http://localhost:8001/api/pixelcalibration/gridDetectTags" | python -m json.tool

# Get info about specific tag
curl "http://localhost:8001/api/pixelcalibration/gridGetTagInfo?tag_id=101" | python -m json.tool

# Test navigation with verbose output
curl "http://localhost:8001/api/pixelcalibration/gridMoveToTag?target_id=50&max_iterations=5" | python -m json.tool
```

## Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Observation camera not available` | Camera not configured or not found | Check config: `ObservationCamera` setting |
| `Need at least 3 valid grid tags` | Not enough tags visible | Move stage to area with more tags |
| `Camera-to-stage transformation not calibrated` | Calibration not run | Run `gridCalibrateTransform` |
| `Tag ID XXX is outside grid range` | Invalid tag ID for grid size | Check grid config (rows×cols) |
| `Max iterations reached` | Navigation not converging | Reduce `step_fraction` or increase `max_iterations` |
| `Lost all tags during navigation` | Tags moved out of view | Check calibration quality; reduce step size |

## Performance Tips

- **Calibration quality**: Use 10-20 visible tags for best results
- **Navigation speed**: Adjust `step_fraction` (0.5-0.9) to balance speed vs. convergence
- **Tolerance**: Use `roi_tolerance_px=8` for general use, `5` for high precision
- **Settle time**: Increase `settle_time` if stage vibrations affect detection
- **Search pattern**: Disable with `search_enabled=false` if tags are always visible

## Advanced: JavaScript/Browser Integration

```javascript
// Detect tags and display results
async function detectTags() {
    const response = await fetch('/api/pixelcalibration/gridDetectTags');
    const data = await response.json();
    
    if (data.success) {
        console.log(`Found ${data.num_tags} tags`);
        data.tags.forEach(tag => {
            console.log(`Tag ${tag.id}: (${tag.cx}, ${tag.cy})`);
        });
    }
}

// Toggle AprilTag overlay on/off
async function toggleOverlay(enabled) {
    const response = await fetch(
        `/api/pixelcalibration/gridSetOverlay?enabled=${enabled}`
    );
    const result = await response.json();
    
    if (result.success) {
        console.log(`Overlay ${enabled ? 'enabled' : 'disabled'}`);
    }
    return result.overlay_enabled;
}

// Get overlay status
async function getOverlayStatus() {
    const response = await fetch('/api/pixelcalibration/gridGetOverlay');
    const data = await response.json();
    return data.overlay_enabled;
}

// Navigate to tag with progress updates
async function navigateToTag(targetId) {
    console.log(`Navigating to tag ${targetId}...`);
    
    const response = await fetch(
        `/api/pixelcalibration/gridMoveToTag?target_id=${targetId}&max_iterations=20`
    );
    const result = await response.json();
    
    if (result.success) {
        console.log(`✓ Navigation successful in ${result.iterations} iterations`);
        console.log(`Final offset: ${result.final_offset_px.toFixed(1)} px`);
        
        // Display trajectory
        result.trajectory.forEach((step, i) => {
            console.log(`Step ${i}: ${step.mode}, moved ${step.move_um} µm`);
        });
    } else {
        console.error(`✗ Navigation failed: ${result.error}`);
    }
}

// Example: Create overlay toggle button
function createOverlayToggleButton() {
    const button = document.createElement('button');
    button.textContent = 'Toggle AprilTag Overlay';
    button.onclick = async () => {
        const currentStatus = await getOverlayStatus();
        const newStatus = await toggleOverlay(!currentStatus);
        button.textContent = `AprilTag Overlay: ${newStatus ? 'ON' : 'OFF'}`;
    };
    return button;
}

// Example: Display stream with overlay controls
function setupStreamViewer() {
    const container = document.createElement('div');
    
    // Stream image
    const img = document.createElement('img');
    img.src = '/api/pixelcalibration/overviewStream';
    img.style.width = '100%';
    
    // Overlay toggle
    const toggleBtn = createOverlayToggleButton();
    
    container.appendChild(toggleBtn);
    container.appendChild(img);
    
    return container;
}

// Usage
detectTags();
toggleOverlay(true);  // Enable overlay
navigateToTag(101);
```

## Configuration File Example

Add to `imcontrol_setups/example_config.json`:

```json
{
  "PixelCalibration": {
    "ObservationCamera": "ObservationCam",
    "ObservationCameraFlip": {
      "flipX": true,
      "flipY": true
    },
    "aprilTagGrid": {
      "rows": 17,
      "cols": 25,
      "start_id": 0,
      "pitch_mm": 40.0,
      "transform": null
    }
  }
}
```

After first calibration, `transform` will be automatically populated with the 2×3 affine matrix.
