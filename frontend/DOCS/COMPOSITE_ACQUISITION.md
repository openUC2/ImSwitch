# Composite Acquisition Implementation Guide

## Overview

The Composite Acquisition feature enables multi-illumination sequential imaging with RGB channel fusion. It captures images under different illumination states (e.g., laser 488, laser 635, LED), fuses them into a single composite RGB JPEG, and streams the result to the frontend.

## Architecture

### Backend (ImSwitch)

**File:** `imswitch/imcontrol/controller/controllers/CompositeController.py`

The CompositeController follows the same pattern as InLineHoloController:
- Background worker thread for non-blocking acquisition
- Frame queue for decoupled processing
- MJPEG streaming for live preview
- RESTful API for control

#### Key Classes

```python
@dataclass
class IlluminationStep:
    """Configuration for a single illumination step"""
    illumination: str    # Source name (e.g., "laser488")
    intensity: float     # 0.0-1.0 or device units
    exposure_ms: float   # Optional exposure override
    settle_ms: float     # Settle time after illumination change
    enabled: bool        # Whether step is active

@dataclass  
class CompositeParams:
    """Acquisition parameters"""
    steps: List[IlluminationStep]
    mapping: Dict[str, str]  # {"R": "laser635", "G": "laser488", "B": "LED"}
    fps_target: float
    jpeg_quality: int
    normalize_channels: bool
    auto_exposure: bool
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/CompositeController/get_illumination_sources_composite` | GET | List available sources |
| `/CompositeController/get_parameters_composite` | GET | Get current parameters |
| `/CompositeController/set_parameters_composite` | POST | Update parameters |
| `/CompositeController/get_state_composite` | GET | Get acquisition state |
| `/CompositeController/start_composite` | POST | Start acquisition |
| `/CompositeController/stop_composite` | POST | Stop acquisition |
| `/CompositeController/mjpeg_stream_composite` | GET | MJPEG stream |
| `/CompositeController/add_step_composite` | POST | Add illumination step |
| `/CompositeController/remove_step_composite` | POST | Remove step by index |
| `/CompositeController/set_mapping_composite` | POST | Set RGB mapping |
| `/CompositeController/capture_single_composite` | GET | One-shot capture |

### Frontend (microscope-app)

#### API Files (`src/backendapi/`)

- `apiCompositeControllerGetIlluminationSources.js`
- `apiCompositeControllerGetParameters.js`
- `apiCompositeControllerSetParameters.js`
- `apiCompositeControllerGetState.js`
- `apiCompositeControllerStart.js`
- `apiCompositeControllerStop.js`
- `apiCompositeControllerAddStep.js`
- `apiCompositeControllerRemoveStep.js`
- `apiCompositeControllerSetMapping.js`
- `apiCompositeControllerCaptureSingle.js`

#### Redux Slice (`src/state/slices/CompositeAcquisitionSlice.js`)

State structure:
```javascript
{
  isRunning: false,
  isStreaming: false,
  currentStep: 0,
  cycleCount: 0,
  averageFps: 0.0,
  illuminationSources: [],
  steps: [],
  mapping: { R: "", G: "", B: "" },
  fpsTarget: 5.0,
  jpegQuality: 85,
  normalizeChannels: true,
  autoExposure: false,
  // ... UI state
}
```

#### React Components (`src/axon/`)

**CompositeAcquisitionComponent.js** - Main control panel
- Configure illumination steps (add/remove/reorder)
- Set RGB channel mapping
- Start/stop acquisition
- Single capture mode
- Advanced settings (FPS, JPEG quality, normalization)

**CompositeStreamViewer.js** - MJPEG stream viewer
- Displays live composite stream
- Status indicators (FPS, resolution)
- Fullscreen support
- Channel mapping overlay

## Usage Examples

### Basic Multi-Color Fluorescence

```javascript
// Configure 3-channel fluorescence
const params = {
  steps: [
    { illumination: "laser488", intensity: 0.3, settle_ms: 20, enabled: true },
    { illumination: "laser561", intensity: 0.4, settle_ms: 20, enabled: true },
    { illumination: "laser635", intensity: 0.2, settle_ms: 20, enabled: true },
  ],
  mapping: {
    R: "laser635",  // Far-red → Red
    G: "laser561",  // Orange → Green  
    B: "laser488",  // Blue → Blue
  },
  fps_target: 3.0,
};
await apiCompositeControllerSetParameters(params);
await apiCompositeControllerStart();
```

### API Example (curl)

```bash
# Start composite acquisition
curl -X POST "http://localhost:8001/CompositeController/start_composite"

# Set parameters
curl -X POST "http://localhost:8001/CompositeController/set_parameters_composite" \
  -d 'params={"fps_target": 5.0, "mapping": {"R": "laser635", "G": "laser488", "B": ""}}'

# View MJPEG stream
# Open in browser: http://localhost:8001/CompositeController/mjpeg_stream_composite?startStream=true

# Stop acquisition
curl -X POST "http://localhost:8001/CompositeController/stop_composite"
```

## Acquisition Loop

1. **Turn off all illumination** - Clean start state
2. **For each enabled step:**
   - Set exposure (if `auto_exposure` enabled)
   - Set illumination source on with configured intensity
   - Wait settle time
   - Capture frame from camera
   - Store frame in channel buffer
3. **Turn off illumination** - After all steps
4. **Fuse channels** - Map to RGB based on `mapping` config
5. **Normalize** (optional) - Per-channel contrast stretch
6. **Encode JPEG** - With configured quality
7. **Add to MJPEG queue** - For streaming to frontend
8. **Rate limit** - Sleep to achieve target FPS

## Integration with ImSwitch

To enable the CompositeController, ensure it's loaded in your ImSwitch configuration. The controller automatically discovers available illumination sources from:
- `self._master.lasersManager` - Lasers
- `self._master.LEDsManager` - LEDs

## Future Extensions

- **EDOF Mode**: Z-stack acquisition with extended depth-of-field fusion
- **Per-channel LUTs**: Custom color mapping and gamma correction
- **Raw frame storage**: Save individual channel frames for analysis
- **Sequence presets**: Save/load acquisition configurations
- **Hardware triggers**: Synchronized acquisition with external hardware
