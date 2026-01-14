# Lightsheet 3D Visualization Integration

## Overview
This document describes the integration of a real-time 3D visualization of the lightsheet microscope assembly into the LightsheetController component. The 3D viewer displays the positions of the objective, sample, and lightsheet based on live stage positions.

## Architecture

### Components Created

1. **Lightsheet3DViewer.jsx** (`src/components/Lightsheet3DViewer.jsx`)
   - Three.js-based 3D visualization component
   - Loads the GLB assembly model
   - Updates positions in real-time based on Redux state
   - Supports axis configuration (offset, scale, invert)

2. **AxisConfigurationMenu.jsx** (`src/components/AxisConfigurationMenu.jsx`)
   - Collapsible accordion menu for axis configuration
   - Allows setting offset, scaling factor, and direction inversion
   - Per-axis configuration for X, Y, Z, and A axes
   - Settings stored in Redux for session persistence

3. **LightsheetPositionControls.jsx** (`src/components/LightsheetPositionControls.jsx`)
   - Joystick-style position controls for all axes
   - Adjustable step size
   - Updates both hardware stage and Redux state
   - Visual feedback with current positions display

### Redux State Extensions

**LightsheetSlice.js** updated with:

```javascript
// Current stage positions for 3D visualization
stagePositions: {
  x: 0,
  y: 0,
  z: 0,
  a: 0
}

// Axis configuration for 3D mapping
axisConfig: {
  x: { offset: 0, scale: 1, invert: false },
  y: { offset: 0, scale: 1, invert: false },
  z: { offset: 0, scale: 1, invert: false },
  a: { offset: 0, scale: 1, invert: false }
}
```

New actions:
- `setStagePosition({ axis, value })`
- `setAllStagePositions(positions)`
- `setAxisConfig({ axis, config })`
- `setAxisOffset({ axis, offset })`
- `setAxisScale({ axis, scale })`
- `setAxisInvert({ axis, invert })`

## Axis Mapping

The stage axes are mapped to 3D model movements as follows:

| Stage Axis | 3D Model Component | Movement Direction | Description |
|------------|-------------------|-------------------|-------------|
| **Z** | Objective 20x | Y-axis | Focus position - objective moves up/down |
| **A** | Sample | Y-axis | Sample vertical position |
| **X** | Sample | X-axis | Sample left/right position |
| **Y** | Sample | Z-axis | Sample depth position |

### Axis Configuration

Each axis supports three configuration parameters:

1. **Offset**: Adds a fixed offset to the position (in µm)
2. **Scale**: Multiplies the position by a scaling factor
3. **Invert**: Reverses the direction of movement

Formula: `displayPosition = (stagePosition * scale + offset) * (invert ? -1 : 1)`

## 3D Model

- **File**: `Assembly_lightsheet_objective_sample_arrangement.glb`
- **Location**: `/public/assets/`
- **Format**: GLTF/GLB (binary GLTF)
- **Source**: Converted from STP CAD file

### Model Structure

The GLB model contains the following named groups:

- `Körper1_9` → Lightsheet
- `Körper2_1`, `Körper1_8` → Sample
- `Körper1_7` → Objective 4x
- `Körper2` to `Körper7`, `Körper1_6` → Objective 20x
- `Körper1` to `Körper1_6` → Chamber (fixed)

## Implementation Details

### Position Polling

The LightsheetController polls the backend for current stage positions every 2 seconds:

```javascript
useEffect(() => {
  const fetchPositions = async () => {
    const positionsData = await apiPositionerControllerGetPositions();
    dispatch(lightsheetSlice.setAllStagePositions(positions));
  };
  
  const interval = setInterval(fetchPositions, 2000);
  return () => clearInterval(interval);
}, [hostIP, hostPort, dispatch]);
```

### Stage Movement

When the user clicks position control buttons:
1. API call to `apiPositionerControllerMovePositioner()` moves the hardware
2. Redux state is updated with the new position
3. 3D viewer automatically re-renders with new positions

### 3D Rendering

The 3D viewer uses:
- **Three.js** for 3D rendering
- **OrbitControls** for camera interaction
- **GLTFLoader** for loading the assembly model
- **React useEffect** hooks for lifecycle management
- **React useRef** for Three.js object persistence

## User Interface Layout

The Scanning Parameters tab now includes:

```
┌─────────────────────────────────────────────┐
│  [2D Live View]    │  [3D Assembly View]    │
├─────────────────────────────────────────────┤
│  [▼ Advanced Axis Configuration]           │
│     X Axis → Sample X                       │
│     [Offset] [Scale] [☐ Invert]            │
│     Y Axis → Sample Z                       │
│     ...                                     │
├─────────────────────────────────────────────┤
│  Stage Position Controls                    │
│     [Step Size: 100µm]                     │
│     X-Y Control │ Z Control │ A Control     │
├─────────────────────────────────────────────┤
│  Scanning Parameters                        │
│     [Min/Max Position, Speed, etc.]        │
└─────────────────────────────────────────────┘
```

## Usage

### Basic Operation

1. **View Live 3D Position**: The 3D viewer automatically shows current stage positions
2. **Move Stage**: Use the position controls to move any axis
3. **Adjust View**: Click and drag in 3D viewer to orbit, right-click to pan, scroll to zoom

### Configure Axis Mapping

1. Click "Advanced Axis Configuration" to expand the menu
2. For each axis, set:
   - **Offset**: If your stage zero is not at the desired model zero
   - **Scale**: If your stage units don't match model units (typically 0.01 for µm to model units)
   - **Invert**: If the axis moves in the opposite direction
3. Click "Reset All to Defaults" to restore default values

### Calibration Workflow

1. Move the stage to a known reference position
2. Observe the 3D visualization
3. Adjust offset/scale/invert until the 3D model matches reality
4. Settings persist during the session

## API Endpoints Used

- `GET /PositionerController/getPositionerPositions` - Get current positions
- `GET /PositionerController/movePositioner` - Move a specific axis
- `GET /LightsheetController/getIsLightsheetRunning` - Check scanning status
- `GET /LightsheetController/performScanningRecording` - Start scanning
- `GET /LightsheetController/setGalvo` - Configure galvo scanner

## Performance Considerations

- **Position polling**: 2-second intervals to avoid overwhelming the backend
- **3D rendering**: 60 FPS with automatic damping for smooth camera movement
- **Model size**: ~1.1 MB GLB file, cached by browser after first load
- **Redux updates**: Debounced position updates to minimize re-renders

## Future Enhancements

Possible improvements:
- [ ] Add recording/playback of stage movements
- [ ] Overlay lightsheet beam path in 3D
- [ ] Add collision detection warnings
- [ ] Export/import axis configuration presets
- [ ] Add screenshot/video capture of 3D view
- [ ] Show scanning path preview in 3D

## Troubleshooting

**3D viewer shows blank or black screen:**
- Check browser console for GLB loading errors
- Verify GLB file exists at `/public/assets/Assembly_lightsheet_objective_sample_arrangement.glb`
- Check that file permissions allow reading

**Positions don't update:**
- Verify backend connection (check Connection Settings)
- Ensure `/PositionerController/getPositionerPositions` endpoint is responding
- Check browser console for API errors

**Movement directions are wrong:**
- Use the "Invert Direction" checkbox for each axis
- Verify axis mapping is correct for your hardware configuration

**Scale is wrong:**
- Adjust the "Scale Factor" in axis configuration
- Typical value is 0.01 for µm positions to model units

## Dependencies

- `three` (^0.160.0) - 3D rendering engine
- `@mui/material` - UI components
- `react-redux` - State management
- `@reduxjs/toolkit` - Redux utilities

## Related Files

- `src/components/LightsheetController.jsx` - Main controller integration
- `src/state/slices/LightsheetSlice.js` - Redux state management
- `src/backendapi/apiPositionerControllerGetPositions.js` - Position API
- `src/backendapi/apiPositionerControllerMovePositioner.js` - Movement API
- `public/assets/Assembly_lightsheet_objective_sample_arrangement.glb` - 3D model
