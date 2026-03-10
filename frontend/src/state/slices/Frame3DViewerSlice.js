// redux/Frame3DViewerSlice.js
// State slice for the FRAME microscope 3D digital twin viewer.
// Stores axis mapping configuration (offset, scale, invert) and camera state.
import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // Axis mapping: how microscope axes (x,y,z,a) map to the 3D model groups.
  // "stage" group is the XY stage (GLB nodes 51..55).
  // "turret" group is the objective turret (GLB nodes 56..63).
  // Offsets are in model units (mm). Scale converts microscope steps (µm) → model mm.
  axisConfig: {
    // Stage group mappings  (microscope axis → 3D model axis of the stage group)
    stageX: { microscopeAxis: "x", modelAxis: "x", offset: -127000/2, scale: 0.001, invert: true },
    stageY: { microscopeAxis: "y", modelAxis: "z", offset: -86000/2, scale: 0.001, invert: true },
    stageZ: { microscopeAxis: "z", modelAxis: "y", offset: 0, scale: 0.001, invert: true },
    // Turret group mapping  (objective / focus axis)
    turretX: { microscopeAxis: "a", modelAxis: "y", offset: -15000, scale: 0.001, invert: false },
  },
  // Persisted camera state so the user's view angle is remembered
  cameraState: {
    position: null,  // { x, y, z }
    target: null,    // { x, y, z }
  },
  // Visibility toggles for each group
  visibility: {
    base: true,
    stage: true,
    turret: true,
  },
};

const frame3DViewerSlice = createSlice({
  name: "frame3DViewer",
  initialState,
  reducers: {
    // Update a single axis config entry, e.g. setAxisConfig({ key: "stageX", config: { offset: 10 } })
    setAxisConfig: (state, action) => {
      const { key, config } = action.payload;
      if (state.axisConfig[key]) {
        state.axisConfig[key] = { ...state.axisConfig[key], ...config };
      }
    },
    // Bulk-set all axis configs
    setAllAxisConfigs: (state, action) => {
      state.axisConfig = { ...state.axisConfig, ...action.payload };
    },
    // Set camera state (debounced from the 3D viewer)
    setCameraState: (state, action) => {
      state.cameraState = { ...state.cameraState, ...action.payload };
    },
    // Toggle visibility for a group
    setVisibility: (state, action) => {
      const { group, visible } = action.payload;
      if (state.visibility[group] !== undefined) {
        state.visibility[group] = visible;
      }
    },
  },
});

// Actions
export const { setAxisConfig, setAllAxisConfigs, setCameraState, setVisibility } =
  frame3DViewerSlice.actions;

// Selector
export const getFrame3DViewerState = (state) => state.frame3DViewer;

// Reducer
export default frame3DViewerSlice.reducer;
