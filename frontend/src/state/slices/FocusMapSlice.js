import { createSlice } from "@reduxjs/toolkit";

/**
 * Redux state slice for Focus Map functionality.
 *
 * Manages:
 *  - focusMapConfig: user-configurable parameters (enabled, grid, fit method, etc.)
 *  - focusMapResults: per-group results from backend (points, fit stats, preview)
 *  - focusMapUI: UI-specific state (loading, errors, selected group)
 */

// Default focus map configuration (matches backend FocusMapConfig)
const defaultFocusMapConfig = {
  enabled: false,

  // Grid generation
  rows: 3,
  cols: 3,
  add_margin: false,

  // Fit strategy
  fit_by_region: true,
  use_manual_map: false, // Reuse manual/global map for all groups via interpolation
  method: "spline", // "spline" | "rbf" | "constant"
  smoothing_factor: 0.1,

  // Runtime behavior
  apply_during_scan: true,
  z_offset: 0.0,
  clamp_enabled: false,
  z_min: 0.0,
  z_max: 0.0,

  // Autofocus settings (synced from frontend ExperimentSlice parameterValue)
  af_range: 100.0,
  af_resolution: 10.0,
  af_cropsize: 2048,
  af_algorithm: "LAPE",
  af_settle_time: 0.1,
  af_static_offset: 0.0,
  af_two_stage: false,
  af_n_gauss: 7,
  af_illumination_channel: "",
  af_mode: "software",

  // Autofocus integration
  autofocus_profile: null,
  settle_ms: 0,
  store_debug_artifacts: true,

  // Per-illumination-channel Z offsets (µm)
  // e.g. { "LED": 0.0, "Fluorescence": -1.5 }
  channel_offsets: {},
};

const initialState = {
  // Configuration (sent to backend as part of Experiment)
  config: { ...defaultFocusMapConfig },

  // Results per group (received from backend)
  // { [groupId]: { group_id, group_name, points, fit_stats, preview_grid, status } }
  results: {},

  // Manual focus points placed by user on the wellplate viewer
  // [{ x, y, z }]
  manualPoints: [],

  // UI state
  ui: {
    isComputing: false,
    computingGroupId: null, // Which group is currently being computed
    selectedGroupId: null, // Which group is selected for visualization
    error: null,
    lastComputeTime: null,
    showOverlayOnWellplate: true, // Toggle: draw focus map points on wellplate canvas
  },
};

const focusMapSlice = createSlice({
  name: "focusMap",
  initialState,
  reducers: {
    // ── Config setters ──────────────────────────────────────

    setFocusMapEnabled: (state, action) => {
      state.config.enabled = action.payload;
    },

    setFocusMapConfig: (state, action) => {
      // Merge partial config updates
      state.config = { ...state.config, ...action.payload };
    },

    setFocusMapRows: (state, action) => {
      state.config.rows = Math.max(1, Math.min(20, action.payload));
    },

    setFocusMapCols: (state, action) => {
      state.config.cols = Math.max(1, Math.min(20, action.payload));
    },

    setFocusMapMethod: (state, action) => {
      state.config.method = action.payload;
    },

    setFocusMapSmoothingFactor: (state, action) => {
      state.config.smoothing_factor = Math.max(0, action.payload);
    },

    setFocusMapFitByRegion: (state, action) => {
      state.config.fit_by_region = action.payload;
    },

    setFocusMapUseManualMap: (state, action) => {
      state.config.use_manual_map = action.payload;
    },

    setFocusMapZOffset: (state, action) => {
      state.config.z_offset = action.payload;
    },

    setFocusMapClampEnabled: (state, action) => {
      state.config.clamp_enabled = action.payload;
    },

    setFocusMapZMin: (state, action) => {
      state.config.z_min = action.payload;
    },

    setFocusMapZMax: (state, action) => {
      state.config.z_max = action.payload;
    },

    setFocusMapSettleMs: (state, action) => {
      state.config.settle_ms = Math.max(0, action.payload);
    },

    setFocusMapAddMargin: (state, action) => {
      state.config.add_margin = action.payload;
    },

    setFocusMapApplyDuringScan: (state, action) => {
      state.config.apply_during_scan = action.payload;
    },

    setFocusMapChannelOffset: (state, action) => {
      // payload: { channel: "LED", offset: -1.5 }
      const { channel, offset } = action.payload;
      state.config.channel_offsets[channel] = offset;
    },

    removeChannelOffset: (state, action) => {
      delete state.config.channel_offsets[action.payload];
    },

    // ── Manual points (user-placed on wellplate) ────────────

    addManualPoint: (state, action) => {
      // payload: { x, y, z }
      if (!state.manualPoints) state.manualPoints = [];
      state.manualPoints.push(action.payload);
    },

    removeManualPoint: (state, action) => {
      if (state.manualPoints) {
        state.manualPoints.splice(action.payload, 1);
      }
    },

    clearManualPoints: (state) => {
      state.manualPoints = [];
    },

    updateManualPointZ: (state, action) => {
      // payload: { index, z }
      if (state.manualPoints && state.manualPoints[action.payload.index]) {
        state.manualPoints[action.payload.index].z = action.payload.z;
      }
    },

    // ── Results management ──────────────────────────────────

    setFocusMapResults: (state, action) => {
      // Replace all results { groupId: resultObj, ... }
      state.results = action.payload;
    },

    updateFocusMapGroupResult: (state, action) => {
      const { groupId, result } = action.payload;
      state.results[groupId] = result;
    },

    clearFocusMapResults: (state, action) => {
      if (action.payload) {
        // Clear specific group
        delete state.results[action.payload];
      } else {
        // Clear all
        state.results = {};
      }
    },

    // ── UI state ────────────────────────────────────────────

    setFocusMapComputing: (state, action) => {
      state.ui.isComputing = action.payload.isComputing;
      state.ui.computingGroupId = action.payload.groupId || null;
      if (!action.payload.isComputing) {
        state.ui.lastComputeTime = Date.now();
      }
    },

    setFocusMapSelectedGroup: (state, action) => {
      state.ui.selectedGroupId = action.payload;
    },

    setFocusMapError: (state, action) => {
      state.ui.error = action.payload;
    },

    clearFocusMapError: (state) => {
      state.ui.error = null;
    },

    setShowOverlayOnWellplate: (state, action) => {
      state.ui.showOverlayOnWellplate = action.payload;
    },

    // ── Reset ───────────────────────────────────────────────

    resetFocusMapState: () => {
      return initialState;
    },
  },
});

// Export actions
export const {
  setFocusMapEnabled,
  setFocusMapConfig,
  setFocusMapRows,
  setFocusMapCols,
  setFocusMapMethod,
  setFocusMapSmoothingFactor,
  setFocusMapFitByRegion,
  setFocusMapUseManualMap,
  setFocusMapZOffset,
  setFocusMapClampEnabled,
  setFocusMapZMin,
  setFocusMapZMax,
  setFocusMapSettleMs,
  setFocusMapAddMargin,
  setFocusMapApplyDuringScan,
  setFocusMapChannelOffset,
  removeChannelOffset,
  addManualPoint,
  removeManualPoint,
  clearManualPoints,
  updateManualPointZ,
  setFocusMapResults,
  updateFocusMapGroupResult,
  clearFocusMapResults,
  setFocusMapComputing,
  setFocusMapSelectedGroup,
  setFocusMapError,
  clearFocusMapError,
  setShowOverlayOnWellplate,
  resetFocusMapState,
} = focusMapSlice.actions;

// Selectors
export const getFocusMapState = (state) => state.focusMap;
export const getFocusMapConfig = (state) => state.focusMap.config;
export const getFocusMapResults = (state) => state.focusMap.results;
export const getFocusMapUI = (state) => state.focusMap.ui;
export const getFocusMapGroupResult = (groupId) => (state) =>
  state.focusMap.results[groupId] || null;
export const isFocusMapEnabled = (state) => state.focusMap.config.enabled;
export const getManualPoints = (state) => state.focusMap.manualPoints || [];
export const getShowOverlayOnWellplate = (state) => state.focusMap.ui?.showOverlayOnWellplate ?? true;

// Export reducer
export default focusMapSlice.reducer;
