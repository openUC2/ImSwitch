// src/state/slices/GoniometerSlice.js
// Redux slice for goniometer / contact angle measurement state

import { createSlice } from "@reduxjs/toolkit";

const initialGoniometerState = {
  // Active tab: "manual" or "auto"
  activeTab: "auto",

  // Snapped image (base64 data URI or null)
  snappedImage: null,
  snappedShape: null,
  isSnapping: false,

  // Annotated result image (base64 data URI or null)
  resultImage: null,

  // Latest measurement result
  lastResult: null, // { success, left_angle, right_angle, ... }

  // Processing config (matches backend DEFAULT_CONFIG)
  config: {
    canny_low: 30,
    canny_high: 100,
    blur_ksize: 5,
    bright_row_thresh: 60,
    roi_x_margin_frac: 0.12,
    roi_y_above_frac: 0.3,
    roi_y_below_px: 60,
    min_contour_length: 100,
    baseline_tolerance: 5,
    local_fit_frac: 0.08,
    poly_degree: 3,
    tangent_delta: 1.0,
    angle_range_low: 5,
    angle_range_high: 175,
  },

  // Manual annotation points (image pixel coords)
  manualPoints: {
    baselinePt1: null, // {x, y}
    baselinePt2: null,
    tangentPt: null,
  },
  manualPlacingMode: null, // "baselinePt1" | "baselinePt2" | "tangentPt" | null

  // Measurement history list
  measurements: [],

  // UI flags
  isMeasuring: false,
  showAdvancedConfig: false,
};

const goniometerSlice = createSlice({
  name: "goniometerState",
  initialState: initialGoniometerState,
  reducers: {
    setActiveTab: (state, action) => {
      state.activeTab = action.payload;
    },
    setSnappedImage: (state, action) => {
      state.snappedImage = action.payload;
    },
    setSnappedShape: (state, action) => {
      state.snappedShape = action.payload;
    },
    setIsSnapping: (state, action) => {
      state.isSnapping = action.payload;
    },
    setResultImage: (state, action) => {
      state.resultImage = action.payload;
    },
    setLastResult: (state, action) => {
      state.lastResult = action.payload;
    },
    setConfig: (state, action) => {
      state.config = { ...state.config, ...action.payload };
    },
    resetConfig: (state) => {
      state.config = initialGoniometerState.config;
    },
    setManualPoints: (state, action) => {
      state.manualPoints = { ...state.manualPoints, ...action.payload };
    },
    resetManualPoints: (state) => {
      state.manualPoints = initialGoniometerState.manualPoints;
    },
    setManualPlacingMode: (state, action) => {
      state.manualPlacingMode = action.payload;
    },
    addMeasurement: (state, action) => {
      state.measurements.push(action.payload);
    },
    setMeasurements: (state, action) => {
      state.measurements = action.payload;
    },
    clearMeasurements: (state) => {
      state.measurements = [];
    },
    setIsMeasuring: (state, action) => {
      state.isMeasuring = action.payload;
    },
    setShowAdvancedConfig: (state, action) => {
      state.showAdvancedConfig = action.payload;
    },
  },
});

// Export actions
export const {
  setActiveTab,
  setSnappedImage,
  setSnappedShape,
  setIsSnapping,
  setResultImage,
  setLastResult,
  setConfig,
  resetConfig,
  setManualPoints,
  resetManualPoints,
  setManualPlacingMode,
  addMeasurement,
  setMeasurements,
  clearMeasurements,
  setIsMeasuring,
  setShowAdvancedConfig,
} = goniometerSlice.actions;

// Selectors
export const getGoniometerState = (state) => state.goniometerState;

export default goniometerSlice.reducer;
