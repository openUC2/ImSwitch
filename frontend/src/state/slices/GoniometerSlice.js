// src/state/slices/GoniometerSlice.js
// Redux slice for goniometer / contact angle measurement state

import { createSlice } from "@reduxjs/toolkit";

const initialGoniometerState = {
  // Active tab: "manual" or "auto"
  activeTab: "auto",

  // Snapped image (base64 data URI or null) — may be cropped
  snappedImage: null,
  snappedShape: null,
  isSnapping: false,

  // Annotated result image (base64 data URI or null)
  resultImage: null,

  // Latest measurement result
  lastResult: null, // { success, left_angle, right_angle, baseline_tilt_deg, ... }

  // Processing config (matches backend DEFAULT_CONFIG - mirror-geometry algorithm)
  config: {
    canny_low: 30,
    canny_high: 100,
    blur_ksize: 5,
    fit_frac: 0.15,
    poly_degree: 2,
    tangent_delta: 1.0,
    angle_range_low: 1,
    angle_range_high: 179,
  },

  // Manual annotation points (image pixel coords in cropped/snapped image space)
  manualPoints: {
    baselinePt1: null, // {x, y}
    baselinePt2: null,
    tangentPt: null,
  },
  manualPlacingMode: null, // "baselinePt1" | "baselinePt2" | "tangentPt" | null

  // Crop ROI — null means full frame; {x1,y1,x2,y2} in full camera pixel space
  cropRoi: null,
  // Whether user is currently drawing a crop rectangle on the live stream
  isCropMode: false,

  // Zoom/pan for the manual result-image viewer
  zoomLevel: 1,        // 1 = 100%, 2 = 200% etc.
  panOffset: { x: 0, y: 0 },  // pixels offset (at natural image scale, pre-zoom)

  // Focus monitor (auto tab)
  showFocusMonitor: false,
  focusMetric: null,        // latest Laplacian-variance value

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
    // Crop
    setCropRoi: (state, action) => {
      state.cropRoi = action.payload; // {x1,y1,x2,y2} or null
    },
    setIsCropMode: (state, action) => {
      state.isCropMode = action.payload;
    },
    // Zoom / pan for result image viewer
    setZoomLevel: (state, action) => {
      state.zoomLevel = Math.max(1, Math.min(8, action.payload));
    },
    setPanOffset: (state, action) => {
      state.panOffset = action.payload;
    },
    resetZoomPan: (state) => {
      state.zoomLevel = 1;
      state.panOffset = { x: 0, y: 0 };
    },
    // Focus monitor
    setShowFocusMonitor: (state, action) => {
      state.showFocusMonitor = action.payload;
    },
    setFocusMetric: (state, action) => {
      state.focusMetric = action.payload;
    },
    // History
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
  setCropRoi,
  setIsCropMode,
  setZoomLevel,
  setPanOffset,
  resetZoomPan,
  setShowFocusMonitor,
  setFocusMetric,
  addMeasurement,
  setMeasurements,
  clearMeasurements,
  setIsMeasuring,
  setShowAdvancedConfig,
} = goniometerSlice.actions;

// Selectors
export const getGoniometerState = (state) => state.goniometerState;

export default goniometerSlice.reducer;
