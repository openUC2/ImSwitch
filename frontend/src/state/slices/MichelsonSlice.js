// src/state/slices/MichelsonSlice.js
// Redux slice for Michelson time-series state management

import { createSlice } from "@reduxjs/toolkit";

// Initial state for Michelson time-series
const initialMichelsonState = {
  // Acquisition status
  isCapturing: false,
  frameCount: 0,
  sampleCount: 0,
  actualFps: 0.0,
  startTime: 0.0,
  
  // ROI parameters
  roiCenter: null, // [x, y] in pixels
  roiSize: 10, // square ROI size (5, 10, 20, etc.)
  
  // Acquisition parameters
  updateFreq: 30.0, // Hz
  bufferDuration: 60.0, // seconds
  decimation: 1,
  
  // Time-series data (for local plotting)
  timestamps: [],
  means: [],
  stds: [],
  
  // Statistics
  dataMin: null,
  dataMax: null,
  dataMean: null,
  dataStd: null,
  
  // UI state
  plotWindowSeconds: 10, // show last N seconds in plot
  autoScale: true,
};

// Create slice
const michelsonSlice = createSlice({
  name: "michelsonState",
  initialState: initialMichelsonState,
  reducers: {
    // Acquisition status
    setIsCapturing: (state, action) => {
      state.isCapturing = action.payload;
    },
    setFrameCount: (state, action) => {
      state.frameCount = action.payload;
    },
    setSampleCount: (state, action) => {
      state.sampleCount = action.payload;
    },
    setActualFps: (state, action) => {
      state.actualFps = action.payload;
    },
    setStartTime: (state, action) => {
      state.startTime = action.payload;
    },
    
    // ROI parameters
    setRoiCenter: (state, action) => {
      state.roiCenter = action.payload;
    },
    setRoiSize: (state, action) => {
      state.roiSize = action.payload;
    },
    
    // Acquisition parameters
    setUpdateFreq: (state, action) => {
      state.updateFreq = action.payload;
    },
    setBufferDuration: (state, action) => {
      state.bufferDuration = action.payload;
    },
    setDecimation: (state, action) => {
      state.decimation = action.payload;
    },
    
    // Time-series data
    setTimeSeriesData: (state, action) => {
      const { timestamps, means, stds } = action.payload;
      state.timestamps = timestamps || [];
      state.means = means || [];
      state.stds = stds || [];
      state.sampleCount = state.timestamps.length;
    },
    
    // Individual setters for time-series arrays
    setTimestamps: (state, action) => {
      state.timestamps = action.payload || [];
      state.sampleCount = state.timestamps.length;
    },
    setMeans: (state, action) => {
      state.means = action.payload || [];
    },
    setStds: (state, action) => {
      state.stds = action.payload || [];
    },
    
    // Append single data point (for live updates)
    appendDataPoint: (state, action) => {
      const { timestamp, mean, std } = action.payload;
      state.timestamps.push(timestamp);
      state.means.push(mean);
      state.stds.push(std);
      state.sampleCount = state.timestamps.length;
      
      // Trim to buffer duration
      const maxSamples = Math.ceil(state.bufferDuration * state.updateFreq);
      if (state.timestamps.length > maxSamples) {
        state.timestamps = state.timestamps.slice(-maxSamples);
        state.means = state.means.slice(-maxSamples);
        state.stds = state.stds.slice(-maxSamples);
      }
    },
    
    clearData: (state) => {
      state.timestamps = [];
      state.means = [];
      state.stds = [];
      state.sampleCount = 0;
    },
    
    // Statistics
    setStatistics: (state, action) => {
      const { data_min, data_max, data_mean, data_std, sample_count } = action.payload;
      state.dataMin = data_min;
      state.dataMax = data_max;
      state.dataMean = data_mean;
      state.dataStd = data_std;
      if (sample_count !== undefined) state.sampleCount = sample_count;
    },
    
    // UI state
    setPlotWindowSeconds: (state, action) => {
      state.plotWindowSeconds = action.payload;
    },
    setAutoScale: (state, action) => {
      state.autoScale = action.payload;
    },
    
    // Bulk update from backend params
    updateParamsFromBackend: (state, action) => {
      const params = action.payload;
      if (params.roi_center !== undefined) state.roiCenter = params.roi_center;
      if (params.roi_size !== undefined) state.roiSize = params.roi_size;
      if (params.update_freq !== undefined) state.updateFreq = params.update_freq;
      if (params.buffer_duration !== undefined) state.bufferDuration = params.buffer_duration;
      if (params.decimation !== undefined) state.decimation = params.decimation;
    },
    
    // Update state from backend
    updateStateFromBackend: (state, action) => {
      const stateData = action.payload;
      if (stateData.is_capturing !== undefined) state.isCapturing = stateData.is_capturing;
      if (stateData.frame_count !== undefined) state.frameCount = stateData.frame_count;
      if (stateData.sample_count !== undefined) state.sampleCount = stateData.sample_count;
      if (stateData.actual_fps !== undefined) state.actualFps = stateData.actual_fps;
      if (stateData.start_time !== undefined) state.startTime = stateData.start_time;
    },
  },
});

// Export actions
export const {
  setIsCapturing,
  setFrameCount,
  setSampleCount,
  setActualFps,
  setStartTime,
  setRoiCenter,
  setRoiSize,
  setUpdateFreq,
  setBufferDuration,
  setDecimation,
  setTimeSeriesData,
  setTimestamps,
  setMeans,
  setStds,
  appendDataPoint,
  clearData,
  setStatistics,
  setPlotWindowSeconds,
  setAutoScale,
  updateParamsFromBackend,
  updateStateFromBackend,
} = michelsonSlice.actions;

// Selector
export const getMichelsonState = (state) => state.michelsonState;

// Export reducer
export default michelsonSlice.reducer;
