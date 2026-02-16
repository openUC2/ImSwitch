// src/state/slices/OffAxisHoloSlice.js
// Redux slice for off-axis hologram processing state management

import { createSlice } from "@reduxjs/toolkit";

// Initial state for off-axis hologram processing
const initialOffAxisHoloState = {
  // Processing status
  isProcessing: false,
  isPaused: false,
  isStreamingFft: false,
  isStreamingMagnitude: false,
  isStreamingPhase: false,
  
  // Optical parameters
  pixelsize: 3.45e-6, // meters
  wavelength: 488e-9, // meters
  na: 0.3,
  dz: 0.0, // digital refocus distance in meters
  
  // Binning and processing rate
  binning: 1,
  updateFreq: 10.0, // Hz
  previewMaxSize: 512,
  
  // Sensor ROI parameters
  roiCenter: null, // [x, y] in pixels
  roiSize: 512,
  
  // Sideband (CC) selection in FFT space
  ccCenter: null, // [x, y] in FFT pixels
  ccSize: [200, 200], // [width, height] or square size
  ccRadius: 100, // legacy: half-size
  
  // Apodization (edge damping)
  apodizationEnabled: false,
  apodizationType: "tukey", // "tukey", "hann", "hamming", "blackman"
  apodizationAlpha: 0.1, // tukey parameter
  
  // Image transforms
  colorChannel: "green",
  flipX: false,
  flipY: false,
  rotation: 0,
  
  // Frame statistics
  fftShape: null, // [height, width] of FFT
  lastProcessTime: 0.0,
  frameCount: 0,
  processedCount: 0,
  
  // UI state
  showDeveloperOptions: false,
  selectedTab: 0, // 0 = camera, 1 = FFT, 2 = magnitude, 3 = phase
};

// Create slice
const offAxisHoloSlice = createSlice({
  name: "offAxisHoloState",
  initialState: initialOffAxisHoloState,
  reducers: {
    // Processing control
    setIsProcessing: (state, action) => {
      state.isProcessing = action.payload;
    },
    setIsPaused: (state, action) => {
      state.isPaused = action.payload;
    },
    setIsStreamingFft: (state, action) => {
      state.isStreamingFft = action.payload;
    },
    setIsStreamingMagnitude: (state, action) => {
      state.isStreamingMagnitude = action.payload;
    },
    setIsStreamingPhase: (state, action) => {
      state.isStreamingPhase = action.payload;
    },
    
    // Optical parameters
    setPixelsize: (state, action) => {
      state.pixelsize = action.payload;
    },
    setWavelength: (state, action) => {
      state.wavelength = action.payload;
    },
    setNa: (state, action) => {
      state.na = action.payload;
    },
    setDz: (state, action) => {
      state.dz = action.payload;
    },
    
    // Processing parameters
    setBinning: (state, action) => {
      state.binning = action.payload;
    },
    setUpdateFreq: (state, action) => {
      state.updateFreq = action.payload;
    },
    setPreviewMaxSize: (state, action) => {
      state.previewMaxSize = action.payload;
    },
    
    // Sensor ROI
    setRoiCenter: (state, action) => {
      state.roiCenter = action.payload;
    },
    setRoiSize: (state, action) => {
      state.roiSize = action.payload;
    },
    
    // Sideband selection
    setCcCenter: (state, action) => {
      state.ccCenter = action.payload;
    },
    setCcSize: (state, action) => {
      state.ccSize = action.payload;
    },
    setCcRadius: (state, action) => {
      state.ccRadius = action.payload;
    },
    
    // Apodization
    setApodizationEnabled: (state, action) => {
      state.apodizationEnabled = action.payload;
    },
    setApodizationType: (state, action) => {
      state.apodizationType = action.payload;
    },
    setApodizationAlpha: (state, action) => {
      state.apodizationAlpha = action.payload;
    },
    
    // Image transforms
    setColorChannel: (state, action) => {
      state.colorChannel = action.payload;
    },
    setFlipX: (state, action) => {
      state.flipX = action.payload;
    },
    setFlipY: (state, action) => {
      state.flipY = action.payload;
    },
    setRotation: (state, action) => {
      state.rotation = action.payload;
    },
    
    // Frame statistics
    setFftShape: (state, action) => {
      state.fftShape = action.payload;
    },
    setLastProcessTime: (state, action) => {
      state.lastProcessTime = action.payload;
    },
    setFrameCount: (state, action) => {
      state.frameCount = action.payload;
    },
    setProcessedCount: (state, action) => {
      state.processedCount = action.payload;
    },
    
    // UI state
    setShowDeveloperOptions: (state, action) => {
      state.showDeveloperOptions = action.payload;
    },
    setSelectedTab: (state, action) => {
      state.selectedTab = action.payload;
    },
    
    // Bulk update from backend
    updateFromBackend: (state, action) => {
      const params = action.payload;
      if (params.pixelsize !== undefined) state.pixelsize = params.pixelsize;
      if (params.wavelength !== undefined) state.wavelength = params.wavelength;
      if (params.na !== undefined) state.na = params.na;
      if (params.dz !== undefined) state.dz = params.dz;
      if (params.binning !== undefined) state.binning = params.binning;
      if (params.update_freq !== undefined) state.updateFreq = params.update_freq;
      if (params.preview_max_size !== undefined) state.previewMaxSize = params.preview_max_size;
      if (params.roi_center !== undefined) state.roiCenter = params.roi_center;
      if (params.roi_size !== undefined) state.roiSize = params.roi_size;
      if (params.cc_center !== undefined) state.ccCenter = params.cc_center;
      if (params.cc_size !== undefined) state.ccSize = params.cc_size;
      if (params.cc_radius !== undefined) state.ccRadius = params.cc_radius;
      if (params.apodization_enabled !== undefined) state.apodizationEnabled = params.apodization_enabled;
      if (params.apodization_type !== undefined) state.apodizationType = params.apodization_type;
      if (params.apodization_alpha !== undefined) state.apodizationAlpha = params.apodization_alpha;
      if (params.color_channel !== undefined) state.colorChannel = params.color_channel;
      if (params.flip_x !== undefined) state.flipX = params.flip_x;
      if (params.flip_y !== undefined) state.flipY = params.flip_y;
      if (params.rotation !== undefined) state.rotation = params.rotation;
    },
    
    // Update state from backend
    updateStateFromBackend: (state, action) => {
      const stateData = action.payload;
      if (stateData.is_processing !== undefined) state.isProcessing = stateData.is_processing;
      if (stateData.is_paused !== undefined) state.isPaused = stateData.is_paused;
      if (stateData.is_streaming_fft !== undefined) state.isStreamingFft = stateData.is_streaming_fft;
      if (stateData.is_streaming_magnitude !== undefined) state.isStreamingMagnitude = stateData.is_streaming_magnitude;
      if (stateData.is_streaming_phase !== undefined) state.isStreamingPhase = stateData.is_streaming_phase;
      if (stateData.fft_shape !== undefined) state.fftShape = stateData.fft_shape;
      if (stateData.last_process_time !== undefined) state.lastProcessTime = stateData.last_process_time;
      if (stateData.frame_count !== undefined) state.frameCount = stateData.frame_count;
      if (stateData.processed_count !== undefined) state.processedCount = stateData.processed_count;
    },
  },
});

// Export actions
export const {
  setIsProcessing,
  setIsPaused,
  setIsStreamingFft,
  setIsStreamingMagnitude,
  setIsStreamingPhase,
  setPixelsize,
  setWavelength,
  setNa,
  setDz,
  setBinning,
  setUpdateFreq,
  setPreviewMaxSize,
  setRoiCenter,
  setRoiSize,
  setCcCenter,
  setCcSize,
  setCcRadius,
  setApodizationEnabled,
  setApodizationType,
  setApodizationAlpha,
  setColorChannel,
  setFlipX,
  setFlipY,
  setRotation,
  setFftShape,
  setLastProcessTime,
  setFrameCount,
  setProcessedCount,
  setShowDeveloperOptions,
  setSelectedTab,
  updateFromBackend,
  updateStateFromBackend,
} = offAxisHoloSlice.actions;

// Selector
export const getOffAxisHoloState = (state) => state.offAxisHoloState;

// Export reducer
export default offAxisHoloSlice.reducer;
