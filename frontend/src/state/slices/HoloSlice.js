// src/state/slices/HoloSlice.js
// Redux slice for inline hologram processing state management

import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for hologram processing
const initialHoloState = {
  // Processing status
  isProcessing: false,
  isPaused: false,
  isStreaming: false,
  
  // Processing parameters
  pixelsize: 3.45e-6, // meters
  wavelength: 488e-9, // meters (488nm = blue-green laser)
  na: 0.3, // numerical aperture
  dz: 0.0, // propagation distance in meters
  binning: 1, // binning factor (1, 2, 4, etc.)
  previousBinning: 1, // store previous binning for pause/resume
  
  // ROI parameters
  roiCenter: [0, 0], // [x, y] in pixels (center of ROI)
  roiSize: 256, // square ROI size in pixels
  
  // Image processing parameters
  colorChannel: "green", // "red", "green", "blue"
  flipX: false,
  flipY: false,
  rotation: 0, // 0, 90, 180, 270 degrees
  
  // Processing rate
  updateFreq: 10.0, // Hz (processing framerate)
  
  // Frame data
  frameSize: [1920, 1080], // Default camera frame size [width, height]
  lastProcessTime: 0.0,
  frameCount: 0,
  processedCount: 0,
  
  // Images
  lastRawImage: null, // Last raw camera image
  lastProcessedImage: null, // Last processed hologram image
  
  // UI state
  showDeveloperOptions: false,
  isLoadingImage: false,
  
  // MJPEG stream URLs
  rawStreamUrl: null,
  processedStreamUrl: null,
};

// Create slice
const holoSlice = createSlice({
  name: "holoState",
  initialState: initialHoloState,
  reducers: {
    // Processing control
    setIsProcessing: (state, action) => {
      state.isProcessing = action.payload;
    },
    setIsPaused: (state, action) => {
      state.isPaused = action.payload;
    },
    setIsStreaming: (state, action) => {
      state.isStreaming = action.payload;
    },
    
    // Basic processing parameters
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
    setBinning: (state, action) => {
      state.binning = action.payload;
    },
    setPreviousBinning: (state, action) => {
      state.previousBinning = action.payload;
    },
    
    // ROI parameters
    setRoiCenter: (state, action) => {
      state.roiCenter = action.payload;
    },
    setRoiSize: (state, action) => {
      state.roiSize = action.payload;
    },
    
    // Image processing parameters
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
    
    // Processing rate
    setUpdateFreq: (state, action) => {
      state.updateFreq = action.payload;
    },
    
    // Frame data
    setFrameSize: (state, action) => {
      state.frameSize = action.payload;
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
    
    // Images
    setLastRawImage: (state, action) => {
      state.lastRawImage = action.payload;
    },
    setLastProcessedImage: (state, action) => {
      state.lastProcessedImage = action.payload;
    },
    
    // UI state
    setShowDeveloperOptions: (state, action) => {
      state.showDeveloperOptions = action.payload;
    },
    setIsLoadingImage: (state, action) => {
      state.isLoadingImage = action.payload;
    },
    
    // MJPEG stream URLs
    setRawStreamUrl: (state, action) => {
      state.rawStreamUrl = action.payload;
    },
    setProcessedStreamUrl: (state, action) => {
      state.processedStreamUrl = action.payload;
    },
    
    // Bulk parameter update
    updateHoloParams: (state, action) => {
      const params = action.payload;
      Object.keys(params).forEach(key => {
        if (state.hasOwnProperty(key)) {
          state[key] = params[key];
        }
      });
    },
    
    // Reset ROI to center
    resetRoiToCenter: (state) => {
      state.roiCenter = [0, 0];
    },
    
    // Reset all state
    resetState: (state) => {
      return { ...initialHoloState };
    },
  },
});

// Export actions
export const {
  setIsProcessing,
  setIsPaused,
  setIsStreaming,
  setPixelsize,
  setWavelength,
  setNa,
  setDz,
  setBinning,
  setPreviousBinning,
  setRoiCenter,
  setRoiSize,
  setColorChannel,
  setFlipX,
  setFlipY,
  setRotation,
  setUpdateFreq,
  setFrameSize,
  setLastProcessTime,
  setFrameCount,
  setProcessedCount,
  setLastRawImage,
  setLastProcessedImage,
  setShowDeveloperOptions,
  setIsLoadingImage,
  setRawStreamUrl,
  setProcessedStreamUrl,
  updateHoloParams,
  resetRoiToCenter,
  resetState,
} = holoSlice.actions;

// Selector helper
export const getHoloState = (state) => state.holoState;

// Export reducer
export default holoSlice.reducer;
