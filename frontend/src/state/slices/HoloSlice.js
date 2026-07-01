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
  dzMax: 30000e-6, // upper bound of dz slider in meters (default 30 mm)
  dzStep: 1e-6, // dz slider step in meters (default 1 µm)
  binning: 1, // binning factor (1, 2, 4, etc.)
  previousBinning: 1, // store previous binning for pause/resume
  fullFrame: false, // reconstruct full sensor (with binning) instead of ROI

  // ROI parameters
  roiCenter: [0, 0], // [x, y] in pixels (center of ROI)
  roiSize: 256, // square ROI size in pixels

  // Image processing parameters
  colorChannel: "red", // "red", "green", "blue", "white"
  flipX: false,
  flipY: false,
  rotation: 0, // 0, 90, 180, 270 degrees

  // Display raw hologram (reconstruct at dz=0) instead of the slider dz
  showRaw: false,


  // Processing rate
  updateFreq: 10.0, // Hz (processing framerate)

  // Backing detector info (populated by get_camera_info_inlineholo)
  cameraName: null,
  isRGB: false,

  // Frame data
  frameSize: [1920, 1080], // Default camera frame size [width, height]
  lastProcessTime: 0.0,
  lastMjpegEmitTime: 0.0, // wall-clock of last server-side MJPEG push
  mjpegClientCount: 0,
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

  // Background normalization (live divide)
  bgEnabled: false, // divide the live frame by the stored background
  hasBackground: false, // a background image is stored in the backend
  backgroundUrl: null, // base64 data URL preview of the stored background
  backgroundMeta: null, // { mode, num_frames, width, height, timestamp }

  // High-quality refinement (button press)
  refineMethod: "phase_retrieval", // "phase_retrieval" | "tv"
  refineIterations: 30,
  refineSupportThreshold: 0.5,
  refineTvWeight: 0.05,
  isRefining: false, // a refinement is currently running
  refinedAmplitudeUrl: null, // base64 data URL of the reconstructed amplitude
  refinedPhaseUrl: null, // base64 data URL of the reconstructed phase
  refineView: "amplitude", // which result to show: "amplitude" | "phase"
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
    setFullFrame: (state, action) => {
      state.fullFrame = action.payload;
    },
    setDzMax: (state, action) => {
      state.dzMax = action.payload;
    },
    setDzStep: (state, action) => {
      state.dzStep = action.payload;
    },
    setCameraName: (state, action) => {
      state.cameraName = action.payload;
    },
    setIsRGB: (state, action) => {
      state.isRGB = action.payload;
    },
    setLastMjpegEmitTime: (state, action) => {
      state.lastMjpegEmitTime = action.payload;
    },
    setMjpegClientCount: (state, action) => {
      state.mjpegClientCount = action.payload;
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
    setShowRaw: (state, action) => {
      state.showRaw = action.payload;
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

    // Background normalization
    setBgEnabled: (state, action) => {
      state.bgEnabled = action.payload;
    },
    setHasBackground: (state, action) => {
      state.hasBackground = action.payload;
    },
    setBackgroundUrl: (state, action) => {
      state.backgroundUrl = action.payload;
    },
    setBackgroundMeta: (state, action) => {
      state.backgroundMeta = action.payload;
    },

    // High-quality refinement
    setRefineMethod: (state, action) => {
      state.refineMethod = action.payload;
    },
    setRefineIterations: (state, action) => {
      state.refineIterations = action.payload;
    },
    setRefineSupportThreshold: (state, action) => {
      state.refineSupportThreshold = action.payload;
    },
    setRefineTvWeight: (state, action) => {
      state.refineTvWeight = action.payload;
    },
    setIsRefining: (state, action) => {
      state.isRefining = action.payload;
    },
    setRefinedAmplitudeUrl: (state, action) => {
      state.refinedAmplitudeUrl = action.payload;
    },
    setRefinedPhaseUrl: (state, action) => {
      state.refinedPhaseUrl = action.payload;
    },
    setRefineView: (state, action) => {
      state.refineView = action.payload;
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
  setFullFrame,
  setDzMax,
  setDzStep,
  setCameraName,
  setIsRGB,
  setLastMjpegEmitTime,
  setMjpegClientCount,
  setRoiCenter,
  setRoiSize,
  setColorChannel,
  setFlipX,
  setFlipY,
  setRotation,
  setShowRaw,
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
  setBgEnabled,
  setHasBackground,
  setBackgroundUrl,
  setBackgroundMeta,
  setRefineMethod,
  setRefineIterations,
  setRefineSupportThreshold,
  setRefineTvWeight,
  setIsRefining,
  setRefinedAmplitudeUrl,
  setRefinedPhaseUrl,
  setRefineView,
  updateHoloParams,
  resetRoiToCenter,
  resetState,
} = holoSlice.actions;

// Selector helper
export const getHoloState = (state) => state.holoState;

// Export reducer
export default holoSlice.reducer;
