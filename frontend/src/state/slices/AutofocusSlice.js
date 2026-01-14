// AutofocusSlice.js
import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for autofocus
const initialAutofocusState = {
  // Autofocus parameters
  rangeZ: 10,
  resolutionZ: 1,
  defocusZ: 0,
  illuminationChannel: "", // Selected illumination channel for autofocus
  tSettle: 0.1, // Settling time between steps (seconds)
  isDebug: false, // Save debug images
  nGauss: 7, // Gaussian blur sigma (0 to disable)
  nCropsize: 2048, // Crop size for focus calculation
  focusAlgorithm: "LAPE", // Focus measurement method (LAPE, GLVA, JPEG)
  staticOffset: 0.0, // Static offset to add to final focus position
  twoStage: false, // Enable two-stage autofocus (coarse + fine scan)
  
  // Runtime state
  isRunning: false,
  
  // Plot data - focus positions and contrast values
  plotData: null, // { x: [...], y: [...] } where x = positions, y = contrast
  
  // UI state
  showPlot: false,
  
  // Live monitoring state
  isLiveMonitoring: false,
  liveFocusValue: null, // Current live focus value
  liveMonitoringPeriod: 0.5, // Update period in seconds
  liveMonitoringMethod: "LAPE", // Focus measurement method
  liveMonitoringCropsize: 2048, // Crop size for live monitoring
};

// Create autofocus slice
const autofocusSlice = createSlice({
  name: "autofocusState",
  initialState: initialAutofocusState,
  reducers: {
    // Parameter setters
    setRangeZ: (state, action) => {
      state.rangeZ = action.payload;
    },
    setResolutionZ: (state, action) => {
      state.resolutionZ = action.payload;
    },
    setDefocusZ: (state, action) => {
      state.defocusZ = action.payload;
    },
    setIlluminationChannel: (state, action) => {
      state.illuminationChannel = action.payload;
    },
    setTSettle: (state, action) => {
      state.tSettle = action.payload;
    },
    setIsDebug: (state, action) => {
      state.isDebug = action.payload;
    },
    setNGauss: (state, action) => {
      state.nGauss = action.payload;
    },
    setNCropsize: (state, action) => {
      state.nCropsize = action.payload;
    },
    setFocusAlgorithm: (state, action) => {
      state.focusAlgorithm = action.payload;
    },
    setStaticOffset: (state, action) => {
      state.staticOffset = action.payload;
    },
    setTwoStage: (state, action) => {
      state.twoStage = action.payload;
    },
    
    // Runtime state
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },
    
    // Plot data
    setPlotData: (state, action) => {
      // action.payload should be { x: [...], y: [...] }
      state.plotData = action.payload;
    },
    clearPlotData: (state) => {
      state.plotData = null;
    },
    
    // UI state
    setShowPlot: (state, action) => {
      state.showPlot = action.payload;
    },
    toggleShowPlot: (state) => {
      state.showPlot = !state.showPlot;
    },
    
    // Live monitoring state
    setIsLiveMonitoring: (state, action) => {
      state.isLiveMonitoring = action.payload;
    },
    setLiveFocusValue: (state, action) => {
      // action.payload should be { focus_value: number, timestamp: number, method: string }
      state.liveFocusValue = action.payload;
    },
    setLiveMonitoringPeriod: (state, action) => {
      state.liveMonitoringPeriod = action.payload;
    },
    setLiveMonitoringMethod: (state, action) => {
      state.liveMonitoringMethod = action.payload;
    },
    setLiveMonitoringCropsize: (state, action) => {
      state.liveMonitoringCropsize = action.payload;
    },
    
    // Reset state
    resetState: (state) => {
      return { ...initialAutofocusState };
    },
  },
});

// Export actions
export const {
  setRangeZ,
  setResolutionZ,
  setDefocusZ,
  setIlluminationChannel,
  setTSettle,
  setIsDebug,
  setNGauss,
  setNCropsize,
  setFocusAlgorithm,
  setStaticOffset,
  setTwoStage,
  setIsRunning,
  setPlotData,
  clearPlotData,
  setShowPlot,
  toggleShowPlot,
  setIsLiveMonitoring,
  setLiveFocusValue,
  setLiveMonitoringPeriod,
  setLiveMonitoringMethod,
  setLiveMonitoringCropsize,
  resetState,
} = autofocusSlice.actions;

// Selector helper
export const getAutofocusState = (state) => state.autofocusState;

// Export reducer
export default autofocusSlice.reducer;
