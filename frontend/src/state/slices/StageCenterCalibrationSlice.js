// src/state/slices/StageCenterCalibrationSlice.js
import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // Wizard state
  isWizardOpen: false,
  activeStep: 0,
  
  // Calibration parameters
  startX: 0,
  startY: 0,
  exposureTimeUs: 3000,
  speed: 5000,
  stepUm: 50.0,
  maxRadiusUm: 2000.0,
  brightnessFactor: 1.4,
  
  // Current position
  currentX: 0,
  currentY: 0,
  
  // Calibration state
  isCalibrationRunning: false,
  calibrationResults: [],
  foundCenterX: null,
  foundCenterY: null,
  
  // Stage map state
  stageMapWidth: 5000,  // in micrometers
  stageMapHeight: 5000,
  stageMapCenterX: 0,
  stageMapCenterY: 0,
  
  // Manual entry state
  manualCenterX: "",
  manualCenterY: "",
  
  // UI state
  isLoading: false,
  error: null,
  successMessage: null,
};

const stageCenterCalibrationSlice = createSlice({
  name: "stageCenterCalibration",
  initialState,
  reducers: {
    // Wizard control
    setWizardOpen: (state, action) => {
      state.isWizardOpen = action.payload;
      if (!action.payload) {
        state.activeStep = 0; // Reset to first step when closing
      }
    },
    setActiveStep: (state, action) => {
      state.activeStep = action.payload;
    },
    nextStep: (state) => {
      state.activeStep = Math.min(state.activeStep + 1, 5); // 6 steps total (0-5)
    },
    previousStep: (state) => {
      state.activeStep = Math.max(state.activeStep - 1, 0);
    },
    
    // Calibration parameters
    setStartX: (state, action) => {
      state.startX = action.payload;
    },
    setStartY: (state, action) => {
      state.startY = action.payload;
    },
    setExposureTimeUs: (state, action) => {
      state.exposureTimeUs = action.payload;
    },
    setSpeed: (state, action) => {
      state.speed = action.payload;
    },
    setStepUm: (state, action) => {
      state.stepUm = action.payload;
    },
    setMaxRadiusUm: (state, action) => {
      state.maxRadiusUm = action.payload;
    },
    setBrightnessFactor: (state, action) => {
      state.brightnessFactor = action.payload;
    },
    setCalibrationParameters: (state, action) => {
      const { startX, startY, exposureTimeUs, speed, stepUm, maxRadiusUm, brightnessFactor } = action.payload;
      if (startX !== undefined) state.startX = startX;
      if (startY !== undefined) state.startY = startY;
      if (exposureTimeUs !== undefined) state.exposureTimeUs = exposureTimeUs;
      if (speed !== undefined) state.speed = speed;
      if (stepUm !== undefined) state.stepUm = stepUm;
      if (maxRadiusUm !== undefined) state.maxRadiusUm = maxRadiusUm;
      if (brightnessFactor !== undefined) state.brightnessFactor = brightnessFactor;
    },
    
    // Current position
    setCurrentX: (state, action) => {
      state.currentX = action.payload;
    },
    setCurrentY: (state, action) => {
      state.currentY = action.payload;
    },
    setCurrentPosition: (state, action) => {
      const { x, y } = action.payload;
      if (x !== undefined) state.currentX = x;
      if (y !== undefined) state.currentY = y;
    },
    
    // Calibration state
    setIsCalibrationRunning: (state, action) => {
      state.isCalibrationRunning = action.payload;
    },
    setCalibrationResults: (state, action) => {
      state.calibrationResults = action.payload;
    },
    setFoundCenter: (state, action) => {
      const { x, y } = action.payload;
      state.foundCenterX = x;
      state.foundCenterY = y;
    },
    
    // Stage map state
    setStageMapDimensions: (state, action) => {
      const { width, height } = action.payload;
      if (width !== undefined) state.stageMapWidth = width;
      if (height !== undefined) state.stageMapHeight = height;
    },
    setStageMapCenter: (state, action) => {
      const { x, y } = action.payload;
      if (x !== undefined) state.stageMapCenterX = x;
      if (y !== undefined) state.stageMapCenterY = y;
    },
    
    // Manual entry state
    setManualCenterX: (state, action) => {
      state.manualCenterX = action.payload;
    },
    setManualCenterY: (state, action) => {
      state.manualCenterY = action.payload;
    },
    setManualCenter: (state, action) => {
      const { x, y } = action.payload;
      if (x !== undefined) state.manualCenterX = x;
      if (y !== undefined) state.manualCenterY = y;
    },
    
    // UI state
    setIsLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
    setSuccessMessage: (state, action) => {
      state.successMessage = action.payload;
    },
    clearMessages: (state) => {
      state.error = null;
      state.successMessage = null;
    },
    
    // Reset actions
    resetCalibrationResults: (state) => {
      state.calibrationResults = [];
      state.foundCenterX = null;
      state.foundCenterY = null;
    },
    resetWizard: (state) => {
      return initialState;
    },
  },
});

// Export actions
export const {
  setWizardOpen,
  setActiveStep,
  nextStep,
  previousStep,
  setStartX,
  setStartY,
  setExposureTimeUs,
  setSpeed,
  setStepUm,
  setMaxRadiusUm,
  setBrightnessFactor,
  setCalibrationParameters,
  setCurrentX,
  setCurrentY,
  setCurrentPosition,
  setIsCalibrationRunning,
  setCalibrationResults,
  setFoundCenter,
  setStageMapDimensions,
  setStageMapCenter,
  setManualCenterX,
  setManualCenterY,
  setManualCenter,
  setIsLoading,
  setError,
  setSuccessMessage,
  clearMessages,
  resetCalibrationResults,
  resetWizard,
} = stageCenterCalibrationSlice.actions;

// Export selector
export const getStageCenterCalibrationState = (state) => state.stageCenterCalibration;

// Export reducer
export default stageCenterCalibrationSlice.reducer;