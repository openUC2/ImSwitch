import { createSlice } from "@reduxjs/toolkit";

// Define the initial state
const initialFocusLockState = {
  // Focus lock status
  isFocusLocked: false,
  isCalibrating: false,
  isMeasuring: false,
  
  // PI Controller parameters
  multiplier: 1.0,
  kp: 0.1,
  ki: 0.01,
  setPoint: 0.0,
  safetyDistanceLimit: 500.0,
  safetyMoveLimit: 3.0,
  minStepThreshold: 0.002,
  safetyMotionActive: false,
  
  // Astigmatism parameters
  gaussian_sigma: 1.0,
  background_threshold: 100.0,
  cropSize: 100,
  cropCenter: [0, 0],
  frameSize: [640, 480], // Default frame size, can be adjusted
  
  // Camera parameters
  exposureTime: 100.0, // Default exposure time in milliseconds
  gain: 0.0, // Default gain value
  
  // Images and streaming
  lastImage: null,
  lastCroppedImage: null,
  pollImageUrl: null,
  
  // Focus values for graph (last 50 points)
  focusValues: [],
  focusTimepoints: [],
  motorPositions: [], // Add motor position tracking
  setPointSignals: [], // Store setPointSignal history for chart
  maxFocusHistory: 50,
  
  // Current focus value
  currentFocusValue: 0.0,
  currentFocusMotorPosition: 0.0, // Add current motor position
  setPointSignal: 0.0,
  
  // One-step autofocus
  storedTargetSetpoint: null, // Stored focus value for one-step AF
  lastAutofocusResult: null, // Last autofocus result
  isPerformingAutofocus: false,
  calibrationRange: null, // [min, max] focus values from calibration
  
  // UI state
  selectedCropRegion: null,
  showImageSelector: false,
  isLoadingImage: false,
};

// Create slice
const focusLockSlice = createSlice({
  name: "focusLockState",
  initialState: initialFocusLockState,
  reducers: {
    // Focus lock control
    setFocusLocked: (state, action) => {
      state.isFocusLocked = action.payload;
    },
    setIsCalibrating: (state, action) => {
      state.isCalibrating = action.payload;
    },
    setIsMeasuring: (state, action) => {
      state.isMeasuring = action.payload;
    },
    
    // PI Controller parameters
    setMultiplier: (state, action) => {
      state.multiplier = action.payload;
    },
    setKp: (state, action) => {
      state.kp = action.payload;
    },
    setKi: (state, action) => {
      state.ki = action.payload;
    },
    setSetPoint: (state, action) => {
      state.setPoint = action.payload;
    },
    setSafetyDistanceLimit: (state, action) => {
      state.safetyDistanceLimit = action.payload;
    },
    setSafetyMoveLimit: (state, action) => {
      state.safetyMoveLimit = action.payload;
    },
    setMinStepThreshold: (state, action) => {
      state.minStepThreshold = action.payload;
    },
    setSafetyMotionActive: (state, action) => {
      state.safetyMotionActive = action.payload;
    },
    
    // Astigmatism parameters
    setgaussian_sigma: (state, action) => {
      state.gaussian_sigma = action.payload;
    },
    setBackgroundThreshold: (state, action) => {
      state.background_threshold = action.payload;
    },
    setCropSize: (state, action) => {
      state.cropSize = action.payload;
    },
    setCropCenter: (state, action) => {
      state.cropCenter = action.payload;
    },
    setFrameSize: (state, action) => {
      state.frameSize = action.payload;
    },
    
    // Camera parameters
    setExposureTime: (state, action) => {
      state.exposureTime = action.payload;
    },
    setGain: (state, action) => {
      state.gain = action.payload;
    },
    
    // Images
    setLastImage: (state, action) => {
      state.lastImage = action.payload;
    },
    setLastCroppedImage: (state, action) => {
      state.lastCroppedImage = action.payload;
    },
    setPollImageUrl: (state, action) => {
      state.pollImageUrl = action.payload;
    },
    
    // Focus values
    addFocusValue: (state, action) => {
      const { focusValue, setPointSignal, currentFocusMotorPosition, timestamp } = action.payload;
      
      // Add new values
      state.focusValues.push(focusValue);
      state.focusTimepoints.push(timestamp);
      state.motorPositions.push(currentFocusMotorPosition || 0);
      state.currentFocusValue = focusValue; // Update current value here too
      state.currentFocusMotorPosition = currentFocusMotorPosition || 0;
      state.setPointSignals.push(setPointSignal ?? 0);
      
      // Keep only last 50 values to prevent memory buildup
      if (state.focusValues.length > state.maxFocusHistory) {
        state.focusValues.shift();
        state.focusTimepoints.shift();
        state.motorPositions.shift();
        state.setPointSignals.shift();
      }
    },
    clearFocusHistory: (state) => {
      state.focusValues = [];
      state.focusTimepoints = [];
      state.motorPositions = [];
    state.setPointSignals = [];
    },
    setCurrentFocusValue: (state, action) => {
      state.currentFocusValue = action.payload;
    },
    setCurrentFocusMotorPosition: (state, action) => {
      state.currentFocusMotorPosition = action.payload;
    },
    
    // One-step autofocus
    setStoredTargetSetpoint: (state, action) => {
      state.storedTargetSetpoint = action.payload;
    },
    setLastAutofocusResult: (state, action) => {
      state.lastAutofocusResult = action.payload;
    },
    setIsPerformingAutofocus: (state, action) => {
      state.isPerformingAutofocus = action.payload;
    },
    setCalibrationRange: (state, action) => {
      state.calibrationRange = action.payload;
    },
    
    // UI state
    setSelectedCropRegion: (state, action) => {
      state.selectedCropRegion = action.payload;
    },
    setShowImageSelector: (state, action) => {
      state.showImageSelector = action.payload;
    },
    setIsLoadingImage: (state, action) => {
      state.isLoadingImage = action.payload;
    },
    
    // Reset crop center coordinates
    resetCropCenter: (state) => {
      state.cropCenter = [0, 0];
      state.selectedCropRegion = null;
    },
    
    // Reset state
    resetState: (state) => {
      return { ...initialFocusLockState };
    },
  },
});

// Export actions
export const {
  setFocusLocked,
  setIsCalibrating,
  setIsMeasuring,
  setMultiplier,
  setKp,
  setKi,
  setSetPoint,
  setSafetyDistanceLimit,
  setSafetyMoveLimit,
  setMinStepThreshold,
  setSafetyMotionActive,
  setgaussian_sigma,
  setBackgroundThreshold,
  setCropSize,
  setFrameSize,
  setCropCenter,
  setExposureTime,
  setGain,
  setLastImage,
  setLastCroppedImage,
  setPollImageUrl,
  addFocusValue,
  clearFocusHistory,
  setCurrentFocusValue,
  setCurrentFocusMotorPosition,
  setStoredTargetSetpoint,
  setLastAutofocusResult,
  setIsPerformingAutofocus,
  setCalibrationRange,
  setSelectedCropRegion,
  setShowImageSelector,
  setIsLoadingImage,
  resetCropCenter,
  resetState,
} = focusLockSlice.actions;

// Selector helper
export const getFocusLockState = (state) => state.focusLockState;

// Export reducer
export default focusLockSlice.reducer;