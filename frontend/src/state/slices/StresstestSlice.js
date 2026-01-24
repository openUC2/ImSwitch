import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for the stresstest
const initialStresstestState = {
  // Parameters
  minPosX: 0.0,
  maxPosX: 10000.0,
  minPosY: 0.0,
  maxPosY: 10000.0,
  numRandomPositions: 5,
  numCycles: 3,
  timeInterval: 10.0,
  illuminationIntensity: 50.0,
  exposureTime: 0.1,
  saveImages: true,
  outputPath: "",
  enableImageBasedError: false,
  numImagesPerPosition: 5,
  imageRegistrationMethod: "fft",
  pixelSizeUM: 0.1,

  // Results
  totalPositions: 0,
  completedPositions: 0,
  averagePositionError: 0.0,
  maxPositionError: 0.0,
  positionErrors: [],
  timestamps: [],
  targetPositions: [],
  actualPositions: [],
  isRunning: false,
  imageBasedErrors: [],
  imageShifts: [],
  imageRegistrationResults: [],
  averageImageError: 0.0,
  maxImageError: 0.0,

  // UI state
  tabIndex: 0,
};

// Create slice
const stresstestSlice = createSlice({
  name: "stresstestState",
  initialState: initialStresstestState,
  reducers: {
    // Parameter setters
    setMinPosX: (state, action) => {
      state.minPosX = action.payload;
    },
    setMaxPosX: (state, action) => {
      state.maxPosX = action.payload;
    },
    setMinPosY: (state, action) => {
      state.minPosY = action.payload;
    },
    setMaxPosY: (state, action) => {
      state.maxPosY = action.payload;
    },
    setNumRandomPositions: (state, action) => {
      state.numRandomPositions = action.payload;
    },
    setNumCycles: (state, action) => {
      state.numCycles = action.payload;
    },
    setTimeInterval: (state, action) => {
      state.timeInterval = action.payload;
    },
    setIlluminationIntensity: (state, action) => {
      state.illuminationIntensity = action.payload;
    },
    setExposureTime: (state, action) => {
      state.exposureTime = action.payload;
    },
    setSaveImages: (state, action) => {
      state.saveImages = action.payload;
    },
    setOutputPath: (state, action) => {
      state.outputPath = action.payload;
    },
    setEnableImageBasedError: (state, action) => {
      state.enableImageBasedError = action.payload;
    },
    setNumImagesPerPosition: (state, action) => {
      state.numImagesPerPosition = action.payload;
    },
    setImageRegistrationMethod: (state, action) => {
      state.imageRegistrationMethod = action.payload;
    },
    setPixelSizeUM: (state, action) => {
      state.pixelSizeUM = action.payload;
    },

    // Results setters
    setTotalPositions: (state, action) => {
      state.totalPositions = action.payload;
    },
    setCompletedPositions: (state, action) => {
      state.completedPositions = action.payload;
    },
    setAveragePositionError: (state, action) => {
      state.averagePositionError = action.payload;
    },
    setMaxPositionError: (state, action) => {
      state.maxPositionError = action.payload;
    },
    setPositionErrors: (state, action) => {
      state.positionErrors = action.payload;
    },
    setTimestamps: (state, action) => {
      state.timestamps = action.payload;
    },
    setTargetPositions: (state, action) => {
      state.targetPositions = action.payload;
    },
    setActualPositions: (state, action) => {
      state.actualPositions = action.payload;
    },
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },
    setImageBasedErrors: (state, action) => {
      state.imageBasedErrors = action.payload;
    },
    setImageShifts: (state, action) => {
      state.imageShifts = action.payload;
    },
    setImageRegistrationResults: (state, action) => {
      state.imageRegistrationResults = action.payload;
    },
    setAverageImageError: (state, action) => {
      state.averageImageError = action.payload;
    },
    setMaxImageError: (state, action) => {
      state.maxImageError = action.payload;
    },

    // UI setters
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },

    // Bulk setters
    setStresstestParams: (state, action) => {
      const params = action.payload;
      state.minPosX = params.minPosX ?? state.minPosX;
      state.maxPosX = params.maxPosX ?? state.maxPosX;
      state.minPosY = params.minPosY ?? state.minPosY;
      state.maxPosY = params.maxPosY ?? state.maxPosY;
      state.numRandomPositions = params.numRandomPositions ?? state.numRandomPositions;
      state.numCycles = params.numCycles ?? state.numCycles;
      state.timeInterval = params.timeInterval ?? state.timeInterval;
      state.illuminationIntensity = params.illuminationIntensity ?? state.illuminationIntensity;
      state.exposureTime = params.exposureTime ?? state.exposureTime;
      state.saveImages = params.saveImages ?? state.saveImages;
      state.outputPath = params.outputPath ?? state.outputPath;
      state.enableImageBasedError = params.enableImageBasedError ?? state.enableImageBasedError;
      state.numImagesPerPosition = params.numImagesPerPosition ?? state.numImagesPerPosition;
      state.imageRegistrationMethod = params.imageRegistrationMethod ?? state.imageRegistrationMethod;
      state.pixelSizeUM = params.pixelSizeUM ?? state.pixelSizeUM;
    },
    setStresstestResults: (state, action) => {
      const results = action.payload;
      state.totalPositions = results.totalPositions ?? state.totalPositions;
      state.completedPositions = results.completedPositions ?? state.completedPositions;
      state.averagePositionError = results.averagePositionError ?? state.averagePositionError;
      state.maxPositionError = results.maxPositionError ?? state.maxPositionError;
      state.positionErrors = results.positionErrors ?? state.positionErrors;
      state.timestamps = results.timestamps ?? state.timestamps;
      state.targetPositions = results.targetPositions ?? state.targetPositions;
      state.actualPositions = results.actualPositions ?? state.actualPositions;
      state.isRunning = results.isRunning ?? state.isRunning;
      state.imageBasedErrors = results.imageBasedErrors ?? state.imageBasedErrors;
      state.imageShifts = results.imageShifts ?? state.imageShifts;
      state.imageRegistrationResults = results.imageRegistrationResults ?? state.imageRegistrationResults;
      state.averageImageError = results.averageImageError ?? state.averageImageError;
      state.maxImageError = results.maxImageError ?? state.maxImageError;
    },

    // Reset state
    resetState: (state) => {
      return initialStresstestState;
    },
  },
});

// Export actions from slice
export const {
  setMinPosX,
  setMaxPosX,
  setMinPosY,
  setMaxPosY,
  setNumRandomPositions,
  setNumCycles,
  setTimeInterval,
  setIlluminationIntensity,
  setExposureTime,
  setSaveImages,
  setOutputPath,
  setEnableImageBasedError,
  setNumImagesPerPosition,
  setImageRegistrationMethod,
  setPixelSizeUM,
  setTotalPositions,
  setCompletedPositions,
  setAveragePositionError,
  setMaxPositionError,
  setPositionErrors,
  setTimestamps,
  setTargetPositions,
  setActualPositions,
  setIsRunning,
  setImageBasedErrors,
  setImageShifts,
  setImageRegistrationResults,
  setAverageImageError,
  setMaxImageError,
  setTabIndex,
  setStresstestParams,
  setStresstestResults,
  resetState,
} = stresstestSlice.actions;

// Selector helper
export const getStresstestState = (state) => state.stresstestState;

// Export reducer from slice
export default stresstestSlice.reducer;