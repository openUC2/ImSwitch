import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // UI state
  tabIndex: 0,

  // Experiment parameters
  timeStamp: "0",
  experimentName: "Test",
  experimentDescription: "Some description",
  uniqueId: 1,

  // Flow parameters
  numImages: 10,
  volumePerImage: 1000,
  timeToStabilize: 0.5,
  pumpSpeed: 10000,

  // Status
  isRunning: false,
  currentImageCount: 0,
};

const flowStopSlice = createSlice({
  name: "flowStop",
  initialState,
  reducers: {
    // UI actions
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },

    // Experiment parameter actions
    setTimeStamp: (state, action) => {
      state.timeStamp = action.payload;
    },
    setExperimentName: (state, action) => {
      state.experimentName = action.payload;
    },
    setExperimentDescription: (state, action) => {
      state.experimentDescription = action.payload;
    },
    setUniqueId: (state, action) => {
      state.uniqueId = action.payload;
    },

    // Flow parameter actions
    setNumImages: (state, action) => {
      state.numImages = action.payload;
    },
    setVolumePerImage: (state, action) => {
      state.volumePerImage = action.payload;
    },
    setTimeToStabilize: (state, action) => {
      state.timeToStabilize = action.payload;
    },
    setPumpSpeed: (state, action) => {
      state.pumpSpeed = action.payload;
    },

    // Status actions
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },
    setCurrentImageCount: (state, action) => {
      state.currentImageCount = action.payload;
    },

    // Batch update actions
    setExperimentInfo: (state, action) => {
      const { name, description, uniqueId } = action.payload;
      state.experimentName = name;
      state.experimentDescription = description;
      state.uniqueId = uniqueId;
    },
    setFlowParameters: (state, action) => {
      const { numImages, volumePerImage, timeToStabilize, pumpSpeed } = action.payload;
      state.numImages = numImages;
      state.volumePerImage = volumePerImage;
      state.timeToStabilize = timeToStabilize;
      state.pumpSpeed = pumpSpeed;
    },

    // Reset actions
    resetExperiment: (state) => {
      state.isRunning = false;
      state.currentImageCount = 0;
      state.timeStamp = "0";
    },
    resetToDefaults: (state) => {
      return { ...initialState };
    },
  },
});

// Export actions
export const {
  setTabIndex,
  setTimeStamp,
  setExperimentName,
  setExperimentDescription,
  setUniqueId,
  setNumImages,
  setVolumePerImage,
  setTimeToStabilize,
  setPumpSpeed,
  setIsRunning,
  setCurrentImageCount,
  setExperimentInfo,
  setFlowParameters,
  resetExperiment,
  resetToDefaults,
} = flowStopSlice.actions;

// Export selector
export const getFlowStopState = (state) => state.flowStop;

// Export reducer
export default flowStopSlice.reducer;