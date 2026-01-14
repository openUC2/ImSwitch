import { createSlice } from "@reduxjs/toolkit";

// Define the initial state
const initialLiveViewState = {
  detectors: [],
  activeTab: 0,
  imageUrls: {},
  pollImageUrl: null,
  pixelSize: null,
  isStreamRunning: false,
  lastSnapPath: null, // Store the last snapped image path
  showPositionController: false, // Persistent toggle for position controller visibility
};

// Create slice
const liveViewSlice = createSlice({
  name: "liveViewState",
  initialState: initialLiveViewState,
  reducers: {
    setDetectors: (state, action) => {
      state.detectors = action.payload;
    },
    setActiveTab: (state, action) => {
      state.activeTab = action.payload;
    },
    setImageUrls: (state, action) => {
      state.imageUrls = action.payload;
    },
    setPollImageUrl: (state, action) => {
      state.pollImageUrl = action.payload;
    },
    setPixelSize: (state, action) => {
      state.pixelSize = action.payload;
    },
    setIsStreamRunning: (state, action) => {
      state.isStreamRunning = action.payload;
    },
    setLastSnapPath: (state, action) => {
      state.lastSnapPath = action.payload;
    },
    setShowPositionController: (state, action) => {
      state.showPositionController = action.payload;
    },
    resetState: (state) => {
      return initialLiveViewState;
    },
  },
});

// Export actions from slice
export const {
  setDetectors,
  setActiveTab,
  setImageUrls,
  setPollImageUrl,
  setPixelSize,
  setIsStreamRunning,
  setLastSnapPath,
  setShowPositionController,
  resetState,
} = liveViewSlice.actions;

// Selector helper
export const getLiveViewState = (state) => state.liveViewState;

// Export reducer from slice
export default liveViewSlice.reducer;
