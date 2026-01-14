import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // Position coordinates
  currentX: "",
  currentY: "",
  targetX: "",
  targetY: "",

  // Calculated offsets
  calculatedOffsetX: "",
  calculatedOffsetY: "",

  // Loaded offsets from server
  loadedOffsetX: "",
  loadedOffsetY: "",

  // Manual offset inputs
  manualOffsetX: "",
  manualOffsetY: "",

  // Image and detector data
  imageUrls: {},
  detectors: [],

  // Utility state
  reloadTrigger: 0,
};

const stageOffsetCalibrationSlice = createSlice({
  name: "stageOffsetCalibration",
  initialState,
  reducers: {
    // Position coordinate actions
    setCurrentX: (state, action) => {
      state.currentX = action.payload;
    },
    setCurrentY: (state, action) => {
      state.currentY = action.payload;
    },
    setTargetX: (state, action) => {
      state.targetX = action.payload;
    },
    setTargetY: (state, action) => {
      state.targetY = action.payload;
    },

    // Calculated offset actions
    setCalculatedOffsetX: (state, action) => {
      state.calculatedOffsetX = action.payload;
    },
    setCalculatedOffsetY: (state, action) => {
      state.calculatedOffsetY = action.payload;
    },

    // Loaded offset actions
    setLoadedOffsetX: (state, action) => {
      state.loadedOffsetX = action.payload;
    },
    setLoadedOffsetY: (state, action) => {
      state.loadedOffsetY = action.payload;
    },

    // Manual offset actions
    setManualOffsetX: (state, action) => {
      state.manualOffsetX = action.payload;
    },
    setManualOffsetY: (state, action) => {
      state.manualOffsetY = action.payload;
    },

    // Image and detector actions
    setImageUrls: (state, action) => {
      state.imageUrls = action.payload;
    },
    updateImageUrl: (state, action) => {
      const { detector, url } = action.payload;
      state.imageUrls[detector] = url;
    },
    setDetectors: (state, action) => {
      state.detectors = action.payload;
    },

    // Utility actions
    incrementReloadTrigger: (state) => {
      state.reloadTrigger += 1;
    },
    setReloadTrigger: (state, action) => {
      state.reloadTrigger = action.payload;
    },

    // Batch update actions
    setCurrentPosition: (state, action) => {
      const { x, y } = action.payload;
      state.currentX = x;
      state.currentY = y;
    },
    setTargetPosition: (state, action) => {
      const { x, y } = action.payload;
      state.targetX = x;
      state.targetY = y;
    },
    setCalculatedOffsets: (state, action) => {
      const { offsetX, offsetY } = action.payload;
      state.calculatedOffsetX = offsetX;
      state.calculatedOffsetY = offsetY;
    },
    setLoadedOffsets: (state, action) => {
      const { offsetX, offsetY } = action.payload;
      state.loadedOffsetX = offsetX;
      state.loadedOffsetY = offsetY;
    },

    // Reset actions
    resetCalculatedOffsets: (state) => {
      state.calculatedOffsetX = "";
      state.calculatedOffsetY = "";
    },
    resetManualOffsets: (state) => {
      state.manualOffsetX = "";
      state.manualOffsetY = "";
    },
  },
});

// Export actions
export const {
  setCurrentX,
  setCurrentY,
  setTargetX,
  setTargetY,
  setCalculatedOffsetX,
  setCalculatedOffsetY,
  setLoadedOffsetX,
  setLoadedOffsetY,
  setManualOffsetX,
  setManualOffsetY,
  setImageUrls,
  updateImageUrl,
  setDetectors,
  incrementReloadTrigger,
  setReloadTrigger,
  setCurrentPosition,
  setTargetPosition,
  setCalculatedOffsets,
  setLoadedOffsets,
  resetCalculatedOffsets,
  resetManualOffsets,
} = stageOffsetCalibrationSlice.actions;

// Export selector
export const getStageOffsetCalibrationState = (state) => state.stageOffsetCalibration;

// Export reducer
export default stageOffsetCalibrationSlice.reducer;