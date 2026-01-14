import { createSlice } from "@reduxjs/toolkit";

// Define the initial state
const initialHistoScanState = {
  illuminationSource: "Laser 1",
  illuminationValue: 128,
  stepSizeX: "300",
  stepSizeY: "300",
  stepsX: "2",
  stepsY: "2",
  menuPosition: {
    mouseX: null,
    mouseY: null,
  },
  path: "Default Path",
  timeInterval: "",
  numberOfScans: "",
  scanStatus: false,
  scanCount: 0,
  scanIndex: 0,
  scanResultAvailable: false,
  isStitchAshlar: false,
  isStitchAshlarFlipX: false,
  isStitchAshlarFlipY: false,
  resizeFactor: 1,
  initPosX: "",
  initPosY: "",
  imageUrl: "",
};

// Create slice
const histoScanSlice = createSlice({
  name: "histoScanState",
  initialState: initialHistoScanState,
  reducers: {
    setIlluminationSource: (state, action) => {
      state.illuminationSource = action.payload;
    },
    setIlluminationValue: (state, action) => {
      state.illuminationValue = action.payload;
    },
    setStepSizeX: (state, action) => {
      state.stepSizeX = action.payload;
    },
    setStepSizeY: (state, action) => {
      state.stepSizeY = action.payload;
    },
    setStepsX: (state, action) => {
      state.stepsX = action.payload;
    },
    setStepsY: (state, action) => {
      state.stepsY = action.payload;
    },
    setMenuPosition: (state, action) => {
      state.menuPosition = action.payload;
    },
    setPath: (state, action) => {
      state.path = action.payload;
    },
    setTimeInterval: (state, action) => {
      state.timeInterval = action.payload;
    },
    setNumberOfScans: (state, action) => {
      state.numberOfScans = action.payload;
    },
    setScanStatus: (state, action) => {
      state.scanStatus = action.payload;
    },
    setScanCount: (state, action) => {
      state.scanCount = action.payload;
    },
    setScanIndex: (state, action) => {
      state.scanIndex = action.payload;
    },
    setScanResultAvailable: (state, action) => {
      state.scanResultAvailable = action.payload;
    },
    setIsStitchAshlar: (state, action) => {
      state.isStitchAshlar = action.payload;
    },
    setIsStitchAshlarFlipX: (state, action) => {
      state.isStitchAshlarFlipX = action.payload;
    },
    setIsStitchAshlarFlipY: (state, action) => {
      state.isStitchAshlarFlipY = action.payload;
    },
    setResizeFactor: (state, action) => {
      state.resizeFactor = action.payload;
    },
    setInitPosX: (state, action) => {
      state.initPosX = action.payload;
    },
    setInitPosY: (state, action) => {
      state.initPosY = action.payload;
    },
    setImageUrl: (state, action) => {
      state.imageUrl = action.payload;
    },
    resetState: (state) => {
      return initialHistoScanState;
    },
  },
});

// Export actions from slice
export const {
  setIlluminationSource,
  setIlluminationValue,
  setStepSizeX,
  setStepSizeY,
  setStepsX,
  setStepsY,
  setMenuPosition,
  setPath,
  setTimeInterval,
  setNumberOfScans,
  setScanStatus,
  setScanCount,
  setScanIndex,
  setScanResultAvailable,
  setIsStitchAshlar,
  setIsStitchAshlarFlipX,
  setIsStitchAshlarFlipY,
  setResizeFactor,
  setInitPosX,
  setInitPosY,
  setImageUrl,
  resetState,
} = histoScanSlice.actions;

// Selector helper
export const getHistoScanState = (state) => state.histoScanState;

// Export reducer from slice
export default histoScanSlice.reducer;