import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for the demo controller
const initialDemoState = {
  // Demo parameters (matching the DemoParams model from the backend)
  maxRangeX: 10000.0,
  maxRangeY: 10000.0,
  scanningScheme: "random",
  illuminationMode: "random",
  
  // Grid/spiral specific parameters
  gridRows: 3,
  gridColumns: 3,
  spiralShells: 3,
  
  // Random specific parameters
  numRandomPositions: 10,
  
  // Demo control parameters
  dwellTime: 0.5,
  totalRunTime: 600000.0,
  
  // Demo results
  isRunning: false,
  currentPosition: { x: 0, y: 0 },
  completedPositions: 0,
  totalPositions: 0,
  elapsedTime: 0,
  remainingTime: 0,
  
  // UI state
  tabIndex: 0,
};

// Create slice
const demoSlice = createSlice({
  name: "demoState",
  initialState: initialDemoState,
  reducers: {
    // Parameter setters
    setMaxRangeX: (state, action) => {
      state.maxRangeX = action.payload;
    },
    setMaxRangeY: (state, action) => {
      state.maxRangeY = action.payload;
    },
    setScanningScheme: (state, action) => {
      state.scanningScheme = action.payload;
    },
    setIlluminationMode: (state, action) => {
      state.illuminationMode = action.payload;
    },
    setGridRows: (state, action) => {
      state.gridRows = action.payload;
    },
    setGridColumns: (state, action) => {
      state.gridColumns = action.payload;
    },
    setSpiralShells: (state, action) => {
      state.spiralShells = action.payload;
    },
    setNumRandomPositions: (state, action) => {
      state.numRandomPositions = action.payload;
    },
    setDwellTime: (state, action) => {
      state.dwellTime = action.payload;
    },
    setTotalRunTime: (state, action) => {
      state.totalRunTime = action.payload;
    },
    
    // Demo state setters
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },
    setCurrentPosition: (state, action) => {
      state.currentPosition = action.payload;
    },
    setCompletedPositions: (state, action) => {
      state.completedPositions = action.payload;
    },
    setTotalPositions: (state, action) => {
      state.totalPositions = action.payload;
    },
    setElapsedTime: (state, action) => {
      state.elapsedTime = action.payload;
    },
    setRemainingTime: (state, action) => {
      state.remainingTime = action.payload;
    },
    
    // UI setters
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },
    
    // Bulk setters
    setDemoParams: (state, action) => {
      const params = action.payload;
      state.maxRangeX = params.maxRangeX ?? state.maxRangeX;
      state.maxRangeY = params.maxRangeY ?? state.maxRangeY;
      state.scanningScheme = params.scanningScheme ?? state.scanningScheme;
      state.illuminationMode = params.illuminationMode ?? state.illuminationMode;
      state.gridRows = params.gridRows ?? state.gridRows;
      state.gridColumns = params.gridColumns ?? state.gridColumns;
      state.spiralShells = params.spiralShells ?? state.spiralShells;
      state.numRandomPositions = params.numRandomPositions ?? state.numRandomPositions;
      state.dwellTime = params.dwellTime ?? state.dwellTime;
      state.totalRunTime = params.totalRunTime ?? state.totalRunTime;
    },
    setDemoResults: (state, action) => {
      const results = action.payload;
      state.isRunning = results.isRunning ?? state.isRunning;
      state.currentPosition = results.currentPosition ?? state.currentPosition;
      state.completedPositions = results.completedPositions ?? state.completedPositions;
      state.totalPositions = results.totalPositions ?? state.totalPositions;
      state.elapsedTime = results.elapsedTime ?? state.elapsedTime;
      state.remainingTime = results.remainingTime ?? state.remainingTime;
    },
    
    // Reset state
    resetState: (state) => {
      return initialDemoState;
    },
  },
});

// Export actions from slice
export const {
  setMaxRangeX,
  setMaxRangeY,
  setScanningScheme,
  setIlluminationMode,
  setGridRows,
  setGridColumns,
  setSpiralShells,
  setNumRandomPositions,
  setDwellTime,
  setTotalRunTime,
  setIsRunning,
  setCurrentPosition,
  setCompletedPositions,
  setTotalPositions,
  setElapsedTime,
  setRemainingTime,
  setTabIndex,
  setDemoParams,
  setDemoResults,
  resetState,
} = demoSlice.actions;

// Selector helper
export const getDemoState = (state) => state.demoState;

// Export reducer from slice
export default demoSlice.reducer;