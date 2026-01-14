import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for the maze game controller
const initialMazeGameState = {
  // Player info
  playerName: "",
  
  // Game parameters
  cropSize: 20,
  jumpLow: 2,
  jumpHigh: 10,
  history: 2,
  downscale: 1,
  pollInterval: 250,
  stepSize: 100, // Joystick step size
  
  // Start position
  startPosition: { x: 0, y: 0 },
  
  // Game state
  running: false,
  counter: 0,
  elapsed: 0,
  smoothMean: 0, // smooth_mean from backend
  
  // XY trace data
  xyTrace: [], // Array of {x, y, timestamp}
  
  // Preview image
  previewImage: null, // Base64 encoded preview image
  
  // Hall of Fame (stored locally)
  hallOfFame: [], // Array of {playerName, time, counter, trace, timestamp}
};

// Create slice
const mazeGameSlice = createSlice({
  name: "mazeGameState",
  initialState: initialMazeGameState,
  reducers: {
    // Player info
    setPlayerName: (state, action) => {
      state.playerName = action.payload;
    },
    
    // Parameter setters
    setCropSize: (state, action) => {
      state.cropSize = action.payload;
    },
    setJumpLow: (state, action) => {
      state.jumpLow = action.payload;
    },
    setJumpHigh: (state, action) => {
      state.jumpHigh = action.payload;
    },
    setHistory: (state, action) => {
      state.history = action.payload;
    },
    setDownscale: (state, action) => {
      state.downscale = action.payload;
    },
    setPollInterval: (state, action) => {
      state.pollInterval = action.payload;
    },
    setStepSize: (state, action) => {
      state.stepSize = action.payload;
    },
    
    // Start position
    setStartPosition: (state, action) => {
      state.startPosition = action.payload;
    },
    
    // Game state setters
    setRunning: (state, action) => {
      state.running = action.payload;
    },
    setCounter: (state, action) => {
      state.counter = action.payload;
    },
    setElapsed: (state, action) => {
      state.elapsed = action.payload;
    },
    setSmoothMean: (state, action) => {
      state.smoothMean = action.payload;
    },
    
    // XY trace
    addTracePoint: (state, action) => {
      state.xyTrace.push(action.payload);
    },
    clearTrace: (state) => {
      state.xyTrace = [];
    },
    
    // Preview image
    setPreviewImage: (state, action) => {
      state.previewImage = action.payload;
    },
    
    // Game state bulk setter
    setGameState: (state, action) => {
      const gameState = action.payload;
      state.running = gameState.running ?? state.running;
      state.counter = gameState.counter ?? state.counter;
      state.elapsed = gameState.elapsed_s ?? state.elapsed;
      state.smoothMean = gameState.smooth_mean ?? state.smoothMean;
    },
    
    // Hall of Fame
    addToHallOfFame: (state, action) => {
      state.hallOfFame.push(action.payload);
      // Sort by time (ascending - best time first)
      state.hallOfFame.sort((a, b) => a.time - b.time);
    },
    clearHallOfFame: (state) => {
      state.hallOfFame = [];
    },
    setHallOfFame: (state, action) => {
      state.hallOfFame = action.payload;
    },
    
    // Reset state
    resetState: () => {
      return initialMazeGameState;
    },
  },
});

// Export actions from slice
export const {
  setPlayerName,
  setCropSize,
  setJumpLow,
  setJumpHigh,
  setHistory,
  setDownscale,
  setPollInterval,
  setStepSize,
  setStartPosition,
  setRunning,
  setCounter,
  setElapsed,
  setSmoothMean,
  addTracePoint,
  clearTrace,
  setPreviewImage,
  setGameState,
  addToHallOfFame,
  clearHallOfFame,
  setHallOfFame,
  resetState,
} = mazeGameSlice.actions;

// Selector helper
export const getMazeGameState = (state) => state.mazeGameState;

// Export reducer from slice
export default mazeGameSlice.reducer;
