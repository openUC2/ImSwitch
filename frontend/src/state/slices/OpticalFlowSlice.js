// OpticalFlowSlice.js
//
// Redux state for the OpticalFlow tab. Mirrors the AutofocusSlice layout:
// parameter inputs, runtime flags, live plot data.
import { createSlice } from "@reduxjs/toolkit";

const initialOpticalFlowState = {
  // Sweep parameters (defaults match the backend)
  distanceUm: 2000,
  speedUmS: 100,
  axis: "X",
  warmupFrames: 3,
  minDisplacementPx: 0.5,
  smoothingWindow: 5,

  // Runtime state -- mirrors FlowState on the backend
  state: "idle", // idle | starting | running | finished | aborted | error
  isRunning: false,

  // Live plot data emitted by sigUpdateFlowAngle
  // times: seconds since measurement start; angles: degrees
  times: [],
  angles: [],

  // Aggregated result emitted by sigFlowResult
  // { meanAngle, std, n, nTotal, distanceUm, speedUmS, axis }
  result: null,

  // Last error message (for UI banner)
  errorMessage: null,
};

const opticalFlowSlice = createSlice({
  name: "opticalFlowState",
  initialState: initialOpticalFlowState,
  reducers: {
    setDistanceUm: (state, action) => {
      state.distanceUm = action.payload;
    },
    setSpeedUmS: (state, action) => {
      state.speedUmS = action.payload;
    },
    setAxis: (state, action) => {
      state.axis = action.payload;
    },
    setWarmupFrames: (state, action) => {
      state.warmupFrames = action.payload;
    },
    setMinDisplacementPx: (state, action) => {
      state.minDisplacementPx = action.payload;
    },
    setSmoothingWindow: (state, action) => {
      state.smoothingWindow = action.payload;
    },

    setState: (state, action) => {
      state.state = action.payload;
      // Keep isRunning in sync with the FSM
      state.isRunning =
        action.payload === "running" || action.payload === "starting";
    },
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },

    setLivePlotData: (state, action) => {
      // payload: { times: number[], angles: number[] }
      state.times = action.payload?.times || [];
      state.angles = action.payload?.angles || [];
    },
    appendSample: (state, action) => {
      // payload: { time: number, angle: number }
      // Stream-style append driven by the backend sigUpdateFlowAngle signal.
      // Mirroring the FocusLock pattern keeps a stable array reference for the
      // plot and avoids the periodic redraw / blinking that comes from
      // replacing the whole arrays on every update.
      const t = action.payload?.time;
      const a = action.payload?.angle;
      if (typeof t === "number" && typeof a === "number") {
        state.times.push(t);
        state.angles.push(a);
      }
    },
    clearLivePlotData: (state) => {
      state.times = [];
      state.angles = [];
    },

    setResult: (state, action) => {
      state.result = action.payload;
    },
    clearResult: (state) => {
      state.result = null;
    },

    setErrorMessage: (state, action) => {
      state.errorMessage = action.payload;
    },

    resetState: () => ({ ...initialOpticalFlowState }),
  },
});

export const {
  setDistanceUm,
  setSpeedUmS,
  setAxis,
  setWarmupFrames,
  setMinDisplacementPx,
  setSmoothingWindow,
  setState,
  setIsRunning,
  setLivePlotData,
  appendSample,
  clearLivePlotData,
  setResult,
  clearResult,
  setErrorMessage,
  resetState,
} = opticalFlowSlice.actions;

export const getOpticalFlowState = (state) => state.opticalFlowState;

export default opticalFlowSlice.reducer;
