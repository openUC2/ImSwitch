// ./state/slices/LEDMatrixSlice.js
import { createSlice } from "@reduxjs/toolkit";

const initialLEDMatrixState = {
  mode: "halves", // could be "halves", "ring", "circle", "all"
  intensity: 255,
  direction: "top",
  circleRadius: 1,
  ringRadius: 3,
  isOn: true
};

const LEDMatrixSlice = createSlice({
  name: "LEDMatrixState",
  initialState: initialLEDMatrixState,
  reducers: {
    setMode: (state, action) => {
      state.mode = action.payload;
    },
    setIntensity: (state, action) => {
      state.intensity = action.payload;
    },
    setDirection: (state, action) => {
      state.direction = action.payload;
    },
    setCircleRadius: (state, action) => {
      state.circleRadius = action.payload;
    },
    setRingRadius: (state, action) => {
      state.ringRadius = action.payload;
    },
    setIsOn: (state, action) => {
      state.isOn = action.payload;
    },
    resetLEDMatrixState: () => {
      return { ...initialLEDMatrixState };
    }
  }
});

export const {
  setMode,
  setIntensity,
  setDirection,
  setCircleRadius,
  setRingRadius,
  setIsOn,
  resetLEDMatrixState
} = LEDMatrixSlice.actions;

export const getLEDMatrixState = (state) => state.LEDMatrixState;

export default LEDMatrixSlice.reducer;
