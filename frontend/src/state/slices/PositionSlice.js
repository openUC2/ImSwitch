// redux/positionSlice.js
import { createSlice } from "@reduxjs/toolkit";

// Define initial state for position
const initialPosition = {
  x: 0,
  y: 0,
  z: 0,
  a: 0,
  stepSizes: {
    X: 100,
    Y: 100,
    Z: 10,
    A: 10,
  },
};

// Create position slice
const positionSlice = createSlice({
  name: "position",
  initialState: initialPosition,
  reducers: {
    setPosition: (state, action) => {
      console.log(
        "PositionSlice.setPosition called with payload:",
        action.payload
      );
      console.log("Current state before update:", {
        x: state.x,
        y: state.y,
        z: state.z,
        a: state.a,
      });
      state.x = action.payload.x ?? state.x;
      state.y = action.payload.y ?? state.y;
      state.z = action.payload.z ?? state.z;
      state.a = action.payload.a ?? state.a;
      console.log("State after update:", {
        x: state.x,
        y: state.y,
        z: state.z,
        a: state.a,
      });
    },
    setStepSize: (state, action) => {
      const { axis, value } = action.payload;
      if (
        state.stepSizes[axis] !== undefined &&
        typeof value === "number" &&
        value > 0
      ) {
        state.stepSizes[axis] = value;
      }
    },
    setAllStepSizes: (state, action) => {
      state.stepSizes = { ...state.stepSizes, ...action.payload };
    },
  },
});

// Export actions from position slice
export const { setPosition, setStepSize, setAllStepSizes } =
  positionSlice.actions;

// Selector helper
export const getPositionState = (state) => state.position;

// Export reducer from position slice
export default positionSlice.reducer;
