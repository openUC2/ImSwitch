// redux/HomingSlice.js
// Runtime-only state (NOT persisted) tracking the collision-safe frame-homing
// procedure. Fed by the backend sigHomingState signal via WebSocketHandler.
import { createSlice } from "@reduxjs/toolkit";

const initialHomingState = {
  active: false,
  cancelled: false,
  phase: "idle", // starting | homing_z | lifting_z | homing_x | homing_y | restoring_z | done | cancelled | error
  axes: {
    Z: "idle", // idle | pending | homing | done
    X: "idle",
    Y: "idle",
    A: "idle",
  },
  message: "",
};

const homingSlice = createSlice({
  name: "homing",
  initialState: initialHomingState,
  reducers: {
    // Replace the whole homing state from a backend sigHomingState payload.
    setHomingState: (state, action) => {
      const p = action.payload || {};
      state.active = p.active ?? state.active;
      state.cancelled = p.cancelled ?? state.cancelled;
      state.phase = p.phase ?? state.phase;
      state.message = p.message ?? state.message;
      if (p.axes) {
        state.axes = { ...state.axes, ...p.axes };
      }
    },
    resetHomingState: () => initialHomingState,
  },
});

// Export actions
export const { setHomingState, resetHomingState } = homingSlice.actions;

// Selector helper
export const getHomingState = (state) => state.homing;

// Export reducer
export default homingSlice.reducer;
