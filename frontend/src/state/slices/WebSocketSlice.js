import { createSlice } from "@reduxjs/toolkit";

// Define the initial state
const initialState = {
  connected: false,
  signalCount: 0,
  testStatus: "idle", // Add test status tracking
  lastTestTime: null,
};

// Create webSocketConnectionSlice slice
const webSocketSlice = createSlice({
  name: "webSocketState",
  initialState: initialState,
  reducers: {
    setConnected: (state, action) => {
      console.log("setConnected", action.payload);
      state.connected = action.payload;
    },
    incrementSignalCount: (state, action) => {
      //console.log("incrementSignalCount");
      state.signalCount++;
    },
    // Add WebSocket test reducers
    setTestStatus: (state, action) => {
      state.testStatus = action.payload; // "idle", "testing", "success", "failed", "timeout"
      if (action.payload === "testing") {
        state.lastTestTime = Date.now();
      }
    },
    resetState: (state) => {
      console.log("resetState");
      return { ...initialState }; // Reset to initial state
    },
  },
});

// Export actions from slice
export const { setConnected, incrementSignalCount, setTestStatus, resetState } =
  webSocketSlice.actions;

// Selector helper
export const getWebSocketState = (state) => state.webSocketState;

// Export reducer from slice
export default webSocketSlice.reducer;
