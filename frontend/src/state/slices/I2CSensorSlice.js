import { createSlice } from "@reduxjs/toolkit";


// Define the initial state
const initialI2CState = {
  // I2C sensor data
  sensorData: {},
  isPolling: false,
  pollPeriod: 1000, // Default polling period in milliseconds
  lastPollTimestamp: null,
  error: null,  
};

// Create slice
const I2CSensorSlice = createSlice({
  name: "i2cState",
  initialState: initialI2CState,
  reducers: {
    updateSensorData: (state, action) => {
      state.sensorData = action.payload;
      state.lastPollTimestamp = new Date().toISOString();
    },
    setPollingStatus: (state, action) => {
      state.isPolling = action.payload;
    },
    setPollPeriod: (state, action) => {
      state.pollPeriod = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
  },
});

// Selector helper
export const getI2CSensorState = (state) => state.i2cState;

// export updateSensorData
export const { updateSensorData, setPollingStatus, setPollPeriod, setError } = I2CSensorSlice.actions;

// Export reducer
export default I2CSensorSlice.reducer;