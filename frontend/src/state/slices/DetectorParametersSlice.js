import { createSlice } from "@reduxjs/toolkit";

// Define the initial state
const initialDetectorParametersState = {
  currentDetectorName: null,
  parameters: {
    exposure: "",
    gain: "",
    pixelSize: "",
    binning: "",
    blacklevel: "",
    isRGB: false,
    mode: "manual",
  },
  lastUpdate: null,
  isLoading: false,
};

// Create slice
const detectorParametersSlice = createSlice({
  name: "detectorParametersState",
  initialState: initialDetectorParametersState,
  reducers: {
    setCurrentDetectorName: (state, action) => {
      state.currentDetectorName = action.payload;
    },
    setParameters: (state, action) => {
      state.parameters = action.payload;
      state.lastUpdate = Date.now();
    },
    updateParameter: (state, action) => {
      const { key, value } = action.payload;
      state.parameters[key] = value;
      state.lastUpdate = Date.now();
    },
    setIsLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    resetState: (state) => {
      return initialDetectorParametersState;
    },
  },
});

// Export actions from slice
export const {
  setCurrentDetectorName,
  setParameters,
  updateParameter,
  setIsLoading,
  resetState,
} = detectorParametersSlice.actions;

// Selector helpers
export const getDetectorParametersState = (state) =>
  state.detectorParametersState;
export const getDetectorParameters = (state) =>
  state.detectorParametersState.parameters;
export const getDetectorCurrentName = (state) =>
  state.detectorParametersState.currentDetectorName;
export const getLastParameterUpdate = (state) =>
  state.detectorParametersState.lastUpdate;

// Export reducer from slice
export default detectorParametersSlice.reducer;
