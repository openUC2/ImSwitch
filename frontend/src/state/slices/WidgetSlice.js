import { createSlice } from "@reduxjs/toolkit";

// Define the initial state (combining WidgetContext and LiveWidgetContext)
const initialWidgetState = {
  // From WidgetContext
  sliderValue: 0,
  generic: { init: "init" },
  
  // From LiveWidgetContext  
  isStreamRunning: false,
};

// Create slice
const widgetSlice = createSlice({
  name: "widgetState",
  initialState: initialWidgetState,
  reducers: {
    setSliderValue: (state, action) => {
      state.sliderValue = action.payload;
    },
    setGeneric: (state, action) => {
      state.generic = action.payload;
    },
    updateGeneric: (state, action) => {
      const { key, value } = action.payload;
      state.generic = {
        ...state.generic,
        [key]: value,
      };
    },
    setIsStreamRunning: (state, action) => {
      state.isStreamRunning = action.payload;
    },
    resetState: (state) => {
      return initialWidgetState;
    },
  },
});

// Export actions from slice
export const {
  setSliderValue,
  setGeneric,
  updateGeneric,
  setIsStreamRunning,
  resetState,
} = widgetSlice.actions;

// Selector helper
export const getWidgetState = (state) => state.widgetState;

// Export reducer from slice
export default widgetSlice.reducer;