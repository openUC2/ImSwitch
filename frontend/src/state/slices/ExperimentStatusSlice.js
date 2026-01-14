import { createSlice } from "@reduxjs/toolkit";


// Define the initial state
const initialExperimentStatusState = {
  status: "idle", 
};

// Create slice
const experimentStatusSlice = createSlice({
  name: "experimentStatusState",
  initialState: initialExperimentStatusState,
  reducers: {
    setStatus: (state, action) => {
      console.log("setStatus");
      state.status = action.payload;
    },
       
    //------------------------ state
    resetState: (state) => {
      console.log("resetState");
      return { ...initialExperimentStatusState };// Reset to initial state
    },
  },
});

// Export actions from slice
export const {
    setStatus,

    resetState
} = experimentStatusSlice.actions;

// Selector helper
export const getExperimentStatusState = (state) => state.experimentStatusState;

// Export reducer from slice
export default experimentStatusSlice.reducer;
