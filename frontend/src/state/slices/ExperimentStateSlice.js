import { createSlice } from "@reduxjs/toolkit";

//  Initial experiment state
const initialExperimentState = {
  status: "",
  stepId: 0,
  stepName: "",
  totalSteps: 0,
};

const experimentStateSlice = createSlice({
  name: "experimentWorkflowState",
  initialState: initialExperimentState,
  reducers: {
    //  Called from the WebSocketHandler
    setStatus: (state, action) => {
      console.log("setStatus", action.payload);
      state.status = action.payload;
    },
    //  Called from the WebSocketHandler
    setStepID: (state, action) => {
      console.log("setStepID", action.payload);
      state.stepId = action.payload;
    },
    //  Called from the WebSocketHandler
    setStepName: (state, action) => {
      console.log("setStepName", action.payload);
      state.stepName = action.payload;
    },
    //  Called from the WebSocketHandler
    setTotalSteps: (state, action) => {
      console.log("setTotalSteps", action.payload);
      state.totalSteps = action.payload;
    },
    resetState: (state) => {
      console.log("resetState");
      return { ...initialExperimentState }; // Reset to initial state
    },
  },
});

//  Export actions and reducer
export const {
  setStatus,
  setStepID,
  setStepName,
  setTotalSteps,
  resetState
} = experimentStateSlice.actions;

//  Selector helper
export const getExperimentState = (state) => state.experimentWorkflowState;

export default experimentStateSlice.reducer;