import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for workflow management
const initialWorkflowState = {
  // Workflow status
  status: "idle", // idle, running, paused, stopped, error
  error: null,
  
  // Workflow content
  currentWorkflow: null, // Current workflow JSON
  workflowJson: "", // Generated JSON string
  workflowUid: null, // UID returned from upload
  
  // UI state
  isGeneratingJson: false,
  isUploading: false,
  isStarting: false,
  isStopping: false,
  isPausing: false,
  isResuming: false,
  
  // Progress tracking
  currentStep: 0,
  totalSteps: 0,
  stepProgress: {},
  
  // History
  executionHistory: [],
  lastExecutionTime: null,
};

// Create slice
const workflowSlice = createSlice({
  name: "workflowState",
  initialState: initialWorkflowState,
  reducers: {
    // Workflow status actions
    setWorkflowStatus: (state, action) => {
      state.status = action.payload;
    },
    setWorkflowError: (state, action) => {
      state.error = action.payload;
      state.status = "error";
    },
    clearWorkflowError: (state) => {
      state.error = null;
    },
    
    // Workflow content actions
    setCurrentWorkflow: (state, action) => {
      state.currentWorkflow = action.payload;
    },
    setWorkflowJson: (state, action) => {
      state.workflowJson = action.payload;
    },
    setWorkflowUid: (state, action) => {
      state.workflowUid = action.payload;
    },
    
    // UI state actions
    setIsGeneratingJson: (state, action) => {
      state.isGeneratingJson = action.payload;
    },
    setIsUploading: (state, action) => {
      state.isUploading = action.payload;
    },
    setIsStarting: (state, action) => {
      state.isStarting = action.payload;
    },
    setIsStopping: (state, action) => {
      state.isStopping = action.payload;
    },
    setIsPausing: (state, action) => {
      state.isPausing = action.payload;
    },
    setIsResuming: (state, action) => {
      state.isResuming = action.payload;
    },
    
    // Progress tracking actions
    setCurrentStep: (state, action) => {
      state.currentStep = action.payload;
    },
    setTotalSteps: (state, action) => {
      state.totalSteps = action.payload;
    },
    updateStepProgress: (state, action) => {
      const { stepId, progress } = action.payload;
      state.stepProgress[stepId] = progress;
    },
    
    // History actions
    addExecutionToHistory: (state, action) => {
      state.executionHistory.push({
        ...action.payload,
        timestamp: Date.now()
      });
      // Keep only last 10 executions
      if (state.executionHistory.length > 10) {
        state.executionHistory = state.executionHistory.slice(-10);
      }
    },
    setLastExecutionTime: (state, action) => {
      state.lastExecutionTime = action.payload;
    },
    
    // Reset actions
    resetWorkflowState: (state) => {
      return initialWorkflowState;
    },
    resetWorkflowProgress: (state) => {
      state.currentStep = 0;
      state.totalSteps = 0;
      state.stepProgress = {};
    },
  },
});

// Export actions from slice
export const {
  setWorkflowStatus,
  setWorkflowError,
  clearWorkflowError,
  setCurrentWorkflow,
  setWorkflowJson,
  setWorkflowUid,
  setIsGeneratingJson,
  setIsUploading,
  setIsStarting,
  setIsStopping,
  setIsPausing,
  setIsResuming,
  setCurrentStep,
  setTotalSteps,
  updateStepProgress,
  addExecutionToHistory,
  setLastExecutionTime,
  resetWorkflowState,
  resetWorkflowProgress,
} = workflowSlice.actions;

// Selector helper
export const getWorkflowState = (state) => state.workflowState;

// Export reducer from slice
export default workflowSlice.reducer;