// src/state/slices/CompositeAcquisitionSlice.js
// Redux slice for composite acquisition (multi-illumination → fused JPEG) state management

import { createSlice } from "@reduxjs/toolkit";

// Define the initial state for composite acquisition
const initialCompositeState = {
  // Acquisition status
  isRunning: false,
  isStreaming: false,
  currentStep: 0,
  cycleCount: 0,
  lastCycleTimeMs: 0.0,
  averageFps: 0.0,
  errorMessage: "",
  
  // Available illumination sources (fetched from backend)
  illuminationSources: [],
  
  // Illumination steps configuration
  steps: [
    // Default example steps (will be replaced by backend config)
    // { illumination: "laser488", intensity: 0.3, exposureMs: null, settleMs: 10.0, enabled: true },
    // { illumination: "laser635", intensity: 0.2, exposureMs: null, settleMs: 10.0, enabled: true },
    // { illumination: "LED_white", intensity: 0.5, exposureMs: null, settleMs: 10.0, enabled: true },
  ],
  
  // RGB channel mapping
  mapping: {
    R: "", // Red channel illumination source
    G: "", // Green channel illumination source
    B: "", // Blue channel illumination source
  },
  
  // Acquisition parameters
  fpsTarget: 1.0, // Target frames per second
  jpegQuality: 100, // JPEG compression quality (0-100)
  normalizeChannels: true, // Normalize each channel before fusion
  autoExposure: false, // Use per-step exposure overrides
  
  // Preview image (base64 for single captures)
  lastCompositeImage: null,
  lastCaptureMetadata: null,
  
  // UI state
  showChannelConfig: true, // Show channel configuration panel
  showAdvancedSettings: false, // Show advanced settings
  selectedStepIndex: -1, // Currently selected step for editing (-1 = none)
  
  // MJPEG stream URL
  streamUrl: null,
  
  // Loading states
  isLoading: false,
  isCapturing: false,
};

// Create slice
const compositeAcquisitionSlice = createSlice({
  name: "compositeAcquisitionState",
  initialState: initialCompositeState,
  reducers: {
    // Acquisition control
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },
    setIsStreaming: (state, action) => {
      state.isStreaming = action.payload;
    },
    setCurrentStep: (state, action) => {
      state.currentStep = action.payload;
    },
    setCycleCount: (state, action) => {
      state.cycleCount = action.payload;
    },
    setLastCycleTimeMs: (state, action) => {
      state.lastCycleTimeMs = action.payload;
    },
    setAverageFps: (state, action) => {
      state.averageFps = action.payload;
    },
    setErrorMessage: (state, action) => {
      state.errorMessage = action.payload;
    },
    
    // Update full state from backend
    updateState: (state, action) => {
      const { is_running, is_streaming, current_step, cycle_count, 
              last_cycle_time_ms, average_fps, error_message } = action.payload;
      if (is_running !== undefined) state.isRunning = is_running;
      if (is_streaming !== undefined) state.isStreaming = is_streaming;
      if (current_step !== undefined) state.currentStep = current_step;
      if (cycle_count !== undefined) state.cycleCount = cycle_count;
      if (last_cycle_time_ms !== undefined) state.lastCycleTimeMs = last_cycle_time_ms;
      if (average_fps !== undefined) state.averageFps = average_fps;
      if (error_message !== undefined) state.errorMessage = error_message;
    },
    
    // Illumination sources
    setIlluminationSources: (state, action) => {
      state.illuminationSources = action.payload;
    },
    
    // Steps management
    setSteps: (state, action) => {
      state.steps = action.payload;
    },
    addStep: (state, action) => {
      state.steps.push(action.payload);
    },
    removeStep: (state, action) => {
      const index = action.payload;
      if (index >= 0 && index < state.steps.length) {
        state.steps.splice(index, 1);
      }
    },
    updateStep: (state, action) => {
      const { index, updates } = action.payload;
      if (index >= 0 && index < state.steps.length) {
        state.steps[index] = { ...state.steps[index], ...updates };
      }
    },
    toggleStepEnabled: (state, action) => {
      const index = action.payload;
      if (index >= 0 && index < state.steps.length) {
        state.steps[index].enabled = !state.steps[index].enabled;
      }
    },
    moveStepUp: (state, action) => {
      const index = action.payload;
      if (index > 0 && index < state.steps.length) {
        const temp = state.steps[index];
        state.steps[index] = state.steps[index - 1];
        state.steps[index - 1] = temp;
      }
    },
    moveStepDown: (state, action) => {
      const index = action.payload;
      if (index >= 0 && index < state.steps.length - 1) {
        const temp = state.steps[index];
        state.steps[index] = state.steps[index + 1];
        state.steps[index + 1] = temp;
      }
    },
    
    // RGB mapping
    setMapping: (state, action) => {
      state.mapping = { ...state.mapping, ...action.payload };
    },
    setMappingR: (state, action) => {
      state.mapping.R = action.payload;
    },
    setMappingG: (state, action) => {
      state.mapping.G = action.payload;
    },
    setMappingB: (state, action) => {
      state.mapping.B = action.payload;
    },
    
    // Acquisition parameters
    setFpsTarget: (state, action) => {
      state.fpsTarget = action.payload;
    },
    setJpegQuality: (state, action) => {
      state.jpegQuality = action.payload;
    },
    setNormalizeChannels: (state, action) => {
      state.normalizeChannels = action.payload;
    },
    setAutoExposure: (state, action) => {
      state.autoExposure = action.payload;
    },
    
    // Update parameters from backend response
    updateParameters: (state, action) => {
      const params = action.payload;
      if (params.steps) {
        // Convert backend format to frontend format
        state.steps = params.steps.map(s => ({
          illumination: s.illumination || "",
          intensity: s.intensity ?? 0.5,
          exposureMs: s.exposure_ms ?? null,
          settleMs: s.settle_ms ?? 100.0,
          enabled: s.enabled ?? true,
        }));
      }
      if (params.mapping) {
        state.mapping = { ...state.mapping, ...params.mapping };
      }
      if (params.fps_target !== undefined) state.fpsTarget = params.fps_target;
      if (params.jpeg_quality !== undefined) state.jpegQuality = params.jpeg_quality;
      if (params.normalize_channels !== undefined) state.normalizeChannels = params.normalize_channels;
      if (params.auto_exposure !== undefined) state.autoExposure = params.auto_exposure;
    },
    
    // Preview image
    setLastCompositeImage: (state, action) => {
      state.lastCompositeImage = action.payload;
    },
    setLastCaptureMetadata: (state, action) => {
      state.lastCaptureMetadata = action.payload;
    },
    
    // UI state
    setShowChannelConfig: (state, action) => {
      state.showChannelConfig = action.payload;
    },
    setShowAdvancedSettings: (state, action) => {
      state.showAdvancedSettings = action.payload;
    },
    setSelectedStepIndex: (state, action) => {
      state.selectedStepIndex = action.payload;
    },
    
    // Stream URL
    setStreamUrl: (state, action) => {
      state.streamUrl = action.payload;
    },
    
    // Loading states
    setIsLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    setIsCapturing: (state, action) => {
      state.isCapturing = action.payload;
    },
    
    // Reset to initial state
    resetState: (state) => {
      return { ...initialCompositeState };
    },
  },
});

// Export actions
export const {
  setIsRunning,
  setIsStreaming,
  setCurrentStep,
  setCycleCount,
  setLastCycleTimeMs,
  setAverageFps,
  setErrorMessage,
  updateState,
  setIlluminationSources,
  setSteps,
  addStep,
  removeStep,
  updateStep,
  toggleStepEnabled,
  moveStepUp,
  moveStepDown,
  setMapping,
  setMappingR,
  setMappingG,
  setMappingB,
  setFpsTarget,
  setJpegQuality,
  setNormalizeChannels,
  setAutoExposure,
  updateParameters,
  setLastCompositeImage,
  setLastCaptureMetadata,
  setShowChannelConfig,
  setShowAdvancedSettings,
  setSelectedStepIndex,
  setStreamUrl,
  setIsLoading,
  setIsCapturing,
  resetState,
} = compositeAcquisitionSlice.actions;

// Selectors
export const getCompositeState = (state) => state.compositeAcquisitionState;
export const getIsRunning = (state) => state.compositeAcquisitionState.isRunning;
export const getIsStreaming = (state) => state.compositeAcquisitionState.isStreaming;
export const getSteps = (state) => state.compositeAcquisitionState.steps;
export const getMapping = (state) => state.compositeAcquisitionState.mapping;
export const getIlluminationSources = (state) => state.compositeAcquisitionState.illuminationSources;
export const getAverageFps = (state) => state.compositeAcquisitionState.averageFps;
export const getLastCompositeImage = (state) => state.compositeAcquisitionState.lastCompositeImage;

// Export reducer
export default compositeAcquisitionSlice.reducer;
