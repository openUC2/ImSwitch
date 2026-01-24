// src/state/slices/dpcSlice.js
// Redux slice for DPC (Differential Phase Contrast) state management

import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // DPC parameters
  pixelsize: 0.2,
  wavelength: 0.53,
  na: 0.3,
  nai: 0.3,
  n: 1.0,
  led_intensity_r: 0,
  led_intensity_g: 255,
  led_intensity_b: 0,
  wait_time: 0.2,
  frame_sync: 2,
  save_images: false,
  save_directory: "",
  reg_u: 0.1,
  reg_p: 0.005,

  // DPC processing state
  is_processing: false,
  is_paused: false,
  frame_count: 0,
  processed_count: 0,
  last_process_time: 0.0,
  processing_fps: 0.0,

  // UI state
  isLoading: false,
  error: null,
};

const dpcSlice = createSlice({
  name: "dpc",
  initialState,
  reducers: {
    // Update DPC parameters
    updateDpcParams: (state, action) => {
      return { ...state, ...action.payload };
    },

    // Update DPC processing state
    updateDpcState: (state, action) => {
      const stateUpdate = action.payload;
      state.is_processing = stateUpdate.is_processing ?? state.is_processing;
      state.is_paused = stateUpdate.is_paused ?? state.is_paused;
      state.frame_count = stateUpdate.frame_count ?? state.frame_count;
      state.processed_count = stateUpdate.processed_count ?? state.processed_count;
      state.last_process_time = stateUpdate.last_process_time ?? state.last_process_time;
      state.processing_fps = stateUpdate.processing_fps ?? state.processing_fps;
    },

    // Update single parameter
    updateSingleParam: (state, action) => {
      const { key, value } = action.payload;
      state[key] = value;
    },

    // Set loading state
    setLoading: (state, action) => {
      state.isLoading = action.payload;
    },

    // Set error
    setError: (state, action) => {
      state.error = action.payload;
    },

    // Clear error
    clearError: (state) => {
      state.error = null;
    },

    // Reset to initial state
    resetDpcState: () => initialState,
  },
});

// Export actions
export const {
  updateDpcParams,
  updateDpcState,
  updateSingleParam,
  setLoading,
  setError,
  clearError,
  resetDpcState,
} = dpcSlice.actions;

// Selectors
export const getDpcState = (state) => state.dpc;
export const getDpcParams = (state) => ({
  pixelsize: state.dpc.pixelsize,
  wavelength: state.dpc.wavelength,
  na: state.dpc.na,
  nai: state.dpc.nai,
  n: state.dpc.n,
  led_intensity_r: state.dpc.led_intensity_r,
  led_intensity_g: state.dpc.led_intensity_g,
  led_intensity_b: state.dpc.led_intensity_b,
  wait_time: state.dpc.wait_time,
  frame_sync: state.dpc.frame_sync,
  save_images: state.dpc.save_images,
  save_directory: state.dpc.save_directory,
  reg_u: state.dpc.reg_u,
  reg_p: state.dpc.reg_p,
});
export const getDpcProcessingState = (state) => ({
  is_processing: state.dpc.is_processing,
  is_paused: state.dpc.is_paused,
  frame_count: state.dpc.frame_count,
  processed_count: state.dpc.processed_count,
  last_process_time: state.dpc.last_process_time,
  processing_fps: state.dpc.processing_fps,
});

export default dpcSlice.reducer;
