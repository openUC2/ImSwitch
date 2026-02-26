import { createSlice } from "@reduxjs/toolkit";

/**
 * UI State for the Experiment Designer
 * Manages dimension bar state: enabled, expanded, and configuration status
 * for progressive disclosure of acquisition parameters
 */

// Dimension identifiers
export const DIMENSIONS = Object.freeze({
  POSITIONS: "positions",
  CHANNELS: "channels",
  Z_FOCUS: "zFocus",
  FOCUS_MAP: "focusMap",
  TIME: "time",
  TILING: "tiling",
  OUTPUT: "output",
});

// Z/Focus mode options
export const Z_FOCUS_MODES = Object.freeze({
  SINGLE_Z: "singleZ",
  AUTOFOCUS: "autofocus",
  Z_STACK: "zStack",
  Z_STACK_AUTOFOCUS: "zStackAutofocus",
});

// Initial state for the experiment UI
const initialExperimentUIState = {
  // Current expanded dimension (only one can be expanded at a time)
  expandedDimension: DIMENSIONS.POSITIONS,
  
  // Dimension enable states
  dimensions: {
    [DIMENSIONS.POSITIONS]: {
      enabled: true, // Always enabled by default
      configured: false,
      summary: "No positions defined",
    },
    [DIMENSIONS.CHANNELS]: {
      enabled: false,
      configured: false,
      summary: "No channels selected",
    },
    [DIMENSIONS.Z_FOCUS]: {
      enabled: false,
      configured: false,
      summary: "Single Z",
      // Sub-state for Z/Focus mode selection
      mode: Z_FOCUS_MODES.SINGLE_Z,
    },
    [DIMENSIONS.FOCUS_MAP]: {
      enabled: false,
      configured: false,
      summary: "Disabled",
    },
    [DIMENSIONS.TIME]: {
      enabled: false,
      configured: false,
      summary: "Single timepoint",
    },
    [DIMENSIONS.TILING]: {
      enabled: false,
      configured: false,
      summary: "No tiling",
    },
    [DIMENSIONS.OUTPUT]: {
      enabled: true, // Always enabled
      configured: false,
      summary: "Default output",
    },
  },

  // Experiment-level metadata estimates (updated dynamically)
  estimates: {
    totalPositions: 0,
    totalChannels: 0,
    totalZPlanes: 1,
    totalTimepoints: 1,
    estimatedDurationMinutes: 0,
    estimatedDataSizeMB: 0,
  },

  // Advanced settings visibility per dimension
  advancedVisible: {
    [DIMENSIONS.POSITIONS]: false,
    [DIMENSIONS.CHANNELS]: false,
    [DIMENSIONS.Z_FOCUS]: false,
    [DIMENSIONS.FOCUS_MAP]: false,
    [DIMENSIONS.TIME]: false,
    [DIMENSIONS.TILING]: false,
    [DIMENSIONS.OUTPUT]: false,
  },
};

// Create the slice
const experimentUISlice = createSlice({
  name: "experimentUI",
  initialState: initialExperimentUIState,
  reducers: {
    // Set the expanded dimension (accordion behavior)
    setExpandedDimension: (state, action) => {
      state.expandedDimension = action.payload;
    },

    // Toggle a dimension's expanded state
    toggleExpandedDimension: (state, action) => {
      const dimension = action.payload;
      if (state.expandedDimension === dimension) {
        state.expandedDimension = null; // Collapse if already expanded
      } else {
        state.expandedDimension = dimension;
      }
    },

    // Enable or disable a dimension
    setDimensionEnabled: (state, action) => {
      const { dimension, enabled } = action.payload;
      if (state.dimensions[dimension]) {
        state.dimensions[dimension].enabled = enabled;
        // Auto-expand when enabling
        if (enabled) {
          state.expandedDimension = dimension;
        }
      }
    },

    // Toggle dimension enabled state
    toggleDimensionEnabled: (state, action) => {
      const dimension = action.payload;
      if (state.dimensions[dimension]) {
        const newEnabled = !state.dimensions[dimension].enabled;
        state.dimensions[dimension].enabled = newEnabled;
        // Auto-expand when enabling
        if (newEnabled) {
          state.expandedDimension = dimension;
        }
      }
    },

    // Update dimension summary text
    setDimensionSummary: (state, action) => {
      const { dimension, summary } = action.payload;
      if (state.dimensions[dimension]) {
        state.dimensions[dimension].summary = summary;
      }
    },

    // Mark dimension as configured/unconfigured
    setDimensionConfigured: (state, action) => {
      const { dimension, configured } = action.payload;
      if (state.dimensions[dimension]) {
        state.dimensions[dimension].configured = configured;
      }
    },

    // Set Z/Focus mode (mutually exclusive)
    setZFocusMode: (state, action) => {
      state.dimensions[DIMENSIONS.Z_FOCUS].mode = action.payload;
      // Update summary based on mode
      const modeSummaries = {
        [Z_FOCUS_MODES.SINGLE_Z]: "Single Z",
        [Z_FOCUS_MODES.AUTOFOCUS]: "Autofocus enabled",
        [Z_FOCUS_MODES.Z_STACK]: "Z-Stack",
        [Z_FOCUS_MODES.Z_STACK_AUTOFOCUS]: "Z-Stack + Autofocus",
      };
      state.dimensions[DIMENSIONS.Z_FOCUS].summary = modeSummaries[action.payload] || "Single Z";
    },

    // Update estimates
    setEstimates: (state, action) => {
      state.estimates = { ...state.estimates, ...action.payload };
    },

    // Toggle advanced settings visibility for a dimension
    toggleAdvancedVisible: (state, action) => {
      const dimension = action.payload;
      if (state.advancedVisible.hasOwnProperty(dimension)) {
        state.advancedVisible[dimension] = !state.advancedVisible[dimension];
      }
    },

    // Set advanced settings visibility
    setAdvancedVisible: (state, action) => {
      const { dimension, visible } = action.payload;
      if (state.advancedVisible.hasOwnProperty(dimension)) {
        state.advancedVisible[dimension] = visible;
      }
    },

    // Reset UI state
    resetUIState: () => {
      return initialExperimentUIState;
    },
  },
});

// Export actions
export const {
  setExpandedDimension,
  toggleExpandedDimension,
  setDimensionEnabled,
  toggleDimensionEnabled,
  setDimensionSummary,
  setDimensionConfigured,
  setZFocusMode,
  setEstimates,
  toggleAdvancedVisible,
  setAdvancedVisible,
  resetUIState,
} = experimentUISlice.actions;

// Selectors
export const getExperimentUIState = (state) => state.experimentUI;
export const getExpandedDimension = (state) => state.experimentUI.expandedDimension;
export const getDimension = (dimension) => (state) => state.experimentUI.dimensions[dimension];
export const getEstimates = (state) => state.experimentUI.estimates;
export const getAdvancedVisible = (dimension) => (state) => state.experimentUI.advancedVisible[dimension];

// Selector for checking if a dimension is enabled
export const isDimensionEnabled = (dimension) => (state) => 
  state.experimentUI.dimensions[dimension]?.enabled ?? false;

// Export reducer
export default experimentUISlice.reducer;
