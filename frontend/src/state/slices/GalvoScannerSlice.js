// GalvoScannerSlice.js - Redux slice for galvo scanner state management
import { createSlice } from '@reduxjs/toolkit';

/**
 * GalvoScannerSlice manages galvo scanner state for the frontend.
 * 
 * State structure:
 * {
 *   scannerNames: ['ESP32Galvo', ...],
 *   selectedScanner: 'ESP32Galvo',
 *   config: { nx, ny, x_min, x_max, y_min, y_max, sample_period_us, frame_count, bidirectional },
 *   status: { running, current_frame, current_line },
 *   loading: false,
 *   error: null,
 *   statusMessage: ''
 * }
 */

const initialState = {
  // Available scanner devices
  scannerNames: [],
  selectedScanner: '',
  
  // Scan configuration (persisted)
  config: {
    nx: 256,
    ny: 256,
    x_min: 500,
    x_max: 3500,
    y_min: 500,
    y_max: 3500,
    sample_period_us: 1,
    frame_count: 0,
    bidirectional: false,
    pre_samples: 0,
    fly_samples: 0,
    trig_delay_us: 0,
    trig_width_us: 0,
    line_settle_samples: 0,
    enable_trigger: 1,
    apply_x_lut: 0
  },
  
  // Current scanner status
  status: {
    running: false,
    current_frame: 0,
    current_line: 0
  },
  
  // UI state
  loading: false,
  error: null,
  statusMessage: '',
  autoRefresh: false,
};

const galvoScannerSlice = createSlice({
  name: 'galvoScanner',
  initialState,
  reducers: {
    /**
     * Set available scanner names
     */
    setScannerNames: (state, action) => {
      state.scannerNames = action.payload;
      // Auto-select first scanner if none selected
      if (state.scannerNames.length > 0 && !state.selectedScanner) {
        state.selectedScanner = state.scannerNames[0];
      }
    },
    
    /**
     * Set selected scanner
     */
    setSelectedScanner: (state, action) => {
      state.selectedScanner = action.payload;
    },
    
    /**
     * Update entire config object
     */
    setConfig: (state, action) => {
      state.config = { ...state.config, ...action.payload };
    },
    
    /**
     * Update a single config parameter
     */
    setConfigParam: (state, action) => {
      const { param, value } = action.payload;
      if (state.config.hasOwnProperty(param)) {
        state.config[param] = value;
      }
    },
    
    /**
     * Set X range (min and max)
     */
    setXRange: (state, action) => {
      const [x_min, x_max] = action.payload;
      state.config.x_min = x_min;
      state.config.x_max = x_max;
    },
    
    /**
     * Set Y range (min and max)
     */
    setYRange: (state, action) => {
      const [y_min, y_max] = action.payload;
      state.config.y_min = y_min;
      state.config.y_max = y_max;
    },
    
    /**
     * Set resolution (nx and ny)
     */
    setResolution: (state, action) => {
      const { nx, ny } = action.payload;
      if (nx !== undefined) state.config.nx = nx;
      if (ny !== undefined) state.config.ny = ny;
    },
    
    /**
     * Toggle bidirectional scanning
     */
    toggleBidirectional: (state) => {
      state.config.bidirectional = !state.config.bidirectional;
    },
    
    /**
     * Update scanner status
     */
    setStatus: (state, action) => {
      state.status = { ...state.status, ...action.payload };
    },
    
    /**
     * Set running state
     */
    setRunning: (state, action) => {
      state.status.running = action.payload;
    },
    
    /**
     * Set loading state
     */
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    
    /**
     * Set error message
     */
    setError: (state, action) => {
      state.error = action.payload;
    },
    
    /**
     * Clear error
     */
    clearError: (state) => {
      state.error = null;
    },
    
    /**
     * Set status message
     */
    setStatusMessage: (state, action) => {
      state.statusMessage = action.payload;
    },
    
    /**
     * Clear status message
     */
    clearStatusMessage: (state) => {
      state.statusMessage = '';
    },
    
    /**
     * Toggle auto-refresh
     */
    toggleAutoRefresh: (state) => {
      state.autoRefresh = !state.autoRefresh;
    },
    
    /**
     * Set auto-refresh
     */
    setAutoRefresh: (state, action) => {
      state.autoRefresh = action.payload;
    },
    
    /**
     * Apply a preset configuration
     */
    applyPreset: (state, action) => {
      const preset = action.payload;
      switch (preset) {
        case '64x64':
          state.config.nx = 64;
          state.config.ny = 64;
          break;
        case '256x256':
          state.config.nx = 256;
          state.config.ny = 256;
          break;
        case '512x512':
          state.config.nx = 512;
          state.config.ny = 512;
          break;
        case 'fullRange':
          state.config.x_min = 0;
          state.config.x_max = 4095;
          state.config.y_min = 0;
          state.config.y_max = 4095;
          break;
        case 'center50':
          state.config.x_min = 1024;
          state.config.x_max = 3072;
          state.config.y_min = 1024;
          state.config.y_max = 3072;
          break;
        default:
          break;
      }
    },
    
    /**
     * Reset config to defaults
     */
    resetConfig: (state) => {
      state.config = initialState.config;
    },
  },
});

// Export actions
export const {
  setScannerNames,
  setSelectedScanner,
  setConfig,
  setConfigParam,
  setXRange,
  setYRange,
  setResolution,
  toggleBidirectional,
  setStatus,
  setRunning,
  setLoading,
  setError,
  clearError,
  setStatusMessage,
  clearStatusMessage,
  toggleAutoRefresh,
  setAutoRefresh,
  applyPreset,
  resetConfig,
} = galvoScannerSlice.actions;

// Default values for when state is not yet initialized
const defaultConfig = {
  nx: 256,
  ny: 256,
  x_min: 500,
  x_max: 3500,
  y_min: 500,
  y_max: 3500,
  sample_period_us: 1,
  frame_count: 0,
  bidirectional: false
};

const defaultStatus = {
  running: false,
  current_frame: 0,
  current_line: 0
};

const defaultState = {
  scannerNames: [],
  selectedScanner: '',
  config: defaultConfig,
  status: defaultStatus,
  loading: false,
  error: null,
  statusMessage: '',
  autoRefresh: false,
};

// Selectors with defensive null checks
export const getGalvoScannerState = (state) => state.galvoScannerState || defaultState;
export const getGalvoConfig = (state) => state.galvoScannerState?.config || defaultConfig;
export const getGalvoStatus = (state) => state.galvoScannerState?.status || defaultStatus;
export const getGalvoScannerNames = (state) => state.galvoScannerState?.scannerNames || [];
export const getSelectedScanner = (state) => state.galvoScannerState?.selectedScanner || '';
export const getGalvoLoading = (state) => state.galvoScannerState?.loading || false;
export const getGalvoError = (state) => state.galvoScannerState?.error || null;
export const getGalvoAutoRefresh = (state) => state.galvoScannerState?.autoRefresh || false;

// Computed selectors
export const getScanInfo = (state) => {
  const config = state.galvoScannerState?.config || defaultConfig;
  const totalPixels = config.nx * config.ny;
  const frameTimeMs = (totalPixels * config.sample_period_us) / 1000;
  const frameRate = config.sample_period_us > 0 ? 1000 / frameTimeMs : Infinity;
  return {
    totalPixels,
    frameTimeMs: frameTimeMs.toFixed(2),
    frameRate: isFinite(frameRate) ? frameRate.toFixed(1) : '∞'
  };
};

// Export reducer
export default galvoScannerSlice.reducer;
