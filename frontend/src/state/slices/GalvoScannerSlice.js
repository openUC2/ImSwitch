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
  
  // Arbitrary points mode
  arbitraryPoints: {
    points: [],        // Array of { x, y, dwell_us, laser_intensity }
    running: false,
    paused: false,
    defaultDwellUs: 500,
    defaultIntensity: 128,
    laserTrigger: 'AUTO',
  },

  // Affine transform (camera -> galvo)
  affineTransform: {
    a11: 1.0, a12: 0.0, tx: 0.0,
    a21: 0.0, a22: 1.0, ty: 0.0,
  },

  // Calibration wizard state
  calibration: {
    active: false,
    currentStep: 0,  // 0, 1, 2
    galvoPoints: [
      { x: 1024, y: 1024, label: 'Top-Left' },
      { x: 3072, y: 1024, label: 'Top-Right' },
      { x: 2048, y: 3072, label: 'Bottom-Center' },
    ],
    camPoints: [null, null, null],  // Filled by user clicks
    completed: false,
  },

  // Active tab index (0 = Raster, 1 = Arbitrary Points)
  activeTab: 0,

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

    // ========================
    // Active Tab
    // ========================

    /**
     * Set active tab index (0=Raster, 1=Arbitrary Points)
     */
    setActiveTab: (state, action) => {
      state.activeTab = action.payload;
    },

    // ========================
    // Arbitrary Points Reducers
    // ========================

    /**
     * Add a point to the arbitrary points list
     */
    addArbitraryPoint: (state, action) => {
      const { x, y, dwell_us, laser_intensity } = action.payload;
      if (state.arbitraryPoints.points.length < 265) {
        state.arbitraryPoints.points.push({
          x,
          y,
          dwell_us: dwell_us ?? state.arbitraryPoints.defaultDwellUs,
          laser_intensity: laser_intensity ?? state.arbitraryPoints.defaultIntensity,
        });
      }
    },

    /**
     * Remove a point by index
     */
    removeArbitraryPoint: (state, action) => {
      const index = action.payload;
      if (index >= 0 && index < state.arbitraryPoints.points.length) {
        state.arbitraryPoints.points.splice(index, 1);
      }
    },

    /**
     * Update a single point's properties
     */
    updateArbitraryPoint: (state, action) => {
      const { index, ...updates } = action.payload;
      if (index >= 0 && index < state.arbitraryPoints.points.length) {
        Object.assign(state.arbitraryPoints.points[index], updates);
      }
    },

    /**
     * Clear all arbitrary points
     */
    clearArbitraryPoints: (state) => {
      state.arbitraryPoints.points = [];
    },

    /**
     * Set all arbitrary points at once
     */
    setArbitraryPoints: (state, action) => {
      state.arbitraryPoints.points = action.payload;
    },

    /**
     * Set default dwell time for new points
     */
    setDefaultDwellUs: (state, action) => {
      state.arbitraryPoints.defaultDwellUs = action.payload;
    },

    /**
     * Set default laser intensity for new points
     */
    setDefaultIntensity: (state, action) => {
      state.arbitraryPoints.defaultIntensity = action.payload;
    },

    /**
     * Apply default dwell time to all existing points
     */
    applyDefaultDwellToAll: (state) => {
      state.arbitraryPoints.points.forEach(pt => {
        pt.dwell_us = state.arbitraryPoints.defaultDwellUs;
      });
    },

    /**
     * Apply default intensity to all existing points
     */
    applyDefaultIntensityToAll: (state) => {
      state.arbitraryPoints.points.forEach(pt => {
        pt.laser_intensity = state.arbitraryPoints.defaultIntensity;
      });
    },

    /**
     * Set laser trigger mode
     */
    setLaserTrigger: (state, action) => {
      state.arbitraryPoints.laserTrigger = action.payload;
    },

    /**
     * Set arbitrary scan running state
     */
    setArbScanRunning: (state, action) => {
      state.arbitraryPoints.running = action.payload;
      if (!action.payload) state.arbitraryPoints.paused = false;
    },

    /**
     * Set arbitrary scan paused state
     */
    setArbScanPaused: (state, action) => {
      state.arbitraryPoints.paused = action.payload;
    },

    // ========================
    // Affine Transform Reducers
    // ========================

    /**
     * Set entire affine transform
     */
    setAffineTransform: (state, action) => {
      state.affineTransform = { ...state.affineTransform, ...action.payload };
    },

    /**
     * Set a single affine transform parameter
     */
    setAffineParam: (state, action) => {
      const { param, value } = action.payload;
      if (state.affineTransform.hasOwnProperty(param)) {
        state.affineTransform[param] = value;
      }
    },

    /**
     * Reset affine transform to identity
     */
    resetAffineTransform: (state) => {
      state.affineTransform = initialState.affineTransform;
    },

    // ========================
    // Calibration Reducers
    // ========================

    /**
     * Start calibration wizard
     */
    startCalibration: (state) => {
      state.calibration.active = true;
      state.calibration.currentStep = 0;
      state.calibration.camPoints = [null, null, null];
      state.calibration.completed = false;
    },

    /**
     * Cancel calibration wizard
     */
    cancelCalibration: (state) => {
      state.calibration.active = false;
      state.calibration.currentStep = 0;
      state.calibration.camPoints = [null, null, null];
      state.calibration.completed = false;
    },

    /**
     * Set camera point for current calibration step
     */
    setCalibrationCamPoint: (state, action) => {
      const { step, x, y } = action.payload;
      state.calibration.camPoints[step] = { x, y };
    },

    /**
     * Advance to the next calibration step
     */
    advanceCalibrationStep: (state) => {
      if (state.calibration.currentStep < 2) {
        state.calibration.currentStep += 1;
      } else {
        state.calibration.completed = true;
      }
    },

    /**
     * Set calibration galvo points (from backend)
     */
    setCalibrationGalvoPoints: (state, action) => {
      state.calibration.galvoPoints = action.payload;
    },

    /**
     * Update a single calibration galvo point by index
     * Payload: { index, x?, y?, label? }
     */
    updateCalibrationGalvoPoint: (state, action) => {
      const { index, ...updates } = action.payload;
      if (index >= 0 && index < state.calibration.galvoPoints.length) {
        Object.assign(state.calibration.galvoPoints[index], updates);
      }
    },

    /**
     * Mark calibration complete
     */
    setCalibrationComplete: (state) => {
      state.calibration.completed = true;
      state.calibration.active = false;
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
  // Tab
  setActiveTab,
  // Arbitrary points
  addArbitraryPoint,
  removeArbitraryPoint,
  updateArbitraryPoint,
  clearArbitraryPoints,
  setArbitraryPoints,
  setDefaultDwellUs,
  setDefaultIntensity,
  applyDefaultDwellToAll,
  applyDefaultIntensityToAll,
  setLaserTrigger,
  setArbScanRunning,
  setArbScanPaused,
  // Affine transform
  setAffineTransform,
  setAffineParam,
  resetAffineTransform,
  // Calibration
  startCalibration,
  cancelCalibration,
  setCalibrationCamPoint,
  advanceCalibrationStep,
  setCalibrationGalvoPoints,
  updateCalibrationGalvoPoint,
  setCalibrationComplete,
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

const defaultAffineTransform = {
  a11: 1.0, a12: 0.0, tx: 0.0,
  a21: 0.0, a22: 1.0, ty: 0.0,
};

const defaultArbitraryPoints = {
  points: [],
  running: false,
  paused: false,
  defaultDwellUs: 500,
  defaultIntensity: 128,
  laserTrigger: 'AUTO',
};

const defaultCalibration = {
  active: false,
  currentStep: 0,
  galvoPoints: [
    { x: 1024, y: 1024, label: 'Top-Left' },
    { x: 3072, y: 1024, label: 'Top-Right' },
    { x: 2048, y: 3072, label: 'Bottom-Center' },
  ],
  camPoints: [null, null, null],
  completed: false,
};

const defaultState = {
  scannerNames: [],
  selectedScanner: '',
  config: defaultConfig,
  status: defaultStatus,
  arbitraryPoints: defaultArbitraryPoints,
  affineTransform: defaultAffineTransform,
  calibration: defaultCalibration,
  activeTab: 0,
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
export const getActiveTab = (state) => state.galvoScannerState?.activeTab ?? 0;
export const getArbitraryPointsState = (state) => state.galvoScannerState?.arbitraryPoints || defaultArbitraryPoints;
export const getArbitraryPointsList = (state) => state.galvoScannerState?.arbitraryPoints?.points || [];
export const getAffineTransformState = (state) => state.galvoScannerState?.affineTransform || defaultAffineTransform;
export const getCalibrationState = (state) => state.galvoScannerState?.calibration || defaultCalibration;

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
