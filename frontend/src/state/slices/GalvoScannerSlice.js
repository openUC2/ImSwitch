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
  // Defaults = the empirically validated FLIM raster (25 µs dwell, eased
  // flyback + settle, overscan) — matches the backend GalvoScanConfig defaults.
  config: {
    nx: 256,
    ny: 256,
    x_min: 500,
    x_max: 3500,
    y_min: 500,
    y_max: 3500,
    sample_period_us: 25,
    frame_count: 0,
    bidirectional: false,
    pre_samples: 64,
    fly_samples: 256,
    trig_delay_us: 0,
    trig_width_us: 0,
    line_settle_samples: 128,
    enable_trigger: 1,
    apply_x_lut: 0,
    overscan_samples: 32,
    laser_blanking: 0,
    hw_pixel_clock: 0
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

  // Camera <-> scanner calibration summary from the backend
  // (getGalvoCameraCalibration): µm per DAC count derived from the affine
  // transform + camera pixel size. umPerDacX/Y are null while uncalibrated.
  cameraCalibration: {
    calibrated: false,
    detectorName: null,
    cameraDetectors: [],
    frameWidth: null,
    frameHeight: null,
    pixelSizeUmX: null,
    pixelSizeUmY: null,
    umPerDacX: null,
    umPerDacY: null,
    rotationDeg: null,
    hint: null,
    flim: {},
  },

  // Active tab index (0 = Raster, 1 = Arbitrary Points)
  activeTab: 0,

  // UI state
  loading: false,
  error: null,
  statusMessage: '',
  autoRefresh: false,
};

const DAC_MAX = 4095;
const clampDac = (v) => Math.max(0, Math.min(DAC_MAX, Math.round(Number(v) || 0)));

/**
 * Place a window of `width` counts centred on `center`, shifted (not
 * shrunk) so it stays inside 0..4095. Returns [min, max].
 */
const windowFromCenterWidth = (center, width) => {
  const w = Math.max(1, Math.min(DAC_MAX, Math.round(Number(width) || 0)));
  let lo = Math.round(Number(center) - w / 2);
  lo = Math.max(0, Math.min(DAC_MAX - w, lo));
  return [lo, lo + w];
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
     * Edit one axis' scan window by centre and/or width instead of min/max.
     * Payload: { axis: 'x' | 'y', center?: number, width?: number }.
     * Omitted values are taken from the current window, so changing the width
     * keeps the centre (and vice versa); the window slides to stay in range.
     */
    setAxisWindow: (state, action) => {
      const { axis, center, width } = action.payload;
      const loKey = axis === 'y' ? 'y_min' : 'x_min';
      const hiKey = axis === 'y' ? 'y_max' : 'x_max';
      const curLo = state.config[loKey];
      const curHi = state.config[hiKey];
      const c = center !== undefined ? center : (curLo + curHi) / 2;
      const w = width !== undefined ? width : curHi - curLo;
      const [lo, hi] = windowFromCenterWidth(c, w);
      state.config[loKey] = lo;
      state.config[hiKey] = hi;
    },

    /**
     * Set the whole scan rectangle at once (drag/resize in the preview).
     * Values are clamped to 0..4095 and ordered so min < max.
     */
    setScanArea: (state, action) => {
      const next = { ...state.config, ...action.payload };
      let xMin = clampDac(next.x_min), xMax = clampDac(next.x_max);
      let yMin = clampDac(next.y_min), yMax = clampDac(next.y_max);
      if (xMin > xMax) [xMin, xMax] = [xMax, xMin];
      if (yMin > yMax) [yMin, yMax] = [yMax, yMin];
      if (xMax === xMin) xMax = Math.min(DAC_MAX, xMin + 1);
      if (yMax === yMin) yMax = Math.min(DAC_MAX, yMin + 1);
      state.config.x_min = xMin; state.config.x_max = xMax;
      state.config.y_min = yMin; state.config.y_max = yMax;
    },

    /**
     * Shift the scan rectangle by (dx, dy) counts, keeping its size; it stops
     * at the DAC limits instead of shrinking.
     */
    moveScanArea: (state, action) => {
      const { dx = 0, dy = 0 } = action.payload;
      const shift = (lo, hi, d) => {
        const w = hi - lo;
        const nlo = Math.max(0, Math.min(DAC_MAX - w, Math.round(lo + d)));
        return [nlo, nlo + w];
      };
      [state.config.x_min, state.config.x_max] = shift(state.config.x_min, state.config.x_max, dx);
      [state.config.y_min, state.config.y_max] = shift(state.config.y_min, state.config.y_max, dy);
    },

    /**
     * Store the camera<->scanner calibration summary from the backend
     */
    setCameraCalibration: (state, action) => {
      state.cameraCalibration = { ...state.cameraCalibration, ...action.payload };
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
  setAxisWindow,
  setScanArea,
  moveScanArea,
  setCameraCalibration,
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

const defaultCameraCalibration = {
  calibrated: false,
  detectorName: null,
  cameraDetectors: [],
  frameWidth: null,
  frameHeight: null,
  pixelSizeUmX: null,
  pixelSizeUmY: null,
  umPerDacX: null,
  umPerDacY: null,
  rotationDeg: null,
  hint: null,
  flim: {},
};

const defaultState = {
  scannerNames: [],
  selectedScanner: '',
  config: defaultConfig,
  status: defaultStatus,
  arbitraryPoints: defaultArbitraryPoints,
  affineTransform: defaultAffineTransform,
  calibration: defaultCalibration,
  cameraCalibration: defaultCameraCalibration,
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
export const getCameraCalibration = (state) => state.galvoScannerState?.cameraCalibration || defaultCameraCalibration;

/**
 * Physical size of the current scan window, if the scanner is calibrated to
 * the camera. Computed client-side from µm/DAC so it follows the sliders live.
 * Returns null while uncalibrated.
 */
export const getScanPhysical = (state) => {
  const config = state.galvoScannerState?.config || defaultConfig;
  const cal = state.galvoScannerState?.cameraCalibration || defaultCameraCalibration;
  if (!cal.calibrated || !cal.umPerDacX || !cal.umPerDacY) return null;
  const spanX = Math.abs(config.x_max - config.x_min);
  const spanY = Math.abs(config.y_max - config.y_min);
  const fovUmX = spanX * cal.umPerDacX;
  const fovUmY = spanY * cal.umPerDacY;
  return {
    umPerDacX: cal.umPerDacX,
    umPerDacY: cal.umPerDacY,
    fovUmX,
    fovUmY,
    pixelUmX: fovUmX / Math.max(1, config.nx),
    pixelUmY: fovUmY / Math.max(1, config.ny),
    fullScaleUmX: 4096 * cal.umPerDacX,
    fullScaleUmY: 4096 * cal.umPerDacY,
  };
};

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
