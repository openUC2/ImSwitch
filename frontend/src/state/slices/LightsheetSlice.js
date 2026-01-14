import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // UI state
  tabIndex: 0,

  // Position and movement parameters
  minPos: -500,
  maxPos: 500,
  speed: 1000,
  axis: "A",
  stepSize: 100, // Step size for step-acquire mode (µm)

  // Illumination parameters
  illuSource: -1,
  illuValue: 512,

  // Galvo scanner parameters
  galvoChannel: 0,
  galvoFrequency: 1000,
  galvoOffset: 0,
  galvoAmplitude: 1000,
  galvoClkDiv: 1,
  galvoPhase: 0,
  galvoInvert: false,

  // Imaging state
  vtkImagePrimary: null,
  isRunning: false,

  // Scan mode and storage settings (NEW)
  scanMode: "continuous", // "continuous" or "step_acquire"
  storageFormat: "ome_zarr", // "tiff", "ome_zarr", or "both"
  experimentName: "lightsheet_scan",

  // Tiling parameters (NEW)
  enableTiling: false,
  tilesXPositive: 1,
  tilesXNegative: 1,
  tilesYPositive: 1,
  tilesYNegative: 1,
  tileStepSizeX: 1000, // µm
  tileStepSizeY: 1000, // µm
  tileOverlap: 0.1, // 10% overlap
  
  // Timelapse parameters (NEW)
  timepoints: 1,
  timeLapsePeriod: 60, // seconds
  
  // Objective FOV (cached from backend)
  objectiveFOV: {
    fovX: 1000,
    fovY: 1000,
    pixelSize: 1.0,
    suggestedOverlap: 0.1
  },

  // Scan status (updated via socket) (NEW)
  scanStatus: {
    isRunning: false,
    scanMode: null,
    currentPosition: 0,
    totalPositions: 0,
    currentFrame: 0,
    progress: 0,
    zarrPath: null,
    tiffPath: null,
    errorMessage: null,
  },

  // Available options from backend (NEW)
  availableScanModes: ["continuous", "step_acquire"],
  availableStorageFormats: ["tiff", "ome_zarr", "both"],

  // Latest zarr path for visualization (NEW)
  latestZarrPath: null,
  latestZarrAbsolutePath: null,

  // Current stage positions for 3D visualization
  stagePositions: {
    x: 0,
    y: 0,
    z: 0,
    a: 0
  },

  // Axis configuration for 3D mapping
  axisConfig: {
    x: { offset: 0, scale: 1, invert: false },
    y: { offset: 0, scale: 1, invert: false },
    z: { offset: 0, scale: 1, invert: false },
    a: { offset: 0, scale: 1, invert: false }
  },

  // 3D viewer camera state (persisted)
  cameraState: {
    position: null, // Will be set on first load
    target: null,   // Camera look-at target
    zoom: null      // Camera zoom/distance
  },
};

const lightsheetSlice = createSlice({
  name: "lightsheet",
  initialState,
  reducers: {
    // UI actions
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },

    // Position and movement actions
    setMinPos: (state, action) => {
      state.minPos = action.payload;
    },
    setMaxPos: (state, action) => {
      state.maxPos = action.payload;
    },
    setSpeed: (state, action) => {
      state.speed = action.payload;
    },
    setAxis: (state, action) => {
      state.axis = action.payload;
    },
    setStepSize: (state, action) => {
      state.stepSize = action.payload;
    },

    // Illumination actions
    setIlluSource: (state, action) => {
      state.illuSource = action.payload;
    },
    setIlluValue: (state, action) => {
      state.illuValue = action.payload;
    },

    // Galvo scanner actions
    setGalvoChannel: (state, action) => {
      state.galvoChannel = action.payload;
    },
    setGalvoFrequency: (state, action) => {
      state.galvoFrequency = action.payload;
    },
    setGalvoOffset: (state, action) => {
      state.galvoOffset = action.payload;
    },
    setGalvoAmplitude: (state, action) => {
      state.galvoAmplitude = action.payload;
    },
    setGalvoClkDiv: (state, action) => {
      state.galvoClkDiv = action.payload;
    },
    setGalvoPhase: (state, action) => {
      state.galvoPhase = action.payload;
    },
    setGalvoInvert: (state, action) => {
      state.galvoInvert = action.payload;
    },

    // Imaging actions
    setVtkImagePrimary: (state, action) => {
      state.vtkImagePrimary = action.payload;
    },
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },

    // NEW: Scan mode and storage actions
    setScanMode: (state, action) => {
      state.scanMode = action.payload;
    },
    setStorageFormat: (state, action) => {
      state.storageFormat = action.payload;
    },
    setExperimentName: (state, action) => {
      state.experimentName = action.payload;
    },

    // NEW: Tiling actions
    setEnableTiling: (state, action) => {
      state.enableTiling = action.payload;
    },
    setTilesXPositive: (state, action) => {
      state.tilesXPositive = action.payload;
    },
    setTilesXNegative: (state, action) => {
      state.tilesXNegative = action.payload;
    },
    setTilesYPositive: (state, action) => {
      state.tilesYPositive = action.payload;
    },
    setTilesYNegative: (state, action) => {
      state.tilesYNegative = action.payload;
    },
    setTileStepSizeX: (state, action) => {
      state.tileStepSizeX = action.payload;
    },
    setTileStepSizeY: (state, action) => {
      state.tileStepSizeY = action.payload;
    },
    setTileOverlap: (state, action) => {
      state.tileOverlap = action.payload;
    },

    // NEW: Timelapse actions
    setTimepoints: (state, action) => {
      state.timepoints = action.payload;
    },
    setTimeLapsePeriod: (state, action) => {
      state.timeLapsePeriod = action.payload;
    },

    // NEW: Objective FOV actions
    setObjectiveFOV: (state, action) => {
      state.objectiveFOV = { ...state.objectiveFOV, ...action.payload };
    },

    // NEW: Scan status actions (updated via socket)
    setScanStatus: (state, action) => {
      state.scanStatus = { ...state.scanStatus, ...action.payload };
      // Also update isRunning for compatibility
      if (typeof action.payload.isRunning !== 'undefined') {
        state.isRunning = action.payload.isRunning;
      }
    },
    updateScanProgress: (state, action) => {
      state.scanStatus.progress = action.payload.progress;
      state.scanStatus.currentFrame = action.payload.currentFrame;
      state.scanStatus.currentPosition = action.payload.currentPosition;
    },

    // NEW: Available options actions
    setAvailableScanModes: (state, action) => {
      state.availableScanModes = action.payload;
    },
    setAvailableStorageFormats: (state, action) => {
      state.availableStorageFormats = action.payload;
    },

    // NEW: Latest zarr path actions
    setLatestZarrPath: (state, action) => {
      state.latestZarrPath = action.payload.zarrPath;
      state.latestZarrAbsolutePath = action.payload.absolutePath;
    },

    // Batch update actions
    setPositionParameters: (state, action) => {
      const { minPos, maxPos, speed, axis } = action.payload;
      state.minPos = minPos;
      state.maxPos = maxPos;
      state.speed = speed;
      state.axis = axis;
    },
    setIlluminationParameters: (state, action) => {
      const { source, value } = action.payload;
      state.illuSource = source;
      state.illuValue = value;
    },
    setGalvoParameters: (state, action) => {
      const { channel, frequency, offset, amplitude, clkDiv, phase, invert } = action.payload;
      state.galvoChannel = channel;
      state.galvoFrequency = frequency;
      state.galvoOffset = offset;
      state.galvoAmplitude = amplitude;
      state.galvoClkDiv = clkDiv;
      state.galvoPhase = phase;
      state.galvoInvert = invert;
    },

    // Stage position actions for 3D visualization
    setStagePosition: (state, action) => {
      const { axis, value } = action.payload;
      state.stagePositions[axis] = value;
    },
    setAllStagePositions: (state, action) => {
      state.stagePositions = { ...state.stagePositions, ...action.payload };
    },

    // Axis configuration actions
    setAxisConfig: (state, action) => {
      const { axis, config } = action.payload;
      state.axisConfig[axis] = { ...state.axisConfig[axis], ...config };
    },
    setAxisOffset: (state, action) => {
      const { axis, offset } = action.payload;
      state.axisConfig[axis].offset = offset;
    },
    setAxisScale: (state, action) => {
      const { axis, scale } = action.payload;
      state.axisConfig[axis].scale = scale;
    },
    setAxisInvert: (state, action) => {
      const { axis, invert } = action.payload;
      state.axisConfig[axis].invert = invert;
    },

    // 3D viewer camera state actions
    setCameraState: (state, action) => {
      state.cameraState = { ...state.cameraState, ...action.payload };
    },

    // Reset actions
    resetToDefaults: (state) => {
      return { ...initialState };
    },
    resetScan: (state) => {
      state.isRunning = false;
      state.vtkImagePrimary = null;
      state.scanStatus = initialState.scanStatus;
    },
  },
});

// Export actions
export const {
  setTabIndex,
  setMinPos,
  setMaxPos,
  setSpeed,
  setAxis,
  setStepSize,
  setIlluSource,
  setIlluValue,
  setGalvoChannel,
  setGalvoFrequency,
  setGalvoOffset,
  setGalvoAmplitude,
  setGalvoClkDiv,
  setGalvoPhase,
  setGalvoInvert,
  setVtkImagePrimary,
  setIsRunning,
  setScanMode,
  setStorageFormat,
  setExperimentName,
  setEnableTiling,
  setTilesXPositive,
  setTilesXNegative,
  setTilesYPositive,
  setTilesYNegative,
  setTileStepSizeX,
  setTileStepSizeY,
  setTileOverlap,
  setTimepoints,
  setTimeLapsePeriod,
  setObjectiveFOV,
  setScanStatus,
  updateScanProgress,
  setAvailableScanModes,
  setAvailableStorageFormats,
  setLatestZarrPath,
  setPositionParameters,
  setIlluminationParameters,
  setGalvoParameters,
  setStagePosition,
  setAllStagePositions,
  setAxisConfig,
  setAxisOffset,
  setAxisScale,
  setAxisInvert,
  setCameraState,
  resetToDefaults,
  resetScan,
} = lightsheetSlice.actions;

// Export selector
export const getLightsheetState = (state) => state.lightsheet;

// Export reducer
export default lightsheetSlice.reducer;