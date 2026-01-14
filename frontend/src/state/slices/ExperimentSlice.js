import { createSlice } from "@reduxjs/toolkit";
import { v4 as uuidv4 } from "uuid";

// Define the initial state
const initialExperimentState = {
  name: "experiment",
  wellLayout: {
    name: "Default",
    unit: "um",
    width: 1000000,
    height: 600000,
    wells: [
      /* Example well
      { x: 200000, y: 200000, shape: "circle", radius: 50000 },
      { x: 400000, y: 200000, shape: "circle", radius: 90000 },
      { x: 600000, y: 200000, shape: "circle", radius: 90000 },
      { x: 800000, y: 200000, shape: "circle", radius: 90000 },
      { x: 200000, y: 400000, shape: "circle", radius: 90000 },
      { x: 400000, y: 400000, shape: "circle", radius: 90000, name: "A1" },
      { x: 600000, y: 400000, shape: "rectangle", width: 90000, height: 180000, },
      { x: 800000, y: 400000, shape: "rectangle", width: 180000, height: 180000, },
      */
    ],
  },
  pointList: [
    /* Example item:
    {
      id: uuidv4(),
      name: "",
      x: 100000,
      y: 100000,
      shape: "",
      rectPlusX: 0,
      rectMinusX: 0,
      rectPlusY: 0,
      rectMinusY: 0,
      circleRadiusX: 0,
      circleRadiusY: 0
    },
    */
  ],
  parameterValue: {
    illumination: [],
    darkfield: false,
    illuIntensities: 0,
    differentialPhaseContrast: false,
    timeLapsePeriod: 0.1,
    numberOfImages: 1,
    autoFocus: false,
    autoFocusMin: 0.0,
    autoFocusMax: 0.0,
    autoFocusStepSize: 0.1,
    autoFocusIlluminationChannel: "", // Selected illumination channel for autofocus
    autoFocusSettleTime: 0.1, // Settling time between steps (seconds)
    autoFocusRange: 100, // Z-range to scan (±range from current position)
    autoFocusResolution: 10, // Step size in Z
    autoFocusCropsize: 2048, // Crop size for focus calculation
    autoFocusStaticOffset: 0.0, // Static offset to add to final focus position
    autoFocusTwoStage: false, // Enable two-stage autofocus (coarse + fine scan)
    autoFocusAlgorithm: "LAPE", // Focus measurement method (LAPE, GLVA, JPEG)
    // Hardware autofocus (FocusLock-based one-shot) parameters
    autoFocusMode: "software", // "software" (Z-sweep) or "hardware" (one-shot using FocusLock)
    autofocus_max_attempts: 3, // Max attempts for hardware autofocus
    autofocus_target_focus_setpoint: 0, // Target focus setpoint for hardware autofocus
    zStack: false,
    zStackMin: 0.0,
    zStackMax: 0.0,
    zStackStepSize: 0.1,
    speed: 0, 
    gains: 0,
    exposureTimes: 0,
    performanceMode: false,
    // Performance mode advanced settings
    performanceTriggerMode: "hardware", // "hardware" (external trigger) or "software" (callback-based)
    performanceTPreMs: 90,   // Pre-exposure settle time in milliseconds
    performanceTPostMs: 50,  // Post-exposure time (exposure) in milliseconds
    ome_write_tiff: false,
    ome_write_zarr: false,
    ome_write_stitched_tiff: true,
    ome_write_individual_tiffs: false,
    // Tile overlap parameters (moved from WellSelectorSlice)
    overlapWidth: 0.0,  // 0.0 = no overlap (100% spacing), 0.1 = 10% overlap (90% spacing)
    overlapHeight: 0.0,  // 0.0 = no overlap (100% spacing), 0.1 = 10% overlap (90% spacing)
    // Snakescan pattern for tiling
    is_snakescan: false,  // Enable snakescan pattern (alternating row directions)
  },
};

// Create slice
const experimentSlice = createSlice({
  name: "experimentState",
  initialState: initialExperimentState,
  reducers: {
    //------------------------ well layout
    setWellLayout: (state, action) => {
      state.wellLayout = action.payload;
    },
    //------------------------ parameter
    setIllumination: (state, action) => {
      console.log("setIllumination", action.payload);
      state.parameterValue.illumination = action.payload;
    },
    setDarkfield: (state, action) => {
      console.log("setDarkfield");
      state.parameterValue.darkfield = action.payload;
    },
    setIlluminationIntensities: (state, action) => {
      console.log("setIlluminationIntensities");
      state.parameterValue.illuIntensities = action.payload;
    },
    setDifferentialPhaseContrast: (state, action) => {
      console.log("setDifferentialPhaseContrast");
      state.parameterValue.differentialPhaseContrast = action.payload;
    },
    setTimeLapsePeriod: (state, action) => {
      console.log("setTimeLapsePeriod");
      state.parameterValue.timeLapsePeriod = action.payload;
    },
    setNumberOfImages: (state, action) => {
      console.log("setNumberOfImages");
      state.parameterValue.numberOfImages = action.payload;
    },
    setAutoFocus: (state, action) => {
      console.log("setAutoFocus");
      state.parameterValue.autoFocus = action.payload;
    },
    setAutoFocusMin: (state, action) => {
      console.log("setAutoFocusMin");
      state.parameterValue.autoFocusMin = action.payload;
    },
    setAutoFocusMax: (state, action) => {
      console.log("setAutoFocusMax");
      state.parameterValue.autoFocusMax = action.payload;
    },
    setAutoFocusStepSize: (state, action) => {
      console.log("setAutoFocusStepSize");
      state.parameterValue.autoFocusStepSize = action.payload;
    },
    setAutoFocusIlluminationChannel: (state, action) => {
      console.log("setAutoFocusIlluminationChannel");
      state.parameterValue.autoFocusIlluminationChannel = action.payload;
    },
    setAutoFocusSettleTime: (state, action) => {
      console.log("setAutoFocusSettleTime");
      state.parameterValue.autoFocusSettleTime = action.payload;
    },
    setAutoFocusRange: (state, action) => {
      console.log("setAutoFocusRange");
      state.parameterValue.autoFocusRange = action.payload;
    },
    setAutoFocusResolution: (state, action) => {
      console.log("setAutoFocusResolution");
      state.parameterValue.autoFocusResolution = action.payload;
    },
    setAutoFocusCropsize: (state, action) => {
      console.log("setAutoFocusCropsize");
      state.parameterValue.autoFocusCropsize = action.payload;
    },
    setAutoFocusStaticOffset: (state, action) => {
      console.log("setAutoFocusStaticOffset");
      state.parameterValue.autoFocusStaticOffset = action.payload;
    },
    setAutoFocusTwoStage: (state, action) => {
      console.log("setAutoFocusTwoStage");
      state.parameterValue.autoFocusTwoStage = action.payload;
    },
    setAutoFocusAlgorithm: (state, action) => {
      console.log("setAutoFocusAlgorithm");
      state.parameterValue.autoFocusAlgorithm = action.payload;
    },
    setAutoFocusMode: (state, action) => {
      console.log("setAutoFocusMode", action.payload);
      state.parameterValue.autoFocusMode = action.payload;
    },
    setAutoFocusMaxAttempts: (state, action) => {
      console.log("setAutoFocusMaxAttempts", action.payload);
      state.parameterValue.autofocus_max_attempts = action.payload;
    },
    setAutoFocusTargetSetpoint: (state, action) => {
      console.log("setAutoFocusTargetSetpoint", action.payload);
      state.parameterValue.autofocus_target_focus_setpoint = action.payload;
    },
    setZStack: (state, action) => {
      console.log("setZStack");
      state.parameterValue.zStack = action.payload;
    },
    setZStackMin: (state, action) => {
      console.log("setZStackMin");
      state.parameterValue.zStackMin = action.payload;
    },
    setZStackMax: (state, action) => {
      console.log("setZStackMax");
      state.parameterValue.zStackMax = action.payload;
    },
    setZStackStepSize: (state, action) => {
      console.log("setZStackStepSize");
      state.parameterValue.zStackStepSize = action.payload;
    },
    setSpeed: (state, action) => {
        console.log("setSpeed");
        state.parameterValue.speed = action.payload;
    },
    setGains: (state, action) => {
      console.log("setGains");
      state.parameterValue.gains = action.payload;
    },
    setExposureTimes: (state, action) => {
      console.log("setExposureTime");
      state.parameterValue.exposureTimes = action.payload;
    },
    //------------------------ generic
    updateParameter: (state, action) => {
      console.log("setParameter", action.payload);
      const { key, value } = action.payload; //Call: updateParameter({parameterName: value})
      if (state.parameterValue.hasOwnProperty(key)) {
        state.parameterValue[key] = value;
      }
    },
    setPerformanceMode: (state, action) => {
      console.log("setPerformanceMode", action.payload);
      state.parameterValue.performanceMode = action.payload;
    },
    setPerformanceTriggerMode: (state, action) => {
      console.log("setPerformanceTriggerMode", action.payload);
      state.parameterValue.performanceTriggerMode = action.payload; // "hardware" or "software"
    },
    setPerformanceTPreMs: (state, action) => {
      console.log("setPerformanceTPreMs", action.payload);
      state.parameterValue.performanceTPreMs = Math.max(0, action.payload);
    },
    setPerformanceTPostMs: (state, action) => {
      console.log("setPerformanceTPostMs", action.payload);
      state.parameterValue.performanceTPostMs = Math.max(0, action.payload);
    },
    setOmeWriteTiff: (state, action) => {
      console.log("setOmeWriteTiff", action.payload);
      state.parameterValue.ome_write_tiff = action.payload;
    },
    setOmeWriteZarr: (state, action) => {
      console.log("setOmeWriteZarr", action.payload);
      state.parameterValue.ome_write_zarr = action.payload;
    },
    setOmeWriteStitchedTiff: (state, action) => {
      console.log("setOmeWriteStitchedTiff", action.payload);
      state.parameterValue.ome_write_stitched_tiff = action.payload;
    },
    setOmeWriteIndividualTiffs: (state, action) => {
      console.log("setOmeWriteIndividualTiffs", action.payload);
      state.parameterValue.ome_write_individual_tiffs = action.payload;
    },
    //------------------------ overlap parameters
    setOverlapWidth: (state, action) => {
      console.log("setOverlapWidth", action.payload);
      state.parameterValue.overlapWidth = Math.max(-15, Math.min(0.5, action.payload)); // Clamp between -0.5 and 0.5 (-50% to 50%)
    },
    setOverlapHeight: (state, action) => {
      console.log("setOverlapHeight", action.payload);
      state.parameterValue.overlapHeight = Math.max(-15, Math.min(0.5, action.payload)); // Clamp between -0.5 and 0.5 (-50% to 50%)
    },
    setIsSnakescan: (state, action) => {
      console.log("setIsSnakescan", action.payload);
      state.parameterValue.is_snakescan = action.payload;
    },
    //------------------------ points
    createPoint: (state, action) => {
      console.log("createPoint", action);
      const newPoint = {
        id: uuidv4(),
        x: action.payload.x,
        y: action.payload.y,
        name: (action.payload.name != null) ? (action.payload.name) : (""),
        shape: (action.payload.shape != null) ? (action.payload.shape) : (""),
        rectPlusX: (action.payload.rectPlusX != null) ? (action.payload.rectPlusX) : (0),
        rectPlusY: (action.payload.rectPlusY != null) ? (action.payload.rectPlusY) : (0),
        rectMinusX: (action.payload.rectMinusX != null) ? (action.payload.rectMinusX) : (0),
        rectMinusY: (action.payload.rectMinusY != null) ? (action.payload.rectMinusY) : (0),
        circleRadiusX: (action.payload.circleRadiusX != null) ? (action.payload.circleRadiusX) : (0),
        circleRadiusY: (action.payload.circleRadiusY != null) ? (action.payload.circleRadiusY) : (0),
      };
      
      console.log("createPoint newPoint", newPoint);
      state.pointList.push(newPoint);
    },
    addPoint: (state, action) => {
      console.log("addPoint");
      state.pointList.push(action.payload);
    },
    removePoint: (state, action) => {
      console.log("removePoint");
      //return state.filter(point => point.id !== action.payload);
      state.pointList.splice(action.payload, 1);
    },
    setPointList: (state, action) => {
      console.log("setPointList");
      state.pointList = action.payload;
    },
    replacePoint: (state, action) => {
      console.log("replacePoint", action.payload);
      const { index, newPoint } = action.payload;
      if (index >= 0 && index < state.pointList.length) {
        state.pointList[index] = newPoint;
      }
    },

    //------------------------ state
    resetState: (state) => {
      console.log("resetState");
      return initialExperimentState; // Reset to initial state
    },
  },
});

// Export actions from slice
export const {
  setWellLayout,

  setIllumination,
  setDarkfield,
  setIlluminationIntensities,
  setDifferentialPhaseContrast,
  setTimeLapsePeriod,
  setNumberOfImages,
  setAutoFocus,
  setAutoFocusMin,
  setAutoFocusMax,
  setAutoFocusStepSize,
  setAutoFocusIlluminationChannel,
  setAutoFocusSettleTime,
  setAutoFocusRange,
  setAutoFocusResolution,
  setAutoFocusCropsize,
  setAutoFocusStaticOffset,
  setAutoFocusTwoStage,
  setAutoFocusAlgorithm,
  setAutoFocusMode,
  setAutoFocusMaxAttempts,
  setAutoFocusTargetSetpoint,
  setZStack,
  setZStackMin,
  setZStackMax,
  setZStackStepSize,
  setSpeed,
  setGains,
  setExposureTimes,
  setPerformanceMode,
  setPerformanceTriggerMode,
  setPerformanceTPreMs,
  setPerformanceTPostMs,
  setOmeWriteTiff,
  setOmeWriteZarr,
  setOmeWriteStitchedTiff,
  setOmeWriteIndividualTiffs,
  setOverlapWidth,
  setOverlapHeight,
  setIsSnakescan,
  createPoint,
  addPoint,
  removePoint,
  setPointList,
  replacePoint,
} = experimentSlice.actions;

// Selector helper
export const getExperimentState = (state) => state.experimentState;

// Export reducer from slice
export default experimentSlice.reducer;
