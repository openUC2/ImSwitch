import { createSlice } from "@reduxjs/toolkit";

// Mirrors DEFAULT_METADATA in FlowStopController.py (EcoTaxa-style sample metadata).
export const EMPTY_METADATA = {
  sample_project: "",
  sample_id: "",
  sample_ship: "",
  sample_operator: "",
  sample_gear: "",
  sample_mesh_size_um: 0,
  sample_depth_min_m: 0,
  sample_depth_max_m: 0,
  sample_latitude: 0,
  sample_longitude: 0,
  sample_date: "",
  sample_time: "",
  sample_total_volume_ml: 0,
  acq_instrument: "openUC2 FlowStop",
  acq_celltype_ul: 0,
  object_notes: "",
};

const initialState = {
  // UI state
  tabIndex: 0,

  // Acquisition parameters (persisted server-side)
  experimentName: "FlowStopExperiment",
  experimentDescription: "",
  uniqueId: "",
  numImages: -1,
  volumePerImage: 1000,
  timeToStabilize: 0.5,
  pumpSpeed: 10000,
  frameRate: 1,
  fileFormat: "JPG",
  isRecordVideo: false,
  wasRunning: false, // resume the acquisition automatically after an ImSwitch restart

  // Sample metadata (persisted server-side)
  metadata: { ...EMPTY_METADATA },

  // Manual controls (UI only)
  focusStep: 100,
  focusSpeed: 10000,
  pumpStep: 1000,
  pumpJogSpeed: 10000,
  illuminationValue: 0,
  illuminationOn: false,
  autoExposure: true,
  exposureTime: 100,

  // Hardware description from the backend
  hardware: null,

  // Status
  isRunning: false,
  currentImageCount: 0,
  progress: -1,
  etaSeconds: -1,
  elapsedSeconds: 0,
  relativePath: "",
  lastError: "",

  // Gallery
  galleryFiles: [],
};

const flowStopSlice = createSlice({
  name: "flowStop",
  initialState,
  reducers: {
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },

    // Accepts a partial {key: value} patch for any top-level field.
    setField: (state, action) => {
      Object.assign(state, action.payload);
    },

    setMetadataField: (state, action) => {
      const { key, value } = action.payload;
      state.metadata[key] = value;
    },
    setMetadata: (state, action) => {
      state.metadata = { ...EMPTY_METADATA, ...action.payload };
    },

    setHardware: (state, action) => {
      state.hardware = action.payload;
    },

    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },
    setCurrentImageCount: (state, action) => {
      state.currentImageCount = action.payload;
    },
    setStatus: (state, action) => {
      const s = action.payload || {};
      state.isRunning = !!s.isRunning;
      state.currentImageCount = s.imagesTaken ?? state.currentImageCount;
      state.progress = s.progress ?? -1;
      state.etaSeconds = s.etaSeconds ?? -1;
      state.elapsedSeconds = s.elapsedSeconds ?? 0;
      state.relativePath = s.relativePath ?? state.relativePath;
      state.lastError = s.lastError ?? "";
    },

    setGalleryFiles: (state, action) => {
      state.galleryFiles = action.payload;
    },

    resetToDefaults: () => ({ ...initialState }),
  },
});

export const {
  setTabIndex,
  setField,
  setMetadataField,
  setMetadata,
  setHardware,
  setIsRunning,
  setCurrentImageCount,
  setStatus,
  setGalleryFiles,
  resetToDefaults,
} = flowStopSlice.actions;

export const getFlowStopState = (state) => state.flowStop;

export default flowStopSlice.reducer;
