import { createSlice } from "@reduxjs/toolkit";

/**
 * Redux slice for Overview Camera Registration Wizard and overlay state.
 *
 * Manages:
 * - Wizard open/close and current step
 * - Per-slide registration status and corner picks
 * - Live stream state
 * - Overlay visibility and opacity for WellSelector canvas
 */

const initialOverviewRegistrationState = {
  // Wizard UI state
  wizardOpen: false,
  wizardStep: 0, // 0=start, 1=slide selection, 2=live align, 3=corner pick, 4=save, 5=finish
  currentSlotId: "1",

  // Layout / camera config from backend
  layoutName: "Heidstar 4x Histosample",
  cameraName: "",
  cameraAvailable: false,
  cornerConvention: "TL,TR,BR,BL",
  cornerLabels: [
    "1: Top-Left",
    "2: Top-Right",
    "3: Bottom-Right",
    "4: Bottom-Left",
  ],
  slots: [], // [{slotId, name, x, y, width, height, corners: [{x,y},...]}]

  // Per-slide status from backend: { "1": {complete, snapshotId, ...}, "2": ... }
  slideStatus: {},

  // Snapshot state (current working snapshot)
  snapshotImage: null, // base64 image data
  snapshotMimeType: "image/jpeg",
  snapshotId: "",
  snapshotTimestamp: "",
  snapshotWidth: 0,
  snapshotHeight: 0,
  snapshotStagePosition: { x: 0, y: 0, z: 0 },

  // Corner picking state
  pickedCorners: [], // [{x, y}, ...] up to 4 points in image pixel coords
  cornerPickingActive: false,

  // Registration result for current slide
  lastRegistrationResult: null,

  // Overlay state for WellSelector canvas
  overlayEnabled: false,
  overlayOpacity: 0.6,
  overlayData: {}, // { slides: { "1": {imageBase64, stageBounds, ...}, ... } }

  // Loading / error state
  isLoading: false,
  error: null,
};

const overviewRegistrationSlice = createSlice({
  name: "overviewRegistrationState",
  initialState: initialOverviewRegistrationState,
  reducers: {
    // Wizard control
    setWizardOpen: (state, action) => {
      state.wizardOpen = action.payload;
    },
    setWizardStep: (state, action) => {
      state.wizardStep = action.payload;
    },
    setCurrentSlotId: (state, action) => {
      state.currentSlotId = action.payload;
    },

    // Config from backend
    setConfig: (state, action) => {
      const cfg = action.payload;
      state.layoutName = cfg.layoutName || state.layoutName;
      state.cameraName = cfg.cameraName || state.cameraName;
      state.cameraAvailable =
        cfg.cameraAvailable !== undefined
          ? cfg.cameraAvailable
          : state.cameraAvailable;
      state.cornerConvention = cfg.cornerConvention || state.cornerConvention;
      state.cornerLabels = cfg.cornerLabels || state.cornerLabels;
      state.slots = cfg.slots || state.slots;
      state.slideStatus = cfg.status || state.slideStatus;
    },

    // Slide status
    setSlideStatus: (state, action) => {
      state.slideStatus = action.payload;
    },
    updateSlideStatus: (state, action) => {
      const { slotId, data } = action.payload;
      state.slideStatus[slotId] = data;
    },

    // Snapshot
    setSnapshot: (state, action) => {
      const snap = action.payload;
      state.snapshotImage = snap.imageBase64 || null;
      state.snapshotMimeType = snap.imageMimeType || "image/jpeg";
      state.snapshotId = snap.snapshotId || "";
      state.snapshotTimestamp = snap.timestamp || "";
      state.snapshotWidth = snap.imageWidth || 0;
      state.snapshotHeight = snap.imageHeight || 0;
      state.snapshotStagePosition = snap.stagePosition || { x: 0, y: 0, z: 0 };
    },
    clearSnapshot: (state) => {
      state.snapshotImage = null;
      state.snapshotId = "";
      state.snapshotTimestamp = "";
      state.snapshotWidth = 0;
      state.snapshotHeight = 0;
      state.pickedCorners = [];
      state.cornerPickingActive = false;
    },

    // Corner picking
    setCornerPickingActive: (state, action) => {
      state.cornerPickingActive = action.payload;
    },
    addPickedCorner: (state, action) => {
      if (state.pickedCorners.length < 4) {
        state.pickedCorners.push(action.payload);
      }
    },
    undoLastCorner: (state) => {
      state.pickedCorners.pop();
    },
    resetPickedCorners: (state) => {
      state.pickedCorners = [];
    },
    setPickedCorners: (state, action) => {
      state.pickedCorners = action.payload;
    },

    // Registration result
    setLastRegistrationResult: (state, action) => {
      state.lastRegistrationResult = action.payload;
    },

    // Overlay
    setOverlayEnabled: (state, action) => {
      state.overlayEnabled = action.payload;
    },
    setOverlayOpacity: (state, action) => {
      state.overlayOpacity = action.payload;
      if (isNaN(state.overlayOpacity)) state.overlayOpacity = 0.6;
    },
    setOverlayData: (state, action) => {
      state.overlayData = action.payload;
    },

    // Loading / error
    setIsLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },

    // Reset wizard state (keep overlay data)
    resetWizard: (state) => {
      state.wizardOpen = false;
      state.wizardStep = 0;
      state.currentSlotId = "1";
      state.snapshotImage = null;
      state.snapshotId = "";
      state.pickedCorners = [];
      state.cornerPickingActive = false;
      state.lastRegistrationResult = null;
      state.error = null;
    },
  },
});

export const {
  setWizardOpen,
  setWizardStep,
  setCurrentSlotId,
  setConfig,
  setSlideStatus,
  updateSlideStatus,
  setSnapshot,
  clearSnapshot,
  setCornerPickingActive,
  addPickedCorner,
  undoLastCorner,
  resetPickedCorners,
  setPickedCorners,
  setLastRegistrationResult,
  setOverlayEnabled,
  setOverlayOpacity,
  setOverlayData,
  setIsLoading,
  setError,
  resetWizard,
} = overviewRegistrationSlice.actions;

export const getOverviewRegistrationState = (state) =>
  state.overviewRegistrationState;

export default overviewRegistrationSlice.reducer;
