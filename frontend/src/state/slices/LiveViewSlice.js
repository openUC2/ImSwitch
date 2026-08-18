import { createSlice } from "@reduxjs/toolkit";

// Define the initial state
const initialLiveViewState = {
  detectors: [],
  activeTab: 0,
  imageUrls: {},
  pollImageUrl: null,
  pixelSize: null,
  isStreamRunning: false,
  lastCapturePath: null, // Store the last capture path (snap/record)
  showPositionController: false, // Persistent toggle for position controller visibility
  snapFormat: 1, // Default: 1 = TIFF. Persisted. Change via setSnapFormat action.
  recordFormat: 4, // Default: 4 = MP4. Persisted. Change via setRecordFormat action.

  // Long-exposure mode: above `longExposureThresholdMs` the backend refuses to
  // stream (a frame arrives slower than the stream worker's grab timeout), so
  // the UI hides/blocks Start and offers Snap instead. Mirrored from
  // LiveViewController.getLongExposureInfo.
  isLongExposure: false,
  exposureMs: 0,
  longExposureThresholdMs: 2000,

  // In-flight snap. The capture runs as a background job on the backend
  // (RecordingController.startSnap) and we poll it, but the progress bar and
  // countdown are driven entirely client-side off `snapStartedAt` /
  // `snapExpectedMs`: a poll that is late or lost must never make the countdown
  // stutter. `snapExpectedMs` is one exposure plus readout/encode overhead; the
  // bar falls back to indeterminate once it is exceeded.
  isSnapping: false,
  snapStartedAt: null,
  snapExpectedMs: 0,
  snapJobId: null,
  snapStatus: null, // pending | running | done | error | cancelled
  snapCancelling: false,
};

// Create slice
const liveViewSlice = createSlice({
  name: "liveViewState",
  initialState: initialLiveViewState,
  reducers: {
    setDetectors: (state, action) => {
      state.detectors = action.payload;
    },
    setActiveTab: (state, action) => {
      state.activeTab = action.payload;
    },
    setImageUrls: (state, action) => {
      state.imageUrls = action.payload;
    },
    setPollImageUrl: (state, action) => {
      state.pollImageUrl = action.payload;
    },
    setPixelSize: (state, action) => {
      state.pixelSize = action.payload;
    },
    setIsStreamRunning: (state, action) => {
      state.isStreamRunning = action.payload;
    },
    setLastCapturePath: (state, action) => {
      state.lastCapturePath = action.payload;
    },
    setLastSnapPath: (state, action) => {
      state.lastCapturePath = action.payload;
    },
    setShowPositionController: (state, action) => {
      state.showPositionController = action.payload;
    },
    setSnapFormat: (state, action) => {
      state.snapFormat = action.payload;
    },
    setRecordFormat: (state, action) => {
      state.recordFormat = action.payload;
    },
    setLongExposureInfo: (state, action) => {
      const payload = action.payload || {};
      state.isLongExposure = Boolean(payload.isLongExposure);
      if (Number.isFinite(Number(payload.exposureMs))) {
        state.exposureMs = Number(payload.exposureMs);
      }
      if (Number.isFinite(Number(payload.thresholdMs))) {
        state.longExposureThresholdMs = Number(payload.thresholdMs);
      }
    },
    startSnap: (state, action) => {
      state.isSnapping = true;
      state.snapStartedAt = Date.now();
      state.snapExpectedMs = Number(action.payload?.expectedMs) || 0;
      state.snapJobId = action.payload?.jobId ?? null;
      state.snapStatus = action.payload?.status ?? "pending";
      state.snapCancelling = false;
    },
    updateSnapJob: (state, action) => {
      // Only the job identity/status is taken from the backend; the timing
      // fields stay client-side so the countdown is immune to poll jitter.
      if (action.payload?.jobId) state.snapJobId = action.payload.jobId;
      if (action.payload?.status) state.snapStatus = action.payload.status;
      if (Number(action.payload?.expectedDurationMs) > 0 && !state.snapExpectedMs) {
        state.snapExpectedMs = Number(action.payload.expectedDurationMs);
      }
    },
    setSnapCancelling: (state, action) => {
      state.snapCancelling = action.payload !== false;
    },
    finishSnap: (state) => {
      state.isSnapping = false;
      state.snapStartedAt = null;
      state.snapExpectedMs = 0;
      state.snapJobId = null;
      state.snapStatus = null;
      state.snapCancelling = false;
    },
    resetState: (state) => {
      return initialLiveViewState;
    },
  },
});

// Export actions from slice
export const {
  setDetectors,
  setActiveTab,
  setImageUrls,
  setPollImageUrl,
  setPixelSize,
  setIsStreamRunning,
  setLastCapturePath,
  setLastSnapPath,
  setShowPositionController,
  setSnapFormat,
  setRecordFormat,
  setLongExposureInfo,
  startSnap,
  updateSnapJob,
  setSnapCancelling,
  finishSnap,
  resetState,
} = liveViewSlice.actions;

// Selector helper
export const getLiveViewState = (state) => state.liveViewState;

// Export reducer from slice
export default liveViewSlice.reducer;
