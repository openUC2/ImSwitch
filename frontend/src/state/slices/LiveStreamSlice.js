import { createSlice } from "@reduxjs/toolkit";

// Define the initial state
const initialLiveStreamState = {
  // Image data removed - handled directly by viewer components
  // liveViewImage: "", // REMOVED: no longer store pixel data in Redux
  minVal: 0,
  maxVal: 255, // Default to 8-bit range for JPEG streaming
  gamma: 1.0, // New: gamma correction
  imageFormat: "jpeg", // Track image format (jpeg, binary, etc.) - default to JPEG
  pixelSize: null,
  fovX: 0,
  fovY: 0, 
  // Backend capability detection
  isLegacyBackend: false,
  backendCapabilities: {
    binaryStreaming: true,
    webglSupported: true
  },
  // Persistent stream settings
  streamSettings: {
    current_compression_algorithm: "jpeg",
    binary: {
      enabled: false,
      compression: { algorithm: "lz4", level: 0 },
      subsampling: { factor: 4 },
      throttle_ms: 100,
      bitdepth_in: 12,
      pixfmt: "GRAY16"
    },
    jpeg: {
      enabled: true,
      quality: 85
    },
    webrtc: {
      enabled: false,
      max_width: 1280,
      throttle_ms: 33,
      subsampling_factor: 1
    }
  },
  // Crop settings (applied before subsampling, shared across all formats)
  cropSize: 0, // 0 = no crop (full FOV), >0 = quadratic crop around center
  // Histogram data
  histogramX: [],
  histogramY: [],
  showHistogram: false,
  // View transform state (optional - can be local to component)
  zoom: 1.0,
  translateX: 0,
  translateY: 0,
  // Stream statistics
  stats: {
    fps: 0,
    bps: 0, // bits per second
    compressionRatio: 0,
    latency_ms: 0, // Current frame latency
    avg_latency_ms: 0, // Moving average latency
    frameCount: 0, // Total frames received
    currentFrameId: null // Current frame ID from metadata
  }
};

// Create slice
const liveStreamSlice = createSlice({
  name: "liveStreamState",
  initialState: initialLiveStreamState,
  reducers: {
    setLiveViewImage: (state, action) => {
      state.liveViewImage = action.payload;
    },
    
    setMinVal: (state, action) => {
      state.minVal = action.payload;
    },

    setMaxVal: (state, action) => {
      state.maxVal = action.payload;
    },

    setImageFormat: (state, action) => {
      state.imageFormat = action.payload;
    },

    setStreamSettings: (state, action) => {
      state.streamSettings = { ...state.streamSettings, ...action.payload };
      // Update imageFormat based on current compression algorithm
      if (action.payload.current_compression_algorithm) {
        state.imageFormat = action.payload.current_compression_algorithm;
      }
    },

    setGamma: (state, action) => {
      state.gamma = action.payload;
    },

    setIsLegacyBackend: (state, action) => {
      state.isLegacyBackend = action.payload;
      // Automatically disable binary streaming for legacy backends
      if (action.payload) {
        state.backendCapabilities.binaryStreaming = false;
        state.backendCapabilities.webglSupported = false;
      }
    },

    setBackendCapabilities: (state, action) => {
      state.backendCapabilities = { ...state.backendCapabilities, ...action.payload };
    },

    setZoom: (state, action) => {
      state.zoom = action.payload;
    },

    setTranslate: (state, action) => {
      state.translateX = action.payload.x;
      state.translateY = action.payload.y;
    },

    setStats: (state, action) => {
      state.stats = { ...state.stats, ...action.payload };
    },
    
    setCurrentFrameId: (state, action) => {
      state.stats.currentFrameId = action.payload;
    },
    
    updateLatency: (state, action) => {
      const latency_ms = action.payload;
      state.stats.latency_ms = latency_ms;
      state.stats.frameCount += 1;
      
      // Calculate moving average (exponential moving average with alpha=0.1)
      if (state.stats.avg_latency_ms === 0) {
        state.stats.avg_latency_ms = latency_ms;
      } else {
        state.stats.avg_latency_ms = 0.1 * latency_ms + 0.9 * state.stats.avg_latency_ms;
      }
    },

    setPixelSize: (state, action) => {
      state.pixelSize = action.payload;
    },


    setHistogramData: (state, action) => {
      state.histogramX = action.payload.x;
      state.histogramY = action.payload.y;
    },

    setShowHistogram: (state, action) => {
      state.showHistogram = action.payload;
    },

    setCropSize: (state, action) => {
      state.cropSize = action.payload;
    },

    resetState: (state) => {
      console.log("resetState");
      return { ...initialLiveStreamState }; // Reset to initial state
    },
  },
});

// Export actions from slice
export const {
  setLiveViewImage,
  setMinVal,
  setMaxVal,
  setImageFormat,
  setStreamSettings,
  setGamma,
  setIsLegacyBackend,
  setBackendCapabilities,
  setZoom,
  setTranslate,
  setStats,
  setCurrentFrameId,
  updateLatency,
  setPixelSize,
  setFovY,
  setHistogramData,
  setShowHistogram,
  setCropSize,
  resetState,
} = liveStreamSlice.actions;

// Selector helper
export const getLiveStreamState = (state) => state.liveStreamState;

// Export reducer from slice
export default liveStreamSlice.reducer;
