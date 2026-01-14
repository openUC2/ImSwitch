import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  // UI state
  tabIndex: 0,

  // Experiment parameters
  experimentName: "STORM_Experiment",
  exposureTime: 50, // milliseconds
  
  // Crop region parameters
  cropRegion: {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  },

  // Laser parameters
  laserIntensities: {},  // Will be populated with available lasers
  
  // Status
  isRunning: false,
  currentFrameNumber: 0,
  
  // STORM reconstruction parameters following the new API structure
  stormParameters: {
    // Processing parameters
    threshold: 0.2,
    fit_roi_size: 13,
    fitting_method: "2D_Phasor_CPU",
    filter_type: "bandpass",
    temporal_median_enabled: false,
    update_rate: 10,
    pixel_size_nm: 117.5,
    super_resolution_pixel_size_nm: 10.0,
    
    // Bandpass filter parameters
    bandpass_filter: {
      center: 40.0,
      width: 90.0,
      filter_type: "gauss",
      show_filter: false
    },
    
    // Blob detector parameters  
    blob_detector: {
      min_threshold: 0.0,
      max_threshold: 255.0,
      min_area: 1.5,
      max_area: 80.0,
      min_circularity: null,
      min_convexity: null,
      min_inertia_ratio: null,
      blob_color: 255,
      min_dist_between_blobs: 0.0
    }
  },
  
  // Acquisition parameters
  acquisitionParameters: {
    session_id: null,
    exposure_time: null,
    max_frames: -1,
    crop_enabled: false,
    crop_x: null,
    crop_y: null,
    crop_width: null,
    crop_height: null,
    save_enabled: false,
    save_directory: "STORM",
    save_format: "tiff",
    process_locally: true,
    process_arkitekt: false,
    processing_parameters: null
  },
  
  // Reconstruction status
  isReconstructing: false,
  
  // Images
  reconstructedImage: null,
  
  // Localizations data
  localizations: [], // Array of {x, y} coordinates
};

const stormSlice = createSlice({
  name: "storm",
  initialState,
  reducers: {
    // UI actions
    setTabIndex: (state, action) => {
      state.tabIndex = action.payload;
    },

    // Experiment parameter actions
    setExperimentName: (state, action) => {
      state.experimentName = action.payload;
    },
    setExposureTime: (state, action) => {
      state.exposureTime = action.payload;
    },

    // Crop region actions
    setCropRegion: (state, action) => {
      state.cropRegion = { ...state.cropRegion, ...action.payload };
    },
    setCropX: (state, action) => {
      state.cropRegion.x = action.payload;
    },
    setCropY: (state, action) => {
      state.cropRegion.y = action.payload;
    },
    setCropWidth: (state, action) => {
      state.cropRegion.width = action.payload;
    },
    setCropHeight: (state, action) => {
      state.cropRegion.height = action.payload;
    },

    // Laser actions
    setLaserIntensities: (state, action) => {
      state.laserIntensities = action.payload;
    },
    setLaserIntensity: (state, action) => {
      const { laserName, intensity } = action.payload;
      state.laserIntensities[laserName] = intensity;
    },

    // Status actions
    setIsRunning: (state, action) => {
      state.isRunning = action.payload;
    },
    setCurrentFrameNumber: (state, action) => {
      state.currentFrameNumber = action.payload;
    },
    
    // STORM reconstruction parameter actions
    setStormParameters: (state, action) => {
      state.stormParameters = { ...state.stormParameters, ...action.payload };
    },
    setThreshold: (state, action) => {
      state.stormParameters.threshold = action.payload;
    },
    setFitRoiSize: (state, action) => {
      state.stormParameters.fit_roi_size = action.payload;
    },
    setFittingMethod: (state, action) => {
      state.stormParameters.fitting_method = action.payload;
    },
    setFilterType: (state, action) => {
      state.stormParameters.filter_type = action.payload;
    },
    setTemporalMedianEnabled: (state, action) => {
      state.stormParameters.temporal_median_enabled = action.payload;
    },
    setUpdateRate: (state, action) => {
      state.stormParameters.update_rate = action.payload;
    },
    setPixelSizeNm: (state, action) => {
      state.stormParameters.pixel_size_nm = action.payload;
    },
    setSuperResolutionPixelSizeNm: (state, action) => {
      state.stormParameters.super_resolution_pixel_size_nm = action.payload;
    },
    setBandpassFilter: (state, action) => {
      state.stormParameters.bandpass_filter = { ...state.stormParameters.bandpass_filter, ...action.payload };
    },
    setBlobDetector: (state, action) => {
      state.stormParameters.blob_detector = { ...state.stormParameters.blob_detector, ...action.payload };
    },
    
    // Acquisition parameter actions
    setAcquisitionParameters: (state, action) => {
      state.acquisitionParameters = { ...state.acquisitionParameters, ...action.payload };
    },
    setSessionId: (state, action) => {
      state.acquisitionParameters.session_id = action.payload;
    },
    setAcquisitionExposureTime: (state, action) => {
      state.acquisitionParameters.exposure_time = action.payload;
    },
    setMaxFrames: (state, action) => {
      state.acquisitionParameters.max_frames = action.payload;
    },
    setCropEnabled: (state, action) => {
      state.acquisitionParameters.crop_enabled = action.payload;
    },
    setSaveEnabled: (state, action) => {
      state.acquisitionParameters.save_enabled = action.payload;
    },
    setSaveDirectory: (state, action) => {
      state.acquisitionParameters.save_directory = action.payload;
    },
    setSaveFormat: (state, action) => {
      state.acquisitionParameters.save_format = action.payload;
    },
    setProcessLocally: (state, action) => {
      state.acquisitionParameters.process_locally = action.payload;
    },
    setProcessArkitekt: (state, action) => {
      state.acquisitionParameters.process_arkitekt = action.payload;
    },
    
    // Reconstruction status actions
    setIsReconstructing: (state, action) => {
      state.isReconstructing = action.payload;
    },

    // Image actions
    setReconstructedImage: (state, action) => {
      state.reconstructedImage = action.payload;
    },

    // Localization actions
    addLocalizations: (state, action) => {
      // action.payload should be an array of {x, y} coordinates
      state.localizations.push(...action.payload);
    },
    resetLocalizations: (state) => {
      state.localizations = [];
    },

    // Reset actions
    resetExperiment: (state) => {
      state.isRunning = false;
      state.currentFrameNumber = 0;
      state.isReconstructing = false;
    },
    resetToDefaults: (state) => {
      return { ...initialState };
    },
  },
});

// Export actions
export const {
  setTabIndex,
  setExperimentName,
  setExposureTime,
  setCropRegion,
  setCropX,
  setCropY,
  setCropWidth,
  setCropHeight,
  setLaserIntensities,
  setLaserIntensity,
  setIsRunning,
  setCurrentFrameNumber,
  setStormParameters,
  setThreshold,
  setFitRoiSize,
  setFittingMethod,
  setFilterType,
  setTemporalMedianEnabled,
  setUpdateRate,
  setPixelSizeNm,
  setSuperResolutionPixelSizeNm,
  setBandpassFilter,
  setBlobDetector,
  setAcquisitionParameters,
  setSessionId,
  setAcquisitionExposureTime,
  setMaxFrames,
  setCropEnabled,
  setSaveEnabled,
  setSaveDirectory,
  setSaveFormat,
  setProcessLocally,
  setProcessArkitekt,
  setIsReconstructing,
  setReconstructedImage,
  addLocalizations,
  resetLocalizations,
  resetExperiment,
  resetToDefaults,
} = stormSlice.actions;

// Export selector
export const getSTORMState = (state) => state.storm;

// Export reducer
export default stormSlice.reducer;