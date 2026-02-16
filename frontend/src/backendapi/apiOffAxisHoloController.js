// src/backendapi/apiOffAxisHoloController.js
// API client functions for Off-Axis Hologram Controller

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get current off-axis hologram processing parameters
 * @returns {Promise<Object>} Promise containing parameters object
 */
export const apiOffAxisHoloControllerGetParams = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/get_parameters_offaxisholo");
    return response.data;
  } catch (error) {
    console.error("Error getting off-axis holo parameters:", error);
    throw error;
  }
};

/**
 * Set off-axis hologram processing parameters
 * @param {Object} params - Parameters to update
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiOffAxisHoloControllerSetParams = async (params) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.post("/OffAxisHoloController/set_parameters_offaxisholo", params);
    return response.data;
  } catch (error) {
    console.error("Error setting off-axis holo parameters:", error);
    throw error;
  }
};

/**
 * Get current processing state
 * @returns {Promise<Object>} Promise containing state object
 */
export const apiOffAxisHoloControllerGetState = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/get_state_offaxisholo");
    return response.data;
  } catch (error) {
    console.error("Error getting off-axis holo state:", error);
    throw error;
  }
};

/**
 * Start off-axis hologram processing
 * @returns {Promise<Object>} Promise containing state
 */
export const apiOffAxisHoloControllerStartProcessing = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/start_processing_offaxisholo");
    return response.data;
  } catch (error) {
    console.error("Error starting off-axis holo processing:", error);
    throw error;
  }
};

/**
 * Stop off-axis hologram processing
 * @returns {Promise<Object>} Promise containing state
 */
export const apiOffAxisHoloControllerStopProcessing = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/stop_processing_offaxisholo");
    return response.data;
  } catch (error) {
    console.error("Error stopping off-axis holo processing:", error);
    throw error;
  }
};

/**
 * Pause off-axis hologram processing
 * @returns {Promise<Object>} Promise containing state
 */
export const apiOffAxisHoloControllerPauseProcessing = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/pause_processing_offaxisholo");
    return response.data;
  } catch (error) {
    console.error("Error pausing off-axis holo processing:", error);
    throw error;
  }
};

/**
 * Resume off-axis hologram processing
 * @returns {Promise<Object>} Promise containing state
 */
export const apiOffAxisHoloControllerResumeProcessing = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/resume_processing_offaxisholo");
    return response.data;
  } catch (error) {
    console.error("Error resuming off-axis holo processing:", error);
    throw error;
  }
};

/**
 * Set digital refocus distance
 * @param {number} dz - Distance in meters
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiOffAxisHoloControllerSetDz = async (dz) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/set_dz_offaxisholo", {
      params: { dz }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting off-axis dz:", error);
    throw error;
  }
};

/**
 * Set sensor ROI
 * @param {Object} roi - ROI parameters {center_x, center_y, size}
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiOffAxisHoloControllerSetRoi = async (roi) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/set_roi_offaxisholo", {
      params: roi
    });
    return response.data;
  } catch (error) {
    console.error("Error setting off-axis ROI:", error);
    throw error;
  }
};

/**
 * Set sideband (cross-correlation) ROI in FFT space
 * @param {Object} ccRoi - CC ROI parameters {center_x, center_y, size_x, size_y}
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiOffAxisHoloControllerSetCcRoi = async (ccRoi) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/set_cc_roi_offaxisholo", {
      params: ccRoi
    });
    return response.data;
  } catch (error) {
    console.error("Error setting off-axis CC ROI:", error);
    throw error;
  }
};

/**
 * Set apodization (edge damping) parameters
 * @param {boolean} enabled - Enable/disable
 * @param {string} windowType - Window type
 * @param {number} alpha - Tukey alpha parameter
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiOffAxisHoloControllerSetApodization = async (enabled, windowType = "tukey", alpha = 0.1) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/set_apodization_offaxisholo", {
      params: { enabled, window_type: windowType, alpha }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting off-axis apodization:", error);
    throw error;
  }
};

/**
 * Set binning factor
 * @param {number} binning - Binning factor (1, 2, 4, etc.)
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiOffAxisHoloControllerSetBinning = async (binning) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/set_binning_offaxisholo", {
      params: { binning }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting off-axis binning:", error);
    throw error;
  }
};

/**
 * Set pixel size
 * @param {number} pixelsize - Pixel size in meters
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiOffAxisHoloControllerSetPixelsize = async (pixelsize) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/set_pixelsize_offaxisholo", {
      params: { pixelsize }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting off-axis pixelsize:", error);
    throw error;
  }
};

/**
 * Set wavelength
 * @param {number} wavelength - Wavelength in meters
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiOffAxisHoloControllerSetWavelength = async (wavelength) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/OffAxisHoloController/set_wavelength_offaxisholo", {
      params: { wavelength }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting off-axis wavelength:", error);
    throw error;
  }
};

export default {
  getParams: apiOffAxisHoloControllerGetParams,
  setParams: apiOffAxisHoloControllerSetParams,
  getState: apiOffAxisHoloControllerGetState,
  startProcessing: apiOffAxisHoloControllerStartProcessing,
  stopProcessing: apiOffAxisHoloControllerStopProcessing,
  pauseProcessing: apiOffAxisHoloControllerPauseProcessing,
  resumeProcessing: apiOffAxisHoloControllerResumeProcessing,
  setDz: apiOffAxisHoloControllerSetDz,
  setRoi: apiOffAxisHoloControllerSetRoi,
  setCcRoi: apiOffAxisHoloControllerSetCcRoi,
  setApodization: apiOffAxisHoloControllerSetApodization,
  setBinning: apiOffAxisHoloControllerSetBinning,
  setPixelsize: apiOffAxisHoloControllerSetPixelsize,
  setWavelength: apiOffAxisHoloControllerSetWavelength,
};
