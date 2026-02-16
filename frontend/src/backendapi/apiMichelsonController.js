// src/backendapi/apiMichelsonController.js
// API client functions for Michelson Time-Series Controller

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get current Michelson capture parameters
 * @returns {Promise<Object>} Promise containing parameters object
 */
export const apiMichelsonControllerGetParams = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/get_parameters_michelson");
    return response.data;
  } catch (error) {
    console.error("Error getting Michelson parameters:", error);
    throw error;
  }
};

/**
 * Set Michelson capture parameters
 * @param {Object} params - Parameters to update
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiMichelsonControllerSetParams = async (params) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.post("/MichelsonTimeSeriesController/set_parameters_michelson", params);
    return response.data;
  } catch (error) {
    console.error("Error setting Michelson parameters:", error);
    throw error;
  }
};

/**
 * Get current capture state
 * @returns {Promise<Object>} Promise containing state object
 */
export const apiMichelsonControllerGetState = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/get_state_michelson");
    return response.data;
  } catch (error) {
    console.error("Error getting Michelson state:", error);
    throw error;
  }
};

/**
 * Start Michelson time-series capture
 * @returns {Promise<Object>} Promise containing state
 */
export const apiMichelsonControllerStartCapture = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/start_capture_michelson");
    return response.data;
  } catch (error) {
    console.error("Error starting Michelson capture:", error);
    throw error;
  }
};

/**
 * Stop Michelson time-series capture
 * @returns {Promise<Object>} Promise containing state
 */
export const apiMichelsonControllerStopCapture = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/stop_capture_michelson");
    return response.data;
  } catch (error) {
    console.error("Error stopping Michelson capture:", error);
    throw error;
  }
};

/**
 * Get time-series data
 * @param {number} lastN - Number of samples to retrieve (0 = all)
 * @returns {Promise<Object>} Promise containing timestamps, means, stds arrays
 */
export const apiMichelsonControllerGetTimeseries = async (lastN = 0) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/get_timeseries_michelson", {
      params: { last_n: lastN }
    });
    return response.data;
  } catch (error) {
    console.error("Error getting Michelson timeseries:", error);
    throw error;
  }
};

/**
 * Export time-series data as CSV (returns download URL or blob)
 * @returns {Promise<Blob>} Promise containing CSV blob
 */
export const apiMichelsonControllerExportCsv = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/export_csv_michelson", {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    console.error("Error exporting Michelson CSV:", error);
    throw error;
  }
};

/**
 * Get statistics of current time-series data
 * @returns {Promise<Object>} Promise containing mean_of_means, std_of_means, etc.
 */
export const apiMichelsonControllerGetStatistics = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/get_statistics_michelson");
    return response.data;
  } catch (error) {
    console.error("Error getting Michelson statistics:", error);
    throw error;
  }
};

/**
 * Clear time-series buffer
 * @returns {Promise<Object>} Promise containing state
 */
export const apiMichelsonControllerClearBuffer = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/clear_buffer_michelson");
    return response.data;
  } catch (error) {
    console.error("Error clearing Michelson buffer:", error);
    throw error;
  }
};

/**
 * Set ROI center and size
 * @param {number} centerX - ROI center X
 * @param {number} centerY - ROI center Y
 * @param {number} size - ROI size (5, 10, 20)
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiMichelsonControllerSetRoi = async (centerX, centerY, size) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/set_roi_michelson", {
      params: { center_x: centerX, center_y: centerY, size }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting Michelson ROI:", error);
    throw error;
  }
};

/**
 * Set update frequency
 * @param {number} freq - Update frequency in Hz
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiMichelsonControllerSetUpdateFreq = async (freq) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/set_update_freq_michelson", {
      params: { freq }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting Michelson update frequency:", error);
    throw error;
  }
};

/**
 * Set buffer duration
 * @param {number} duration - Buffer duration in seconds
 * @returns {Promise<Object>} Promise containing updated parameters
 */
export const apiMichelsonControllerSetBufferDuration = async (duration) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/MichelsonTimeSeriesController/set_buffer_duration_michelson", {
      params: { duration }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting Michelson buffer duration:", error);
    throw error;
  }
};

export default {
  getParams: apiMichelsonControllerGetParams,
  setParams: apiMichelsonControllerSetParams,
  getState: apiMichelsonControllerGetState,
  startCapture: apiMichelsonControllerStartCapture,
  stopCapture: apiMichelsonControllerStopCapture,
  getTimeseries: apiMichelsonControllerGetTimeseries,
  exportCsv: apiMichelsonControllerExportCsv,
  getStatistics: apiMichelsonControllerGetStatistics,
  clearBuffer: apiMichelsonControllerClearBuffer,
  setRoi: apiMichelsonControllerSetRoi,
  setUpdateFreq: apiMichelsonControllerSetUpdateFreq,
  setBufferDuration: apiMichelsonControllerSetBufferDuration,
};
