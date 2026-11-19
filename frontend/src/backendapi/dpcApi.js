// src/backendapi/dpcApi.js
// DPC (Differential Phase Contrast) API module

import axios from "axios";

/**
 * Get current DPC parameters
 * @param {string} baseUrl - Base URL of the backend API (e.g., "http://localhost/imswitch/api")
 * @returns {Promise<Object>} DPC parameters
 */
export const getDpcParams = async (baseUrl) => {
  const response = await axios.get(
    `${baseUrl}/DPCController/get_dpc_params`
  );
  return response.data;
};

/**
 * Set DPC parameters
 * @param {string} baseUrl - Base URL of the backend API
 * @param {Object} params - DPC parameters to update
 * @returns {Promise<Object>} Updated DPC parameters
 */
export const setDpcParams = async (baseUrl, params) => {
  const response = await axios.post(
    `${baseUrl}/DPCController/set_dpc_params`,
    params
  );
  return response.data;
};

/**
 * Get current DPC processing state
 * @param {string} baseUrl - Base URL of the backend API
 * @returns {Promise<Object>} DPC state
 */
export const getDpcState = async (baseUrl) => {
  const response = await axios.get(
    `${baseUrl}/DPCController/get_dpc_state`
  );
  return response.data;
};

/**
 * Start DPC processing
 * @param {string} baseUrl - Base URL of the backend API
 * @returns {Promise<Object>} Start response with status and state
 */
export const startDpcProcessing = async (baseUrl) => {
  const response = await axios.get(
    `${baseUrl}/DPCController/start_dpc_processing`
  );
  return response.data;
};

/**
 * Stop DPC processing
 * @param {string} baseUrl - Base URL of the backend API
 * @returns {Promise<Object>} Stop response with status and state
 */
export const stopDpcProcessing = async (baseUrl) => {
  const response = await axios.get(
    `${baseUrl}/DPCController/stop_dpc_processing`
  );
  return response.data;
};

/**
 * Build DPC MJPEG stream URL
 * @param {string} baseUrl - Base URL of the backend API
 * @param {number} jpegQuality - JPEG quality (1-100)
 * @returns {string} MJPEG stream URL
 */
export const getDpcStreamUrl = (baseUrl, jpegQuality = 85) => {
  return `${baseUrl}/DPCController/mjpeg_stream_dpc?startStream=true&jpeg_quality=${jpegQuality}`;
};

/** PNG of the DPC weak-object transfer functions; params: wavelength, na, nai, nai_inner (unset: current) */
export const getDpcWotfUrl = (baseUrl, params) =>
  `${baseUrl}/DPCController/get_dpc_wotf?${new URLSearchParams(params)}`;

const get = async (url) => (await axios.get(url)).data;

/** LEDs with illumination NA (camera frame), DPC membership, brightfield flag, calibration */
export const getLedLayout = (baseUrl) => get(`${baseUrl}/DPCController/get_led_layout`);

/** Manual LED -> camera calibration: {rotation_deg, flip, offset_x_mm, offset_y_mm, offset_z_mm} */
export const setLedCalibration = (baseUrl, cal) =>
  get(`${baseUrl}/DPCController/set_led_calibration?${new URLSearchParams(cal)}`);

/** LED scans running in the backend; progress via get_dpc_state (task, task_progress, task_message) */
export const startLedCalibration = (baseUrl) => get(`${baseUrl}/DPCController/start_led_calibration`);
export const startFpmAcquisition = (baseUrl) => get(`${baseUrl}/DPCController/start_fpm_acquisition`);
export const snapDpc = (baseUrl) => get(`${baseUrl}/DPCController/snap_dpc`);
export const stopDpcTask = (baseUrl) => get(`${baseUrl}/DPCController/stop_task`);

/** Light exactly these LED chain indices (others off) */
export const setLeds = (baseUrl, ids, [r, g, b]) =>
  get(`${baseUrl}/LEDMatrixController/setLEDs?${new URLSearchParams({
    ledIds: ids.join(","), intensity_r: r, intensity_g: g, intensity_b: b })}`);
