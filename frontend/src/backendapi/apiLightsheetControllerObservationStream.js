// src/backendapi/apiLightsheetControllerObservationStream.js
// API functions for Lightsheet Observation Camera Stream control
import createAxiosInstance from "./createAxiosInstance";

/**
 * Start or stop the observation camera MJPEG stream.
 * 
 * @param {boolean} start - true to start, false to stop the stream
 * @returns {Promise<Object>} Response with stream status
 */
export const apiLightsheetControllerObservationStreamControl = async (start = true) => {
  try {
    const axiosInstance = createAxiosInstance();
    const endpoint = start 
      ? "/LightsheetController/startObservationStream"
      : "/LightsheetController/stopObservationStream";
    const response = await axiosInstance.get(endpoint);
    return response.data;
  } catch (error) {
    console.error(`Failed to ${start ? 'start' : 'stop'} observation stream:`, error);
    throw error;
  }
};

/**
 * Set observation camera exposure time.
 * 
 * @param {number} exposureTime - Exposure time in milliseconds
 * @returns {Promise<Object>} Response data
 */
export const apiLightsheetControllerSetObservationExposure = async (exposureTime) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/setObservationExposure", {
      params: { exposure_time: exposureTime }
    });
    return response.data;
  } catch (error) {
    console.error("Failed to set observation exposure:", error);
    throw error;
  }
};

/**
 * Get observation camera exposure time.
 * 
 * @returns {Promise<number>} Exposure time in milliseconds
 */
export const apiLightsheetControllerGetObservationExposure = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/getObservationExposure");
    return response.data;
  } catch (error) {
    console.error("Failed to get observation exposure:", error);
    throw error;
  }
};

/**
 * Set observation camera gain.
 * 
 * @param {number} gain - Gain value
 * @returns {Promise<Object>} Response data
 */
export const apiLightsheetControllerSetObservationGain = async (gain) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/setObservationGain", {
      params: { gain }
    });
    return response.data;
  } catch (error) {
    console.error("Failed to set observation gain:", error);
    throw error;
  }
};

/**
 * Get observation camera gain.
 * 
 * @returns {Promise<number>} Gain value
 */
export const apiLightsheetControllerGetObservationGain = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/getObservationGain");
    return response.data;
  } catch (error) {
    console.error("Failed to get observation gain:", error);
    throw error;
  }
};

/**
 * Set stream image transformation parameters (flip/rotate).
 * 
 * @param {Object} params - Transformation parameters
 * @param {boolean} params.flipX - Flip horizontally
 * @param {boolean} params.flipY - Flip vertically
 * @param {number} params.rotation - Rotation angle (0, 90, 180, 270)
 * @returns {Promise<Object>} Response data
 */
export const apiLightsheetControllerSetStreamTransform = async ({ flipX = false, flipY = false, rotation = 0 }) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LightsheetController/setObservationStreamTransform", {
      params: { flip_x: flipX, flip_y: flipY, rotation }
    });
    return response.data;
  } catch (error) {
    console.error("Failed to set stream transform:", error);
    throw error;
  }
};

export default {
  apiLightsheetControllerObservationStreamControl,
  apiLightsheetControllerSetObservationExposure,
  apiLightsheetControllerGetObservationExposure,
  apiLightsheetControllerSetObservationGain,
  apiLightsheetControllerGetObservationGain,
  apiLightsheetControllerSetStreamTransform
};
