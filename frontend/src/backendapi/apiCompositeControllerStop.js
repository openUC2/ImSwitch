// src/backendapi/apiCompositeControllerStop.js
// Stop composite acquisition

import createAxiosInstance from "./createAxiosInstance";

/**
 * Stop composite acquisition
 * Stops the acquisition loop and turns off all illumination.
 * 
 * @returns {Promise<Object>} Current state dictionary
 * 
 * @example
 * const state = await apiCompositeControllerStop();
 * // Returns: { is_running: false, is_streaming: false, ... }
 */
const apiCompositeControllerStop = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.post("/CompositeController/stop_composite");
    return response.data;
  } catch (error) {
    console.error("Error stopping composite acquisition:", error);
    throw error;
  }
};

export default apiCompositeControllerStop;
