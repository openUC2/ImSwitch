// src/backendapi/apiCompositeControllerStart.js
// Start composite acquisition

import createAxiosInstance from "./createAxiosInstance";

/**
 * Start composite acquisition
 * Begins the acquisition loop that cycles through illumination steps,
 * captures frames, fuses them into composite, and streams results.
 * 
 * @returns {Promise<Object>} Current state dictionary
 * 
 * @example
 * const state = await apiCompositeControllerStart();
 * // Returns: { is_running: true, is_streaming: false, ... }
 */
const apiCompositeControllerStart = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.post("/CompositeController/start_composite");
    return response.data;
  } catch (error) {
    console.error("Error starting composite acquisition:", error);
    throw error;
  }
};

export default apiCompositeControllerStart;
