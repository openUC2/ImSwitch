// src/backendapi/apiCompositeControllerGetState.js
// Get current composite acquisition state

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get current composite acquisition state
 * @returns {Promise<Object>} State object with running status and statistics
 * 
 * @example
 * const state = await apiCompositeControllerGetState();
 * // Returns:
 * // {
 * //   is_running: true,
 * //   is_streaming: true,
 * //   current_step: 1,
 * //   cycle_count: 42,
 * //   last_cycle_time_ms: 180.5,
 * //   average_fps: 4.8,
 * //   error_message: ""
 * // }
 */
const apiCompositeControllerGetState = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/CompositeController/get_state_composite");
    return response.data;
  } catch (error) {
    console.error("Error getting composite state:", error);
    throw error;
  }
};

export default apiCompositeControllerGetState;
