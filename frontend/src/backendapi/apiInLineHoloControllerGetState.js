// src/backendapi/apiInLineHoloControllerGetState.js
// Get current inline hologram processing state from backend

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get current inline hologram processing state
 * @returns {Promise<Object>} Promise containing the state object with processing status
 */
const apiInLineHoloControllerGetState = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/InLineHoloController/get_state_inlineholo");
    return response.data;
  } catch (error) {
    console.error("Error getting inline holo state:", error);
    throw error;
  }
};

export default apiInLineHoloControllerGetState;
