// src/backendapi/apiInLineHoloControllerPauseProcessing.js
// Pause inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Pause inline hologram processing
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerPauseProcessing = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/InLineHoloController/pause_processing_inlineholo");
    return response.data;
  } catch (error) {
    console.error("Error pausing inline holo processing:", error);
    throw error;
  }
};

export default apiInLineHoloControllerPauseProcessing;
