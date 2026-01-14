// src/backendapi/apiInLineHoloControllerStopProcessing.js
// Stop inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Stop inline hologram processing
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerStopProcessing = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/InLineHoloController/stop_processing_inlineholo");
    return response.data;
  } catch (error) {
    console.error("Error stopping inline holo processing:", error);
    throw error;
  }
};

export default apiInLineHoloControllerStopProcessing;
