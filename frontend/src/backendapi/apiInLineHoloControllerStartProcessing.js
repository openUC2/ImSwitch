// src/backendapi/apiInLineHoloControllerStartProcessing.js
// Start inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Start inline hologram processing
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerStartProcessing = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/InLineHoloController/start_processing_inlineholo");
    return response.data;
  } catch (error) {
    console.error("Error starting inline holo processing:", error);
    throw error;
  }
};

export default apiInLineHoloControllerStartProcessing;
