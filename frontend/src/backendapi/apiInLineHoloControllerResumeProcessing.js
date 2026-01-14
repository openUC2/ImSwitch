// src/backendapi/apiInLineHoloControllerResumeProcessing.js
// Resume inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Resume inline hologram processing
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerResumeProcessing = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/InLineHoloController/resume_processing_inlineholo");
    return response.data;
  } catch (error) {
    console.error("Error resuming inline holo processing:", error);
    throw error;
  }
};

export default apiInLineHoloControllerResumeProcessing;
