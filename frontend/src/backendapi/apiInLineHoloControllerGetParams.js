// src/backendapi/apiInLineHoloControllerGetParams.js
// Get current inline hologram processing parameters from backend

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get current inline hologram processing parameters
 * @returns {Promise<Object>} Promise containing the parameters object
 */
const apiInLineHoloControllerGetParams = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/InLineHoloController/get_parameters_inlineholo");
    return response.data;
  } catch (error) {
    console.error("Error getting inline holo parameters:", error);
    throw error;
  }
};

export default apiInLineHoloControllerGetParams;
