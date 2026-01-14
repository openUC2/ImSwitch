// src/backendapi/apiInLineHoloControllerSetDz.js
// Set propagation distance (dz) for inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Set propagation distance in meters
 * @param {number} dz - Propagation distance in meters
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerSetDz = async (dz) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(`/InLineHoloController/set_dz_inlineholo?dz=${dz}`);
    return response.data;
  } catch (error) {
    console.error("Error setting dz:", error);
    throw error;
  }
};

export default apiInLineHoloControllerSetDz;
