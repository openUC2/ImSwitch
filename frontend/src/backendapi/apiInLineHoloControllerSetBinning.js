// src/backendapi/apiInLineHoloControllerSetBinning.js
// Set binning factor for inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Set binning factor (1, 2, 4, etc.)
 * Note: Pixel size in reconstruction kernel is automatically adjusted
 * @param {number} binning - Binning factor
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerSetBinning = async (binning) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(`/InLineHoloController/set_binning_inlineholo?binning=${binning}`);
    return response.data;
  } catch (error) {
    console.error("Error setting binning:", error);
    throw error;
  }
};

export default apiInLineHoloControllerSetBinning;
