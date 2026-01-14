// src/backendapi/apiInLineHoloControllerSetRoi.js
// Set ROI parameters for inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Set ROI parameters
 * @param {Object} params - ROI parameters
 * @param {number} params.center_x - ROI center X in pixels (optional)
 * @param {number} params.center_y - ROI center Y in pixels (optional)
 * @param {number} params.size - ROI size (square) in pixels (optional, default 256)
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerSetRoi = async (params) => {
  const instance = createAxiosInstance();
  try {
    const queryParams = new URLSearchParams();
    if (params.center_x !== undefined) queryParams.append('center_x', params.center_x);
    if (params.center_y !== undefined) queryParams.append('center_y', params.center_y);
    if (params.size !== undefined) queryParams.append('size', params.size);
    
    const response = await instance.get(`/InLineHoloController/set_roi_inlineholo?${queryParams.toString()}`);
    return response.data;
  } catch (error) {
    console.error("Error setting inline holo ROI:", error);
    throw error;
  }
};

export default apiInLineHoloControllerSetRoi;
