// src/backendapi/apiInLineHoloControllerSetParams.js
// Set inline hologram processing parameters

import createAxiosInstance from "./createAxiosInstance";

/**
 * Set inline hologram processing parameters
 * @param {Object} params - Parameters object
 * @param {number} params.pixelsize - Pixel size in meters
 * @param {number} params.wavelength - Wavelength in meters
 * @param {number} params.na - Numerical aperture
 * @param {number} params.dz - Propagation distance in meters
 * @param {Array<number>} params.roi_center - ROI center [x, y]
 * @param {number} params.roi_size - ROI size (square)
 * @param {string} params.color_channel - Color channel ("red", "green", "blue")
 * @param {boolean} params.flip_x - Flip horizontally
 * @param {boolean} params.flip_y - Flip vertically
 * @param {number} params.rotation - Rotation angle (0, 90, 180, 270)
 * @param {number} params.update_freq - Update frequency in Hz
 * @param {number} params.binning - Binning factor
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerSetParams = async (params) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.post("/InLineHoloController/set_parameters_inlineholo", params);
    return response.data;
  } catch (error) {
    console.error("Error setting inline holo parameters:", error);
    throw error;
  }
};

export default apiInLineHoloControllerSetParams;
