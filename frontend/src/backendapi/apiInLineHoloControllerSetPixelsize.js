// src/backendapi/apiInLineHoloControllerSetPixelsize.js
// Set pixel size for inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Set pixel size in meters
 * @param {number} pixelsize - Pixel size in meters
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerSetPixelsize = async (pixelsize) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(`/InLineHoloController/set_pixelsize_inlineholo?pixelsize=${pixelsize}`);
    return response.data;
  } catch (error) {
    console.error("Error setting pixelsize:", error);
    throw error;
  }
};

export default apiInLineHoloControllerSetPixelsize;
