// src/backendapi/apiInLineHoloControllerSetWavelength.js
// Set wavelength for inline hologram processing

import createAxiosInstance from "./createAxiosInstance";

/**
 * Set wavelength in meters
 * @param {number} wavelength - Wavelength in meters
 * @returns {Promise<Object>} Promise containing success status
 */
const apiInLineHoloControllerSetWavelength = async (wavelength) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(`/InLineHoloController/set_wavelength_inlineholo?wavelength=${wavelength}`);
    return response.data;
  } catch (error) {
    console.error("Error setting wavelength:", error);
    throw error;
  }
};

export default apiInLineHoloControllerSetWavelength;
