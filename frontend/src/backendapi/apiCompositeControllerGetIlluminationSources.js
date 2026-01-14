// src/backendapi/apiCompositeControllerGetIlluminationSources.js
// Get all available illumination sources for composite acquisition

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get list of all available illumination sources (lasers + LEDs)
 * @returns {Promise<Array<string>>} Array of illumination source names
 * 
 * @example
 * const sources = await apiCompositeControllerGetIlluminationSources();
 * // Returns: ["laser488", "laser635", "LED_white", "LED_blue"]
 */
const apiCompositeControllerGetIlluminationSources = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/CompositeController/get_illumination_sources_composite");
    return response.data;
  } catch (error) {
    console.error("Error getting illumination sources:", error);
    throw error;
  }
};

export default apiCompositeControllerGetIlluminationSources;
