// src/backendapi/apiCompositeControllerSetMapping.js
// Set RGB channel mapping for composite acquisition

import createAxiosInstance from "./createAxiosInstance";

/**
 * Set RGB channel mapping
 * Maps illumination sources to RGB output channels
 * 
 * @param {Object} mapping - Channel mapping configuration
 * @param {string} [mapping.r_source=""] - Illumination source for Red channel
 * @param {string} [mapping.g_source=""] - Illumination source for Green channel
 * @param {string} [mapping.b_source=""] - Illumination source for Blue channel
 * @returns {Promise<Object>} Updated parameters dictionary
 * 
 * @example
 * const params = await apiCompositeControllerSetMapping({
 *   r_source: "laser635",  // Red fluorescence
 *   g_source: "laser488",  // Green fluorescence
 *   b_source: "LED_white"  // Transmitted light
 * });
 */
const apiCompositeControllerSetMapping = async ({ r_source = "", g_source = "", b_source = "" }) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.post("/CompositeController/set_mapping_composite", null, {
      params: { r_source, g_source, b_source }
    });
    return response.data;
  } catch (error) {
    console.error("Error setting composite mapping:", error);
    throw error;
  }
};

export default apiCompositeControllerSetMapping;
