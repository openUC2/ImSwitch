// src/backendapi/apiCompositeControllerGetParameters.js
// Get current composite acquisition parameters

import createAxiosInstance from "./createAxiosInstance";

/**
 * Get current composite acquisition parameters
 * @returns {Promise<Object>} Parameters object with steps, mapping, fps_target, etc.
 * 
 * @example
 * const params = await apiCompositeControllerGetParameters();
 * // Returns:
 * // {
 * //   steps: [
 * //     { illumination: "laser488", intensity: 0.3, exposure_ms: 50, settle_ms: 10, enabled: true },
 * //     { illumination: "laser635", intensity: 0.2, exposure_ms: 80, settle_ms: 10, enabled: true }
 * //   ],
 * //   mapping: { R: "laser635", G: "laser488", B: "" },
 * //   fps_target: 5.0,
 * //   jpeg_quality: 85,
 * //   normalize_channels: true,
 * //   auto_exposure: false
 * // }
 */
const apiCompositeControllerGetParameters = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get("/CompositeController/get_parameters_composite");
    return response.data;
  } catch (error) {
    console.error("Error getting composite parameters:", error);
    throw error;
  }
};

export default apiCompositeControllerGetParameters;
