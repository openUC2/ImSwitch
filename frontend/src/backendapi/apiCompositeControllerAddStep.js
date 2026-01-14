// src/backendapi/apiCompositeControllerAddStep.js
// Add a new illumination step to composite sequence

import createAxiosInstance from "./createAxiosInstance";

/**
 * Add a new illumination step to the composite acquisition sequence
 * @param {Object} stepConfig - Step configuration
 * @param {string} stepConfig.illumination - Name of illumination source
 * @param {number} [stepConfig.intensity=0.5] - Intensity value (0.0-1.0 or device units)
 * @param {number} [stepConfig.exposure_ms] - Optional exposure override in ms
 * @param {number} [stepConfig.settle_ms=10.0] - Settle time in ms after illumination change
 * @returns {Promise<Object>} Updated parameters dictionary
 * 
 * @example
 * const params = await apiCompositeControllerAddStep({
 *   illumination: "laser488",
 *   intensity: 0.5,
 *   settle_ms: 20
 * });
 */
const apiCompositeControllerAddStep = async ({ illumination, intensity = 0.5, exposure_ms = null, settle_ms = 10.0 }) => {
  const instance = createAxiosInstance();
  try {
    const params = {
      illumination,
      intensity,
      settle_ms
    };
    if (exposure_ms !== null) {
      params.exposure_ms = exposure_ms;
    }
    const response = await instance.post("/CompositeController/add_step_composite", null, { params });
    return response.data;
  } catch (error) {
    console.error("Error adding composite step:", error);
    throw error;
  }
};

export default apiCompositeControllerAddStep;
