// src/backendapi/apiCompositeControllerSetParameters.js
// Update composite acquisition parameters

import createAxiosInstance from "./createAxiosInstance";

/**
 * Update composite acquisition parameters
 * @param {Object} params - Parameters to update (partial update supported)
 * @param {Array<Object>} [params.steps] - Illumination steps array
 * @param {Object} [params.mapping] - RGB channel mapping { R: string, G: string, B: string }
 * @param {number} [params.fps_target] - Target FPS (0.1-30.0)
 * @param {number} [params.jpeg_quality] - JPEG quality (1-100)
 * @param {boolean} [params.normalize_channels] - Normalize channels before fusion
 * @param {boolean} [params.auto_exposure] - Use per-step exposure overrides
 * @returns {Promise<Object>} Updated parameters
 * 
 * @example
 * const updatedParams = await apiCompositeControllerSetParameters({
 *   steps: [
 *     { illumination: "laser488", intensity: 0.5, settle_ms: 20 },
 *     { illumination: "LED_white", intensity: 1.0 }
 *   ],
 *   mapping: { R: "", G: "laser488", B: "LED_white" },
 *   fps_target: 3.0
 * });
 */
const apiCompositeControllerSetParameters = async (params) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.post("/CompositeController/set_parameters_composite", params);
    return response.data;
  } catch (error) {
    console.error("Error setting composite parameters:", error);
    throw error;
  }
};

export default apiCompositeControllerSetParameters;
