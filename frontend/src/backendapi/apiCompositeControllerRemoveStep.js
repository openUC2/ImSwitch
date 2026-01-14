// src/backendapi/apiCompositeControllerRemoveStep.js
// Remove an illumination step from composite sequence

import createAxiosInstance from "./createAxiosInstance";

/**
 * Remove an illumination step by index
 * @param {number} index - Index of step to remove (0-based)
 * @returns {Promise<Object>} Updated parameters dictionary
 * 
 * @example
 * const params = await apiCompositeControllerRemoveStep(0);
 * // Removes the first step
 */
const apiCompositeControllerRemoveStep = async (index) => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.post("/CompositeController/remove_step_composite", null, {
      params: { index }
    });
    return response.data;
  } catch (error) {
    console.error("Error removing composite step:", error);
    throw error;
  }
};

export default apiCompositeControllerRemoveStep;
