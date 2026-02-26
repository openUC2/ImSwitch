// src/backendapi/apiExperimentControllerInterruptFocusMap.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Interrupt an ongoing focus map computation.
 *
 * @returns {Promise<Object>} Status message
 */
const apiExperimentControllerInterruptFocusMap = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.post(
      "/ExperimentController/interruptFocusMap"
    );
    return response.data;
  } catch (error) {
    console.error("Error interrupting focus map:", error);
    throw error;
  }
};

export default apiExperimentControllerInterruptFocusMap;
