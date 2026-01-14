// src/backendapi/apiAutofocusControllerStartLiveMonitoring.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Start live autofocus monitoring mode
 * @param {number} period - Update period in seconds (default 0.5s)
 * @param {string} method - Focus measurement method ("LAPE", "GLVA", or "JPEG")
 * @param {number} nCropsize - Crop size for focus calculation (default 2048)
 * @returns {Promise<Object>} Response containing status and configuration
 */
const apiAutofocusControllerStartLiveMonitoring = async (period = 0.5, method = "LAPE", nCropsize = 2048) => {
  try {
    const axiosInstance = createAxiosInstance();
    
    const response = await axiosInstance.get("/AutofocusController/startLiveMonitoring", {
      params: {
        period,
        method,
        nCropsize
      }
    });
    
    return response.data;
  } catch (error) {
    console.error("Failed to start live autofocus monitoring:", error);
    throw error;
  }
};

export default apiAutofocusControllerStartLiveMonitoring;
