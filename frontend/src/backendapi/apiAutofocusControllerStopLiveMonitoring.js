// src/backendapi/apiAutofocusControllerStopLiveMonitoring.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Stop live autofocus monitoring mode
 * @returns {Promise<Object>} Response containing status
 */
const apiAutofocusControllerStopLiveMonitoring = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    
    const response = await axiosInstance.get("/AutofocusController/stopLiveMonitoring");
    
    return response.data;
  } catch (error) {
    console.error("Failed to stop live autofocus monitoring:", error);
    throw error;
  }
};

export default apiAutofocusControllerStopLiveMonitoring;
