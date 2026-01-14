// src/backendapi/apiAutofocusControllerGetLiveMonitoringStatus.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Get current status of live autofocus monitoring
 * @returns {Promise<Object>} Response containing monitoring status and configuration
 */
const apiAutofocusControllerGetLiveMonitoringStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    
    const response = await axiosInstance.get("/AutofocusController/getLiveMonitoringStatus");
    
    return response.data;
  } catch (error) {
    console.error("Failed to get live monitoring status:", error);
    throw error;
  }
};

export default apiAutofocusControllerGetLiveMonitoringStatus;
