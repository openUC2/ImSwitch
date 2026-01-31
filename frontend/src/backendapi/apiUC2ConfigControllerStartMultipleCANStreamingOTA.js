// src/backendapi/apiUC2ConfigControllerStartMultipleCANStreamingOTA.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Start CAN streaming OTA for multiple devices sequentially.
 * 
 * This method uses the CAN bus streaming protocol to upload firmware directly
 * from the master device via USB->CAN, without requiring WiFi connectivity.
 * 
 * @param {Array<number>} canIds - List of CAN IDs (e.g., [11, 12, 13, 20, 30])
 * @param {number} delayBetween - Delay in seconds between devices (for reboot time)
 * @returns {Promise<Object>} - Status response with results for each device
 */
const apiUC2ConfigControllerStartMultipleCANStreamingOTA = async (canIds, delayBetween = 5) => {
  try {
    const axiosInstance = createAxiosInstance();
    
    const params = {
      delay_between: delayBetween
    };
    
    // Send can_ids as JSON array in the request body
    const body = Array.isArray(canIds) ? canIds : [canIds];
    
    const response = await axiosInstance.post("/UC2ConfigController/startMultipleCANStreamingOTA", body, { params });
    return response.data;
  } catch (error) {
    console.error("Error starting multiple CAN streaming OTA:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerStartMultipleCANStreamingOTA;
