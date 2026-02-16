// src/backendapi/apiUC2ConfigControllerStartCANStreamingOTA.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Start CAN-based OTA streaming upload (USB-based, no WiFi required).
 * 
 * This method uses the CAN bus streaming protocol to upload firmware directly
 * from the master device via USB->CAN, without requiring WiFi connectivity.
 * 
 * @param {number} canId - CAN ID of the target device (e.g., 11=Motor X, 20=Laser)
 * @param {string} firmwareUrl - Optional URL to download firmware
 * @returns {Promise<Object>} - Status response
 */
const apiUC2ConfigControllerStartCANStreamingOTA = async (canId, firmwareUrl = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    
    const params = { can_id: canId };
    if (firmwareUrl) {
      params.firmware_url = firmwareUrl;
    }
    
    const response = await axiosInstance.get("/UC2ConfigController/startCANStreamingOTA", { params });
    return response.data;
  } catch (error) {
    console.error("Error starting CAN streaming OTA:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerStartCANStreamingOTA;
