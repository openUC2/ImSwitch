// src/backendapi/apiUC2ConfigControllerStartMultipleDeviceOTA.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerStartMultipleDeviceOTA = async (canIds, ssid = null, password = null, timeout = 300000, delayBetween = 2) => {
  try {
    const axiosInstance = createAxiosInstance();
    
    // Backend expects POST with can_ids array in body and other params as query params
    const params = { 
      timeout
    };
    if (delayBetween) params.delay_between = delayBetween;
    if (ssid) params.ssid = ssid;
    if (password) params.password = password;
    
    // Send can_ids as JSON array in the request body
    const body = Array.isArray(canIds) ? canIds : [canIds];
    
    /*
    curl -X 'POST' \
      'http://localhost/imswitch/api/UC2ConfigController/startMultipleDeviceOTA?timeout=300000' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '[10, 20]'
    */
    const response = await axiosInstance.post("/UC2ConfigController/startMultipleDeviceOTA", body, { params });
    return response.data; // { status: "success", message: "...", results: [...] }
  } catch (error) {
    console.error("Error starting multiple device OTA:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerStartMultipleDeviceOTA;
