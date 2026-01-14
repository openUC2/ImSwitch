// src/backendapi/apiUC2ConfigControllerStartSingleDeviceOTA.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerStartSingleDeviceOTA = async (canId, ssid = null, password = null, timeout = 300000) => {
  try {
    const axiosInstance = createAxiosInstance();
    const payload = { can_id: canId, timeout };
    if (ssid) payload.ssid = ssid;
    if (password) payload.password = password;
    
    const response = await axiosInstance.post("/UC2ConfigController/startSingleDeviceOTA", payload);
    return response.data; // { status: "success", message: "...", can_id: X, command_response: {...} }
  } catch (error) {
    console.error("Error starting single device OTA:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerStartSingleDeviceOTA;
