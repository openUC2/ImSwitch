// src/backendapi/apiUC2ConfigControllerSetOTAWiFiCredentials.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerSetOTAWiFiCredentials = async (ssid, password) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/UC2ConfigController/setOTAWiFiCredentials", { 
      params: {
      ssid,
      password,
  }});
    return response.data; // { status: "success", message: "..." }
  } catch (error) {
    console.error("Error setting OTA WiFi credentials:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerSetOTAWiFiCredentials;
