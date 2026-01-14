// src/backendapi/apiUC2ConfigControllerGetOTAWiFiCredentials.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetOTAWiFiCredentials = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/UC2ConfigController/getOTAWiFiCredentials");
    return response.data; // { ssid: "...", password: "..." }
  } catch (error) {
    console.error("Error fetching OTA WiFi credentials:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerGetOTAWiFiCredentials;
