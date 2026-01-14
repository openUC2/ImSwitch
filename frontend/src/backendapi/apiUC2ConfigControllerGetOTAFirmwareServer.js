// src/backendapi/apiUC2ConfigControllerGetOTAFirmwareServer.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetOTAFirmwareServer = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/UC2ConfigController/getOTAFirmwareServer");
    return response.data; // { firmware_server_url: "http://localhost:9000" }
  } catch (error) {
    console.error("Error fetching OTA firmware server:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerGetOTAFirmwareServer;
