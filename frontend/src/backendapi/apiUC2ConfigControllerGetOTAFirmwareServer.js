// src/backendapi/apiUC2ConfigControllerGetOTAFirmwareServer.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetOTAFirmwareServer = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/UC2ConfigController/getOTAFirmwareServer");
    return response.data; // { source: "local"|"github", firmware_server_url, github_repo, github_release }
  } catch (error) {
    console.error("Error fetching OTA firmware server:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerGetOTAFirmwareServer;
