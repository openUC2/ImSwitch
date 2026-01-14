// src/backendapi/apiUC2ConfigControllerSetOTAFirmwareServer.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerSetOTAFirmwareServer = async (serverUrl) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/UC2ConfigController/setOTAFirmwareServer", {
      params: {
        server_url: serverUrl,
      },
    });
    return response.data; // { status: "success", firmware_count: X, firmware_files: [...] }
  } catch (error) {
    console.error("Error setting OTA firmware server:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerSetOTAFirmwareServer;
