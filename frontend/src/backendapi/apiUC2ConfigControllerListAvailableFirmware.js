// src/backendapi/apiUC2ConfigControllerListAvailableFirmware.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerListAvailableFirmware = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/UC2ConfigController/listAvailableFirmware");
    return response.data; // { status: "success", firmware_server: "...", firmware_count: X, firmware: {...} }
  } catch (error) {
    console.error("Error listing available firmware:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerListAvailableFirmware;
