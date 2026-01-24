// src/backendapi/apiUC2ConfigControllerScanCanbus.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerScanCanbus = async (timeout = 5) => {
  try {
    const axiosInstance = createAxiosInstance(timeout + 5);
    const response = await axiosInstance.get("/UC2ConfigController/scan_canbus", {
      params: {
        timeout,
      },
    });
    return response.data; // Array of devices: [{ canId, deviceType, status, deviceTypeStr, statusStr }]
  } catch (error) {
    console.error("Error scanning CAN bus:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerScanCanbus;
