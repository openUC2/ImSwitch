// src/backendapi/apiUC2ConfigControllerScanCanbus.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerScanCanbus = async (timeout = 5, probeRange = false) => {
  try {
    const axiosInstance = createAxiosInstance(timeout + 5);
    const response = await axiosInstance.get("/UC2ConfigController/scan_canbus", {
      params: {
        timeout,
        probe_range: probeRange,
      },
    });
    // { master: {...}, scan: [{ canId, deviceTypeStr, statusStr, build, fwVersion, mac }], detected_ids, count }
    return response.data;
  } catch (error) {
    console.error("Error scanning CAN bus:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerScanCanbus;
