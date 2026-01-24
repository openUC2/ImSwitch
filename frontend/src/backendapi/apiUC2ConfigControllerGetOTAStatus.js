// src/backendapi/apiUC2ConfigControllerGetOTAStatus.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetOTAStatus = async (canId = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = canId !== null ? { can_id: canId } : {};
    const response = await axiosInstance.get("/UC2ConfigController/getOTAStatus", { params });
    return response.data; // { status: "...", ota_status: {...} } or { status: "...", all_ota_status: {...} }
  } catch (error) {
    console.error("Error getting OTA status:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerGetOTAStatus;
