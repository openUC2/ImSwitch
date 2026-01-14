// src/backendapi/apiUC2ConfigControllerClearOTAStatus.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerClearOTAStatus = async (canId = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const payload = canId !== null ? { can_id: canId } : {};
    const response = await axiosInstance.post("/UC2ConfigController/clearOTAStatus", payload);
    return response.data; // { status: "success", message: "..." }
  } catch (error) {
    console.error("Error clearing OTA status:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerClearOTAStatus;
