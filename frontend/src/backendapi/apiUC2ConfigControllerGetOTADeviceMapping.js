// src/backendapi/apiUC2ConfigControllerGetOTADeviceMapping.js
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetOTADeviceMapping = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/UC2ConfigController/getOTADeviceMapping");
    return response.data; // CAN_ADDRESS_MAP object
  } catch (error) {
    console.error("Error getting OTA device mapping:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerGetOTADeviceMapping;
