// src/backendapi/apiMMCoreControllerGetMMCoreTemperature.js
import createAxiosInstance from "./createAxiosInstance";

const apiMMCoreControllerGetMMCoreTemperature = async (detectorName) => {
  const axiosInstance = createAxiosInstance();
  const params = detectorName ? { detectorName } : {};
  const response = await axiosInstance.get(
    "/MMCoreController/getMMCoreTemperature",
    { params },
  );
  return response.data;
};

export default apiMMCoreControllerGetMMCoreTemperature;
