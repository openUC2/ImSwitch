// src/backendapi/apiMMCoreControllerGetMMCoreDetectors.js
import createAxiosInstance from "./createAxiosInstance";

const apiMMCoreControllerGetMMCoreDetectors = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/MMCoreController/getMMCoreDetectors");
  return response.data;
};

export default apiMMCoreControllerGetMMCoreDetectors;
