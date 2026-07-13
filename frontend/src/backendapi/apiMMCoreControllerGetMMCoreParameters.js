// src/backendapi/apiMMCoreControllerGetMMCoreParameters.js
import createAxiosInstance from "./createAxiosInstance";

const apiMMCoreControllerGetMMCoreParameters = async (detectorName) => {
  const axiosInstance = createAxiosInstance();
  const params = detectorName ? { detectorName } : {};
  const response = await axiosInstance.get(
    "/MMCoreController/getMMCoreParameters",
    { params },
  );
  return response.data;
};

export default apiMMCoreControllerGetMMCoreParameters;
