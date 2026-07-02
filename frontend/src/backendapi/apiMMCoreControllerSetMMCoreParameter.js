// src/backendapi/apiMMCoreControllerSetMMCoreParameter.js
import createAxiosInstance from "./createAxiosInstance";

const apiMMCoreControllerSetMMCoreParameter = async ({
  detectorName,
  name,
  value,
}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/MMCoreController/setMMCoreParameter",
    { detectorName, name, value },
    { headers: { "Content-Type": "application/json" } },
  );
  return response.data;
};

export default apiMMCoreControllerSetMMCoreParameter;
