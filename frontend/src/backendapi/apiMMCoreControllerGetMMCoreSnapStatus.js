// src/backendapi/apiMMCoreControllerGetMMCoreSnapStatus.js
import createAxiosInstance from "./createAxiosInstance";

const apiMMCoreControllerGetMMCoreSnapStatus = async (jobId) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/MMCoreController/getMMCoreSnapStatus",
    { params: { jobId } },
  );
  return response.data;
};

export default apiMMCoreControllerGetMMCoreSnapStatus;
