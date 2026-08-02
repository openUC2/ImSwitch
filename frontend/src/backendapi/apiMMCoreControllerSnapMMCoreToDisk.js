// src/backendapi/apiMMCoreControllerSnapMMCoreToDisk.js
import createAxiosInstance from "./createAxiosInstance";

const apiMMCoreControllerSnapMMCoreToDisk = async ({
  detectorName,
  exposureMs,
  fileName,
  saveFormat = "tiff",
}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/MMCoreController/snapMMCoreToDisk",
    { detectorName, exposureMs, fileName, saveFormat },
    { headers: { "Content-Type": "application/json" } },
  );
  return response.data;
};

export default apiMMCoreControllerSnapMMCoreToDisk;
