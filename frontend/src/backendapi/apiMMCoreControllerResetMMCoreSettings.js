// src/backendapi/apiMMCoreControllerResetMMCoreSettings.js
import createAxiosInstance from "./createAxiosInstance";

// Reset the camera to factory defaults and drop any persisted settings.
// Returns the refreshed parameter tree.
const apiMMCoreControllerResetMMCoreSettings = async ({
  detectorName,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/MMCoreController/resetMMCoreSettings",
    { detectorName },
    { headers: { "Content-Type": "application/json" } },
  );
  return response.data;
};

export default apiMMCoreControllerResetMMCoreSettings;
