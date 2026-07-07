// src/backendapi/apiMMCoreControllerSaveMMCoreSettings.js
import createAxiosInstance from "./createAxiosInstance";

// Persist the camera's current editable parameters (or an explicit `values`
// map) to the setup JSON so they are re-applied on the next startup.
const apiMMCoreControllerSaveMMCoreSettings = async ({
  detectorName,
  values,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/MMCoreController/saveMMCoreSettings",
    { detectorName, values: values ?? null },
    { headers: { "Content-Type": "application/json" } },
  );
  return response.data;
};

export default apiMMCoreControllerSaveMMCoreSettings;
