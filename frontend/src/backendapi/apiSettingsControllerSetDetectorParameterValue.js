import createAxiosInstance from "./createAxiosInstance";

// Sets a single detector parameter (any JSON type) and returns the refreshed
// parameter tree as the detector sees it after the change.
const apiSettingsControllerSetDetectorParameterValue = async ({
  detectorName = null,
  name,
  value,
}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/SettingsController/setDetectorParameterValue",
    { detectorName, name, value },
    { headers: { "Content-Type": "application/json" } },
  );
  return response.data;
};

export default apiSettingsControllerSetDetectorParameterValue;
