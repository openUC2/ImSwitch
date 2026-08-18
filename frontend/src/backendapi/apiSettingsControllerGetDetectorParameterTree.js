import createAxiosInstance from "./createAxiosInstance";

// Full camera parameter tree of a detector (hardware info + grouped parameters),
// used by the advanced camera settings dialog.
const apiSettingsControllerGetDetectorParameterTree = async (
  detectorName = null,
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/SettingsController/getDetectorParameterTree",
    { params: detectorName ? { detectorName } : {} },
  );
  return response.data;
};

export default apiSettingsControllerGetDetectorParameterTree;
