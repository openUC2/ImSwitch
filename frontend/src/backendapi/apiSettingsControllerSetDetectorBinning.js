import createAxiosInstance from "./createAxiosInstance";

// Applies a binning factor. The backend answers with the resulting binning and
// frame shape, since binning changes the delivered image size.
const apiSettingsControllerSetDetectorBinning = async ({
  binning,
  detectorName = null,
}) => {
  const axiosInstance = createAxiosInstance();
  const params = { binning };
  if (detectorName) params.detectorName = detectorName;
  const response = await axiosInstance.get(
    "/SettingsController/setDetectorBinning",
    { params },
  );
  return response.data;
};

export default apiSettingsControllerSetDetectorBinning;
