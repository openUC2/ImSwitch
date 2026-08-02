import createAxiosInstance from "./createAxiosInstance";

/**
 * Run the autonomous overview scan: visit each registered slot,
 * snap, re-warp and return the full overlay payload.
 * POST /ExperimentController/runAutonomousOverviewScan
 */
const apiRunAutonomousOverviewScan = async (
  cameraName = "",
  layoutName = "Heidstar 4x Histosample",
  settleTimeS = 0.5
) => {
  const axiosInstance = createAxiosInstance();
  const params = [];
  if (cameraName) params.push(`camera_name=${encodeURIComponent(cameraName)}`);
  if (layoutName) params.push(`layout_name=${encodeURIComponent(layoutName)}`);
  if (settleTimeS != null)
    params.push(`settle_time_s=${encodeURIComponent(settleTimeS)}`);
  const url =
    "/ExperimentController/runAutonomousOverviewScan?" + params.join("&");
  const response = await axiosInstance.post(url);
  return response.data;
};

export default apiRunAutonomousOverviewScan;
