import createAxiosInstance from "./createAxiosInstance";

/**
 * Get all overlay data for WellSelector canvas rendering.
 * GET /ExperimentController/getOverviewOverlayData
 */
const apiGetOverviewOverlayData = async (
  cameraName = "",
  layoutName = "Heidstar 4x Histosample"
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/ExperimentController/getOverviewOverlayData",
    {
      params: { camera_name: cameraName, layout_name: layoutName },
    }
  );
  return response.data;
};

export default apiGetOverviewOverlayData;
