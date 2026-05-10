import createAxiosInstance from "./createAxiosInstance";

/**
 * Load the persisted overview-registration config for editing.
 * GET /ExperimentController/getOverviewRegistrationConfigData
 */
const apiGetOverviewRegistrationConfigData = async (
  cameraName = "",
  layoutName = "Heidstar 4x Histosample"
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/ExperimentController/getOverviewRegistrationConfigData",
    {
      params: { camera_name: cameraName, layout_name: layoutName },
    }
  );
  return response.data;
};

export default apiGetOverviewRegistrationConfigData;
