import createAxiosInstance from "./createAxiosInstance";

/**
 * Get overview registration wizard config with slot definitions.
 * GET /ExperimentController/getOverviewRegistrationConfig
 */
const apiGetOverviewRegistrationConfig = async (
  layoutName = "Heidstar 4x Histosample"
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/ExperimentController/getOverviewRegistrationConfig",
    {
      params: { layout_name: layoutName },
    }
  );
  return response.data;
};

export default apiGetOverviewRegistrationConfig;
