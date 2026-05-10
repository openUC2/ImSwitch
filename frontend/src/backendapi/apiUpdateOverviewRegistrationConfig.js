import createAxiosInstance from "./createAxiosInstance";

/**
 * Persist edits to the overview-registration config (XYZ positions etc).
 * POST /ExperimentController/updateOverviewRegistrationConfig
 */
const apiUpdateOverviewRegistrationConfig = async (configData) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/ExperimentController/updateOverviewRegistrationConfig",
    configData
  );
  return response.data;
};

export default apiUpdateOverviewRegistrationConfig;
