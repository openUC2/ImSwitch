import createAxiosInstance from "./createAxiosInstance";

/**
 * Get registration status for all slides.
 * GET /ExperimentController/getOverviewRegistrationStatus
 */
const apiGetOverviewRegistrationStatus = async (
  cameraName = "",
  layoutName = "Heidstar 4x Histosample"
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/ExperimentController/getOverviewRegistrationStatus",
    {
      params: { camera_name: cameraName, layout_name: layoutName },
    }
  );
  return response.data;
};

export default apiGetOverviewRegistrationStatus;
