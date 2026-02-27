import createAxiosInstance from "./createAxiosInstance";

/**
 * Get overview registration wizard config with slot definitions.
 * POST /ExperimentController/getOverviewRegistrationConfig
 *
 * Sends the frontend's current wellLayout (with offsets applied) so
 * that slot corners returned by the backend match the canvas exactly.
 *
 * @param {object|null} layoutData - Full wellLayout object from Redux (preferred)
 * @param {string}      layoutName - Fallback layout name if layoutData is null
 */
const apiGetOverviewRegistrationConfig = async (
  layoutData = null,
  layoutName = "Heidstar 4x Histosample"
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/ExperimentController/getOverviewRegistrationConfig",
    {
      layout_data: layoutData,
      layout_name: layoutName,
    }
  );
  return response.data;
};

export default apiGetOverviewRegistrationConfig;
