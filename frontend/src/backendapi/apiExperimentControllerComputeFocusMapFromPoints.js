import createAxiosInstance from "./createAxiosInstance";

/**
 * Compute a focus map from manually specified XYZ reference points.
 *
 * @param {Object} params
 * @param {Array<{x: number, y: number, z: number}>} params.points - Reference points
 * @param {string} [params.group_id="manual"] - Group identifier
 * @param {string} [params.group_name="Manual Points"] - Display name
 * @param {string} [params.method="rbf"] - Fit method
 * @param {number} [params.smoothing_factor=0.1]
 * @param {number} [params.z_offset=0]
 * @param {boolean} [params.clamp_enabled=false]
 * @param {number} [params.z_min=0]
 * @param {number} [params.z_max=0]
 * @returns {Promise<Object>} Focus map result
 */
const apiExperimentControllerComputeFocusMapFromPoints = async (params) => {
  const axiosInstance = createAxiosInstance();

  // Send all parameters as POST JSON body, not as query params.
  // The backend @APIExport(requestType="POST") parses the body.
  const response = await axiosInstance.post(
    "/ExperimentController/computeFocusMapFromPoints",
    params,
  );
  return response.data;
};

export default apiExperimentControllerComputeFocusMapFromPoints;
