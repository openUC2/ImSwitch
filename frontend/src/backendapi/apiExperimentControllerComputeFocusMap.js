// src/backendapi/apiExperimentControllerComputeFocusMap.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Compute focus map for one or all scan groups.
 *
 * @param {Object} focusMapConfig - FocusMapConfig object (optional, uses experiment default if null)
 * @param {string|null} groupId - If provided, compute for this group only
 * @returns {Promise<Object>} Focus map results per group
 */
const apiExperimentControllerComputeFocusMap = async (focusMapConfig = null, groupId = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = {};
    if (groupId) params.group_id = groupId;

    const response = await axiosInstance.post(
      "/ExperimentController/computeFocusMap",
      focusMapConfig,
      { params }
    );
    return response.data;
  } catch (error) {
    console.error("Error computing focus map:", error);
    throw error;
  }
};

export default apiExperimentControllerComputeFocusMap;
