// src/backendapi/apiExperimentControllerGetFocusMap.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Get saved focus maps from backend.
 *
 * @param {string|null} groupId - If provided, get only this group. Otherwise get all.
 * @returns {Promise<Object>} Focus map data per group
 */
const apiExperimentControllerGetFocusMap = async (groupId = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = {};
    if (groupId) params.group_id = groupId;

    const response = await axiosInstance.get(
      "/ExperimentController/getFocusMap",
      { params }
    );
    return response.data;
  } catch (error) {
    console.error("Error getting focus map:", error);
    throw error;
  }
};

export default apiExperimentControllerGetFocusMap;
