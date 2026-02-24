// src/backendapi/apiExperimentControllerClearFocusMap.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Clear focus map(s).
 *
 * @param {string|null} groupId - If provided, clear only this group. Otherwise clear all.
 * @returns {Promise<Object>} Status message
 */
const apiExperimentControllerClearFocusMap = async (groupId = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = {};
    if (groupId) params.group_id = groupId;

    const response = await axiosInstance.post(
      "/ExperimentController/clearFocusMap",
      null,
      { params }
    );
    return response.data;
  } catch (error) {
    console.error("Error clearing focus map:", error);
    throw error;
  }
};

export default apiExperimentControllerClearFocusMap;
