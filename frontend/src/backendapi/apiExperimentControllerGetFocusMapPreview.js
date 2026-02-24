// src/backendapi/apiExperimentControllerGetFocusMapPreview.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Get a preview grid for visualization of a focus map.
 *
 * @param {string} groupId - Group to preview
 * @param {number} resolution - Grid resolution for preview (default 30)
 * @returns {Promise<Object>} Preview data with x, y, z arrays and raw points
 */
const apiExperimentControllerGetFocusMapPreview = async (groupId, resolution = 30) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.post(
      "/ExperimentController/getFocusMapPreview",
      null,
      { params: { group_id: groupId, resolution } }
    );
    return response.data;
  } catch (error) {
    console.error("Error getting focus map preview:", error);
    throw error;
  }
};

export default apiExperimentControllerGetFocusMapPreview;
