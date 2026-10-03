// src/backendapi/apiExperimentControllerGetFocusMapSummary.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Which focus map each scan region would actually use, and how good it is.
 *
 * @returns {Promise<Object>} { regions, focus_map_active, channel_offsets_um, autofocus_offsets_um }
 */
const apiExperimentControllerGetFocusMapSummary = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/ExperimentController/getFocusMapSummary",
    );
    return response.data;
  } catch (error) {
    console.error("Error getting focus map summary:", error);
    throw error;
  }
};

export default apiExperimentControllerGetFocusMapSummary;
