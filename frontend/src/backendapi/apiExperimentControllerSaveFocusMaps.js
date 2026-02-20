// src/backendapi/apiExperimentControllerSaveFocusMaps.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Save all current focus maps to disk as JSON files.
 *
 * @param {string|null} path - Directory to save into. Null = default ~/ImSwitch/focus_maps
 * @returns {Promise<Object>} { saved_files, count, path }
 */
const apiExperimentControllerSaveFocusMaps = async (path = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = {};
    if (path) params.path = path;

    const response = await axiosInstance.get(
      "/ExperimentController/saveFocusMaps",
      { params }
    );
    return response.data;
  } catch (error) {
    console.error("Error saving focus maps:", error);
    throw error;
  }
};

export default apiExperimentControllerSaveFocusMaps;
