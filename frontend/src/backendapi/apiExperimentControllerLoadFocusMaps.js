// src/backendapi/apiExperimentControllerLoadFocusMaps.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Load focus maps from disk, restoring previously saved maps.
 *
 * @param {string|null} path - Directory to load from. Null = default ~/ImSwitch/focus_maps
 * @returns {Promise<Object>} { loaded_count, path, maps }
 */
const apiExperimentControllerLoadFocusMaps = async (path = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = {};
    if (path) params.path = path;

    const response = await axiosInstance.get(
      "/ExperimentController/loadFocusMaps",
      { params }
    );
    return response.data;
  } catch (error) {
    console.error("Error loading focus maps:", error);
    throw error;
  }
};

export default apiExperimentControllerLoadFocusMaps;
