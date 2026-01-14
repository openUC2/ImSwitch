// src/backendapi/apiExperimentControllerGetWellplateLayouts.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Get list of available pre-defined wellplate layouts from server
 * @returns {Promise} Promise resolving to object with layout names and metadata
 */
const apiExperimentControllerGetWellplateLayouts = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/ExperimentController/getAvailableWellplateLayouts");
    return response.data;
  } catch (error) {
    console.error("Error fetching wellplate layouts:", error);
    throw error;
  }
};

export default apiExperimentControllerGetWellplateLayouts;

/*
// Example usage:

const fetchLayouts = async () => {
  try {
    const layouts = await apiExperimentControllerGetWellplateLayouts();
    console.log("Available layouts:", layouts);
    // layouts will be: { "96-well-standard": { name, description, rows, cols, ... }, ... }
  } catch (err) {
    console.error("Failed to fetch layouts:", err);
  }
};
*/
