// src/backendapi/apiExperimentControllerGetWellplateLayout.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Get a specific wellplate layout from server with optional offsets
 * @param {string} layoutName - Name of the layout (e.g., '96-well-standard')
 * @param {number} offsetX - X offset in micrometers (default: 0)
 * @param {number} offsetY - Y offset in micrometers (default: 0)
 * @returns {Promise} Promise resolving to complete layout with all wells
 */
const apiExperimentControllerGetWellplateLayout = async (layoutName, offsetX = 0, offsetY = 0) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/ExperimentController/getWellplateLayout", {
      params: {
        layout_name: layoutName,
        offset_x: offsetX,
        offset_y: offsetY
      }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching wellplate layout '${layoutName}':`, error);
    throw error;
  }
};

export default apiExperimentControllerGetWellplateLayout;

/*
// Example usage:

const loadWellplate = async () => {
  try {
    const layout = await apiExperimentControllerGetWellplateLayout('96-well-standard', 1000, 500);
    console.log("Layout:", layout);
    // layout will contain: { name, description, rows, cols, wells: [...], ... }
    dispatch(experimentSlice.setWellLayout(layout));
  } catch (err) {
    console.error("Failed to load wellplate:", err);
  }
};
*/
