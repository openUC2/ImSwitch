// src/backendapi/apiExperimentControllerGenerateCustomWellplate.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Generate a custom wellplate layout on the server
 * @param {Object} layoutParams - Layout parameters
 * @param {string} layoutParams.name - Name of the layout
 * @param {number} layoutParams.rows - Number of rows
 * @param {number} layoutParams.cols - Number of columns
 * @param {number} layoutParams.well_spacing_x - X spacing in micrometers
 * @param {number} layoutParams.well_spacing_y - Y spacing in micrometers
 * @param {string} layoutParams.well_shape - 'circle' or 'rectangle'
 * @param {number} [layoutParams.well_radius] - Radius for circular wells
 * @param {number} [layoutParams.well_width] - Width for rectangular wells
 * @param {number} [layoutParams.well_height] - Height for rectangular wells
 * @param {number} [layoutParams.offset_x] - X offset (default: 0)
 * @param {number} [layoutParams.offset_y] - Y offset (default: 0)
 * @param {string} [layoutParams.description] - Description of layout
 * @returns {Promise} Promise resolving to complete layout definition
 */
const apiExperimentControllerGenerateCustomWellplate = async (layoutParams) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.post("/ExperimentController/generateCustomWellplateLayout", layoutParams, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    return response.data;
  } catch (error) {
    console.error("Error generating custom wellplate layout:", error);
    throw error;
  }
};

export default apiExperimentControllerGenerateCustomWellplate;

/*
// Example usage:

const createCustomWellplate = async () => {
  try {
    const customLayout = await apiExperimentControllerGenerateCustomWellplate({
      name: "My Custom 48-well",
      rows: 6,
      cols: 8,
      well_spacing_x: 13000,  // 13mm spacing
      well_spacing_y: 13000,
      well_shape: "circle",
      well_radius: 5000,  // 5mm radius
      offset_x: 2000,
      offset_y: 2000,
      description: "Custom 48-well plate with 13mm spacing"
    });
    console.log("Custom layout:", customLayout);
    dispatch(experimentSlice.setWellLayout(customLayout));
  } catch (err) {
    console.error("Failed to create custom wellplate:", err);
  }
};
*/
