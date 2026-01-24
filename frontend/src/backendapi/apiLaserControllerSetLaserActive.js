import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the active state for a specific laser
 * 
 * @param {string} laserName - Name of the laser
 * @param {boolean} active - Whether to activate or deactivate the laser
 * @returns {Promise<any>} Response from the backend
 * 
 * @example
 * await apiLaserControllerSetLaserActive("LED", true);
 */
const apiLaserControllerSetLaserActive = async (laserName, active) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LaserController/setLaserActive", {
      params: { laserName, active }
    });
    return response.data;
  } catch (error) {
    console.error(`Error setting laser active state for ${laserName}:`, error);
    throw error;
  }
};

export default apiLaserControllerSetLaserActive;
