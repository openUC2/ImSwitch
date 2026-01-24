import createAxiosInstance from "./createAxiosInstance";

/**
 * Get the active state of a specific laser
 * 
 * @param {string} laserName - Name of the laser
 * @returns {Promise<boolean>} Whether the laser is active
 * 
 * @example
 * const isActive = await apiLaserControllerGetLaserActive("LED");
 * // Returns: true or false
 */
const apiLaserControllerGetLaserActive = async (laserName) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LaserController/getLaserActive", {
      params: { laserName }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching laser active state for ${laserName}:`, error);
    throw error;
  }
};

export default apiLaserControllerGetLaserActive;
