import createAxiosInstance from "./createAxiosInstance";

/**
 * Get the current intensity value of a specific laser
 * 
 * @param {string} laserName - Name of the laser
 * @returns {Promise<number>} Current intensity value
 * 
 * @example
 * const value = await apiLaserControllerGetLaserValue("LED");
 * // Returns: 128
 */
const apiLaserControllerGetLaserValue = async (laserName) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LaserController/getLaserValue", {
      params: { laserName }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching laser value for ${laserName}:`, error);
    throw error;
  }
};

export default apiLaserControllerGetLaserValue;
