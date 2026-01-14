import createAxiosInstance from "./createAxiosInstance";

/**
 * Get the min and max intensity range for a specific laser
 * 
 * @param {string} laserName - Name of the laser
 * @returns {Promise<[number, number]>} Array with [min, max] values
 * 
 * @example
 * const [min, max] = await apiLaserControllerGetLaserValueRanges("LED");
 * // Returns: [0, 255]
 */
const apiLaserControllerGetLaserValueRanges = async (laserName) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LaserController/getLaserValueRanges", {
      params: { laserName }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching laser value ranges for ${laserName}:`, error);
    throw error;
  }
};

export default apiLaserControllerGetLaserValueRanges;
