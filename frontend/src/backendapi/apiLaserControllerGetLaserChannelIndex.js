import createAxiosInstance from "./createAxiosInstance";

/**
 * Get the channel index for a specific laser
 * 
 * @param {string} laserName - Name of the laser
 * @returns {Promise<number|string|null>} Channel index (number, "LED", or null if not found)
 * 
 * @example
 * const channelIndex = await apiLaserControllerGetLaserChannelIndex("Laser488");
 * // Returns: 0, 1, 2, "LED", etc.
 */
const apiLaserControllerGetLaserChannelIndex = async (laserName) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LaserController/getLaserChannelIndex", {
      params: { laserName }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching channel index for laser ${laserName}:`, error);
    throw error;
  }
};

export default apiLaserControllerGetLaserChannelIndex;
