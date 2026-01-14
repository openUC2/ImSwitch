import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the intensity value for a specific laser
 * 
 * @param {string} laserName - Name of the laser
 * @param {number} value - Intensity value to set
 * @returns {Promise<any>} Response from the backend
 * 
 * @example
 * await apiLaserControllerSetLaserValue("LED", 128);
 */
const apiLaserControllerSetLaserValue = async (laserName, value) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LaserController/setLaserValue", {
      params: { laserName, value }
    });
    return response.data;
  } catch (error) {
    console.error(`Error setting laser value for ${laserName}:`, error);
    throw error;
  }
};

export default apiLaserControllerSetLaserValue;
