import createAxiosInstance from "./createAxiosInstance";

/**
 * Get all available laser names from the backend
 * 
 * @returns {Promise<Array<string>>} Array of laser names
 * 
 * @example
 * const laserNames = await apiLaserControllerGetLaserNames();
 * // Returns: ["Laser488", "Laser635", "LaserRed", ...]
 */
const apiLaserControllerGetLaserNames = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/LaserController/getLaserNames");
    return response.data;
  } catch (error) {
    console.error("Error fetching laser names:", error);
    throw error;
  }
};

export default apiLaserControllerGetLaserNames;
