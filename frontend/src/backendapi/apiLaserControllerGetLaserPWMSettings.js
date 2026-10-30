import createAxiosInstance from "./createAxiosInstance";

/**
 * Firmware PWM parameters of a laser (from /laser_get on the ESP32)
 *
 * @param {string} laserName - Name of the laser
 * @returns {Promise<Object|null>} e.g. {LASERid: 1, LASERval: 0, LASERmax: 1024, LASERRes: 10, LASERFreq: 5000}
 */
const apiLaserControllerGetLaserPWMSettings = async (laserName) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/LaserController/getLaserPWMSettings", {
    params: { laserName },
  });
  return response.data;
};

export default apiLaserControllerGetLaserPWMSettings;
