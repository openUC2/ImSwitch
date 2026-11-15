import createAxiosInstance from "./createAxiosInstance";

// Set PWM frequency (Hz) and/or resolution (bits); the laser is switched off. Returns the new settings.
const apiLaserControllerSetLaserPWMSettings = async (laserName, frequency, resolution) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/LaserController/setLaserPWMSettings", {
    params: { laserName, frequency, resolution },
  });
  return response.data;
};

export default apiLaserControllerSetLaserPWMSettings;
