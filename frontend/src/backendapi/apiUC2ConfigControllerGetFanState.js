import createAxiosInstance from "./createAxiosInstance";

/**
 * Read the current fan state from the firmware.
 *
 * @returns {Promise<Object>} { mode, wiper, manual, rpm, stalled, kick, tempC, curve }
 */
const apiUC2ConfigControllerGetFanState = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/getFanState");
  return response.data;
};

export default apiUC2ConfigControllerGetFanState;
