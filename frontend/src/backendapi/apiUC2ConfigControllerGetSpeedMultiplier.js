import createAxiosInstance from "./createAxiosInstance";

/**
 * Read the joystick-jog speed multiplier per axis.
 *
 * @returns {Promise<Object>} e.g. { A: 1, X: 15, Y: 5, Z: 15 }
 */
const apiUC2ConfigControllerGetSpeedMultiplier = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/getSpeedMultiplier",
  );
  return response.data;
};

export default apiUC2ConfigControllerGetSpeedMultiplier;
