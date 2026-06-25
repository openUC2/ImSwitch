import createAxiosInstance from "./createAxiosInstance";

/**
 * Read the PS-controller joystick inversion per axis.
 *
 * @returns {Promise<Object>} e.g. { A: false, X: false, Y: true, Z: false }
 */
const apiUC2ConfigControllerGetJoystickDirection = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/getJoystickDirection",
  );
  return response.data;
};

export default apiUC2ConfigControllerGetJoystickDirection;
