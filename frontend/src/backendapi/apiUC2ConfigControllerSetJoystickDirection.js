import createAxiosInstance from "./createAxiosInstance";

/**
 * Invert (or un-invert) the PS-controller joystick for one motor axis.
 *
 * @param {string} axis - Axis name ("A", "X", "Y", "Z").
 * @param {boolean} inverted - True to reverse joystick movement for that axis.
 * @returns {Promise<Object>} { status, axis, inverted }
 */
const apiUC2ConfigControllerSetJoystickDirection = async (
  axis = "X",
  inverted = false,
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/setJoystickDirection",
    { params: { axis, inverted } },
  );
  return response.data;
};

export default apiUC2ConfigControllerSetJoystickDirection;
