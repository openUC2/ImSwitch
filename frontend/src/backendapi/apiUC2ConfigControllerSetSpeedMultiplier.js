import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the joystick-jog speed multiplier for one motor axis.
 *
 * @param {string} axis - Axis name ("A", "X", "Y", "Z").
 * @param {number} multiplier - Speed multiplier applied on the device for joystick jogging.
 * @returns {Promise<Object>} { status, axis, multiplier }
 */
const apiUC2ConfigControllerSetSpeedMultiplier = async (
  axis = "X",
  multiplier = 1,
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/setSpeedMultiplier",
    { params: { axis, multiplier } },
  );
  return response.data;
};

export default apiUC2ConfigControllerSetSpeedMultiplier;
