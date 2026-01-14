/**
 * API wrapper for setting motor settings for a specific axis
 * @param {string} axis - Axis name (X, Y, Z, or A)
 * @param {object} settings - Settings object with motion, homing, and limits
 */
import createAxiosInstance from "./createAxiosInstance";

const apiMotorSettingsSetForAxis = async (axis, settings) => {
  const api = createAxiosInstance();
  const response = await api.post(
    `/UC2ConfigController/setMotorSettingsForAxis?axis=${axis}`,
    settings
  );
  return response.data;
};

export default apiMotorSettingsSetForAxis;
