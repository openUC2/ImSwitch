/**
 * API wrapper for getting motor settings for a specific axis
 * @param {string} axis - Axis name (X, Y, Z, or A)
 */
import createAxiosInstance from "./createAxiosInstance";

const apiMotorSettingsGetForAxis = async (axis) => {
  const api = createAxiosInstance();
  const response = await api.get(
    `/UC2ConfigController/getMotorSettingsForAxis?axis=${axis}`
  );
  return response.data;
};

export default apiMotorSettingsGetForAxis;
