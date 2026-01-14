/**
 * API wrapper for setting TMC stepper driver settings for a specific axis
 * @param {string} axis - Axis name (X, Y, Z, or A)
 * @param {object} settings - TMC settings object
 */
import createAxiosInstance from "./createAxiosInstance";

const apiTMCSettingsSetForAxis = async (axis, settings) => {
  const api = createAxiosInstance();
  const response = await api.post(
    `/UC2ConfigController/setTMCSettingsForAxis?axis=${axis}`,
    settings
  );
  return response.data;
};

export default apiTMCSettingsSetForAxis;
