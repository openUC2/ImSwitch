/**
 * API wrapper for setting all motor settings at once
 * @param {object} settings - Settings object with global and axes sub-objects
 */
import createAxiosInstance from "./createAxiosInstance";

const apiMotorSettingsSet = async (settings) => {
  const api = createAxiosInstance();
  const response = await api.post(
    "/UC2ConfigController/setMotorSettings",
    settings
  );
  return response.data;
};

export default apiMotorSettingsSet;
