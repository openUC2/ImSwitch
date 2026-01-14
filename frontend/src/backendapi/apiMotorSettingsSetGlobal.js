/**
 * API wrapper for setting global motor settings
 * @param {object} settings - Global settings object (axisOrder, isCoreXY, etc.)
 */
import createAxiosInstance from "./createAxiosInstance";

const apiMotorSettingsSetGlobal = async (settings) => {
  const api = createAxiosInstance();
  const response = await api.post(
    "/UC2ConfigController/setGlobalMotorSettings",
    settings
  );
  return response.data;
};

export default apiMotorSettingsSetGlobal;
