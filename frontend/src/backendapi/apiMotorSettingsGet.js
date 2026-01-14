/**
 * API wrapper for getting all motor settings from UC2ConfigController
 * Returns motor settings for all axes and global configuration
 */
import createAxiosInstance from "./createAxiosInstance";

const apiMotorSettingsGet = async () => {
  const api = createAxiosInstance();
  const response = await api.get("/UC2ConfigController/getMotorSettings");
  return response.data;
};

export default apiMotorSettingsGet;
