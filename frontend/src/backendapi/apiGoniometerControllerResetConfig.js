// src/backendapi/apiGoniometerControllerResetConfig.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerResetConfig = async () => {
  const instance = createAxiosInstance();
  const response = await instance.post("/GoniometerController/reset_config_goniometer");
  return response.data;
};

export default apiGoniometerControllerResetConfig;
