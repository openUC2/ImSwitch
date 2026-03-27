// src/backendapi/apiGoniometerControllerGetConfig.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerGetConfig = async () => {
  const instance = createAxiosInstance();
  const response = await instance.get("/GoniometerController/get_config_goniometer");
  return response.data;
};

export default apiGoniometerControllerGetConfig;
