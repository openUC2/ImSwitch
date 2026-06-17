// src/backendapi/apiGoniometerControllerSetConfig.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerSetConfig = async (params) => {
  const instance = createAxiosInstance();
  const response = await instance.post("/GoniometerController/set_config_goniometer", params);
  return response.data;
};

export default apiGoniometerControllerSetConfig;
