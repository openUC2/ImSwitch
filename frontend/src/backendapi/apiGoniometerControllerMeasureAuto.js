// src/backendapi/apiGoniometerControllerMeasureAuto.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerMeasureAuto = async () => {
  const instance = createAxiosInstance();
  const response = await instance.get("/GoniometerController/measure_auto_goniometer");
  return response.data;
};

export default apiGoniometerControllerMeasureAuto;
