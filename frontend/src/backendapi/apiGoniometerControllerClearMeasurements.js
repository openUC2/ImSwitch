// src/backendapi/apiGoniometerControllerClearMeasurements.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerClearMeasurements = async () => {
  const instance = createAxiosInstance();
  const response = await instance.post("/GoniometerController/clear_measurements_goniometer");
  return response.data;
};

export default apiGoniometerControllerClearMeasurements;
