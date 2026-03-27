// src/backendapi/apiGoniometerControllerGetMeasurements.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerGetMeasurements = async () => {
  const instance = createAxiosInstance();
  const response = await instance.get("/GoniometerController/get_measurements_goniometer");
  return response.data;
};

export default apiGoniometerControllerGetMeasurements;
