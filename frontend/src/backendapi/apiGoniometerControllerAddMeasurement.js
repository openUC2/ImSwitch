// src/backendapi/apiGoniometerControllerAddMeasurement.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerAddMeasurement = async (measurement) => {
  const instance = createAxiosInstance();
  const response = await instance.post(
    "/GoniometerController/add_measurement_goniometer",
    measurement
  );
  return response.data;
};

export default apiGoniometerControllerAddMeasurement;
