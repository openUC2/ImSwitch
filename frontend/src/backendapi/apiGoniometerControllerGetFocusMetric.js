// src/backendapi/apiGoniometerControllerGetFocusMetric.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerGetFocusMetric = async () => {
  const instance = createAxiosInstance();
  const response = await instance.get(
    "/GoniometerController/get_focus_metric_goniometer"
  );
  return response.data;
};

export default apiGoniometerControllerGetFocusMetric;
