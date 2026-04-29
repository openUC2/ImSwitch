// src/backendapi/apiGoniometerControllerSetCameraSettings.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerSetCameraSettings = async ({ exposure_time, gain } = {}) => {
  const instance = createAxiosInstance();
  const params = {};
  if (exposure_time != null) params.exposure_time = exposure_time;
  if (gain != null) params.gain = gain;
  const response = await instance.post(
    "/GoniometerController/set_camera_settings_goniometer",
    null,
    { params }
  );
  return response.data;
};

export default apiGoniometerControllerSetCameraSettings;
