// src/backendapi/apiGoniometerControllerGetCameraSettings.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerGetCameraSettings = async () => {
  const instance = createAxiosInstance();
  const response = await instance.get(
    "/GoniometerController/get_camera_settings_goniometer"
  );
  return response.data;
};

export default apiGoniometerControllerGetCameraSettings;
