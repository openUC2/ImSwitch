// src/backendapi/apiGoniometerControllerResetCrop.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerResetCrop = async () => {
  const instance = createAxiosInstance();
  const response = await instance.post(
    "/GoniometerController/reset_crop_goniometer"
  );
  return response.data;
};

export default apiGoniometerControllerResetCrop;
