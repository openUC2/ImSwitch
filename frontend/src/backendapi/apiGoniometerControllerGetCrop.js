// src/backendapi/apiGoniometerControllerGetCrop.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerGetCrop = async () => {
  const instance = createAxiosInstance();
  const response = await instance.get(
    "/GoniometerController/get_crop_goniometer"
  );
  return response.data;
};

export default apiGoniometerControllerGetCrop;
