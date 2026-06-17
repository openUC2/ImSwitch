// src/backendapi/apiGoniometerControllerSetCrop.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerSetCrop = async ({ x1, y1, x2, y2 }) => {
  const instance = createAxiosInstance();
  const response = await instance.post(
    "/GoniometerController/set_crop_goniometer",
    null,
    { params: { x1, y1, x2, y2 } }
  );
  return response.data;
};

export default apiGoniometerControllerSetCrop;
