// src/backendapi/apiGoniometerControllerSnap.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerSnap = async () => {
  const instance = createAxiosInstance();
  const response = await instance.get("/GoniometerController/snap_goniometer");
  return response.data;
};

export default apiGoniometerControllerSnap;
