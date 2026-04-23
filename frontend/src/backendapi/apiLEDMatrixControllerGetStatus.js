import createAxiosInstance from "./createAxiosInstance";

const apiLEDMatrixControllerGetStatus = async () => {
  const instance = createAxiosInstance();
  const response = await instance.get("/LEDMatrixController/getstatus");
  return response.data;
};

export default apiLEDMatrixControllerGetStatus;
