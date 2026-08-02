import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerGetHomingStatus = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/PositionerController/getHomingStatus",
  );
  return response.data;
};

export default apiPositionerControllerGetHomingStatus;
