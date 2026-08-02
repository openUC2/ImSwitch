// Cancel an in-progress Z-stage sync run and halt all motors.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerCancelZStageSync = async ({
  positionerName = null,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/cancelZStageSync",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerCancelZStageSync;
