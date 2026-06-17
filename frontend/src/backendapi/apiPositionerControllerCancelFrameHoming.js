// Cancel an in-progress frame-homing run.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerCancelFrameHoming = async ({
  positionerName = null,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/cancelFrameHoming",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerCancelFrameHoming;
