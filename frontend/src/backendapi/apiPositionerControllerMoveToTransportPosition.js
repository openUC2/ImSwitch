// Move the stage to the stored transportation (locking) position.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerMoveToTransportPosition = async ({
  positionerName = null,
  speed = 10000,
  isBlocking = true,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = { speed, isBlocking };
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/moveToTransportPosition",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerMoveToTransportPosition;
