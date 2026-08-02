// Immediately stop all axes of the positioner.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerStopAllAxes = async ({
  positionerName = null,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/stopAllAxes",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerStopAllAxes;
