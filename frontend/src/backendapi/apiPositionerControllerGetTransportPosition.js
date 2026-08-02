// Get the stored transportation position (A/X/Y/Z).
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerGetTransportPosition = async ({
  positionerName = null,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/getTransportPosition",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerGetTransportPosition;
