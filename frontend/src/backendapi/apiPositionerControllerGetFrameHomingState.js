// Get the current frame-homing progress state (fallback poll; live updates come
// through the sigHomingState websocket signal).
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerGetFrameHomingState = async ({
  positionerName = null,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/getFrameHomingState",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerGetFrameHomingState;
