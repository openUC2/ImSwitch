// Start the collision-safe global homing procedure (Z first, lift, then XY).
// Per-axis progress is pushed back via the sigHomingState websocket signal.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerStartFrameHoming = async ({
  positionerName = null,
  isBlocking = false,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = { isBlocking };
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/startFrameHoming",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerStartFrameHoming;
