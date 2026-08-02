// Fetch the current Z-stage sync progress state. The frontend polls this
// (every ~2 s) to track progress; there is no websocket signal.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerGetZStageSyncState = async ({
  positionerName = null,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (positionerName) params.positionerName = positionerName;
  const response = await axiosInstance.get(
    "/PositionerController/getZStageSyncState",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerGetZStageSyncState;
