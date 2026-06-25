// Start the Z-stage motor re-sync procedure (drive Z into the mechanical stop,
// back off half, restore the limit switch, re-home Z, return to previous Z).
// Progress is read by polling getZStageSyncState.
import createAxiosInstance from "./createAxiosInstance";

const apiPositionerControllerStartZStageSync = async ({
  positionerName = null,
  steps = null,
  isBlocking = false,
} = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = { isBlocking };
  if (positionerName) params.positionerName = positionerName;
  if (steps !== null && steps !== undefined && steps !== "") params.steps = steps;
  const response = await axiosInstance.get(
    "/PositionerController/startZStageSync",
    { params },
  );
  return response.data;
};

export default apiPositionerControllerStartZStageSync;
