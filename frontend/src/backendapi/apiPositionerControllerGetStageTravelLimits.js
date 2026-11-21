import createAxiosInstance from "./createAxiosInstance";

/**
 * Travel limits per positioner and axis, in the frame of getPositionerPositions:
 * { positioner: { X: { min, max, enabled }, ... } }; min/max null = open side.
 */
const apiPositionerControllerGetStageTravelLimits = async () => {
  const response = await createAxiosInstance().get(
    "/PositionerController/getStageTravelLimits",
  );
  return response.data;
};

export default apiPositionerControllerGetStageTravelLimits;
