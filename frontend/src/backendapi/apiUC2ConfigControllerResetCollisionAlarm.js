import createAxiosInstance from "./createAxiosInstance";

/**
 * Clear the latched crash state after the user has verified the situation is
 * resolved. Does not move any motor — re-homing after a crash is the user's
 * decision.
 *
 * @returns {Promise<Object>} { trip, latched, armed, event, timestamp }
 */
const apiUC2ConfigControllerResetCollisionAlarm = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/resetCollisionAlarm",
  );
  return response.data;
};

export default apiUC2ConfigControllerResetCollisionAlarm;
