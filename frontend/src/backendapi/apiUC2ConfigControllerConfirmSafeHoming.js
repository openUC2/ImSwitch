import createAxiosInstance from "./createAxiosInstance";

/**
 * Clear the post-crash "requires homing" flag after a safe frame-homing has
 * been completed. Call once the stage has been re-homed and its position is
 * trustworthy again.
 *
 * @returns {Promise<Object>} { trip, latched, armed, requiresHoming, ... }
 */
const apiUC2ConfigControllerConfirmSafeHoming = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/confirmSafeHoming",
  );
  return response.data;
};

export default apiUC2ConfigControllerConfirmSafeHoming;
