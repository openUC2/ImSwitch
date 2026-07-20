import createAxiosInstance from "./createAxiosInstance";

/**
 * Select the collision-detection algorithm on the GPIO slave.
 *
 * @param {string} mode - "auto" (adaptive, parameter-free, recommended) or
 *   "manual" (fixed reference +/- threshold).
 * @returns {Promise<Object>} Firmware acknowledgement.
 */
const apiUC2ConfigControllerSetCollisionMode = async (mode = "auto") => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/setCollisionMode",
    { params: { mode } },
  );
  return response.data;
};

export default apiUC2ConfigControllerSetCollisionMode;
