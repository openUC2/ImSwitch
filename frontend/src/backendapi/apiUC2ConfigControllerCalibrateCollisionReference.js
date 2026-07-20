import createAxiosInstance from "./createAxiosInstance";

/**
 * Tell the GPIO slave to take its CURRENT rolling mean as the new collision
 * reference (persisted in NVS). Call while the system is idle and
 * collision-free.
 *
 * @returns {Promise<Object>} Firmware acknowledgement.
 */
const apiUC2ConfigControllerCalibrateCollisionReference = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/calibrateCollisionReference",
  );
  return response.data;
};

export default apiUC2ConfigControllerCalibrateCollisionReference;
