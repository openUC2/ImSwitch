import createAxiosInstance from "./createAxiosInstance";

/**
 * Arm/disarm automatic motor stop on collision. While armed, a pushed
 * collision event immediately stops ALL positioners; the crash state latches
 * until resetCollisionAlarm is called.
 *
 * @param {boolean} arm - True to arm auto-stop.
 * @returns {Promise<Object>} { trip, latched, armed, event, timestamp }
 */
const apiUC2ConfigControllerArmCollisionProtection = async (arm = true) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/armCollisionProtection",
    { params: { arm } },
  );
  return response.data;
};

export default apiUC2ConfigControllerArmCollisionProtection;
