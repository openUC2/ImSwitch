import createAxiosInstance from "./createAxiosInstance";

/**
 * Explicitly set the collision-detector idle baseline (ADC counts) —
 * typically the polled "mean" value. Persisted on the GPIO slave in NVS.
 *
 * @param {number} reference - Idle baseline in ADC counts.
 * @returns {Promise<Object>} Firmware acknowledgement.
 */
const apiUC2ConfigControllerSetCollisionReference = async (reference) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/setCollisionReference",
    { params: { reference } },
  );
  return response.data;
};

export default apiUC2ConfigControllerSetCollisionReference;
