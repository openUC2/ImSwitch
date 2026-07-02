import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the collision-detector deviation band (ADC counts around the
 * reference). Persisted on the GPIO slave in NVS.
 *
 * @param {number} threshold - Allowed deviation in ADC counts (e.g. 150).
 * @returns {Promise<Object>} Firmware acknowledgement.
 */
const apiUC2ConfigControllerSetCollisionThreshold = async (threshold) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/setCollisionThreshold",
    { params: { threshold } },
  );
  return response.data;
};

export default apiUC2ConfigControllerSetCollisionThreshold;
