import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the collision-detector sensitivity: the number of CONSECUTIVE
 * out-of-band samples (50 Hz) required to trip — and in-band samples to
 * clear. 3-4 rejects single-sample spikes while confirming a real collision
 * within ~60-80 ms. Persisted on the GPIO slave in NVS.
 *
 * @param {number} sensitivity - Consecutive samples (1-50).
 * @returns {Promise<Object>} Firmware acknowledgement.
 */
const apiUC2ConfigControllerSetCollisionSensitivity = async (sensitivity) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/setCollisionSensitivity",
    { params: { sensitivity } },
  );
  return response.data;
};

export default apiUC2ConfigControllerSetCollisionSensitivity;
