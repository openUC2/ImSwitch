import createAxiosInstance from "./createAxiosInstance";

/**
 * Poll the collision detector on the GPIO slave (triggers SDO reads on the
 * CAN bus — the slave never broadcasts sensor values on its own).
 *
 * @returns {Promise<Object>} { mean, filtered, raw, reference, threshold,
 *   sensitivity, trip, estop, latched, armed }
 */
const apiUC2ConfigControllerGetGpioStatus = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/getGpioStatus");
  return response.data;
};

export default apiUC2ConfigControllerGetGpioStatus;
