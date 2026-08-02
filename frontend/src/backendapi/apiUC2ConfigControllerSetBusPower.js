import createAxiosInstance from "./createAxiosInstance";

/**
 * Enable/disable the high-current CAN-bus power that feeds the slaves.
 *
 * @param {boolean} enable - True to power the bus, False to cut power.
 * @returns {Promise<Object>} { status, power }
 */
const apiUC2ConfigControllerSetBusPower = async (enable = true) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/setBusPower", {
    params: { enable },
  });
  return response.data;
};

export default apiUC2ConfigControllerSetBusPower;
