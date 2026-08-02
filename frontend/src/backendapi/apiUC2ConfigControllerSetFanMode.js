import createAxiosInstance from "./createAxiosInstance";

/**
 * Set the fan operating mode.
 *
 * @param {string} mode - 'auto' (curve-driven), 'manual' (fixed wiper) or 'off'.
 * @param {number|null} wiper - 0-127 PWM wiper, used for 'manual' mode.
 * @returns {Promise<Object>} { status, mode, wiper }
 */
const apiUC2ConfigControllerSetFanMode = async (mode = "auto", wiper = null) => {
  const axiosInstance = createAxiosInstance();
  const params = { mode };
  if (wiper !== null && wiper !== undefined) {
    params.wiper = wiper;
  }
  const response = await axiosInstance.get("/UC2ConfigController/setFanMode", {
    params,
  });
  return response.data;
};

export default apiUC2ConfigControllerSetFanMode;
