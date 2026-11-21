import createAxiosInstance from "./createAxiosInstance";

/**
 * Switch the status OLED on or off entirely (stored on the ESP32, survives a reboot).
 *
 * @param {boolean} enabled - True to show the display, False to turn it dark.
 * @returns {Promise<Object>} { status, enabled }
 */
const apiUC2ConfigControllerSetDisplayEnabled = async (enabled = true) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/setDisplayEnabled", {
    params: { enabled },
  });
  return response.data;
};

export default apiUC2ConfigControllerSetDisplayEnabled;
