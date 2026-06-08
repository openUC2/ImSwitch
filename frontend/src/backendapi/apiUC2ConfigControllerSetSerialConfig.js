import createAxiosInstance from "./createAxiosInstance";

/**
 * Apply (and optionally persist) the serial port / baudrate used to talk
 * to the ESP32. Either argument may be omitted to keep the current value.
 *
 * @param {string|null} port - e.g. "/dev/ttyACM0" or null to keep current.
 * @param {number|null} baudrate - e.g. 115200, or null to keep current.
 * @param {boolean} persist - When true, writes back to the setup JSON.
 * @returns {Promise<Object>} { status, connected, port, baudrate, persisted }
 */
const apiUC2ConfigControllerSetSerialConfig = async (
  port = null,
  baudrate = null,
  persist = true
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/UC2ConfigController/setSerialConfig",
    null,
    {
      params: {
        port: port,
        baudrate: baudrate,
        persist: persist,
      },
      timeout: 30000,
    }
  );
  return response.data;
};

export default apiUC2ConfigControllerSetSerialConfig;
