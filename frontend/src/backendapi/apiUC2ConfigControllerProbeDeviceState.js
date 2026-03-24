import createAxiosInstance from "./createAxiosInstance";

/**
 * Send {"task":"/state_get"} to a device and return the raw response.
 * Use this to verify that firmware is running correctly after flashing.
 * @param {string} port - Serial port device path (e.g. "/dev/ttyACM0")
 * @param {number} baud - Serial baudrate (115200 or 921600)
 * @param {number} timeout - Serial read timeout in seconds (default 2.0)
 * @returns {Promise<Object>} Result with status, state_response, and firmware_ok
 */
const apiUC2ConfigControllerProbeDeviceState = async (
  port,
  baud = 115200,
  timeout = 2.0
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/probeDeviceState",
    {
      params: {
        port: port,
        baud: baud,
        timeout: timeout,
      },
      timeout: 15000, // 15 seconds
    }
  );
  return response.data;
};

export default apiUC2ConfigControllerProbeDeviceState;
