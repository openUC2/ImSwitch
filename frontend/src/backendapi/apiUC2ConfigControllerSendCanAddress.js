import createAxiosInstance from "./createAxiosInstance";

/**
 * Send a CAN bus address assignment to a freshly-flashed device via serial.
 * Known addresses: master=1, a=10, x=11, y=12, z=13, led=30.
 * @param {string} port - Serial port device path (e.g. "/dev/ttyACM0")
 * @param {number} address - CAN bus address to assign
 * @param {number} baud - Serial baudrate for communication (default 115200)
 * @param {number} timeout - Serial read timeout in seconds (default 2.0)
 * @returns {Promise<Object>} Result with status and response
 */
const apiUC2ConfigControllerSendCanAddress = async (
  port,
  address,
  baud = 115200,
  timeout = 2.0
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/UC2ConfigController/sendCanAddress",
    null,
    {
      params: {
        port: port,
        address: address,
        baud: baud,
        timeout: timeout,
      },
      timeout: 30000, // 30 seconds
    }
  );
  return response.data;
};

export default apiUC2ConfigControllerSendCanAddress;
