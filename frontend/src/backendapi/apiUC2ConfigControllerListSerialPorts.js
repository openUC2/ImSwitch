import createAxiosInstance from "./createAxiosInstance";

/**
 * List available serial ports for USB flashing
 * @returns {Promise<Array>} Array of serial port objects
 */
const apiUC2ConfigControllerListSerialPorts = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/listSerialPorts"
  );
  return response.data;
};

export default apiUC2ConfigControllerListSerialPorts;
