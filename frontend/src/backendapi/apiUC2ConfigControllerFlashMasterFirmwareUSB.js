import createAxiosInstance from "./createAxiosInstance";

/**
 * Flash firmware to an ESP32 via USB using esptool
 * @param {string|null} port - Serial port (auto-detect if null)
 * @param {string} match - Substring to identify HAT in port metadata (default "HAT")
 * @param {number} baud - Flashing baudrate (default 921600)
 * @param {string|null} firmwareFilename - Name of .bin file on server (auto-detect master if null)
 * @param {boolean} reconnectAfter - Whether to reconnect after flashing (default true)
 * @returns {Promise<Object>} Flash result status
 */
const apiUC2ConfigControllerFlashMasterFirmwareUSB = async (
  port = null,
  match = "HAT",
  baud = 921600,
  firmwareFilename = null,
  reconnectAfter = true
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/UC2ConfigController/flashMasterFirmwareUSB",
    null,
    {
      params: {
        port: port,
        match: match,
        baud: baud,
        firmware_filename: firmwareFilename,
        reconnect_after: reconnectAfter,
      },
      timeout: 300000, // 5 minutes timeout for flashing
    }
  );
  return response.data;
};

export default apiUC2ConfigControllerFlashMasterFirmwareUSB;
