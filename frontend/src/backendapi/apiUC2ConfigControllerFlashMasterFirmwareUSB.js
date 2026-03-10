import createAxiosInstance from "./createAxiosInstance";

/**
 * Flash firmware to an ESP32/S2/S3 via USB using esptool
 * @param {string|null} port - Serial port (auto-detect if null)
 * @param {string} match - Substring to identify device in port metadata (default "HAT")
 * @param {number} baud - Flashing baudrate (default 921600)
 * @param {string|null} firmwareFilename - Name of .bin file on server (auto-detect master if null)
 * @param {boolean} reconnectAfter - Whether to reconnect after flashing (default true)
 * @param {string} chip - Chip type: "auto", "esp32", "esp32s3", "esp32s2", "esp32c3" (default "auto")
 * @param {boolean} eraseFlash - Whether to erase flash before writing (default false)
 * @param {boolean} skipDisconnect - Skip ImSwitch serial disconnect before flashing (default false)
 * @returns {Promise<Object>} Flash result status
 */
const apiUC2ConfigControllerFlashMasterFirmwareUSB = async (
  port = null,
  match = "HAT",
  baud = 921600,
  firmwareFilename = null,
  reconnectAfter = true,
  chip = "auto",
  eraseFlash = false,
  skipDisconnect = false
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
        chip: chip,
        erase_flash: eraseFlash,
        skip_disconnect: skipDisconnect,
      },
      timeout: 300000, // 5 minutes timeout for flashing
    }
  );
  return response.data;
};

export default apiUC2ConfigControllerFlashMasterFirmwareUSB;
