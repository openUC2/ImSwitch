import createAxiosInstance from "./createAxiosInstance";

/**
 * Flash master CAN HAT firmware via USB using esptool
 * @param {string|null} port - Serial port (auto-detect if null)
 * @param {string} match - Substring to identify HAT in port metadata (default "HAT")
 * @param {number} baud - Flashing baudrate (default 921600)
 * @param {number} flashOffset - Address to flash firmware to (default 0x0)
 * @param {boolean} eraseFlash - Whether to erase flash before writing (default false)
 * @param {boolean} reconnectAfter - Whether to reconnect after flashing (default true)
 * @returns {Promise<Object>} Flash result status
 */
const apiUC2ConfigControllerFlashMasterFirmwareUSB = async (
  port = null,
  match = "HAT",
  baud = 921600,
  flashOffset = 0x0,
  eraseFlash = false,
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
        flash_offset: flashOffset,
        erase_flash: eraseFlash,
        reconnect_after: reconnectAfter,
      },
      timeout: 300000, // 5 minutes timeout for flashing
    }
  );
  return response.data;
};

export default apiUC2ConfigControllerFlashMasterFirmwareUSB;
