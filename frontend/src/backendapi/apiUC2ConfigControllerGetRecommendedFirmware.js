import createAxiosInstance from "./createAxiosInstance";

/**
 * Which firmware image on the server fits a board, from what its firmware
 * reports over serial (/state_get: fwImage, pindef, CAN id). Read-only.
 * @param {string} port - "" for the board ImSwitch is connected to (read over
 *   the open link), or another serial port to probe (may reset that board)
 * @param {number} baud - Serial baudrate for probing another port
 * @returns {Promise<Object>} {status, source: "imswitch"|"port", port, chip,
 *   identity: {fwImage, fwVersion, pindef, isMaster, canId, ...},
 *   server_version, recommended: {filename, merged, source, reason,
 *   candidates: [{filename, source, on_server}], file}}
 */
const apiUC2ConfigControllerGetRecommendedFirmware = async (port = "", baud = 115200) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/getRecommendedFirmware", {
    params: { port, baud },
    timeout: 20000, // probing another port waits out its boot log
  });
  return response.data;
};

export default apiUC2ConfigControllerGetRecommendedFirmware;
