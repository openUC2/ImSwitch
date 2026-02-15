import createAxiosInstance from "./createAxiosInstance";

/**
 * List ALL .bin firmware files from the configured firmware server (flat list).
 * Unlike listAvailableFirmware which maps files to CAN IDs, this returns every
 * .bin file so the user can pick any firmware for USB flashing.
 * @returns {Promise<Object>} { status, firmware_server, files: [{ filename, size, mod_time, url }] }
 */
const apiUC2ConfigControllerListAllFirmwareFiles = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/listAllFirmwareFiles"
  );
  return response.data;
};

export default apiUC2ConfigControllerListAllFirmwareFiles;
