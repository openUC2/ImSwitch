// src/backendapi/apiUC2ConfigControllerListAvailableFirmware.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * List firmware files mapped to CAN IDs, optionally filtered to specific IDs.
 *
 * Delegates to the unified listAvailableFirmware endpoint (which itself uses
 * listAllFirmwareFiles under the hood).  The response format is unchanged:
 *   { status, firmware_server, firmware_count, firmware: {can_id: {...}} }
 *
 * @param {number[]} [canIds] - Optional array of CAN IDs to filter the result.
 * @returns {Promise<Object>}
 */
const apiUC2ConfigControllerListAvailableFirmware = async (canIds) => {
  try {
    const axiosInstance = createAxiosInstance();
    const params = canIds && canIds.length > 0 ? { can_ids: canIds } : {};
    const response = await axiosInstance.get(
      "/UC2ConfigController/listAvailableFirmware",
      { params }
    );
    return response.data;
  } catch (error) {
    console.error("Error listing available firmware:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerListAvailableFirmware;
