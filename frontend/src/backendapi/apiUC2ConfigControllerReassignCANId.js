// src/backendapi/apiUC2ConfigControllerReassignCANId.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Reassign a CAN node's id over the bus (no reflash / USB needed).
 *
 * Identify the node EITHER by its MAC (`mac` — recommended: the master finds
 * whichever id currently advertises that MAC, so the current id need not be
 * known) OR by its current id (`target`, optionally guarded by `expectMac`).
 * `newId` is always the desired new id. The device persists it and reappears
 * at `newId` after ~0.3 s, so re-scan afterwards to confirm.
 *
 * @param {number} newId - desired CAN id (1..127)
 * @param {string|null} mac - target MAC "AA:BB:CC:DD:EE:FF" (preferred)
 * @param {number|null} target - current CAN id of the node (alternative to mac)
 * @param {string|null} expectMac - when using target, verify MAC before committing
 * @param {number} timeout - command timeout in seconds
 * @returns {Promise<Object>} e.g. { status: "ok", target: 60, newId: 70, mac: "..." }
 *                            or { status: "error", error: "MAC not found on bus" }
 */
const apiUC2ConfigControllerReassignCANId = async (
  newId,
  mac = null,
  target = null,
  expectMac = null,
  timeout = 5
) => {
  try {
    const axiosInstance = createAxiosInstance(timeout + 5);
    const response = await axiosInstance.post(
      "/UC2ConfigController/reassignCANId",
      null,
      {
        params: {
          new_id: newId,
          mac: mac,
          target: target,
          expect_mac: expectMac,
          timeout: timeout,
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error reassigning CAN id:", error);
    throw error;
  }
};

export default apiUC2ConfigControllerReassignCANId;
