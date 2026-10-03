// Compare every connected board's firmware with the firmware server's
// version.json. Read-only (no flashing); on a CAN master this runs a bus scan.
// -> { status, firmware_server, server_version, server_commit_time,
//      updates_available, devices: [{ canId, deviceTypeStr, connection,
//      installed_version, build, mac, filename, available_version,
//      update_status }] }
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerCheckFirmwareUpdates = async (timeout = 5, probeRange = false) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/checkFirmwareUpdates", {
    params: { timeout, probe_range: probeRange },
  });
  return response.data;
};

export default apiUC2ConfigControllerCheckFirmwareUpdates;
