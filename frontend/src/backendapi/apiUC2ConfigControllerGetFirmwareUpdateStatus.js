// State of the current/last firmware update run:
// { state: idle|running|success|failed|cancelled, server_version, message,
//   current, homing_required, steps: [{ canId, connection, deviceTypeStr,
//   filename, from_version, to_version, status, message }] }
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetFirmwareUpdateStatus = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/getFirmwareUpdateStatus");
  return response.data;
};

export default apiUC2ConfigControllerGetFirmwareUpdateStatus;
