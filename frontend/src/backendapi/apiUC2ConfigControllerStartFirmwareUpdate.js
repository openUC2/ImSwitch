// Start the unattended update of the given CAN nodes (then the USB master if
// includeMaster). Returns { status: "started", steps } or
// { status: "refused", reasons: [...] } (nothing was started).
// Progress: apiUC2ConfigControllerGetFirmwareUpdateStatus.
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerStartFirmwareUpdate = async (canIds, includeMaster) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/UC2ConfigController/startFirmwareUpdate",
    canIds,
    { params: { include_master: includeMaster } },
  );
  return response.data;
};

export default apiUC2ConfigControllerStartFirmwareUpdate;
