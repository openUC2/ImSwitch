// Stop the running firmware update after the current board (a USB flash of
// the master in progress is not interrupted).
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerCancelFirmwareUpdate = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post("/UC2ConfigController/cancelFirmwareUpdate");
  return response.data;
};

export default apiUC2ConfigControllerCancelFirmwareUpdate;
