// Result of the (opt-in) firmware check after startup: the checkFirmwareUpdates
// payload when it found updates, else { updates_available: 0 }.
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetFirmwareUpdatePrompt = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/getFirmwareUpdatePrompt");
  return response.data;
};

export default apiUC2ConfigControllerGetFirmwareUpdatePrompt;
