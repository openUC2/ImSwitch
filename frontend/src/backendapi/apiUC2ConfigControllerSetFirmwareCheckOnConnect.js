// Enable/disable the firmware check after startup (saved in the setup JSON,
// takes effect on the next start). Returns { enabled }.
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerSetFirmwareCheckOnConnect = async (enabled) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/setFirmwareCheckOnConnect", {
    params: { enabled },
  });
  return response.data;
};

export default apiUC2ConfigControllerSetFirmwareCheckOnConnect;
