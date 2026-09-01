// Soft-restart the ESP32 master board (firmware reboot, ~2s downtime).
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerEspRestart = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/espRestart");
  return response.data;
};

export default apiUC2ConfigControllerEspRestart;
