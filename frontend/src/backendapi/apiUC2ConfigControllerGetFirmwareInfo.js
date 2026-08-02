// Read the firmware identity of the USB-connected ESP32 master:
// { name, version, date, author, pindef, isMaster, connected, serialport }.
// The build date and pindef are the most useful fields for telling boards apart.
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetFirmwareInfo = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/getFirmwareInfo",
  );
  return response.data;
};

export default apiUC2ConfigControllerGetFirmwareInfo;
