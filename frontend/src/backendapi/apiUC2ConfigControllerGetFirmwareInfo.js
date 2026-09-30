// Read the firmware identity of the USB-connected ESP32 master:
// { name, version, fwVersion, date, author, pindef, isMaster, connected, serialport }.
// fwVersion is the release the firmware was built from (matches version.json on
// the firmware server); version is the fixed API generation ("V2.0").
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetFirmwareInfo = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/getFirmwareInfo",
  );
  return response.data;
};

export default apiUC2ConfigControllerGetFirmwareInfo;
