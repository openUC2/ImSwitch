// Whether ImSwitch compares the boards with the firmware server after startup: { enabled }.
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerGetFirmwareCheckOnConnect = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/getFirmwareCheckOnConnect");
  return response.data;
};

export default apiUC2ConfigControllerGetFirmwareCheckOnConnect;
