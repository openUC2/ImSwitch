import createAxiosInstance from "./createAxiosInstance";

/**
 * Look up this machine's name (the Wi-Fi hotspot SSID of an openUC2 OS device)
 * and send it to the status OLED.
 *
 * @returns {Promise<Object>} { status: "success"|"unsupported"|"error", name, message? }
 */
const apiUC2ConfigControllerUpdateDisplayDeviceName = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/updateDisplayDeviceName");
  return response.data;
};

export default apiUC2ConfigControllerUpdateDisplayDeviceName;
