import createAxiosInstance from "./createAxiosInstance";

/**
 * Read the status OLED of the CAN-HAT master.
 *
 * @returns {Promise<Object>} { present, on, addr, name, status, progress, fw, flip, contrast, splash },
 *   or {} when the firmware has no display module.
 */
const apiUC2ConfigControllerGetDisplayState = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/getDisplayState");
  return response.data;
};

export default apiUC2ConfigControllerGetDisplayState;
