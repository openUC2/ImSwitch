import createAxiosInstance from "./createAxiosInstance";

/**
 * Get USB flash status for master device
 * @returns {Promise<Object>} Flash status object
 */
const apiUC2ConfigControllerGetUSBFlashStatus = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/getUSBFlashStatus"
  );
  return response.data;
};

export default apiUC2ConfigControllerGetUSBFlashStatus;
