import createAxiosInstance from "./createAxiosInstance";

/**
 * Cancel a USB-flash run that is currently in progress.
 * Safe to call when nothing is running.
 * @returns {Promise<Object>} { status: "cancelled" | "idle", ... }
 */
const apiUC2ConfigControllerCancelUSBFlash = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/UC2ConfigController/cancelUSBFlash",
    null,
    { timeout: 10000 }
  );
  return response.data;
};

export default apiUC2ConfigControllerCancelUSBFlash;
