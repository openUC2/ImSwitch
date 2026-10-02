import createAxiosInstance from "./createAxiosInstance";

/**
 * State and terminal output of the pallet upgrade.
 *
 * @returns {Promise<Object>} { state: "activating" | "inactive" | "failed" | ..., log }
 */
const apiUC2ConfigControllerGetPalletUpgradeStatus = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/getPalletUpgradeStatus",
  );
  return response.data;
};

export default apiUC2ConfigControllerGetPalletUpgradeStatus;
