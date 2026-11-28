import createAxiosInstance from "./createAxiosInstance";

/**
 * Branches and tags of github.com/openUC2/os-rpi to pick for the pallet
 * upgrade (needs internet on the ImSwitch host).
 *
 * @returns {Promise<Object>} { status: "success", branches, tags } or { status: "error", message }
 */
const apiUC2ConfigControllerListPalletVersions = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/listPalletVersions",
  );
  return response.data;
};

export default apiUC2ConfigControllerListPalletVersions;
