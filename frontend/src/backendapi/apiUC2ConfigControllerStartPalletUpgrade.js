import createAxiosInstance from "./createAxiosInstance";

/**
 * EXPERIMENTAL (openUC2 OS only): run `forklift plt upgrade --force &&
 * forklift stage apply` on the host. This restarts the ImSwitch container.
 *
 * @param {string} [version] branch, tag or commit of github.com/openUC2/os-rpi
 *   to switch to (e.g. "edge"); empty = the pallet's stored upgrade query
 * @returns {Promise<Object>} { status: "started", unit } or { status: "error", message }
 */
const apiUC2ConfigControllerStartPalletUpgrade = async (version = "") => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/startPalletUpgrade",
    { params: { version } },
  );
  return response.data;
};

export default apiUC2ConfigControllerStartPalletUpgrade;
