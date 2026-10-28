import createAxiosInstance from "./createAxiosInstance";

/**
 * EXPERIMENTAL (openUC2 OS only): run `forklift plt upgrade --force &&
 * forklift stage apply` on the host. This restarts the ImSwitch container.
 *
 * @returns {Promise<Object>} { status: "started" } or { status: "error", message }
 */
const apiUC2ConfigControllerStartPalletUpgrade = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/startPalletUpgrade",
  );
  return response.data;
};

export default apiUC2ConfigControllerStartPalletUpgrade;
