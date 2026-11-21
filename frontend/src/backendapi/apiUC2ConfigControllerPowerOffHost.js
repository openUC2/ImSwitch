import createAxiosInstance from "./createAxiosInstance";

/**
 * Shut down (reboot=false) or restart (reboot=true) the computer ImSwitch runs
 * on. Motors stop and lights go off first. openUC2 OS only.
 *
 * @returns {Promise<Object>} { status: "shutting down" | "rebooting" } or { status: "error", message }
 */
const apiUC2ConfigControllerPowerOffHost = async (reboot = false) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/UC2ConfigController/powerOffHost",
    null,
    { params: { reboot } },
  );
  return response.data;
};

export default apiUC2ConfigControllerPowerOffHost;
