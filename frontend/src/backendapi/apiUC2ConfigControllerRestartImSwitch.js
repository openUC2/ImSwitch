// Restart the ImSwitch backend service. The API (and every stream/socket)
// goes away for the duration of the restart — callers should expect the
// request itself to succeed but the connection to drop right after.
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerRestartImSwitch = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/UC2ConfigController/restartImSwitch",
  );
  return response.data;
};

export default apiUC2ConfigControllerRestartImSwitch;
