// Re-open the serial connection to the ESP32 master (optionally on a
// different port/baudrate). Returns { status: "started", port, baudrate }.
import createAxiosInstance from "./createAxiosInstance";

const apiUC2ConfigControllerReconnect = async ({ port = null, baudrate = null } = {}) => {
  const axiosInstance = createAxiosInstance();
  const params = {};
  if (port) params.port = port;
  if (baudrate) params.baudrate = baudrate;
  const response = await axiosInstance.get("/UC2ConfigController/reconnect", {
    params,
  });
  return response.data;
};

export default apiUC2ConfigControllerReconnect;
