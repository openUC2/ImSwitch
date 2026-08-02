// src/backendapi/apiUC2ConfigControllerCancelCANStreamingOTA.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Request cancellation of any running CAN streaming OTA upload.
 * The backend worker will abort at the next chunk boundary and restore
 * the serial connection.
 * @returns {Promise<Object>} { status, message }
 */
const apiUC2ConfigControllerCancelCANStreamingOTA = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/UC2ConfigController/cancelCANStreamingOTA"
  );
  return response.data;
};

export default apiUC2ConfigControllerCancelCANStreamingOTA;
