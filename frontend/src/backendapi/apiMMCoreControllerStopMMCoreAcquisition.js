// src/backendapi/apiMMCoreControllerStopMMCoreAcquisition.js
import createAxiosInstance from "./createAxiosInstance";

// Hard stop for a detector: cancels every running snap job, aborts the
// exposure in the driver and releases a camera claim left behind by a worker
// that died, so the camera is usable again without restarting the backend.
const apiMMCoreControllerStopMMCoreAcquisition = async ({ detectorName } = {}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/MMCoreController/stopMMCoreAcquisition",
    { detectorName },
    { headers: { "Content-Type": "application/json" } },
  );
  return response.data;
};

export default apiMMCoreControllerStopMMCoreAcquisition;
