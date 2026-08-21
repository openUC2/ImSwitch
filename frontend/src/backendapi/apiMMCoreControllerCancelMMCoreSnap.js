// src/backendapi/apiMMCoreControllerCancelMMCoreSnap.js
import createAxiosInstance from "./createAxiosInstance";

// Aborts a running snap job. The backend stops the exposure in the driver, so
// a long integration ends immediately instead of running out its full time.
// jobId is optional -- omitting it cancels the most recently started job.
const apiMMCoreControllerCancelMMCoreSnap = async ({ jobId, detectorName } = {}) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.post(
    "/MMCoreController/cancelMMCoreSnap",
    { jobId, detectorName },
    { headers: { "Content-Type": "application/json" } },
  );
  return response.data;
};

export default apiMMCoreControllerCancelMMCoreSnap;
