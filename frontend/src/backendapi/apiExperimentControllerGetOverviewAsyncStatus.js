// src/backendapi/apiExperimentControllerGetOverviewAsyncStatus.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * GET /ExperimentController/getOverviewAsyncStatus
 *
 * Polls the background-overview-task state used by ``recaptureSlot`` and
 * ``runAutonomousOverviewScan``.
 */
const apiExperimentControllerGetOverviewAsyncStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      "/ExperimentController/getOverviewAsyncStatus"
    );
    return response.data;
  } catch (error) {
    console.error("Error polling overview async status:", error);
    throw error;
  }
};

export default apiExperimentControllerGetOverviewAsyncStatus;
