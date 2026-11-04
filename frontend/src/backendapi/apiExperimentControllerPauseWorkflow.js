// src/backendapi/apiExperimentControllerPauseWorkflow.js
import createAxiosInstance from "./createAxiosInstance";

// Pauses a running ExperimentController run (normal mode). The backend answers
// 200 with {status: "error", message} when it cannot pause (performance mode,
// nothing running), so that case is turned into a rejection for the caller.
const apiExperimentControllerPauseWorkflow = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/ExperimentController/pauseWorkflow");
  const data = response.data;
  if (data && data.status === "error") {
    throw new Error(data.message || "Experiment could not be paused");
  }
  return data;
};

export default apiExperimentControllerPauseWorkflow;
