// src/api/apiExperimentControllerPauseWorkflow.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerPauseWorkflow = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/WorkflowController/pause_workflow"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error pausing workflow:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerPauseWorkflow;

/*
// Example usage:

const pauseWorkflow = () => {
  apiExperimentControllerPauseWorkflow()
    .then((data) => {
      setWorkflowStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to pause workflow"); // Handle the error
    });
};
*/
