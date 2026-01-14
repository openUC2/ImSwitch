// src/api/apiExperimentControllerStopWorkflow.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerStopWorkflow = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/WorkflowController/stop_workflow"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error stopping workflow:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerStopWorkflow;

/*
// Example usage:

const stopWorkflow = () => {
  apiExperimentControllerStopWorkflow()
    .then((data) => {
      setWorkflowStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to stop workflow"); // Handle the error
    });
};
*/