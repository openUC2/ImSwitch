// src/api/apiExperimentControllerGetWorkflowStatus.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerGetWorkflowStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/WorkflowController/getWorkflowStatus"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error getting workflow status:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerGetWorkflowStatus;

/*
// Example usage:

const getWorkflowStatus = () => {
  apiExperimentControllerGetWorkflowStatus()
    .then((data) => {
      setWorkflowStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to get workflow status"); // Handle the error
    });
};
*/