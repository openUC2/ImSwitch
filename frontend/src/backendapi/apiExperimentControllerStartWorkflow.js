// src/api/apiExperimentControllerStartWorkflow.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerStartWorkflow = async (workflowUid) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const requestBody = {
      workflow_id: workflowUid // Use the UID from uploaded protocol
    };
    const response = await axiosInstance.post("/WorkflowController/start_workflow_api", requestBody, {
      headers: {
        'Content-Type': 'application/json'
      }
    }); // Send POST request with workflow UID
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error starting workflow:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerStartWorkflow;

/*
// Example usage:

const startWorkflow = () => {
  apiExperimentControllerStartWorkflow()
    .then((data) => {
      setWorkflowStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to start workflow"); // Handle the error
    });
};
*/