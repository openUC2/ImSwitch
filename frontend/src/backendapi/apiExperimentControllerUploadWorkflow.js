// src/api/apiExperimentControllerUploadWorkflow.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerUploadWorkflow = async (workflowJson) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.post("/WorkflowController/create_workflow_definition_api", workflowJson, {
      headers: {
        'Content-Type': 'application/json'
      }
    }); // Send POST request with workflow JSON
    return response.data; // Return the data from the response (should include UID)
  } catch (error) {
    console.error("Error uploading workflow:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerUploadWorkflow;

/*
// Example usage:

const uploadWorkflow = (workflowData) => {
  apiExperimentControllerUploadWorkflow(workflowData)
    .then((data) => {
      setWorkflowUploadStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to upload workflow"); // Handle the error
    });
};
*/