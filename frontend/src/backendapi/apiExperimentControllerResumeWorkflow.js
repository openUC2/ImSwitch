// src/api/apiExperimentControllerResumeWorkflow.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerResumeWorkflow = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/WorkflowController/resume_workflow"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error resuming workflow:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerResumeWorkflow;

/*
// Example usage:

const resumeWorkflow = () => {
  apiExperimentControllerResumeWorkflow()
    .then((data) => {
      setWorkflowStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to resume workflow"); // Handle the error
    });
};
*/