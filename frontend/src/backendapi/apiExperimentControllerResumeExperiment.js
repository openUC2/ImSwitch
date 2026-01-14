// src/api/apiExperimentControllerResumeExperiment.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerResumeExperiment = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/ExperimentController/resumeExperiment"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error resuming experiment:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerResumeExperiment;

/*
// Example usage:

const resumeExperiment = () => {
  apiExperimentControllerResumeExperiment()
    .then((data) => {
      setExperimentStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to resume experiment"); // Handle the error
    });
};
*/
