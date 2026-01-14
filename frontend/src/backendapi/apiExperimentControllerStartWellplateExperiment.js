// src/api/apiExperimentControllerStartWellplateExperiment.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerStartWellplateExperiment = async (experimentData) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.post("/ExperimentController/startWellplateExperiment", experimentData); // Send POST request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error starting wellplate experiment:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerStartWellplateExperiment;


/*
// Example usage:

const startExperiment = (experimentData) => {
  apiExperimentControllerStartWellplateExperiment(experimentData)
    .then((data) => {
      setExperimentStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to start wellplate experiment"); // Handle the error
    });
};
*/
