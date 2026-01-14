// src/api/apiExperimentControllerStopExperiment.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerStopExperiment = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/ExperimentController/stopExperiment"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error stopping experiment:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerStopExperiment;

/*
// Example usage:

const stopExperiment = () => {
  apiExperimentControllerStopExperiment()
    .then((data) => {
      setExperimentStatus(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to stop experiment"); // Handle the error
    });
};
*/
