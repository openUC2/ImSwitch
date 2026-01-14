// src/api/apiExperimentControllerGetCurrentExperimentParams.js
import createAxiosInstance from "./createAxiosInstance";

const apiExperimentControllerGetCurrentExperimentParams = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/ExperimentController/getHardwareParameters"); // Send GET request
    console.log("Request apiExperimentControllerGetCurrentExperimentParams");
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching experiment parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiExperimentControllerGetCurrentExperimentParams;
