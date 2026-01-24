// src/api/apiStresstestControllerStart.js
import createAxiosInstance from "./createAxiosInstance";

const apiStresstestControllerStart = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/StresstestController/startStresstest"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error starting stresstest:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiStresstestControllerStart;

/*
// Example usage:

const startStresstest = () => {
  apiStresstestControllerStart()
    .then((data) => {
      console.log("Stresstest started:", data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to start stresstest"); // Handle the error
    });
};
*/