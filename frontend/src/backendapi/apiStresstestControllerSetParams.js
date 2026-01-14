// src/api/apiStresstestControllerSetParams.js
import createAxiosInstance from "./createAxiosInstance";

const apiStresstestControllerSetParams = async (params) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.post("/StresstestController/setStresstestParams", params); // Send POST request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error setting stresstest parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiStresstestControllerSetParams;

/*
// Example usage:

const updateStresstestParams = (params) => {
  apiStresstestControllerSetParams(params)
    .then((data) => {
      console.log("Parameters updated successfully:", data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to update stresstest parameters"); // Handle the error
    });
};
*/