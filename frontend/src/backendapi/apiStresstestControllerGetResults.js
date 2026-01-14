// src/api/apiStresstestControllerGetResults.js
import createAxiosInstance from "./createAxiosInstance";

const apiStresstestControllerGetResults = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/StresstestController/getStresstestResults"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching stresstest results:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiStresstestControllerGetResults;

/*
// Example usage:

const fetchStresstestResults = () => {
  apiStresstestControllerGetResults()
    .then((data) => {
      setStresstestResults(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to fetch stresstest results"); // Handle the error
    });
};
*/