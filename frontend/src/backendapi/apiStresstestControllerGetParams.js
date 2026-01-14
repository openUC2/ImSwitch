// src/api/apiStresstestControllerGetParams.js
import createAxiosInstance from "./createAxiosInstance";

const apiStresstestControllerGetParams = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/StresstestController/getStresstestParams"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching stresstest parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiStresstestControllerGetParams;

/*
// Example usage:

const fetchStresstestParams = () => {
  apiStresstestControllerGetParams()
    .then((data) => {
      setStresstestParams(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to fetch stresstest parameters"); // Handle the error
    });
};
*/