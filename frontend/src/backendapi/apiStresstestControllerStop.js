// src/api/apiStresstestControllerStop.js
import createAxiosInstance from "./createAxiosInstance";

const apiStresstestControllerStop = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/StresstestController/stopStresstest"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error stopping stresstest:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiStresstestControllerStop;

/*
// Example usage:

const stopStresstest = () => {
  apiStresstestControllerStop()
    .then((data) => {
      console.log("Stresstest stopped:", data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to stop stresstest"); // Handle the error
    });
};
*/