// src/api/apiSTORMControllerGetStatus.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerGetStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/STORMReconController/getSTORMStatus"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error fetching STORM status:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerGetStatus;

/*
// Example usage:

const fetchSTORMStatus = () => {
  apiSTORMControllerGetStatus()
    .then((data) => {
      console.log("STORM status:", data); // Handle success response
      // Expected data format: { isRunning: boolean, currentFrame: number }
    })
    .catch((err) => {
      console.error("Failed to fetch STORM status:", err); // Handle the error
    });
};
*/