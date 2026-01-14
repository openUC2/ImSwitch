// src/api/apiSTORMControllerStop.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerStop = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get("/STORMReconController/stopFastSTORMAcquisition"); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error stopping STORM experiment:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerStop;

/*
// Example usage:

const stopSTORMExperiment = () => {
  apiSTORMControllerStop()
    .then((data) => {
      console.log("STORM experiment stopped:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to stop STORM experiment:", err); // Handle the error
    });
};
*/