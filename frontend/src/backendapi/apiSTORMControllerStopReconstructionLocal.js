// src/backendapi/apiSTORMControllerStopReconstructionLocal.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerStopReconstructionLocal = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    const url = `/STORMReconController/stopSTORMReconstructionLocal`;
    const response = await axiosInstance.get(url); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error stopping STORM reconstruction local:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerStopReconstructionLocal;

/*
// Example usage:

const stopReconstructionLocal = () => {
  apiSTORMControllerStopReconstructionLocal()
    .then((data) => {
      console.log("STORM reconstruction local stopped:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to stop STORM reconstruction local:", err); // Handle the error
    });
};
*/