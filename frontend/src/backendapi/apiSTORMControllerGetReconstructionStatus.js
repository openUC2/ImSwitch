// src/backendapi/apiSTORMControllerGetReconstructionStatus.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerGetReconstructionStatus = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    const url = `/STORMReconController/getSTORMReconstructionStatus`;
    const response = await axiosInstance.get(url); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error getting STORM reconstruction status:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerGetReconstructionStatus;

/*
// Example usage:

const getReconstructionStatus = () => {
  apiSTORMControllerGetReconstructionStatus()
    .then((data) => {
      console.log("STORM reconstruction status retrieved:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to get STORM reconstruction status:", err); // Handle the error
    });
};
*/