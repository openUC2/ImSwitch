// src/backendapi/apiSTORMControllerGetProcessingParameters.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerGetProcessingParameters = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    const url = `/STORMReconController/getSTORMProcessingParameters`;
    const response = await axiosInstance.get(url); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error getting STORM processing parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerGetProcessingParameters;

/*
// Example usage:

const getProcessingParameters = () => {
  apiSTORMControllerGetProcessingParameters()
    .then((data) => {
      console.log("STORM processing parameters:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to get STORM processing parameters:", err); // Handle the error
    });
};
*/