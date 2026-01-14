// src/backendapi/apiSTORMControllerGetLastReconstructedImagePath.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerGetLastReconstructedImagePath = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    const url = `/STORMReconController/getLastReconstructedImagePath`;
    const response = await axiosInstance.get(url); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error getting last reconstructed image path:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerGetLastReconstructedImagePath;

/*
// Example usage:

const getLastReconstructedImagePath = () => {
  apiSTORMControllerGetLastReconstructedImagePath()
    .then((data) => {
      console.log("Last reconstructed image path retrieved:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to get last reconstructed image path:", err); // Handle the error
    });
};
*/