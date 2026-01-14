// src/api/apiSTORMControllerStart.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerStart = async (params = {}) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    // Build query parameters if provided
    const queryParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        queryParams.append(key, value);
      }
    });
    
    const url = `/STORMReconController/startFastSTORMAcquisition${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    const response = await axiosInstance.get(url); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error starting STORM experiment:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerStart;

/*
// Example usage:

const startSTORMExperiment = () => {
  apiSTORMControllerStart({
    exposureTime: 50,
    cropX: 0,
    cropY: 0,
    cropWidth: 512,
    cropHeight: 512
  })
    .then((data) => {
      console.log("STORM experiment started:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to start STORM experiment:", err); // Handle the error
    });
};
*/