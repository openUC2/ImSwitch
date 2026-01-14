// src/backendapi/apiSTORMControllerSetParameters.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerSetParameters = async (params = {}) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    // Build query parameters if provided
    const queryParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        queryParams.append(key, value);
      }
    });
    
    const url = `/STORMReconController/setSTORMParameters${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    const response = await axiosInstance.get(url); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error setting STORM parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerSetParameters;

/*
// Example usage:

const setSTORMParameters = () => {
  apiSTORMControllerSetParameters({
    threshold: 100,
    roi_size: 15,
    update_rate: 10
  })
    .then((data) => {
      console.log("STORM parameters set:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to set STORM parameters:", err); // Handle the error
    });
};
*/