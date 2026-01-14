// src/backendapi/apiSTORMControllerTriggerReconstruction.js
import createAxiosInstance from "./createAxiosInstance";

const apiSTORMControllerTriggerReconstruction = async (params = {}) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    // Build query parameters if provided
    const queryParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        queryParams.append(key, value);
      }
    });
    
    const url = `/STORMReconController/triggerSTORMReconstruction${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
    const response = await axiosInstance.get(url); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error triggering STORM reconstruction:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSTORMControllerTriggerReconstruction;

/*
// Example usage:

const triggerSTORMReconstruction = () => {
  apiSTORMControllerTriggerReconstruction({
    frame: 100  // Optional frame parameter
  })
    .then((data) => {
      console.log("STORM reconstruction triggered:", data); // Handle success response
    })
    .catch((err) => {
      console.error("Failed to trigger STORM reconstruction:", err); // Handle the error
    });
};
*/