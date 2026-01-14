// src/backendapi/apiSettingsControllerSetStreamParams.js
import createAxiosInstance from "./createAxiosInstance";

const apiSettingsControllerSetStreamParams = async (params = {}) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    // Extract parameters from the nested structure
    // Params come in as: { binary: { compression, subsampling, throttle_ms, enabled }, jpeg: { quality, enabled } }
    // Backend expects: { compression, subsampling, throttle_ms }
    
    const requestBody = {};
    
    // Check if params has binary/jpeg structure (from StreamSettings/StreamControlOverlay)
    if (params.binary !== undefined) {
      // Extract from nested structure
      if (params.binary.compression !== undefined) {
        if (params.jpeg["enabled"]) {
          requestBody.compression = {algorithm: 'jpeg', level: 80};
        }
        if (params.binary["enabled"]) {
          requestBody.compression = params.binary.compression;
        }
        
      }
      
      if (params.binary.subsampling !== undefined) {
        requestBody.subsampling = params.binary.subsampling;
      }
      
      if (params.binary.throttle_ms !== undefined) {
        requestBody.throttlems = params.binary.throttle_ms;
      }
    } else {
      // Fallback: accept direct parameters (for backward compatibility)
      if (params.compression !== undefined) {
        requestBody.compression = params.compression;
      }
      
      if (params.subsampling !== undefined) {
        requestBody.subsampling = params.subsampling;
      }
      
      if (params.throttle_ms !== undefined) {
        requestBody.throttlems = params.throttle_ms;
      }
    }
    
    console.log('apiSettingsControllerSetStreamParams - sending:', requestBody);
    
    const url = `/SettingsController/setStreamParams`;
    
    // Send POST request with JSON body
    const response = await axiosInstance.post(url, requestBody, {
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error setting stream parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSettingsControllerSetStreamParams;

/*
// Example usage:

const setStreamParams = async () => {
  try {
    const result = await apiSettingsControllerSetStreamParams({
      throttle_ms: 50,
      compression: {
        algorithm: "lz4",
        level: 0
      },
      subsampling: {
        factor: 2,
      }
    });
    console.log("Stream parameters updated:", result);
  } catch (err) {
    console.error("Failed to set stream parameters:", err);
  }
};
*/