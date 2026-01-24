// src/backendapi/apiSettingsControllerGetStreamParams.js
import createAxiosInstance from "./createAxiosInstance";

const apiSettingsControllerGetStreamParams = async () => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    
    const url = `/SettingsController/getStreamParams`;
    const response = await axiosInstance.get(url); // Send GET request
    return response.data; // Return the data from the response
  } catch (error) {
    console.error("Error getting stream parameters:", error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiSettingsControllerGetStreamParams;

/*
// Example usage:

const getStreamParams = async () => {
  try {
    const params = await apiSettingsControllerGetStreamParams();
    console.log("Stream parameters:", params);
  } catch (err) {
    console.error("Failed to get stream parameters:", err);
  }
};

// Expected response format:
{
  "binary": {
    "enabled": true,
    "compression": {
      "algorithm": "lz4",
      "level": 0
    },
    "subsampling": {
      "factor": 1,
    },
    "throttle_ms": 50,
    "bitdepth_in": 12,
    "pixfmt": "GRAY16"
  },
  "jpeg": {
    "enabled": false
  }
}
*/