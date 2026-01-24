// src/backendapi/apiSettingsControllerSetJpegQuality.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Set JPEG compression quality for legacy streaming
 * @param {number} quality - JPEG quality (1-100)
 * @returns {Promise<object>} Response data
 */
const apiSettingsControllerSetJpegQuality = async (quality = 85) => {
  try {
    const axiosInstance = createAxiosInstance();
    
    const url = `/SettingsController/setJpegQuality?quality=${quality}`;
    const response = await axiosInstance.post(url);
    
    return response.data;
  } catch (error) {
    console.error("Error setting JPEG quality:", error);
    throw error;
  }
};

export default apiSettingsControllerSetJpegQuality;

/*
// Example usage:

const setJpegQuality = async () => {
  try {
    const result = await apiSettingsControllerSetJpegQuality(85);
    console.log("JPEG quality updated:", result);
  } catch (err) {
    console.error("Failed to set JPEG quality:", err);
  }
};
*/
