// src/backendapi/apiLiveViewControllerStopLiveView.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Stop live streaming for a specific detector
 * If detectorName is null, stops the first active stream
 * Protocol is ignored - we only care about the detector
 * GET /LiveViewController/stopLiveView
 * 
 * @param {string|null} detectorName - Name of detector (null = stop first active detector)
 * 
 * Returns: Dictionary with status
 */
const apiLiveViewControllerStopLiveView = async (detectorName = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const url = `/LiveViewController/stopLiveView`;
    
    const params = {};
    if (detectorName !== null && detectorName !== undefined) {
      params.detectorName = detectorName;
    }
    
    const response = await axiosInstance.get(url, { params });
    return response.data;
  } catch (error) {
    console.error('Error stopping live view:', error);
    throw error;
  }
};

export default apiLiveViewControllerStopLiveView;

/**
 * Example usage:
 * 
 * // Stop first active stream
 * await apiLiveViewControllerStopLiveView();
 * 
 * // Stop specific detector
 * await apiLiveViewControllerStopLiveView('Camera1');
 */
