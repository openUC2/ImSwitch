// src/backendapi/apiLiveViewControllerGetActiveStreams.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Get list of currently active streams
 * GET /LiveViewController/getActiveStreams
 * 
 * Returns: Dictionary with active stream information
 * Example response:
 * {
 *   "detector1": {
 *     "protocol": "binary",
 *     "params": {...}
 *   }
 * }
 */
const apiLiveViewControllerGetActiveStreams = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const url = `/LiveViewController/getActiveStreams`;
    const response = await axiosInstance.get(url);
    return response.data;
  } catch (error) {
    console.error('Error getting active streams:', error);
    throw error;
  }
};

export default apiLiveViewControllerGetActiveStreams;

/**
 * Example usage:
 * const streams = await apiLiveViewControllerGetActiveStreams();
 * console.log('Active streams:', streams);
 */
