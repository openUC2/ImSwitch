// src/backendapi/apiViewControllerGetLiveViewActive.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Check if any live view stream is currently active
 * GET /LiveViewController/getLiveViewActive
 * 
 * Returns: boolean (true if any stream is active, false otherwise)
 */
const apiViewControllerGetLiveViewActive = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const url = `/LiveViewController/getLiveViewActive`;
    const response = await axiosInstance.get(url);
    
    // Backend returns boolean directly
    if (typeof response.data === 'boolean') {
      return response.data;
    }
    
    // Fallback for legacy format
    if (typeof response.data === 'object' && response.data !== null) {
      return response.data.active || response.data.isActive || false;
    }
    
    return false;
  } catch (error) {
    console.warn('Failed to get live view active status:', error.message);
    return false;
  }
};

export default apiViewControllerGetLiveViewActive;

/**
 * Example usage:
 * const isActive = await apiViewControllerGetLiveViewActive();
 * console.log('Stream is active:', isActive);
 */
