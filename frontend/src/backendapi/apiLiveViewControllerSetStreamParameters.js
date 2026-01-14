// src/backendapi/apiLiveViewControllerSetStreamParameters.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Configure streaming parameters for a protocol (global settings)
 * POST /LiveViewController/setStreamParameters
 * 
 * @param {string} protocol - Streaming protocol ('binary', 'jpeg', 'mjpeg', 'webrtc')
 * @param {object} params - Dictionary of parameters to set
 * 
 * Example params:
 * For binary:
 *   {
 *     compression_algorithm: 'lz4',
 *     compression_level: 0,
 *     subsampling_factor: 4,
 *     throttle_ms: 50
 *   }
 * 
 * For jpeg/mjpeg:
 *   {
 *     jpeg_quality: 80,
 *     subsampling_factor: 4,
 *     throttle_ms: 50
 *   }
 * 
 * For webrtc:
 *   {
 *     ice_servers: [{ urls: ['stun:stun.l.google.com:19302'] }],
 *     media_constraints: { video: true, audio: true }
 *   }
 * 
 * Returns: Dictionary with status and updated params
 */
const apiLiveViewControllerSetStreamParameters = async (protocol, params) => {
  try {
    const axiosInstance = createAxiosInstance();
    const url = `/LiveViewController/setStreamParameters`;
    
    // Backend expects POST with protocol as query param and params as JSON body
    const response = await axiosInstance.post(url, params, {
      params: { protocol },
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error setting stream parameters:', error);
    throw error;
  }
};

export default apiLiveViewControllerSetStreamParameters;

/**
 * Example usage:
 * 
 * // Set binary stream parameters
 * await apiLiveViewControllerSetStreamParameters('binary', {
 *   compression_algorithm: 'lz4',
 *   compression_level: 0,
 *   subsampling_factor: 4,
 *   throttle_ms: 50
 * });
 * 
 * // Set JPEG stream parameters
 * await apiLiveViewControllerSetStreamParameters('jpeg', {
 *   jpeg_quality: 80,
 *   subsampling_factor: 4,
 *   throttle_ms: 50
 * });
 */
