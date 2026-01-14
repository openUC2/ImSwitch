// src/backendapi/apiLiveViewControllerStartLiveView.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Start live streaming for a specific detector
 * Only one protocol can be active per detector at a time
 * POST /LiveViewController/startLiveView
 * 
 * @param {string|null} detectorName - Name of detector to stream from (null = first available)
 * @param {string} protocol - Streaming protocol ('binary', 'jpeg', 'mjpeg', 'webrtc')
 * @param {object|null} params - Optional parameters to override defaults
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
 * For jpeg:
 *   {
 *     jpeg_quality: 80,
 *     subsampling_factor: 4,
 *     throttle_ms: 50
 *   }
 * 
 * For mjpeg:
 *   {
 *     jpeg_quality: 80,
 *     throttle_ms: 50
 *   }
 * 
 * For webrtc:
 *   {
 *     stun_servers: ['stun:stun.l.google.com:19302'],
 *     turn_servers: []
 *   }
 * 
 * Returns: Dictionary with status and stream info
 */
const apiLiveViewControllerStartLiveView = async (detectorName = null, protocol = 'binary', params = null) => {
  try {
    const axiosInstance = createAxiosInstance();
    const url = `/LiveViewController/startLiveView`;
    
    const queryParams = {};
    if (detectorName !== null && detectorName !== undefined) {
      queryParams.detectorName = detectorName;
    }
    if (protocol !== null && protocol !== undefined) {
      queryParams.protocol = protocol;
    }
    
    const response = await axiosInstance.post(url, params, {
      params: queryParams,
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error starting live view:', error);
    throw error;
  }
};

export default apiLiveViewControllerStartLiveView;

/**
 * Example usage:
 * 
 * // Start binary stream with default params
 * await apiLiveViewControllerStartLiveView(null, 'binary');
 * 
 * // Start binary stream with custom params
 * await apiLiveViewControllerStartLiveView(null, 'binary', {
 *   compression_algorithm: 'lz4',
 *   compression_level: 0,
 *   subsampling_factor: 4,
 *   throttle_ms: 50
 * });
 * 
 * // Start JPEG stream
 * await apiLiveViewControllerStartLiveView(null, 'jpeg', {
 *   jpeg_quality: 80,
 *   subsampling_factor: 4,
 *   throttle_ms: 50
 * });
 * 
 * // Start for specific detector
 * await apiLiveViewControllerStartLiveView('Camera1', 'binary');
 */
