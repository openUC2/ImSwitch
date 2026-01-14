/**
 * API call to get camera information
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Get information about available cameras
 * @returns {Promise} Response with camera specifications
 */
export default async function apiAcceptanceTestControllerGetCameraInfo() {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/AcceptanceTestController/getCameraInfo');
  return response.data;
}
