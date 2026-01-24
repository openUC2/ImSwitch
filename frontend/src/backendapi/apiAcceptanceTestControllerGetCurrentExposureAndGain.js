/**
 * API call to get current exposure and gain settings
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Get current exposure time and gain
 * @param {string} detectorName - Name of the detector (optional)
 * @returns {Promise} Response with exposure and gain values
 */
export default async function apiAcceptanceTestControllerGetCurrentExposureAndGain(detectorName = null) {
  const axiosInstance = createAxiosInstance();
  const params = detectorName ? { detectorName } : {};
  
  const response = await axiosInstance.get('/AcceptanceTestController/getCurrentExposureAndGain', { params });
  return response.data;
}
