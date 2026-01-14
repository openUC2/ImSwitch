/**
 * API call to get available light sources
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Get list of available light sources
 * @returns {Promise} Response with list of light sources
 */
export default async function apiAcceptanceTestControllerGetAvailableLightSources() {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/AcceptanceTestController/getAvailableLightSources');
  return response.data;
}
