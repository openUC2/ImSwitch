/**
 * API call to home Y axis for acceptance testing
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Home Y axis
 * @param {string} positionerName - Name of the positioner (optional)
 * @returns {Promise} Response with status and message
 */
export default async function apiAcceptanceTestControllerHomeAxisY(positionerName = null) {
  const axiosInstance = createAxiosInstance();
  const params = positionerName ? { positionerName } : {};
  
  const response = await axiosInstance.get('/AcceptanceTestController/homeAxisY', { params });
  return response.data;
}
