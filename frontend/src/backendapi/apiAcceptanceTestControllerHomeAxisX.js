/**
 * API call to home X axis for acceptance testing
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Home X axis
 * @param {string} positionerName - Name of the positioner (optional)
 * @returns {Promise} Response with status and message
 */
export default async function apiAcceptanceTestControllerHomeAxisX(positionerName = null) {
  const axiosInstance = createAxiosInstance();
  const params = positionerName ? { positionerName } : {};
  
  const response = await axiosInstance.get('/AcceptanceTestController/homeAxisX', { params });
  return response.data;
}
