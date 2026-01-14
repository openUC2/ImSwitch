/**
 * API call to home A axis for acceptance testing
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Home A axis
 * @param {string} positionerName - Name of the positioner (optional)
 * @returns {Promise} Response with status and message
 */
export default async function apiAcceptanceTestControllerHomeAxisA(positionerName = null) {
  const axiosInstance = createAxiosInstance();
  const params = positionerName ? { positionerName } : {};
  
  const response = await axiosInstance.get('/AcceptanceTestController/homeAxisA', { params });
  return response.data;
}
