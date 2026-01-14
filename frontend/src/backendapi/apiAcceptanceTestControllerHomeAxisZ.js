/**
 * API call to home Z axis for acceptance testing
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Home Z axis
 * @param {string} positionerName - Name of the positioner (optional)
 * @returns {Promise} Response with status and message
 */
export default async function apiAcceptanceTestControllerHomeAxisZ(positionerName = null) {
  const axiosInstance = createAxiosInstance();
  const params = positionerName ? { positionerName } : {};
  
  const response = await axiosInstance.get('/AcceptanceTestController/homeAxisZ', { params });
  return response.data;
}
