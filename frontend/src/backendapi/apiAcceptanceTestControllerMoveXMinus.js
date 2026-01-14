/**
 * API call to move in negative X direction
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Move stage in negative X direction
 * @param {string} positionerName - Name of the positioner (optional)
 * @param {number} distance - Distance to move in micrometers
 * @returns {Promise} Response with status and movement info
 */
export default async function apiAcceptanceTestControllerMoveXMinus(
  positionerName = null,
  distance = 1000
) {
  const axiosInstance = createAxiosInstance();
  const params = { distance };
  if (positionerName) params.positionerName = positionerName;
  
  const response = await axiosInstance.get('/AcceptanceTestController/moveXMinus', { params });
  return response.data;
}
