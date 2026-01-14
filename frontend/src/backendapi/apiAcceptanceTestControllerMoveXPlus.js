/**
 * API call to move in positive X direction
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Move stage in positive X direction
 * @param {string} positionerName - Name of the positioner (optional)
 * @param {number} distance - Distance to move in micrometers
 * @returns {Promise} Response with status and movement info
 */
export default async function apiAcceptanceTestControllerMoveXPlus(
  positionerName = null,
  distance = 1000
) {
  const axiosInstance = createAxiosInstance();
  const params = { distance };
  if (positionerName) params.positionerName = positionerName;
  
  const response = await axiosInstance.get('/AcceptanceTestController/moveXPlus', { params });
  return response.data;
}
