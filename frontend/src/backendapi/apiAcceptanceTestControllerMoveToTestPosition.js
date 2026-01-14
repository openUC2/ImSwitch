/**
 * API call to move to a test position
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Move stage to test position
 * @param {string} positionerName - Name of the positioner (optional)
 * @param {number} x - X position in micrometers
 * @param {number} y - Y position in micrometers
 * @param {number} z - Z position in micrometers
 * @returns {Promise} Response with status and position info
 */
export default async function apiAcceptanceTestControllerMoveToTestPosition(
  positionerName = null,
  x = 2000,
  y = 2000,
  z = 2000
) {
  const axiosInstance = createAxiosInstance();
  const params = { x, y, z };
  if (positionerName) params.positionerName = positionerName;
  
  const response = await axiosInstance.get('/AcceptanceTestController/moveToTestPosition', { params });
  return response.data;
}
