/**
 * API call to turn on a light source
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Turn on a specific light source
 * @param {string} laserName - Name of the laser/light source
 * @returns {Promise} Response with status
 */
export default async function apiAcceptanceTestControllerTurnOnLight(laserName) {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/AcceptanceTestController/turnOnLight', {
    params: { laserName }
  });
  return response.data;
}
