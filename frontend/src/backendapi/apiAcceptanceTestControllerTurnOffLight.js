/**
 * API call to turn off a light source
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Turn off a specific light source
 * @param {string} laserName - Name of the laser/light source
 * @returns {Promise} Response with status
 */
export default async function apiAcceptanceTestControllerTurnOffLight(laserName) {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/AcceptanceTestController/turnOffLight', {
    params: { laserName }
  });
  return response.data;
}
