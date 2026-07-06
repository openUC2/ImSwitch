import createAxiosInstance from './createAxiosInstance';

/**
 * Start continuous I2C sensor polling. Optionally set the period (seconds).
 * @param {number} [period] Poll period in seconds
 * @returns {Promise} { running, period }
 */
export default async function apiI2CSensorControllerStartPolling(period) {
  const axios = createAxiosInstance();
  try {
    const params = period !== undefined && period !== null ? { period } : {};
    const response = await axios.get('/I2CSensorController/startI2CSensorPolling', { params });
    return response.data;
  } catch (error) {
    console.error('Failed to start I2C sensor polling:', error);
    throw error;
  }
}
