import createAxiosInstance from './createAxiosInstance';

/**
 * Stop continuous I2C sensor polling.
 * @returns {Promise} { running }
 */
export default async function apiI2CSensorControllerStopPolling() {
  const axios = createAxiosInstance();
  try {
    const response = await axios.get('/I2CSensorController/stopI2CSensorPolling');
    return response.data;
  } catch (error) {
    console.error('Failed to stop I2C sensor polling:', error);
    throw error;
  }
}
