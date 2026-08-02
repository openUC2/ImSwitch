import createAxiosInstance from './createAxiosInstance';

/**
 * Change the I2C sensor poll period (seconds).
 * @param {number} period Poll period in seconds (min 0.2)
 * @returns {Promise} { period }
 */
export default async function apiI2CSensorControllerSetPollPeriod(period) {
  const axios = createAxiosInstance();
  try {
    const response = await axios.get('/I2CSensorController/setI2CSensorPollPeriod', {
      params: { period },
    });
    return response.data;
  } catch (error) {
    console.error('Failed to set I2C sensor poll period:', error);
    throw error;
  }
}
