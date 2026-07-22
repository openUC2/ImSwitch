import createAxiosInstance from './createAxiosInstance';

/**
 * Get the current I2C sensor polling status and configuration.
 * @returns {Promise} { running, period, bufferSize, available, enableSHT45, enableTSL2591, csvPath }
 */
export default async function apiI2CSensorControllerGetStatus() {
  const axios = createAxiosInstance();
  try {
    const response = await axios.get('/I2CSensorController/getI2CSensorStatus');
    return response.data;
  } catch (error) {
    console.error('Failed to get I2C sensor status:', error);
    throw error;
  }
}
