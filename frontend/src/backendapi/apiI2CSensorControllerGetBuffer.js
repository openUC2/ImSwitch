import createAxiosInstance from './createAxiosInstance';

/**
 * Get the rolling window of recent sensor samples (to seed the chart on load).
 * @returns {Promise} { buffer: Array<record> }
 */
export default async function apiI2CSensorControllerGetBuffer() {
  const axios = createAxiosInstance();
  try {
    const response = await axios.get('/I2CSensorController/getI2CSensorBuffer');
    return response.data;
  } catch (error) {
    console.error('Failed to get I2C sensor buffer:', error);
    throw error;
  }
}
