import createAxiosInstance from './createAxiosInstance';

/**
 * One-shot read of all enabled sensors (works whether or not polling is on).
 * @returns {Promise} record { timestamp, datetime, temperature_c, humidity_pct, lux, ch0_full, ch1_ir, ok }
 */
export default async function apiI2CSensorControllerGetLatest() {
  const axios = createAxiosInstance();
  try {
    const response = await axios.get('/I2CSensorController/getLatestI2CSensorValues');
    return response.data;
  } catch (error) {
    console.error('Failed to get latest I2C sensor values:', error);
    throw error;
  }
}
