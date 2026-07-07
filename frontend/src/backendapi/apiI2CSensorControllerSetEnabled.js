import createAxiosInstance from './createAxiosInstance';

/**
 * Enable/disable individual sensors.
 * @param {boolean} [sht45]
 * @param {boolean} [tsl2591]
 * @returns {Promise} { enableSHT45, enableTSL2591 }
 */
export default async function apiI2CSensorControllerSetEnabled(sht45, tsl2591) {
  const axios = createAxiosInstance();
  try {
    const params = {};
    if (sht45 !== undefined && sht45 !== null) params.sht45 = sht45;
    if (tsl2591 !== undefined && tsl2591 !== null) params.tsl2591 = tsl2591;
    const response = await axios.get('/I2CSensorController/setI2CSensorEnabled', { params });
    return response.data;
  } catch (error) {
    console.error('Failed to set I2C sensor enabled flags:', error);
    throw error;
  }
}
